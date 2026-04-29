from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio.features import rasterize

from .config import RainfallConfig, SimulationConfig
from .preprocessing import IntermediateOutputs
from .simulation import RasterGrid, SimulationState

matplotlib.use("Agg")
LOGGER = logging.getLogger(__name__)
GRAVITY_METERS_PER_SECOND_SQUARED = 9.80665
DRY_DEPTH_METERS = 0.001


@dataclass(frozen=True)
class SimulationRunResult:
    """Files and timing produced by a simulation run."""

    snapshot_directory: Path
    snapshots: tuple[Path, ...]
    duration_seconds: float


def run_simulation(
    config: SimulationConfig,
    intermediates: IntermediateOutputs,
) -> SimulationRunResult:
    """Run the simulation loop and write PNG snapshots."""
    grid = RasterGrid.from_dem(intermediates.clipped_dem)
    state = SimulationState.dry(grid, manning_n=0.08, runoff_coefficient=1.0)
    storm_mask = storm_footprint_mask(config.inputs.storm_footprint, intermediates.clipped_dem)
    duration_seconds = rainfall_duration_seconds(config.rainfall)
    snapshot_interval_seconds = config.output.snapshots.interval_minutes * 60

    snapshot_directory = resolve_snapshot_directory(
        config.output.snapshots.directory,
        intermediates.workspace,
    )
    snapshot_directory.mkdir(parents=True, exist_ok=True)

    snapshots = [
        write_snapshot(
            grid,
            storm_mask,
            state,
            snapshot_directory,
            index=0,
        )
    ]
    next_snapshot_seconds = min(snapshot_interval_seconds, duration_seconds)
    with simulation_progress(duration_seconds) as progress:
        while state.hydraulic.time_seconds < duration_seconds:
            previous_time_seconds = state.hydraulic.time_seconds
            dt_seconds = timestep_duration_seconds(
                current_time_seconds=state.hydraulic.time_seconds,
                duration_seconds=duration_seconds,
                next_snapshot_seconds=next_snapshot_seconds,
                nominal_time_step_seconds=config.time_step.seconds,
            )
            run_timestep(
                state,
                config.rainfall,
                storm_mask,
                dt_seconds,
            )
            progress.update(state.hydraulic.time_seconds - previous_time_seconds)

            if state.hydraulic.time_seconds >= next_snapshot_seconds:
                snapshots.append(
                    write_snapshot(
                        grid,
                        storm_mask,
                        state,
                        snapshot_directory,
                        index=len(snapshots),
                    )
                )
                progress.set_postfix(snapshots=len(snapshots))
                next_snapshot_seconds = min(
                    next_snapshot_seconds + snapshot_interval_seconds,
                    duration_seconds,
                )

    LOGGER.info("Wrote %s simulation snapshots to %s", len(snapshots), snapshot_directory)
    return SimulationRunResult(
        snapshot_directory=snapshot_directory,
        snapshots=tuple(snapshots),
        duration_seconds=state.hydraulic.time_seconds,
    )


def resolve_snapshot_directory(snapshot_directory: Path, workspace: Path) -> Path:
    """Resolve relative snapshot directories beneath the run workspace."""
    if snapshot_directory.is_absolute():
        return snapshot_directory
    return workspace / snapshot_directory


@contextmanager
def simulation_progress(duration_seconds: float) -> Iterator[object]:
    """Yield a model-time progress bar for long-running simulation loops."""
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("tqdm is required for simulation progress reporting.") from exc

    with tqdm(
        total=duration_seconds,
        desc="Simulating",
        unit="model s",
        disable=not sys.stderr.isatty(),
    ) as progress:
        yield progress


def rainfall_duration_seconds(rainfall: RainfallConfig) -> float:
    """Return the duration implied by the rainfall series."""
    return max(point.time_minutes for point in rainfall.series) * 60


def timestep_duration_seconds(
    current_time_seconds: float,
    duration_seconds: float,
    next_snapshot_seconds: float,
    nominal_time_step_seconds: float,
) -> float:
    """Return the next timestep without stepping past a snapshot or event end."""
    return min(
        nominal_time_step_seconds,
        duration_seconds - current_time_seconds,
        next_snapshot_seconds - current_time_seconds,
    )


def storm_footprint_mask(storm_footprint: Path, template_raster: Path) -> np.ndarray:
    """Rasterize the storm footprint onto the clipped DEM grid."""
    storm = gpd.read_file(storm_footprint)
    if storm.empty:
        raise RuntimeError("Storm footprint does not contain any features.")

    with rasterio.open(template_raster) as template:
        storm = storm.to_crs(template.crs)
        geometries = [geometry for geometry in storm.geometry if geometry is not None and not geometry.is_empty]
        if not geometries:
            raise RuntimeError("Storm footprint does not contain any valid geometries.")

        mask = rasterize(
            ((geometry, 1) for geometry in geometries),
            out_shape=(template.height, template.width),
            transform=template.transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)
        valid_cells = template.read_masks(1) > 0

    mask &= valid_cells
    if not mask.any():
        raise RuntimeError("Storm footprint does not overlap the clipped DEM.")
    return mask


def run_timestep(
    state: SimulationState,
    rainfall: RainfallConfig,
    storm_mask: np.ndarray,
    dt_seconds: float,
) -> None:
    """Advance the full model by one timestep."""
    timestep_start_seconds = state.hydraulic.time_seconds
    apply_rainfall_forcing(state, rainfall, storm_mask, timestep_start_seconds, dt_seconds)
    water_timestep(state, dt_seconds)
    state.hydraulic.time_seconds += dt_seconds
    update_diagnostics(state, dt_seconds)


def apply_rainfall_forcing(
    state: SimulationState,
    rainfall: RainfallConfig,
    storm_mask: np.ndarray,
    timestep_start_seconds: float,
    dt_seconds: float,
) -> None:
    """Add rainfall depth for one timestep when the event is active."""
    rate_m_per_second = average_rainfall_rate_m_per_second(
        rainfall,
        timestep_start_seconds,
        dt_seconds,
    )
    if rate_m_per_second <= 0:
        return

    add_rainfall_depth(state, storm_mask, rate_m_per_second, dt_seconds)


def average_rainfall_rate_m_per_second(
    rainfall: RainfallConfig,
    timestep_start_seconds: float,
    dt_seconds: float,
) -> float:
    """Return the mean rainfall rate across a timestep."""
    start_rate = rainfall_rate_m_per_second(rainfall, timestep_start_seconds)
    end_rate = rainfall_rate_m_per_second(rainfall, timestep_start_seconds + dt_seconds)
    return 0.5 * (start_rate + end_rate)


def rainfall_rate_m_per_second(rainfall: RainfallConfig, elapsed_seconds: float) -> float:
    """Linearly interpolate rainfall intensity and return meters per second."""
    elapsed_minutes = elapsed_seconds / 60
    points = rainfall.series

    if elapsed_minutes <= points[0].time_minutes:
        return points[0].rate_mm_per_hr / 1000 / 3600

    for previous, current in zip(points, points[1:]):
        if elapsed_minutes <= current.time_minutes:
            elapsed_fraction = (
                (elapsed_minutes - previous.time_minutes)
                / (current.time_minutes - previous.time_minutes)
            )
            rate_mm_per_hr = previous.rate_mm_per_hr + (
                (current.rate_mm_per_hr - previous.rate_mm_per_hr) * elapsed_fraction
            )
            return rate_mm_per_hr / 1000 / 3600

    return points[-1].rate_mm_per_hr / 1000 / 3600


def add_rainfall_depth(
    state: SimulationState,
    storm_mask: np.ndarray,
    rate_m_per_second: float,
    dt_seconds: float,
) -> None:
    """Add rainfall depth to valid cells covered by the storm footprint."""
    receiving_cells = storm_mask & state.grid.valid_cells
    depth_delta = rate_m_per_second * dt_seconds
    state.hydraulic.depth[receiving_cells] += (
        depth_delta * state.surface.runoff_coefficient[receiving_cells]
    )


def water_timestep(state: SimulationState, dt_seconds: float) -> None:
    """Advance water depth with vectorized local-inertial face fluxes."""
    update_face_fluxes(state, dt_seconds)
    limit_outgoing_fluxes(state, dt_seconds)
    update_depth_from_fluxes(state, dt_seconds)


def update_face_fluxes(state: SimulationState, dt_seconds: float) -> None:
    """Update interior face fluxes from water-surface slope and Manning friction."""
    grid = state.grid
    hydraulic = state.hydraulic
    surface = state.surface
    eta = hydraulic.water_surface(grid)

    qx_old = hydraulic.qx[:, 1:-1]
    # Face depth uses the higher adjacent terrain cell as the sill elevation.
    h_face_x = np.maximum(
        0,
        np.maximum(eta[:, :-1], eta[:, 1:])
        - np.maximum(grid.elevation[:, :-1], grid.elevation[:, 1:]),
    )
    valid_x = grid.valid_cells[:, :-1] & grid.valid_cells[:, 1:]
    slope_x = (eta[:, 1:] - eta[:, :-1]) / grid.dx
    n_face_x = 0.5 * (surface.manning_n[:, :-1] + surface.manning_n[:, 1:])
    hydraulic.qx[:, 1:-1] = local_inertial_flux_update(
        old_flux=qx_old,
        face_depth=h_face_x,
        slope=slope_x,
        manning_n=n_face_x,
        valid_faces=valid_x,
        dt_seconds=dt_seconds,
    )

    # Open boundaries are dry outside cells; outward flux leaves the domain.
    left_boundary_flux = local_inertial_flux_update(
        old_flux=hydraulic.qx[:, 0],
        face_depth=hydraulic.depth[:, 0],
        slope=(eta[:, 0] - grid.elevation[:, 0]) / grid.dx,
        manning_n=surface.manning_n[:, 0],
        valid_faces=grid.valid_cells[:, 0],
        dt_seconds=dt_seconds,
    )
    hydraulic.qx[:, 0] = np.minimum(left_boundary_flux, 0)
    right_boundary_flux = local_inertial_flux_update(
        old_flux=hydraulic.qx[:, -1],
        face_depth=hydraulic.depth[:, -1],
        slope=(grid.elevation[:, -1] - eta[:, -1]) / grid.dx,
        manning_n=surface.manning_n[:, -1],
        valid_faces=grid.valid_cells[:, -1],
        dt_seconds=dt_seconds,
    )
    hydraulic.qx[:, -1] = np.maximum(right_boundary_flux, 0)

    qy_old = hydraulic.qy[1:-1, :]
    # Face depth uses the higher adjacent terrain cell as the sill elevation.
    h_face_y = np.maximum(
        0,
        np.maximum(eta[:-1, :], eta[1:, :])
        - np.maximum(grid.elevation[:-1, :], grid.elevation[1:, :]),
    )
    valid_y = grid.valid_cells[:-1, :] & grid.valid_cells[1:, :]
    slope_y = (eta[1:, :] - eta[:-1, :]) / grid.dy
    n_face_y = 0.5 * (surface.manning_n[:-1, :] + surface.manning_n[1:, :])
    hydraulic.qy[1:-1, :] = local_inertial_flux_update(
        old_flux=qy_old,
        face_depth=h_face_y,
        slope=slope_y,
        manning_n=n_face_y,
        valid_faces=valid_y,
        dt_seconds=dt_seconds,
    )

    # Open boundaries are dry outside cells; outward flux leaves the domain.
    top_boundary_flux = local_inertial_flux_update(
        old_flux=hydraulic.qy[0, :],
        face_depth=hydraulic.depth[0, :],
        slope=(eta[0, :] - grid.elevation[0, :]) / grid.dy,
        manning_n=surface.manning_n[0, :],
        valid_faces=grid.valid_cells[0, :],
        dt_seconds=dt_seconds,
    )
    hydraulic.qy[0, :] = np.minimum(top_boundary_flux, 0)
    bottom_boundary_flux = local_inertial_flux_update(
        old_flux=hydraulic.qy[-1, :],
        face_depth=hydraulic.depth[-1, :],
        slope=(grid.elevation[-1, :] - eta[-1, :]) / grid.dy,
        manning_n=surface.manning_n[-1, :],
        valid_faces=grid.valid_cells[-1, :],
        dt_seconds=dt_seconds,
    )
    hydraulic.qy[-1, :] = np.maximum(bottom_boundary_flux, 0)


def local_inertial_flux_update(
    *,
    old_flux: np.ndarray,
    face_depth: np.ndarray,
    slope: np.ndarray,
    manning_n: np.ndarray,
    valid_faces: np.ndarray,
    dt_seconds: float,
) -> np.ndarray:
    """Return the semi-implicit local-inertial flux update for face arrays."""
    depth_safe = np.maximum(face_depth, DRY_DEPTH_METERS)
    denominator = (
        1
        + GRAVITY_METERS_PER_SECOND_SQUARED
        * dt_seconds
        * manning_n**2
        * np.abs(old_flux)
        / depth_safe ** (7 / 3)
    )
    flux = (
        old_flux
        - GRAVITY_METERS_PER_SECOND_SQUARED * face_depth * dt_seconds * slope
    ) / denominator
    return np.where(valid_faces & (face_depth >= DRY_DEPTH_METERS), flux, 0)


def limit_outgoing_fluxes(state: SimulationState, dt_seconds: float) -> None:
    """Scale outgoing face fluxes so no cell can lose more water than it stores."""
    depth = state.hydraulic.depth
    qx = state.hydraulic.qx
    qy = state.hydraulic.qy
    valid_cells = state.grid.valid_cells

    outgoing_depth = dt_seconds * (
        np.maximum(qx[:, 1:], 0) / state.grid.dx
        + np.maximum(-qx[:, :-1], 0) / state.grid.dx
        + np.maximum(qy[1:, :], 0) / state.grid.dy
        + np.maximum(-qy[:-1, :], 0) / state.grid.dy
    )
    scale = np.ones_like(depth)
    needs_limit = valid_cells & (outgoing_depth > depth) & (outgoing_depth > 0)
    scale[needs_limit] = depth[needs_limit] / outgoing_depth[needs_limit]

    qx[:, 1:-1] *= np.where(qx[:, 1:-1] >= 0, scale[:, :-1], scale[:, 1:])
    qx[:, 0] *= np.where(qx[:, 0] >= 0, 1, scale[:, 0])
    qx[:, -1] *= np.where(qx[:, -1] >= 0, scale[:, -1], 1)

    qy[1:-1, :] *= np.where(qy[1:-1, :] >= 0, scale[:-1, :], scale[1:, :])
    qy[0, :] *= np.where(qy[0, :] >= 0, 1, scale[0, :])
    qy[-1, :] *= np.where(qy[-1, :] >= 0, scale[-1, :], 1)


def update_depth_from_fluxes(state: SimulationState, dt_seconds: float) -> None:
    """Update cell depths from vectorized face-flux divergence."""
    depth = state.hydraulic.depth
    qx = state.hydraulic.qx
    qy = state.hydraulic.qy
    depth += dt_seconds * (
        (qx[:, :-1] - qx[:, 1:]) / state.grid.dx
        + (qy[:-1, :] - qy[1:, :]) / state.grid.dy
    )
    np.maximum(depth, 0, out=depth)
    depth[~state.grid.valid_cells] = 0


def update_diagnostics(state: SimulationState, dt_seconds: float) -> None:
    """Update diagnostic rasters after a timestep has been applied."""
    state.diagnostics.max_depth = np.maximum(
        state.diagnostics.max_depth,
        state.hydraulic.depth,
    )
    wet_cells = state.hydraulic.depth > 0
    newly_wet = np.isnan(state.diagnostics.arrival_time) & wet_cells
    state.diagnostics.arrival_time[newly_wet] = state.hydraulic.time_seconds
    state.diagnostics.flood_duration[wet_cells] += dt_seconds


def write_snapshot(
    grid: RasterGrid,
    storm_mask: np.ndarray,
    state: SimulationState,
    snapshot_directory: Path,
    index: int,
) -> Path:
    """Write one simulation snapshot and return its path."""
    snapshot = snapshot_directory / f"snapshot_{index:04d}.png"
    render_snapshot(
        grid,
        storm_mask,
        state.hydraulic.depth,
        snapshot,
        elapsed_minutes=state.hydraulic.time_seconds / 60,
    )
    return snapshot


def render_snapshot(
    grid: RasterGrid,
    storm_mask: np.ndarray,
    depth: np.ndarray,
    path: Path,
    elapsed_minutes: float,
) -> None:
    """Render DEM, storm footprint, and water depth into a PNG image."""
    figure, axis = plt.subplots(figsize=(8, 6), dpi=150)
    axis.imshow(np.ma.masked_invalid(grid.elevation), cmap="gray")
    axis.imshow(np.where(storm_mask, 1.0, np.nan), cmap="Blues", alpha=0.22, vmin=0, vmax=1)

    depth_layer = np.ma.masked_where(depth <= 0, depth)
    water = axis.imshow(depth_layer, cmap="turbo", alpha=0.75)
    if np.nanmax(depth) > 0:
        colorbar = figure.colorbar(water, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Water depth (m)")

    axis.set_title(f"Simulated inundation at {elapsed_minutes:.1f} minutes")
    axis.axis("off")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
