from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.features import rasterize

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .config import RainfallConfig, SimulationConfig
from .preprocessing import IntermediateOutputs
from .simulation import RasterGrid, SimulationState

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RainfallSimulationResult:
    """Files produced by the rainfall-only simulation pass."""

    snapshot_directory: Path
    snapshots: tuple[Path, ...]
    duration_seconds: float


def run_rainfall_only_simulation(
    config: SimulationConfig,
    intermediates: IntermediateOutputs,
) -> RainfallSimulationResult:
    """Run a bucket-style rainfall simulation and write PNG snapshots."""
    grid = RasterGrid.from_dem(intermediates.clipped_dem)
    state = SimulationState.dry(grid, manning_n=0.08, runoff_coefficient=1.0)
    storm_mask = storm_footprint_mask(config.inputs.storm_footprint, intermediates.clipped_dem)

    snapshot_directory = resolve_snapshot_directory(
        config.output.snapshots.directory,
        intermediates.workspace,
    )
    snapshot_directory.mkdir(parents=True, exist_ok=True)

    snapshots: list[Path] = []
    for index, snapshot_time in enumerate(
        snapshot_times_seconds(config.rainfall, config.output.snapshots.interval_minutes)
    ):
        advance_rainfall_only_state(
            state,
            config.rainfall,
            storm_mask,
            snapshot_time,
            config.time_step.seconds,
        )
        snapshot = snapshot_directory / f"snapshot_{index:04d}.png"
        render_snapshot(
            grid,
            storm_mask,
            state.hydraulic.depth,
            snapshot,
            elapsed_minutes=state.hydraulic.time_seconds / 60,
        )
        snapshots.append(snapshot)

    LOGGER.info("Wrote %s rainfall snapshots to %s", len(snapshots), snapshot_directory)
    return RainfallSimulationResult(
        snapshot_directory=snapshot_directory,
        snapshots=tuple(snapshots),
        duration_seconds=state.hydraulic.time_seconds,
    )


def resolve_snapshot_directory(snapshot_directory: Path, workspace: Path) -> Path:
    """Resolve relative snapshot directories beneath the run workspace."""
    if snapshot_directory.is_absolute():
        return snapshot_directory
    return workspace / snapshot_directory


def snapshot_times_seconds(rainfall: RainfallConfig, interval_minutes: float) -> tuple[float, ...]:
    """Return the elapsed model times at which snapshots should be written."""
    duration_seconds = max(point.time_minutes for point in rainfall.series) * 60
    interval_seconds = interval_minutes * 60
    times = list(np.arange(0, duration_seconds + interval_seconds, interval_seconds))
    times = [time for time in times if time <= duration_seconds]
    if not times or times[-1] < duration_seconds:
        times.append(duration_seconds)
    return tuple(float(time) for time in times)


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


def advance_rainfall_only_state(
    state: SimulationState,
    rainfall: RainfallConfig,
    storm_mask: np.ndarray,
    target_time_seconds: float,
    time_step_seconds: float,
) -> None:
    """Advance state to a target time by adding rain only under the storm mask."""
    while state.hydraulic.time_seconds < target_time_seconds:
        dt_seconds = min(time_step_seconds, target_time_seconds - state.hydraulic.time_seconds)
        rate_m_per_second = rainfall_rate_m_per_second(rainfall, state.hydraulic.time_seconds)
        add_rainfall_depth(state, storm_mask, rate_m_per_second, dt_seconds)


def rainfall_rate_m_per_second(rainfall: RainfallConfig, elapsed_seconds: float) -> float:
    """Linearly interpolate rainfall intensity and return meters per second."""
    times = np.array([point.time_minutes for point in rainfall.series], dtype=np.float64)
    rates = np.array([point.rate_mm_per_hr for point in rainfall.series], dtype=np.float64)
    elapsed_minutes = elapsed_seconds / 60
    return float(np.interp(elapsed_minutes, times, rates) / 1000 / 3600)


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
    state.hydraulic.time_seconds += dt_seconds
    state.diagnostics.max_depth = np.maximum(
        state.diagnostics.max_depth,
        state.hydraulic.depth,
    )
    wet_cells = state.hydraulic.depth > 0
    newly_wet = np.isnan(state.diagnostics.arrival_time) & wet_cells
    state.diagnostics.arrival_time[newly_wet] = state.hydraulic.time_seconds
    state.diagnostics.flood_duration[wet_cells] += dt_seconds


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

    axis.set_title(f"Rainfall-only inundation at {elapsed_minutes:.1f} minutes")
    axis.axis("off")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
