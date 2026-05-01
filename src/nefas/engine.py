import logging
import sys
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.features import rasterize
from line_profiler import profile
from numba import boolean, float64, njit, prange, vectorize

from .config import RainfallConfig, SimulationConfig
from .preprocessing import IntermediateOutputs
from .simulation import RasterGrid, SimulationState


matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt

LOGGER = logging.getLogger(__name__)
GRAVITY_METERS_PER_SECOND_SQUARED = 9.80665
DRY_DEPTH_METERS = 0.001
MAX_PENDING_SNAPSHOT_RENDERS = 10
WATER_DEPTH_ALPHA = 0.82
WATER_DEPTH_COLORMAP = LinearSegmentedColormap.from_list(
    "nefas_water_depth",
    ("#d8fbff", "#86d4ff", "#1f8be3", "#08306b"),
)
WATER_DEPTH_COLORMAP.set_bad((0, 0, 0, 0))

_SNAPSHOT_RENDER_GRID: RasterGrid | None = None
_SNAPSHOT_RENDER_STORM_MASK: np.ndarray | None = None


@dataclass(frozen=True)
class SimulationRunResult:
    """Files and timing produced by a simulation run."""

    snapshot_directory: Path
    snapshots: tuple[Path, ...]
    duration_seconds: float


@dataclass
class SnapshotRenderQueue:
    """Bounded asynchronous snapshot renderer."""

    executor: ProcessPoolExecutor
    pending: list[Future[Path]]

    def submit(
        self,
        *,
        depth: np.ndarray,
        path: Path,
        elapsed_minutes: float,
        max_depth_meters: float | None,
    ) -> Path:
        """Queue a render job and apply backpressure when too many are pending."""
        while len(self.pending) >= MAX_PENDING_SNAPSHOT_RENDERS:
            self.pending.pop(0).result()

        self.pending.append(
            self.executor.submit(
                render_snapshot_from_context,
                depth.copy(),
                path,
                elapsed_minutes,
                max_depth_meters,
            )
        )
        return path

    def wait(self) -> None:
        """Wait for all queued render jobs, surfacing worker exceptions."""
        while self.pending:
            self.pending.pop(0).result()


def run_simulation(
    config: SimulationConfig,
    intermediates: IntermediateOutputs,
) -> SimulationRunResult:
    """Run the simulation loop and write PNG snapshots."""
    grid = RasterGrid.from_dem(intermediates.clipped_dem)
    state = SimulationState.dry(grid, manning_n=0.08, runoff_coefficient=1.0)
    storm_mask = storm_footprint_mask(
        config.inputs.storm_footprint, intermediates.clipped_dem
    )
    duration_seconds = simulation_duration_seconds(config)
    snapshot_interval_seconds = config.output.snapshots.interval_minutes * 60

    snapshot_directory = resolve_snapshot_directory(
        config.output.snapshots.directory,
        intermediates.workspace,
    )
    snapshot_directory.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(
        max_workers=1,
        initializer=initialize_snapshot_renderer,
        initargs=(
            grid.elevation,
            grid.dx,
            grid.dy,
            grid.valid_cells,
            storm_mask,
        ),
    ) as executor:
        render_queue = SnapshotRenderQueue(executor=executor, pending=[])
        snapshots = [
            queue_snapshot(
                render_queue,
                state,
                snapshot_directory,
                index=0,
                max_depth_meters=config.output.snapshots.max_depth_meters,
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
                    nominal_time_step_seconds=effective_time_step_seconds(config),
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
                        queue_snapshot(
                            render_queue,
                            state,
                            snapshot_directory,
                            index=len(snapshots),
                            max_depth_meters=config.output.snapshots.max_depth_meters,
                        )
                    )
                    progress.set_postfix(snapshots=len(snapshots))
                    next_snapshot_seconds = min(
                        next_snapshot_seconds + snapshot_interval_seconds,
                        duration_seconds,
                    )

        render_queue.wait()

    LOGGER.info(
        "Wrote %s simulation snapshots to %s",
        len(snapshots),
        snapshot_directory,
    )
    return SimulationRunResult(
        snapshot_directory=snapshot_directory,
        snapshots=tuple(snapshots),
        duration_seconds=state.hydraulic.time_seconds,
    )


def initialize_snapshot_renderer(
    elevation: np.ndarray,
    dx: float,
    dy: float,
    valid_cells: np.ndarray,
    storm_mask: np.ndarray,
) -> None:
    """Initialize read-only render context inside a worker process."""
    global _SNAPSHOT_RENDER_GRID, _SNAPSHOT_RENDER_STORM_MASK
    _SNAPSHOT_RENDER_GRID = RasterGrid(
        elevation=elevation,
        dx=dx,
        dy=dy,
        valid_cells=valid_cells,
    )
    _SNAPSHOT_RENDER_STORM_MASK = storm_mask


def render_snapshot_from_context(
    depth: np.ndarray,
    path: Path,
    elapsed_minutes: float,
    max_depth_meters: float | None,
) -> Path:
    """Render a queued snapshot using worker-local static context."""
    if _SNAPSHOT_RENDER_GRID is None or _SNAPSHOT_RENDER_STORM_MASK is None:
        raise RuntimeError("Snapshot renderer has not been initialized.")

    render_snapshot(
        _SNAPSHOT_RENDER_GRID,
        _SNAPSHOT_RENDER_STORM_MASK,
        depth,
        path,
        elapsed_minutes=elapsed_minutes,
        max_depth_meters=max_depth_meters,
    )
    return path


def resolve_snapshot_directory(snapshot_directory: Path, workspace: Path) -> Path:
    """Resolve relative snapshot directories beneath the run workspace."""
    if snapshot_directory.is_absolute():
        return snapshot_directory
    return workspace / snapshot_directory


@contextmanager
def simulation_progress(duration_seconds: float) -> Iterator[object]:
    """Yield a model-time progress bar for long-running simulation loops."""
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


def simulation_duration_seconds(config: SimulationConfig) -> float:
    """Return the configured simulation duration or the rainfall event duration."""
    if config.simulation_time.total_runtime_seconds is not None:
        return config.simulation_time.total_runtime_seconds
    return rainfall_duration_seconds(config.rainfall)


def effective_time_step_seconds(config: SimulationConfig) -> float:
    """Return the timestep duration after applying the optional maximum cap."""
    time_step_seconds = config.simulation_time.time_step_seconds
    max_time_step_seconds = config.simulation_time.max_time_step_seconds
    if max_time_step_seconds is None:
        return time_step_seconds
    return min(time_step_seconds, max_time_step_seconds)


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
        geometries = [
            geometry
            for geometry in storm.geometry
            if geometry is not None and not geometry.is_empty
        ]
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
    apply_rainfall_forcing(
        state, rainfall, storm_mask, timestep_start_seconds, dt_seconds
    )
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


def rainfall_rate_m_per_second(
    rainfall: RainfallConfig, elapsed_seconds: float
) -> float:
    """Linearly interpolate rainfall intensity and return meters per second."""
    elapsed_minutes = elapsed_seconds / 60
    points = rainfall.series

    if elapsed_minutes <= points[0].time_minutes:
        return points[0].rate_mm_per_hr / 1000 / 3600

    for previous, current in zip(points, points[1:]):
        if elapsed_minutes <= current.time_minutes:
            elapsed_fraction = (elapsed_minutes - previous.time_minutes) / (
                current.time_minutes - previous.time_minutes
            )
            rate_mm_per_hr = previous.rate_mm_per_hr + (
                (current.rate_mm_per_hr - previous.rate_mm_per_hr) * elapsed_fraction
            )
            return rate_mm_per_hr / 1000 / 3600

    return 0


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


@profile
def water_timestep(state: SimulationState, dt_seconds: float) -> None:
    """Advance water depth with vectorized local-inertial face fluxes."""
    update_face_fluxes(state, dt_seconds)
    limit_outgoing_fluxes(
        state.hydraulic.depth,
        state.hydraulic.qx,
        state.hydraulic.qy,
        state.grid.valid_cells,
        state.grid.dx,
        state.grid.dy,
        dt_seconds,
    )
    update_depth_from_fluxes(state, dt_seconds)


@profile
def update_face_fluxes(state: SimulationState, dt_seconds: float) -> None:
    """Update interior face fluxes from water-surface slope and Manning friction."""
    grid = state.grid
    hydraulic = state.hydraulic
    surface = state.surface
    eta = hydraulic.water_surface(grid)

    update_interior_x_fluxes_numba(
        hydraulic.qx,
        eta,
        grid.elevation,
        surface.manning_n,
        grid.valid_cells,
        grid.dx,
        dt_seconds,
    )
    update_x_boundary_fluxes_numba(
        hydraulic.qx,
        hydraulic.depth,
        eta,
        grid.elevation,
        surface.manning_n,
        grid.valid_cells,
        grid.dx,
        dt_seconds,
    )

    update_interior_y_fluxes_numba(
        hydraulic.qy,
        eta,
        grid.elevation,
        surface.manning_n,
        grid.valid_cells,
        grid.dy,
        dt_seconds,
    )
    update_y_boundary_fluxes_numba(
        hydraulic.qy,
        hydraulic.depth,
        eta,
        grid.elevation,
        surface.manning_n,
        grid.valid_cells,
        grid.dy,
        dt_seconds,
    )


@njit(
    float64(float64, float64, float64, float64, boolean, float64),
    cache=True,
    inline="always",
)
def local_inertial_flux_update_value(
    old_flux: float,
    face_depth: float,
    slope: float,
    manning_n: float,
    valid_face: bool,
    dt_seconds: float,
) -> float:
    """Return one local-inertial flux update value."""
    if not valid_face or face_depth < DRY_DEPTH_METERS:
        return 0.0

    depth_safe = max(face_depth, DRY_DEPTH_METERS)
    denominator = (
        1.0
        + GRAVITY_METERS_PER_SECOND_SQUARED
        * dt_seconds
        * manning_n**2
        * abs(old_flux)
        / depth_safe ** (7.0 / 3.0)
    )
    return (
        old_flux - GRAVITY_METERS_PER_SECOND_SQUARED * face_depth * dt_seconds * slope
    ) / denominator


@njit(cache=True, parallel=True)
def update_interior_x_fluxes_numba(
    qx: np.ndarray,
    eta: np.ndarray,
    elevation: np.ndarray,
    manning_n: np.ndarray,
    valid_cells: np.ndarray,
    dx: float,
    dt_seconds: float,
) -> None:
    """Update east-west interior face fluxes in one compiled pass."""
    rows, cols = eta.shape
    for row in prange(rows):
        for col in range(cols - 1):
            face_depth = max(
                0.0,
                max(eta[row, col], eta[row, col + 1])
                - max(elevation[row, col], elevation[row, col + 1]),
            )
            slope = (eta[row, col + 1] - eta[row, col]) / dx
            face_manning_n = 0.5 * (
                manning_n[row, col] + manning_n[row, col + 1]
            )
            valid_face = valid_cells[row, col] and valid_cells[row, col + 1]
            qx[row, col + 1] = local_inertial_flux_update_value(
                qx[row, col + 1],
                face_depth,
                slope,
                face_manning_n,
                valid_face,
                dt_seconds,
            )


@njit(cache=True, parallel=True)
def update_interior_y_fluxes_numba(
    qy: np.ndarray,
    eta: np.ndarray,
    elevation: np.ndarray,
    manning_n: np.ndarray,
    valid_cells: np.ndarray,
    dy: float,
    dt_seconds: float,
) -> None:
    """Update north-south interior face fluxes in one compiled pass."""
    rows, cols = eta.shape
    for row in prange(rows - 1):
        for col in range(cols):
            face_depth = max(
                0.0,
                max(eta[row, col], eta[row + 1, col])
                - max(elevation[row, col], elevation[row + 1, col]),
            )
            slope = (eta[row + 1, col] - eta[row, col]) / dy
            face_manning_n = 0.5 * (
                manning_n[row, col] + manning_n[row + 1, col]
            )
            valid_face = valid_cells[row, col] and valid_cells[row + 1, col]
            qy[row + 1, col] = local_inertial_flux_update_value(
                qy[row + 1, col],
                face_depth,
                slope,
                face_manning_n,
                valid_face,
                dt_seconds,
            )


@njit(cache=True, parallel=True)
def update_x_boundary_fluxes_numba(
    qx: np.ndarray,
    depth: np.ndarray,
    eta: np.ndarray,
    elevation: np.ndarray,
    manning_n: np.ndarray,
    valid_cells: np.ndarray,
    dx: float,
    dt_seconds: float,
) -> None:
    """Update open east-west boundary fluxes in one compiled pass."""
    rows = eta.shape[0]
    right_col = eta.shape[1] - 1
    right_face = qx.shape[1] - 1
    for row in prange(rows):
        left_flux = local_inertial_flux_update_value(
            qx[row, 0],
            depth[row, 0],
            (eta[row, 0] - elevation[row, 0]) / dx,
            manning_n[row, 0],
            valid_cells[row, 0],
            dt_seconds,
        )
        qx[row, 0] = min(left_flux, 0.0)

        right_flux = local_inertial_flux_update_value(
            qx[row, right_face],
            depth[row, right_col],
            (elevation[row, right_col] - eta[row, right_col]) / dx,
            manning_n[row, right_col],
            valid_cells[row, right_col],
            dt_seconds,
        )
        qx[row, right_face] = max(right_flux, 0.0)


@njit(cache=True, parallel=True)
def update_y_boundary_fluxes_numba(
    qy: np.ndarray,
    depth: np.ndarray,
    eta: np.ndarray,
    elevation: np.ndarray,
    manning_n: np.ndarray,
    valid_cells: np.ndarray,
    dy: float,
    dt_seconds: float,
) -> None:
    """Update open north-south boundary fluxes in one compiled pass."""
    cols = eta.shape[1]
    bottom_row = eta.shape[0] - 1
    bottom_face = qy.shape[0] - 1
    for col in prange(cols):
        top_flux = local_inertial_flux_update_value(
            qy[0, col],
            depth[0, col],
            (eta[0, col] - elevation[0, col]) / dy,
            manning_n[0, col],
            valid_cells[0, col],
            dt_seconds,
        )
        qy[0, col] = min(top_flux, 0.0)

        bottom_flux = local_inertial_flux_update_value(
            qy[bottom_face, col],
            depth[bottom_row, col],
            (elevation[bottom_row, col] - eta[bottom_row, col]) / dy,
            manning_n[bottom_row, col],
            valid_cells[bottom_row, col],
            dt_seconds,
        )
        qy[bottom_face, col] = max(bottom_flux, 0.0)


@vectorize(
    [float64(float64, float64, float64, float64, boolean, float64)],
    nopython=True,
    target="parallel",
    cache=True,
)
def local_inertial_flux_update_numba(
    old_flux: float,
    face_depth: float,
    slope: float,
    manning_n: float,
    valid_face: bool,
    dt_seconds: float,
) -> float:
    """Return the compiled scalar local-inertial flux update for face arrays."""
    if not valid_face or face_depth < DRY_DEPTH_METERS:
        return 0.0

    depth_safe = max(face_depth, DRY_DEPTH_METERS)
    denominator = (
        1.0
        + GRAVITY_METERS_PER_SECOND_SQUARED
        * dt_seconds
        * manning_n**2
        * abs(old_flux)
        / depth_safe ** (7.0 / 3.0)
    )
    return (
        old_flux - GRAVITY_METERS_PER_SECOND_SQUARED * face_depth * dt_seconds * slope
    ) / denominator


@profile
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
        old_flux - GRAVITY_METERS_PER_SECOND_SQUARED * face_depth * dt_seconds * slope
    ) / denominator
    return np.where(valid_faces & (face_depth >= DRY_DEPTH_METERS), flux, 0)


@njit(cache=True, parallel=True)
def limit_outgoing_fluxes(
    depth: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    valid_cells: np.ndarray,
    dx: float,
    dy: float,
    dt_seconds: float,
) -> None:
    """Scale outgoing face fluxes in compiled loops."""
    rows, cols = depth.shape
    scale = np.ones(depth.shape, dtype=np.float64)

    for row in prange(rows):
        for col in range(cols):
            outgoing_depth = dt_seconds * (
                max(qx[row, col + 1], 0.0) / dx
                + max(-qx[row, col], 0.0) / dx
                + max(qy[row + 1, col], 0.0) / dy
                + max(-qy[row, col], 0.0) / dy
            )
            if valid_cells[row, col] and outgoing_depth > depth[row, col]:
                if outgoing_depth > 0.0:
                    scale[row, col] = depth[row, col] / outgoing_depth

    for row in prange(rows):
        for col in range(1, cols):
            if qx[row, col] >= 0.0:
                qx[row, col] *= scale[row, col - 1]
            else:
                qx[row, col] *= scale[row, col]

    for row in prange(rows):
        if qx[row, 0] < 0.0:
            qx[row, 0] *= scale[row, 0]
        if qx[row, cols] >= 0.0:
            qx[row, cols] *= scale[row, cols - 1]

    for row in prange(1, rows):
        for col in range(cols):
            if qy[row, col] >= 0.0:
                qy[row, col] *= scale[row - 1, col]
            else:
                qy[row, col] *= scale[row, col]

    for col in prange(cols):
        if qy[0, col] < 0.0:
            qy[0, col] *= scale[0, col]
        if qy[rows, col] >= 0.0:
            qy[rows, col] *= scale[rows - 1, col]


@profile
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


def queue_snapshot(
    render_queue: SnapshotRenderQueue,
    state: SimulationState,
    snapshot_directory: Path,
    index: int,
    max_depth_meters: float | None = None,
) -> Path:
    """Queue one simulation snapshot render and return its eventual path."""
    snapshot = snapshot_directory / f"snapshot_{index:04d}.png"
    return render_queue.submit(
        depth=state.hydraulic.depth,
        path=snapshot,
        elapsed_minutes=state.hydraulic.time_seconds / 60,
        max_depth_meters=max_depth_meters,
    )


def write_snapshot(
    grid: RasterGrid,
    storm_mask: np.ndarray,
    state: SimulationState,
    snapshot_directory: Path,
    index: int,
    max_depth_meters: float | None = None,
) -> Path:
    """Write one simulation snapshot and return its path."""
    snapshot = snapshot_directory / f"snapshot_{index:04d}.png"
    render_snapshot(
        grid,
        storm_mask,
        state.hydraulic.depth,
        snapshot,
        elapsed_minutes=state.hydraulic.time_seconds / 60,
        max_depth_meters=max_depth_meters,
    )
    return snapshot


@profile
def render_snapshot(
    grid: RasterGrid,
    storm_mask: np.ndarray,
    depth: np.ndarray,
    path: Path,
    elapsed_minutes: float,
    max_depth_meters: float | None = None,
) -> None:
    """Render DEM, storm footprint, and water depth into a PNG image."""
    figure, axis = plt.subplots(figsize=(8, 6), dpi=150)
    axis.imshow(np.ma.masked_invalid(grid.elevation), cmap="gray")
    axis.imshow(
        np.where(storm_mask, 1.0, np.nan),
        cmap="Blues",
        alpha=0.22,
        vmin=0,
        vmax=1,
    )

    depth_layer = np.ma.masked_where(depth <= 0, depth)
    depth_scale = {}
    if max_depth_meters is not None:
        depth_scale = {"vmin": 0, "vmax": max_depth_meters}
    water = axis.imshow(
        depth_layer,
        cmap=WATER_DEPTH_COLORMAP,
        alpha=WATER_DEPTH_ALPHA,
        **depth_scale,
    )
    if np.nanmax(depth) > 0:
        colorbar = figure.colorbar(water, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Water depth (m)")

    axis.set_title(f"Simulated inundation at {elapsed_minutes:.1f} minutes")
    axis.axis("off")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
