from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class RasterGrid:
    """Terrain grid used by the finite-volume solver.

    The simulation itself only needs the terrain elevations, cell spacing, and
    active-cell mask. Geospatial metadata stays outside this object so the
    numerical state remains focused on array calculations.
    """

    # Terrain elevation at cell centers, in meters.
    elevation: FloatArray
    # Cell width in the x direction, in meters.
    dx: float
    # Cell height in the y direction, in meters.
    dy: float
    # True where the DEM has valid data and the solver should update cells.
    active: BoolArray | None = None

    def __post_init__(self) -> None:
        """Normalize elevation and active-mask arrays to solver dtypes."""
        elevation = np.asarray(self.elevation, dtype=np.float64)
        object.__setattr__(self, "elevation", elevation)
        if self.active is None:
            object.__setattr__(self, "active", np.isfinite(elevation))
        else:
            object.__setattr__(self, "active", np.asarray(self.active, dtype=bool))

    @property
    def shape(self) -> tuple[int, int]:
        """Cell-centered grid shape as ``(ny, nx)``."""
        return self.elevation.shape

    @property
    def ny(self) -> int:
        """Number of rows in the cell-centered grid."""
        return self.shape[0]

    @property
    def nx(self) -> int:
        """Number of columns in the cell-centered grid."""
        return self.shape[1]

    @property
    def x_face_shape(self) -> tuple[int, int]:
        """Shape for east-west face fluxes, including boundary faces."""
        return self.ny, self.nx + 1

    @property
    def y_face_shape(self) -> tuple[int, int]:
        """Shape for north-south face fluxes, including boundary faces."""
        return self.ny + 1, self.nx

    @classmethod
    def from_dem(cls, path: Path) -> RasterGrid:
        """Build a terrain grid from the first band of a DEM raster."""
        with rasterio.open(path) as source:
            elevation = source.read(1).astype(np.float64)
            active = source.read_masks(1) > 0
            if source.nodata is not None:
                active &= elevation != source.nodata

            return cls(
                elevation=np.where(active, elevation, np.nan),
                dx=abs(source.transform.a),
                dy=abs(source.transform.e),
                active=active,
            )


@dataclass(slots=True)
class SurfaceFields:
    """Cell-centered surface parameters used by forcing and friction terms."""

    # Manning roughness coefficient for each grid cell.
    manning_n: FloatArray
    # Fraction of rainfall that becomes surface-water input for each cell.
    runoff_coefficient: FloatArray

    @classmethod
    def uniform(
        cls,
        grid: RasterGrid,
        manning_n: float,
        runoff_coefficient: float,
    ) -> SurfaceFields:
        """Create spatially uniform surface parameters over a grid."""
        return cls(
            manning_n=np.full(grid.shape, manning_n, dtype=np.float64),
            runoff_coefficient=np.full(grid.shape, runoff_coefficient, dtype=np.float64),
        )


@dataclass(slots=True)
class HydraulicState:
    """Time-varying hydraulic arrays for a local-inertial simulation."""

    # Water depth at cell centers, in meters.
    depth: FloatArray
    # East-west face discharge or unit flux, depending on solver convention.
    qx: FloatArray
    # North-south face discharge or unit flux, depending on solver convention.
    qy: FloatArray
    # Elapsed model time from the beginning of the event, in seconds.
    time_seconds: float = 0.0

    @classmethod
    def dry(cls, grid: RasterGrid) -> HydraulicState:
        """Create a zero-depth, zero-flux hydraulic state for a grid."""
        return cls(
            depth=np.zeros(grid.shape, dtype=np.float64),
            qx=np.zeros(grid.x_face_shape, dtype=np.float64),
            qy=np.zeros(grid.y_face_shape, dtype=np.float64),
        )

    def water_surface(self, grid: RasterGrid) -> FloatArray:
        """Return water-surface elevation as terrain elevation plus depth."""
        return grid.elevation + self.depth


@dataclass(slots=True)
class DiagnosticState:
    """Accumulated outputs tracked while the simulation advances."""

    # First time each cell exceeds the configured arrival-depth threshold.
    arrival_time: FloatArray
    # Maximum simulated water depth at each cell, in meters.
    max_depth: FloatArray
    # Maximum simulated flow speed at each cell, in meters per second.
    max_velocity: FloatArray
    # Total time each cell remains above the flood-depth threshold, in seconds.
    flood_duration: FloatArray

    @classmethod
    def initialize(cls, grid: RasterGrid) -> DiagnosticState:
        """Create empty diagnostic arrays for a new simulation."""
        return cls(
            arrival_time=np.full(grid.shape, np.nan, dtype=np.float64),
            max_depth=np.zeros(grid.shape, dtype=np.float64),
            max_velocity=np.zeros(grid.shape, dtype=np.float64),
            flood_duration=np.zeros(grid.shape, dtype=np.float64),
        )


@dataclass(slots=True)
class SimulationState:
    """Complete in-memory state for one simulation run."""

    grid: RasterGrid
    surface: SurfaceFields
    hydraulic: HydraulicState
    diagnostics: DiagnosticState

    @classmethod
    def dry(
        cls,
        grid: RasterGrid,
        manning_n: float,
        runoff_coefficient: float,
    ) -> SimulationState:
        """Create a simulation state with dry hydraulics and empty diagnostics."""
        return cls(
            grid=grid,
            surface=SurfaceFields.uniform(
                grid,
                manning_n=manning_n,
                runoff_coefficient=runoff_coefficient,
            ),
            hydraulic=HydraulicState.dry(grid),
            diagnostics=DiagnosticState.initialize(grid),
        )
