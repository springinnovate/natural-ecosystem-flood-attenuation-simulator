from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class RasterGrid:
    elevation: FloatArray
    dx: float
    dy: float
    transform: Any = None
    crs: Any = None
    active: BoolArray | None = None

    def __post_init__(self) -> None:
        elevation = np.asarray(self.elevation, dtype=np.float64)
        object.__setattr__(self, "elevation", elevation)
        if self.active is None:
            object.__setattr__(self, "active", np.isfinite(elevation))
        else:
            object.__setattr__(self, "active", np.asarray(self.active, dtype=bool))

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    @property
    def ny(self) -> int:
        return self.shape[0]

    @property
    def nx(self) -> int:
        return self.shape[1]

    @property
    def x_face_shape(self) -> tuple[int, int]:
        return self.ny, self.nx + 1

    @property
    def y_face_shape(self) -> tuple[int, int]:
        return self.ny + 1, self.nx

    @classmethod
    def from_dem(cls, path: Path) -> RasterGrid:
        with rasterio.open(path) as source:
            elevation = source.read(1).astype(np.float64)
            active = source.read_masks(1) > 0
            if source.nodata is not None:
                active &= elevation != source.nodata

            return cls(
                elevation=np.where(active, elevation, np.nan),
                dx=abs(source.transform.a),
                dy=abs(source.transform.e),
                transform=source.transform,
                crs=source.crs,
                active=active,
            )


@dataclass(slots=True)
class SurfaceFields:
    manning_n: FloatArray
    runoff_coefficient: FloatArray

    @classmethod
    def uniform(
        cls,
        grid: RasterGrid,
        *,
        manning_n: float,
        runoff_coefficient: float,
    ) -> SurfaceFields:
        return cls(
            manning_n=np.full(grid.shape, manning_n, dtype=np.float64),
            runoff_coefficient=np.full(grid.shape, runoff_coefficient, dtype=np.float64),
        )


@dataclass(slots=True)
class HydraulicState:
    depth: FloatArray
    qx: FloatArray
    qy: FloatArray
    time_seconds: float = 0.0

    @classmethod
    def dry(cls, grid: RasterGrid) -> HydraulicState:
        return cls(
            depth=np.zeros(grid.shape, dtype=np.float64),
            qx=np.zeros(grid.x_face_shape, dtype=np.float64),
            qy=np.zeros(grid.y_face_shape, dtype=np.float64),
        )

    def water_surface(self, grid: RasterGrid) -> FloatArray:
        return grid.elevation + self.depth


@dataclass(slots=True)
class DiagnosticState:
    arrival_time: FloatArray
    max_depth: FloatArray
    max_velocity: FloatArray
    flood_duration: FloatArray

    @classmethod
    def initialize(cls, grid: RasterGrid) -> DiagnosticState:
        return cls(
            arrival_time=np.full(grid.shape, np.nan, dtype=np.float64),
            max_depth=np.zeros(grid.shape, dtype=np.float64),
            max_velocity=np.zeros(grid.shape, dtype=np.float64),
            flood_duration=np.zeros(grid.shape, dtype=np.float64),
        )


@dataclass(slots=True)
class SimulationState:
    grid: RasterGrid
    surface: SurfaceFields
    hydraulic: HydraulicState
    diagnostics: DiagnosticState

    @classmethod
    def dry(
        cls,
        grid: RasterGrid,
        *,
        manning_n: float,
        runoff_coefficient: float,
    ) -> SimulationState:
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
