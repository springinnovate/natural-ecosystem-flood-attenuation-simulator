from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import shape
from shapely.ops import unary_union

from .config import RainfallConfig, RainfallPoint
from .simulation import RasterGrid

DEFAULT_CELL_SIZE_METERS = 30.0
DEFAULT_SHAPE = (80, 160)
DEFAULT_CRS = "EPSG:3857"
NODATA_VALUE = -9999.0


@dataclass(frozen=True)
class SyntheticCaseExport:
    """Paths written for an exported synthetic case."""

    workspace: Path
    dem: Path
    area_of_interest: Path
    storm_footprint: Path
    config: Path


@dataclass(frozen=True)
class SyntheticCase:
    """In-memory synthetic terrain and forcing for solver development."""

    name: str
    elevation: np.ndarray
    valid_cells: np.ndarray
    storm_mask: np.ndarray
    rainfall: RainfallConfig
    cell_size: float
    expected_behavior: str
    manning_n: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Normalize arrays to the dtypes used by the simulation engine."""
        elevation = np.asarray(self.elevation, dtype=np.float64)
        valid_cells = np.asarray(self.valid_cells, dtype=bool)
        storm_mask = np.asarray(self.storm_mask, dtype=bool)
        object.__setattr__(self, "elevation", elevation)
        object.__setattr__(self, "valid_cells", valid_cells)
        object.__setattr__(self, "storm_mask", storm_mask)
        if self.manning_n is not None:
            object.__setattr__(self, "manning_n", np.asarray(self.manning_n, dtype=np.float64))

    @property
    def shape(self) -> tuple[int, int]:
        """Return the raster shape as ``(rows, columns)``."""
        return self.elevation.shape

    def grid(self) -> RasterGrid:
        """Return this case as a solver-ready raster grid."""
        return RasterGrid(
            elevation=np.where(self.valid_cells, self.elevation, np.nan),
            dx=self.cell_size,
            dy=self.cell_size,
            valid_cells=self.valid_cells,
        )

    def export(
        self,
        workspace: Path,
        *,
        crs: str = DEFAULT_CRS,
        time_step_seconds: float = 5.0,
        total_runtime_seconds: float | None = None,
        snapshot_interval_minutes: float = 15.0,
    ) -> SyntheticCaseExport:
        """Write this case to DEM, vector, and YAML files for CLI testing."""
        workspace.mkdir(parents=True, exist_ok=True)
        transform = from_origin(
            0,
            self.shape[0] * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

        dem_path = workspace / "dem.tif"
        aoi_path = workspace / "area_of_interest.gpkg"
        storm_path = workspace / "storm_footprint.gpkg"
        config_path = workspace / "config.yaml"

        write_dem(dem_path, self, transform, crs)
        write_vector(aoi_path, "area_of_interest", self.name, self.valid_cells, transform, crs)
        write_vector(storm_path, "storm_footprint", self.name, self.storm_mask, transform, crs)
        write_config(
            config_path,
            case=self,
            dem_path=dem_path,
            aoi_path=aoi_path,
            storm_path=storm_path,
            time_step_seconds=time_step_seconds,
            total_runtime_seconds=total_runtime_seconds,
            snapshot_interval_minutes=snapshot_interval_minutes,
        )

        return SyntheticCaseExport(
            workspace=workspace,
            dem=dem_path,
            area_of_interest=aoi_path,
            storm_footprint=storm_path,
            config=config_path,
        )


def flat_plain(
    *,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    cell_size: float = DEFAULT_CELL_SIZE_METERS,
    elevation_meters: float = 0.0,
) -> SyntheticCase:
    """Create a flat floodplain where rainfall should accumulate uniformly."""
    return SyntheticCase(
        name="flat_plain",
        elevation=np.full(shape, elevation_meters, dtype=np.float64),
        valid_cells=np.ones(shape, dtype=bool),
        storm_mask=np.ones(shape, dtype=bool),
        rainfall=default_rainfall(),
        cell_size=cell_size,
        expected_behavior=(
            "Uniform rainfall should produce nearly uniform depth until open "
            "boundaries or solver diffusion affect the edges."
        ),
    )


def long_slope(
    *,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    cell_size: float = DEFAULT_CELL_SIZE_METERS,
    slope: float = 0.0005,
) -> SyntheticCase:
    """Create a long gentle plane sloping from west to east."""
    _, columns = shape
    elevation_profile = np.arange(columns - 1, -1, -1, dtype=np.float64) * cell_size * slope
    elevation = np.broadcast_to(elevation_profile, shape).copy()
    return SyntheticCase(
        name="long_slope",
        elevation=elevation,
        valid_cells=np.ones(shape, dtype=bool),
        storm_mask=np.ones(shape, dtype=bool),
        rainfall=default_rainfall(),
        cell_size=cell_size,
        expected_behavior=(
            "Water should drift eastward down the low-gradient floodplain, with "
            "arrival times increasing away from the storm source."
        ),
    )


def bowl_with_spillway(
    *,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    cell_size: float = DEFAULT_CELL_SIZE_METERS,
) -> SyntheticCase:
    """Create a closed depression with a low spillway on the eastern rim."""
    rows, columns = shape
    row_coords, col_coords = np.indices(shape)
    center_row = (rows - 1) / 2
    center_col = (columns - 1) / 2
    radius = np.hypot(row_coords - center_row, col_coords - center_col)
    radius /= radius.max()
    elevation = 2.0 * radius

    rim_col = int(columns * 0.75)
    elevation[:, rim_col:] += 0.75
    gap = slice(int(rows * 0.45), int(rows * 0.55))
    elevation[gap, rim_col] = elevation[gap, rim_col].min() - 0.5

    return SyntheticCase(
        name="bowl_with_spillway",
        elevation=elevation,
        valid_cells=np.ones(shape, dtype=bool),
        storm_mask=np.ones(shape, dtype=bool),
        rainfall=default_rainfall(),
        cell_size=cell_size,
        expected_behavior=(
            "The central depression should pond first, then spill through the "
            "lower eastern notch once the bowl fills."
        ),
    )


def ridge_with_gap(
    *,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    cell_size: float = DEFAULT_CELL_SIZE_METERS,
    slope: float = 0.0005,
) -> SyntheticCase:
    """Create a gentle slope interrupted by a ridge with one breach."""
    elevation = long_slope(shape=shape, cell_size=cell_size, slope=slope).elevation.copy()
    rows, columns = shape
    ridge_col = columns // 2
    gap = slice(int(rows * 0.43), int(rows * 0.57))
    elevation[:, ridge_col] += 2.0
    elevation[gap, ridge_col] -= 2.2

    return SyntheticCase(
        name="ridge_with_gap",
        elevation=elevation,
        valid_cells=np.ones(shape, dtype=bool),
        storm_mask=np.ones(shape, dtype=bool),
        rainfall=default_rainfall(),
        cell_size=cell_size,
        expected_behavior=(
            "Water should back up behind the ridge and preferentially pass "
            "through the lower breach."
        ),
    )


def incised_floodplain(
    *,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    cell_size: float = DEFAULT_CELL_SIZE_METERS,
    slope: float = 0.0003,
) -> SyntheticCase:
    """Create a broad floodplain with a shallow incised low-flow path."""
    elevation = long_slope(shape=shape, cell_size=cell_size, slope=slope).elevation.copy()
    rows, columns = shape
    row_coords, col_coords = np.indices(shape)
    centerline = rows / 2 + np.sin(col_coords / max(columns, 1) * 2 * np.pi) * rows * 0.08
    distance_to_path = np.abs(row_coords - centerline)
    swale = np.clip(1 - distance_to_path / max(rows * 0.08, 1), 0, 1)
    elevation -= swale * 0.75

    return SyntheticCase(
        name="incised_floodplain",
        elevation=elevation,
        valid_cells=np.ones(shape, dtype=bool),
        storm_mask=np.ones(shape, dtype=bool),
        rainfall=default_rainfall(),
        cell_size=cell_size,
        expected_behavior=(
            "Water should preferentially occupy the shallow swale, then spread "
            "onto the floodplain as depth increases."
        ),
    )


def roughness_patch(
    *,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    cell_size: float = DEFAULT_CELL_SIZE_METERS,
) -> SyntheticCase:
    """Create a sloped floodplain with a high-roughness central patch."""
    base = long_slope(shape=shape, cell_size=cell_size)
    manning_n = np.full(shape, 0.04, dtype=np.float64)
    rows, columns = shape
    manning_n[
        int(rows * 0.3) : int(rows * 0.7),
        int(columns * 0.4) : int(columns * 0.65),
    ] = 0.15

    return SyntheticCase(
        name="roughness_patch",
        elevation=base.elevation,
        valid_cells=base.valid_cells,
        storm_mask=base.storm_mask,
        rainfall=base.rainfall,
        cell_size=cell_size,
        manning_n=manning_n,
        expected_behavior=(
            "Once roughness is active in the solver, the central patch should "
            "slow flow and increase upstream ponding."
        ),
    )


def open_boundary_drainage(
    *,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    cell_size: float = DEFAULT_CELL_SIZE_METERS,
) -> SyntheticCase:
    """Create a small storm near an open eastern boundary."""
    elevation = long_slope(shape=shape, cell_size=cell_size, slope=0.001).elevation
    rows, columns = shape
    storm_mask = np.zeros(shape, dtype=bool)
    storm_mask[:, int(columns * 0.75) :] = True

    return SyntheticCase(
        name="open_boundary_drainage",
        elevation=elevation,
        valid_cells=np.ones(shape, dtype=bool),
        storm_mask=storm_mask,
        rainfall=default_rainfall(),
        cell_size=cell_size,
        expected_behavior=(
            "Water should route toward the eastern edge and leave the domain "
            "through the open boundary."
        ),
    )


def all_cases() -> tuple[SyntheticCase, ...]:
    """Return the standard synthetic cases for exploratory solver testing."""
    return (
        flat_plain(),
        long_slope(),
        bowl_with_spillway(),
        ridge_with_gap(),
        incised_floodplain(),
        roughness_patch(),
        open_boundary_drainage(),
    )


def default_rainfall() -> RainfallConfig:
    """Return a compact rainfall event shared by synthetic cases."""
    return RainfallConfig(
        series=(
            RainfallPoint(time_minutes=0, rate_mm_per_hr=0),
            RainfallPoint(time_minutes=15, rate_mm_per_hr=25),
            RainfallPoint(time_minutes=60, rate_mm_per_hr=25),
            RainfallPoint(time_minutes=90, rate_mm_per_hr=0),
        )
    )


def write_dem(path: Path, case: SyntheticCase, transform: rasterio.Affine, crs: str) -> None:
    """Write a synthetic case DEM to a GeoTIFF."""
    dem = np.where(case.valid_cells, case.elevation, NODATA_VALUE).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=case.shape[0],
        width=case.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=NODATA_VALUE,
    ) as dataset:
        dataset.write(dem, 1)


def write_vector(
    path: Path,
    layer_name: str,
    case_name: str,
    mask: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
) -> None:
    """Write a boolean raster mask to a single-feature GeoPackage."""
    geometry = mask_geometry(mask, transform)
    gpd.GeoDataFrame(
        {"name": [case_name], "kind": [layer_name]},
        geometry=[geometry],
        crs=crs,
    ).to_file(path, driver="GPKG")


def mask_geometry(mask: np.ndarray, transform: rasterio.Affine) -> object:
    """Convert true cells in a boolean mask to one vector geometry."""
    if not mask.any():
        raise ValueError("Cannot export an empty mask.")

    geometries = [
        shape(geometry)
        for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=transform)
        if value == 1
    ]
    if not geometries:
        raise ValueError("Cannot export an empty mask.")
    return unary_union(geometries)


def write_config(
    path: Path,
    *,
    case: SyntheticCase,
    dem_path: Path,
    aoi_path: Path,
    storm_path: Path,
    time_step_seconds: float,
    total_runtime_seconds: float | None,
    snapshot_interval_minutes: float,
) -> None:
    """Write a runnable NEFAS YAML configuration for an exported case."""
    runtime_seconds = total_runtime_seconds
    if runtime_seconds is None:
        runtime_seconds = max(point.time_minutes for point in case.rainfall.series) * 60

    config = {
        "inputs": {
            "dem": str(dem_path.resolve()),
            "area_of_interest": str(aoi_path.resolve()),
            "storm_footprint": str(storm_path.resolve()),
        },
        "rainfall": {
            "series": [
                {
                    "time_minutes": point.time_minutes,
                    "rate_mm_per_hr": point.rate_mm_per_hr,
                }
                for point in case.rainfall.series
            ]
        },
        "simulation_time": {
            "time_step_seconds": time_step_seconds,
            "total_runtime_seconds": runtime_seconds,
        },
        "output": {
            "directory": str((path.parent / "outputs").resolve()),
            "snapshots": {
                "directory": "snapshots",
                "interval_minutes": snapshot_interval_minutes,
            },
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
