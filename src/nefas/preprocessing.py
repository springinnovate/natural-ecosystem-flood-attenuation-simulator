from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SimulationConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntermediateOutputs:
    workspace: Path
    selected_aoi: Path
    projected_aoi: Path
    clipped_dem: Path


def intermediate_workspace(config_path: Path, output_directory: Path) -> Path:
    return output_directory / config_path.stem


def prepare_intermediates(config: SimulationConfig, config_path: Path) -> IntermediateOutputs:
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.mask import mask
    except ImportError as exc:
        raise RuntimeError("geopandas and rasterio are required for input preprocessing.") from exc

    workspace = intermediate_workspace(config_path, config.output.directory)
    workspace.mkdir(parents=True, exist_ok=True)

    selected_aoi_path = workspace / "aoi_selected.gpkg"
    projected_aoi_path = workspace / "aoi_dem_crs.gpkg"
    clipped_dem_path = workspace / "dem_clipped.tif"

    LOGGER.info("Reading AOI from %s", config.inputs.area_of_interest)
    area_of_interest = gpd.read_file(config.inputs.area_of_interest, fid_as_index=True)
    selected_aoi = filter_area_of_interest(
        area_of_interest,
        config.processing.area_of_interest.filters,
    )
    if selected_aoi.empty:
        raise RuntimeError("AOI filters did not select any features.")
    selected_aoi.to_file(selected_aoi_path, driver="GPKG")
    LOGGER.info("Wrote selected AOI features to %s", selected_aoi_path)

    with rasterio.open(config.inputs.dem) as dem:
        LOGGER.info("Reading DEM from %s", config.inputs.dem)
        projected_aoi = selected_aoi.to_crs(dem.crs)
        projected_aoi.to_file(projected_aoi_path, driver="GPKG")
        LOGGER.info("Wrote DEM-projected AOI features to %s", projected_aoi_path)

        clipped, transform = mask(dem, projected_aoi.geometry, crop=True)
        profile = dem.profile.copy()
        if not profile.get("tiled"):
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        profile.update(
            driver="GTiff",
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=transform,
        )

    with rasterio.open(clipped_dem_path, "w", **profile) as output:
        output.write(clipped)
    LOGGER.info("Wrote clipped DEM to %s", clipped_dem_path)

    return IntermediateOutputs(
        workspace=workspace,
        selected_aoi=selected_aoi_path,
        projected_aoi=projected_aoi_path,
        clipped_dem=clipped_dem_path,
    )


def filter_area_of_interest(area_of_interest: Any, filters: dict[str, str | int | float | bool]) -> Any:
    selected = area_of_interest
    for field, expected in filters.items():
        if field in selected.columns:
            selected = selected[selected[field] == expected]
        elif field.lower() == "fid":
            selected = selected[selected.index == expected]
        else:
            selected = selected.iloc[0:0]
    return selected
