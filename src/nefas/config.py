from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a simulation configuration does not match the schema."""


@dataclass(frozen=True)
class InputConfig:
    dem: Path
    area_of_interest: Path
    storm_footprint: Path


@dataclass(frozen=True)
class RainfallPoint:
    time_minutes: float
    rate_mm_per_hr: float


@dataclass(frozen=True)
class RainfallConfig:
    series: tuple[RainfallPoint, ...]


@dataclass(frozen=True)
class TimeStepConfig:
    seconds: float
    max_seconds: float | None = None


@dataclass(frozen=True)
class OutputConfig:
    directory: Path


@dataclass(frozen=True)
class AreaOfInterestProcessingConfig:
    filters: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class ProcessingConfig:
    area_of_interest: AreaOfInterestProcessingConfig


@dataclass(frozen=True)
class SimulationConfig:
    inputs: InputConfig
    rainfall: RainfallConfig
    time_step: TimeStepConfig
    output: OutputConfig
    processing: ProcessingConfig


def load_config(path: Path) -> SimulationConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read simulation configuration files.") from exc

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    return parse_config(raw)


def parse_config(raw: Any) -> SimulationConfig:
    root = _mapping(raw, "configuration")
    inputs = _mapping(root.get("inputs"), "inputs")
    rainfall = _mapping(root.get("rainfall"), "rainfall")
    time_step = _mapping(root.get("time_step"), "time_step")
    output = _mapping(root.get("output"), "output")
    processing = _optional_mapping(root.get("processing"), "processing")

    return SimulationConfig(
        inputs=InputConfig(
            dem=_path(inputs, "dem"),
            area_of_interest=_path(inputs, "area_of_interest"),
            storm_footprint=_path(inputs, "storm_footprint"),
        ),
        rainfall=RainfallConfig(series=_rainfall_series(rainfall)),
        time_step=TimeStepConfig(
            seconds=_float(time_step, "seconds", positive=True),
            max_seconds=_optional_float(time_step, "max_seconds", positive=True),
        ),
        output=OutputConfig(directory=_path(output, "directory")),
        processing=ProcessingConfig(
            area_of_interest=AreaOfInterestProcessingConfig(
                filters=_area_of_interest_filters(processing)
            )
        ),
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping.")
    return value


def _optional_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _mapping(value, name)


def _path(section: dict[str, Any], key: str) -> Path:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty path string.")
    return Path(value)


def _area_of_interest_filters(section: dict[str, Any]) -> dict[str, str | int | float | bool]:
    area_of_interest = _optional_mapping(section.get("area_of_interest"), "processing.area_of_interest")
    filters = area_of_interest.get("filters", {})
    if filters is None:
        return {}
    filters = _mapping(filters, "processing.area_of_interest.filters")

    for key, value in filters.items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigError("AOI filter names must be non-empty strings.")
        if not isinstance(value, str | int | float | bool):
            raise ConfigError(f"AOI filter {key} must be a scalar value.")

    return filters


def _rainfall_series(section: dict[str, Any]) -> tuple[RainfallPoint, ...]:
    series = section.get("series")
    if not isinstance(series, list) or not series:
        raise ConfigError("rainfall.series must be a non-empty list.")

    points: list[RainfallPoint] = []
    for index, item in enumerate(series):
        point = _mapping(item, f"rainfall.series[{index}]")
        points.append(
            RainfallPoint(
                time_minutes=_float(point, "time_minutes"),
                rate_mm_per_hr=_float(point, "rate_mm_per_hr"),
            )
        )

    return tuple(points)


def _optional_float(section: dict[str, Any], key: str, *, positive: bool = False) -> float | None:
    if key not in section or section[key] is None:
        return None
    return _float(section, key, positive=positive)


def _float(section: dict[str, Any], key: str, *, positive: bool = False) -> float:
    try:
        value = float(section[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be numeric.") from exc

    if positive and value <= 0:
        raise ConfigError(f"{key} must be greater than zero.")
    if not positive and value < 0:
        raise ConfigError(f"{key} must be zero or greater.")
    return value
