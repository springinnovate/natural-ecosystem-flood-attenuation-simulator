from __future__ import annotations

import unittest
from pathlib import Path

from nefas.config import DEFAULT_SNAPSHOT_INTERVAL_MINUTES, ConfigError, parse_config


class ConfigParsingTests(unittest.TestCase):
    def test_parses_minimal_configuration(self) -> None:
        config = parse_config(
            {
                "inputs": {
                    "dem": "data/dem.tif",
                    "area_of_interest": "data/aoi.gpkg",
                    "storm_footprint": "data/storm.gpkg",
                },
                "rainfall": {
                    "series": [
                        {"time_minutes": 0, "rate_mm_per_hr": 0},
                        {"time_minutes": 15, "rate_mm_per_hr": 25},
                    ]
                },
                "simulation_time": {
                    "time_step_seconds": 5,
                    "max_time_step_seconds": 30,
                    "total_runtime_seconds": 7200,
                },
                "output": {
                    "directory": "outputs/example",
                    "snapshots": {
                        "directory": "frames",
                        "interval_minutes": 10,
                        "max_depth_meters": 1.5,
                    },
                },
                "processing": {
                    "area_of_interest": {
                        "filters": {
                            "name": "United States",
                            "FID": 3283,
                        }
                    }
                },
            }
        )

        self.assertEqual(config.inputs.dem, Path("data/dem.tif"))
        self.assertEqual(config.rainfall.series[1].rate_mm_per_hr, 25)
        self.assertEqual(config.simulation_time.time_step_seconds, 5)
        self.assertEqual(config.simulation_time.max_time_step_seconds, 30)
        self.assertEqual(config.simulation_time.total_runtime_seconds, 7200)
        self.assertEqual(config.time_step.seconds, 5)
        self.assertEqual(config.time_step.max_seconds, 30)
        self.assertEqual(config.output.directory, Path("outputs/example"))
        self.assertEqual(config.output.snapshots.directory, Path("frames"))
        self.assertEqual(config.output.snapshots.interval_minutes, 10)
        self.assertEqual(config.output.snapshots.max_depth_meters, 1.5)
        self.assertEqual(
            config.processing.area_of_interest.filters,
            {"name": "United States", "FID": 3283},
        )

    def test_aoi_filters_are_optional(self) -> None:
        config = parse_config(
            {
                "inputs": {
                    "dem": "data/dem.tif",
                    "area_of_interest": "data/aoi.gpkg",
                    "storm_footprint": "data/storm.gpkg",
                },
                "rainfall": {
                    "series": [
                        {"time_minutes": 0, "rate_mm_per_hr": 0},
                    ]
                },
                "simulation_time": {"time_step_seconds": 5},
                "output": {"directory": "outputs/example"},
            }
        )

        self.assertEqual(config.processing.area_of_interest.filters, {})
        self.assertEqual(config.output.snapshots.directory, Path("snapshots"))
        self.assertEqual(
            config.output.snapshots.interval_minutes,
            DEFAULT_SNAPSHOT_INTERVAL_MINUTES,
        )
        self.assertIsNone(config.output.snapshots.max_depth_meters)

    def test_requires_rainfall_series(self) -> None:
        with self.assertRaises(ConfigError):
            parse_config(
                {
                    "inputs": {
                        "dem": "data/dem.tif",
                        "area_of_interest": "data/aoi.gpkg",
                        "storm_footprint": "data/storm.gpkg",
                    },
                    "rainfall": {"series": []},
                    "simulation_time": {"time_step_seconds": 5},
                    "output": {"directory": "outputs/example"},
                }
            )

    def test_accepts_legacy_time_step_configuration(self) -> None:
        config = parse_config(
            {
                "inputs": {
                    "dem": "data/dem.tif",
                    "area_of_interest": "data/aoi.gpkg",
                    "storm_footprint": "data/storm.gpkg",
                },
                "rainfall": {
                    "series": [
                        {"time_minutes": 0, "rate_mm_per_hr": 0},
                    ]
                },
                "time_step": {"seconds": 10, "max_seconds": 60},
                "output": {"directory": "outputs/example"},
            }
        )

        self.assertEqual(config.simulation_time.time_step_seconds, 10)
        self.assertEqual(config.simulation_time.max_time_step_seconds, 60)
        self.assertIsNone(config.simulation_time.total_runtime_seconds)


if __name__ == "__main__":
    unittest.main()
