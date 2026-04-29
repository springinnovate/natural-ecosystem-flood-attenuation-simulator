from __future__ import annotations

import unittest
from pathlib import Path

from nefas.config import ConfigError, parse_config


class ConfigParsingTests(unittest.TestCase):
    def test_parses_minimal_configuration(self) -> None:
        config = parse_config(
            {
                "inputs": {
                    "dem": "data/dem.tif",
                    "area_of_interest": "data/aoi.geojson",
                    "storm_footprint": "data/storm.geojson",
                },
                "rainfall": {
                    "series": [
                        {"time_minutes": 0, "rate_mm_per_hr": 0},
                        {"time_minutes": 15, "rate_mm_per_hr": 25},
                    ]
                },
                "time_step": {"seconds": 5},
                "output": {"directory": "outputs/example"},
            }
        )

        self.assertEqual(config.inputs.dem, Path("data/dem.tif"))
        self.assertEqual(config.rainfall.series[1].rate_mm_per_hr, 25)
        self.assertEqual(config.time_step.seconds, 5)
        self.assertEqual(config.output.directory, Path("outputs/example"))

    def test_requires_rainfall_series(self) -> None:
        with self.assertRaises(ConfigError):
            parse_config(
                {
                    "inputs": {
                        "dem": "data/dem.tif",
                        "area_of_interest": "data/aoi.geojson",
                        "storm_footprint": "data/storm.geojson",
                    },
                    "rainfall": {"series": []},
                    "time_step": {"seconds": 5},
                    "output": {"directory": "outputs/example"},
                }
            )


if __name__ == "__main__":
    unittest.main()
