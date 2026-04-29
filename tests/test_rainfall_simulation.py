from __future__ import annotations

import unittest

import numpy as np

from nefas.config import RainfallConfig, RainfallPoint
from nefas.rainfall_simulation import (
    add_rainfall_depth,
    rainfall_rate_m_per_second,
    snapshot_times_seconds,
)
from nefas.simulation import RasterGrid, SimulationState


class RainfallSimulationTests(unittest.TestCase):
    def test_snapshot_times_include_start_interval_and_end(self) -> None:
        rainfall = RainfallConfig(
            series=(
                RainfallPoint(time_minutes=0, rate_mm_per_hr=0),
                RainfallPoint(time_minutes=90, rate_mm_per_hr=5),
            )
        )

        self.assertEqual(
            snapshot_times_seconds(rainfall, interval_minutes=30),
            (0, 1800, 3600, 5400),
        )

    def test_rainfall_rate_is_interpolated_and_converted(self) -> None:
        rainfall = RainfallConfig(
            series=(
                RainfallPoint(time_minutes=0, rate_mm_per_hr=0),
                RainfallPoint(time_minutes=60, rate_mm_per_hr=60),
            )
        )

        self.assertAlmostEqual(
            rainfall_rate_m_per_second(rainfall, elapsed_seconds=1800),
            0.03 / 3600,
        )

    def test_adds_depth_only_to_valid_storm_cells(self) -> None:
        grid = RasterGrid(
            elevation=np.ones((2, 3), dtype=np.float64),
            dx=30,
            dy=30,
            valid_cells=np.array(
                [
                    [True, True, False],
                    [True, True, True],
                ]
            ),
        )
        state = SimulationState.dry(
            grid,
            manning_n=0.08,
            runoff_coefficient=0.5,
        )
        storm_mask = np.array(
            [
                [True, False, True],
                [False, True, False],
            ]
        )

        add_rainfall_depth(
            state,
            storm_mask,
            rate_m_per_second=0.001,
            dt_seconds=10,
        )

        expected_depth = np.array(
            [
                [0.005, 0, 0],
                [0, 0.005, 0],
            ]
        )
        np.testing.assert_allclose(state.hydraulic.depth, expected_depth)
        self.assertEqual(state.hydraulic.time_seconds, 10)
        np.testing.assert_allclose(state.diagnostics.max_depth, expected_depth)


if __name__ == "__main__":
    unittest.main()
