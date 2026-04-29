from __future__ import annotations

import unittest

import numpy as np

from nefas.config import RainfallConfig, RainfallPoint
from nefas.engine import (
    add_rainfall_depth,
    apply_rainfall_forcing,
    rainfall_rate_m_per_second,
    timestep_duration_seconds,
    water_timestep,
)
from nefas.simulation import RasterGrid, SimulationState


class EngineTests(unittest.TestCase):
    def test_timestep_stops_at_snapshot_and_event_end(self) -> None:
        self.assertEqual(
            timestep_duration_seconds(
                current_time_seconds=1798,
                duration_seconds=5400,
                next_snapshot_seconds=1800,
                nominal_time_step_seconds=5,
            ),
            2,
        )
        self.assertEqual(
            timestep_duration_seconds(
                current_time_seconds=5398,
                duration_seconds=5400,
                next_snapshot_seconds=7200,
                nominal_time_step_seconds=5,
            ),
            2,
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

    def test_apply_rainfall_forcing_adds_depth_only_to_valid_storm_cells(self) -> None:
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
        rainfall = RainfallConfig(
            series=(
                RainfallPoint(time_minutes=0, rate_mm_per_hr=360),
                RainfallPoint(time_minutes=10, rate_mm_per_hr=360),
            )
        )

        apply_rainfall_forcing(
            state,
            rainfall,
            storm_mask,
            timestep_start_seconds=0,
            dt_seconds=10,
        )

        expected_depth = np.array(
            [
                [0.0005, 0, 0],
                [0, 0.0005, 0],
            ]
        )
        np.testing.assert_allclose(state.hydraulic.depth, expected_depth)
        self.assertEqual(state.hydraulic.time_seconds, 0)
        np.testing.assert_allclose(state.diagnostics.max_depth, np.zeros(grid.shape))

    def test_add_rainfall_depth_uses_supplied_rate_directly(self) -> None:
        grid = RasterGrid(
            elevation=np.ones((1, 2), dtype=np.float64),
            dx=30,
            dy=30,
        )
        state = SimulationState.dry(
            grid,
            manning_n=0.08,
            runoff_coefficient=1.0,
        )

        add_rainfall_depth(
            state,
            storm_mask=np.array([[True, False]]),
            rate_m_per_second=0.001,
            dt_seconds=10,
        )

        np.testing.assert_allclose(state.hydraulic.depth, np.array([[0.01, 0]]))

    def test_water_timestep_updates_east_west_flux_from_surface_slope(self) -> None:
        grid = RasterGrid(
            elevation=np.zeros((1, 2), dtype=np.float64),
            dx=30,
            dy=30,
        )
        state = SimulationState.dry(
            grid,
            manning_n=0.08,
            runoff_coefficient=1.0,
        )
        state.hydraulic.depth[0, 0] = 1.0

        water_timestep(state, dt_seconds=1)

        self.assertGreater(state.hydraulic.qx[0, 1], 0)
        self.assertEqual(state.hydraulic.qx[0, 0], 0)
        self.assertEqual(state.hydraulic.qx[0, 2], 0)
        self.assertLess(state.hydraulic.depth[0, 0], 1.0)
        self.assertGreater(state.hydraulic.depth[0, 1], 0)
        self.assertAlmostEqual(float(state.hydraulic.depth.sum()), 1.0)

    def test_water_timestep_updates_north_south_flux_from_surface_slope(self) -> None:
        grid = RasterGrid(
            elevation=np.zeros((2, 1), dtype=np.float64),
            dx=30,
            dy=30,
        )
        state = SimulationState.dry(
            grid,
            manning_n=0.08,
            runoff_coefficient=1.0,
        )
        state.hydraulic.depth[0, 0] = 1.0

        water_timestep(state, dt_seconds=1)

        self.assertGreater(state.hydraulic.qy[1, 0], 0)
        self.assertEqual(state.hydraulic.qy[0, 0], 0)
        self.assertEqual(state.hydraulic.qy[2, 0], 0)
        self.assertLess(state.hydraulic.depth[0, 0], 1.0)
        self.assertGreater(state.hydraulic.depth[1, 0], 0)
        self.assertAlmostEqual(float(state.hydraulic.depth.sum()), 1.0)

    def test_water_timestep_zeroes_dry_and_invalid_face_fluxes(self) -> None:
        grid = RasterGrid(
            elevation=np.array(
                [
                    [0, np.nan],
                ],
                dtype=np.float64,
            ),
            dx=30,
            dy=30,
            valid_cells=np.array(
                [
                    [True, False],
                ]
            ),
        )
        state = SimulationState.dry(
            grid,
            manning_n=0.08,
            runoff_coefficient=1.0,
        )
        state.hydraulic.qx[0, 1] = 1.0

        water_timestep(state, dt_seconds=1)

        self.assertEqual(state.hydraulic.qx[0, 1], 0)
        np.testing.assert_allclose(state.hydraulic.depth, np.array([[0, 0]]))

    def test_water_timestep_limits_flux_to_available_depth(self) -> None:
        grid = RasterGrid(
            elevation=np.zeros((1, 2), dtype=np.float64),
            dx=30,
            dy=30,
        )
        state = SimulationState.dry(
            grid,
            manning_n=0.0,
            runoff_coefficient=1.0,
        )
        state.hydraulic.depth[0, 0] = 0.01
        state.hydraulic.qx[0, 1] = 10.0

        water_timestep(state, dt_seconds=10)

        self.assertAlmostEqual(state.hydraulic.depth[0, 0], 0)
        self.assertAlmostEqual(state.hydraulic.depth[0, 1], 0.01)
        self.assertAlmostEqual(float(state.hydraulic.depth.sum()), 0.01)


if __name__ == "__main__":
    unittest.main()
