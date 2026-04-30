from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from nefas.config import RainfallConfig, RainfallPoint
from nefas.engine import (
    add_rainfall_depth,
    apply_rainfall_forcing,
    rainfall_rate_m_per_second,
    render_snapshot,
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
            elevation=np.zeros((3, 4), dtype=np.float64),
            dx=30,
            dy=30,
        )
        state = SimulationState.dry(
            grid,
            manning_n=0.08,
            runoff_coefficient=1.0,
        )
        state.hydraulic.depth[1, 1] = 1.0

        water_timestep(state, dt_seconds=1)

        self.assertGreater(state.hydraulic.qx[1, 2], 0)
        self.assertLess(state.hydraulic.depth[1, 1], 1.0)
        self.assertGreater(state.hydraulic.depth[1, 2], 0)
        self.assertAlmostEqual(float(state.hydraulic.depth.sum()), 1.0)

    def test_water_timestep_updates_north_south_flux_from_surface_slope(self) -> None:
        grid = RasterGrid(
            elevation=np.zeros((4, 3), dtype=np.float64),
            dx=30,
            dy=30,
        )
        state = SimulationState.dry(
            grid,
            manning_n=0.08,
            runoff_coefficient=1.0,
        )
        state.hydraulic.depth[1, 1] = 1.0

        water_timestep(state, dt_seconds=1)

        self.assertGreater(state.hydraulic.qy[2, 1], 0)
        self.assertLess(state.hydraulic.depth[1, 1], 1.0)
        self.assertGreater(state.hydraulic.depth[2, 1], 0)
        self.assertAlmostEqual(float(state.hydraulic.depth.sum()), 1.0)

    def test_water_timestep_routes_boundary_flux_out_of_domain(self) -> None:
        grid = RasterGrid(
            elevation=np.zeros((1, 1), dtype=np.float64),
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

        self.assertLess(state.hydraulic.qx[0, 0], 0)
        self.assertGreater(state.hydraulic.qx[0, 1], 0)
        self.assertLess(state.hydraulic.qy[0, 0], 0)
        self.assertGreater(state.hydraulic.qy[1, 0], 0)
        self.assertLess(float(state.hydraulic.depth.sum()), 1.0)

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

    def test_render_snapshot_uses_configured_depth_scale(self) -> None:
        grid = RasterGrid(
            elevation=np.zeros((1, 1), dtype=np.float64),
            dx=30,
            dy=30,
        )
        figure = MagicMock()
        axis = MagicMock()
        water_layer = MagicMock()
        axis.imshow.side_effect = [MagicMock(), MagicMock(), water_layer]

        with patch("nefas.engine.plt.subplots", return_value=(figure, axis)):
            render_snapshot(
                grid,
                storm_mask=np.array([[True]]),
                depth=np.array([[0.5]], dtype=np.float64),
                path=Path("snapshot.png"),
                elapsed_minutes=15,
                max_depth_meters=2.0,
            )

        _, water_kwargs = axis.imshow.call_args_list[2]
        self.assertEqual(water_kwargs["vmin"], 0)
        self.assertEqual(water_kwargs["vmax"], 2.0)
        figure.colorbar.assert_called_once_with(water_layer, ax=axis, fraction=0.046, pad=0.04)

    def test_render_snapshot_autoscales_when_depth_scale_is_omitted(self) -> None:
        grid = RasterGrid(
            elevation=np.zeros((1, 1), dtype=np.float64),
            dx=30,
            dy=30,
        )
        figure = MagicMock()
        axis = MagicMock()
        axis.imshow.side_effect = [MagicMock(), MagicMock(), MagicMock()]

        with patch("nefas.engine.plt.subplots", return_value=(figure, axis)):
            render_snapshot(
                grid,
                storm_mask=np.array([[True]]),
                depth=np.array([[0.5]], dtype=np.float64),
                path=Path("snapshot.png"),
                elapsed_minutes=15,
            )

        _, water_kwargs = axis.imshow.call_args_list[2]
        self.assertNotIn("vmin", water_kwargs)
        self.assertNotIn("vmax", water_kwargs)


if __name__ == "__main__":
    unittest.main()
