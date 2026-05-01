from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from nefas.config import (
    AreaOfInterestProcessingConfig,
    InputConfig,
    OutputConfig,
    ProcessingConfig,
    RainfallConfig,
    RainfallPoint,
    SimulationConfig,
    SimulationTimeConfig,
    SnapshotConfig,
)
from nefas.engine import (
    WATER_DEPTH_ALPHA,
    WATER_DEPTH_COLORMAP,
    add_rainfall_depth,
    apply_rainfall_forcing,
    effective_time_step_seconds,
    limit_outgoing_fluxes,
    local_inertial_flux_update,
    local_inertial_flux_update_numba,
    rainfall_rate_m_per_second,
    render_snapshot,
    simulation_duration_seconds,
    timestep_duration_seconds,
    update_interior_x_fluxes_numba,
    update_interior_y_fluxes_numba,
    water_timestep,
)
from nefas.simulation import RasterGrid, SimulationState


def make_simulation_config(
    *,
    simulation_time: SimulationTimeConfig,
) -> SimulationConfig:
    return SimulationConfig(
        inputs=InputConfig(
            dem=Path("data/dem.tif"),
            area_of_interest=Path("data/aoi.gpkg"),
            storm_footprint=Path("data/storm.gpkg"),
        ),
        rainfall=RainfallConfig(
            series=(
                RainfallPoint(time_minutes=0, rate_mm_per_hr=0),
                RainfallPoint(time_minutes=60, rate_mm_per_hr=60),
            )
        ),
        simulation_time=simulation_time,
        output=OutputConfig(
            directory=Path("outputs/example"),
            snapshots=SnapshotConfig(
                directory=Path("snapshots"),
                interval_minutes=15,
            ),
        ),
        processing=ProcessingConfig(
            area_of_interest=AreaOfInterestProcessingConfig(filters={})
        ),
    )


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

    def test_rainfall_rate_is_zero_after_final_rainfall_point(self) -> None:
        rainfall = RainfallConfig(
            series=(
                RainfallPoint(time_minutes=0, rate_mm_per_hr=0),
                RainfallPoint(time_minutes=60, rate_mm_per_hr=60),
            )
        )

        self.assertEqual(rainfall_rate_m_per_second(rainfall, elapsed_seconds=3601), 0)

    def test_simulation_duration_uses_configured_runtime_when_present(self) -> None:
        config = make_simulation_config(
            simulation_time=SimulationTimeConfig(
                time_step_seconds=5,
                total_runtime_seconds=7200,
            )
        )

        self.assertEqual(simulation_duration_seconds(config), 7200)

    def test_simulation_duration_falls_back_to_rainfall_duration(self) -> None:
        config = make_simulation_config(
            simulation_time=SimulationTimeConfig(time_step_seconds=5)
        )

        self.assertEqual(simulation_duration_seconds(config), 3600)

    def test_effective_time_step_applies_maximum_cap(self) -> None:
        config = make_simulation_config(
            simulation_time=SimulationTimeConfig(
                time_step_seconds=60,
                max_time_step_seconds=30,
            )
        )

        self.assertEqual(effective_time_step_seconds(config), 30)

    def test_numba_flux_update_matches_numpy_reference(self) -> None:
        old_flux = np.array(
            [
                [0.0, 0.5, -0.2],
                [1.5, -1.0, 0.25],
            ],
            dtype=np.float64,
        )
        face_depth = np.array(
            [
                [0.0, 0.002, 0.5],
                [1.25, 0.0005, 0.9],
            ],
            dtype=np.float64,
        )
        slope = np.array(
            [
                [0.0, 0.01, -0.02],
                [0.03, -0.01, 0.005],
            ],
            dtype=np.float64,
        )
        manning_n = np.full_like(old_flux, 0.08)
        valid_faces = np.array(
            [
                [True, True, False],
                [True, True, True],
            ]
        )

        expected = local_inertial_flux_update(
            old_flux=old_flux,
            face_depth=face_depth,
            slope=slope,
            manning_n=manning_n,
            valid_faces=valid_faces,
            dt_seconds=5.0,
        )
        actual = local_inertial_flux_update_numba(
            old_flux,
            face_depth,
            slope,
            manning_n,
            valid_faces,
            5.0,
        )

        np.testing.assert_allclose(actual, expected)

    def test_fused_x_flux_kernel_matches_numpy_reference(self) -> None:
        elevation = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.2, 0.0, 0.3],
            ],
            dtype=np.float64,
        )
        depth = np.array(
            [
                [0.3, 0.0, 0.5],
                [0.0, 0.4, 0.2],
            ],
            dtype=np.float64,
        )
        eta = elevation + depth
        manning_n = np.array(
            [
                [0.04, 0.08, 0.12],
                [0.05, 0.07, 0.09],
            ],
            dtype=np.float64,
        )
        valid_cells = np.array(
            [
                [True, True, False],
                [True, True, True],
            ]
        )
        qx = np.array(
            [
                [0.0, 0.1, -0.2, 0.0],
                [0.0, 0.3, -0.4, 0.0],
            ],
            dtype=np.float64,
        )

        h_face_x = np.maximum(
            0,
            np.maximum(eta[:, :-1], eta[:, 1:])
            - np.maximum(elevation[:, :-1], elevation[:, 1:]),
        )
        slope_x = (eta[:, 1:] - eta[:, :-1]) / 30.0
        n_face_x = 0.5 * (manning_n[:, :-1] + manning_n[:, 1:])
        valid_x = valid_cells[:, :-1] & valid_cells[:, 1:]
        expected = local_inertial_flux_update(
            old_flux=qx[:, 1:-1],
            face_depth=h_face_x,
            slope=slope_x,
            manning_n=n_face_x,
            valid_faces=valid_x,
            dt_seconds=5.0,
        )

        update_interior_x_fluxes_numba(
            qx,
            eta,
            elevation,
            manning_n,
            valid_cells,
            30.0,
            5.0,
        )

        np.testing.assert_allclose(qx[:, 1:-1], expected)

    def test_fused_y_flux_kernel_matches_numpy_reference(self) -> None:
        elevation = np.array(
            [
                [0.0, 0.1],
                [0.2, 0.0],
                [0.1, 0.3],
            ],
            dtype=np.float64,
        )
        depth = np.array(
            [
                [0.3, 0.0],
                [0.0, 0.4],
                [0.2, 0.1],
            ],
            dtype=np.float64,
        )
        eta = elevation + depth
        manning_n = np.array(
            [
                [0.04, 0.08],
                [0.05, 0.07],
                [0.06, 0.09],
            ],
            dtype=np.float64,
        )
        valid_cells = np.array(
            [
                [True, True],
                [True, False],
                [True, True],
            ]
        )
        qy = np.array(
            [
                [0.0, 0.0],
                [0.1, -0.2],
                [0.3, -0.4],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )

        h_face_y = np.maximum(
            0,
            np.maximum(eta[:-1, :], eta[1:, :])
            - np.maximum(elevation[:-1, :], elevation[1:, :]),
        )
        slope_y = (eta[1:, :] - eta[:-1, :]) / 30.0
        n_face_y = 0.5 * (manning_n[:-1, :] + manning_n[1:, :])
        valid_y = valid_cells[:-1, :] & valid_cells[1:, :]
        expected = local_inertial_flux_update(
            old_flux=qy[1:-1, :],
            face_depth=h_face_y,
            slope=slope_y,
            manning_n=n_face_y,
            valid_faces=valid_y,
            dt_seconds=5.0,
        )

        update_interior_y_fluxes_numba(
            qy,
            eta,
            elevation,
            manning_n,
            valid_cells,
            30.0,
            5.0,
        )

        np.testing.assert_allclose(qy[1:-1, :], expected)

    def test_outgoing_flux_limiter_scales_boundary_and_interior_outflows(self) -> None:
        depth = np.array([[0.10, 0.20]], dtype=np.float64)
        qx = np.array([[-0.5, 0.4, 0.6]], dtype=np.float64)
        qy = np.array(
            [
                [-0.2, -0.1],
                [0.3, 0.2],
            ],
            dtype=np.float64,
        )

        limit_outgoing_fluxes(
            depth,
            qx,
            qy,
            np.array([[True, True]]),
            10.0,
            20.0,
            5.0,
        )

        np.testing.assert_allclose(
            qx,
            np.array([[-0.08695652, 0.06956522, 0.32]]),
        )
        np.testing.assert_allclose(
            qy,
            np.array([[-0.03478261, -0.05333333], [0.05217391, 0.10666667]]),
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

    def test_render_snapshot_uses_water_colormap(self) -> None:
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
        self.assertIs(water_kwargs["cmap"], WATER_DEPTH_COLORMAP)
        self.assertEqual(water_kwargs["alpha"], WATER_DEPTH_ALPHA)


if __name__ == "__main__":
    unittest.main()
