from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from nefas.simulation import RasterGrid, SimulationState


class RasterGridTests(unittest.TestCase):
    def test_loads_grid_from_dem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dem.tif"
            transform = from_origin(100, 200, 30, 30)
            data = np.array(
                [
                    [1, 2, -9999],
                    [3, 4, 5],
                ],
                dtype=np.float32,
            )

            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=2,
                width=3,
                count=1,
                dtype="float32",
                crs="EPSG:5070",
                transform=transform,
                nodata=-9999,
            ) as dataset:
                dataset.write(data, 1)

            grid = RasterGrid.from_dem(path)

        self.assertEqual(grid.shape, (2, 3))
        self.assertEqual(grid.x_face_shape, (2, 4))
        self.assertEqual(grid.y_face_shape, (3, 3))
        self.assertEqual(grid.dx, 30)
        self.assertEqual(grid.dy, 30)
        self.assertFalse(bool(grid.active[0, 2]))
        self.assertTrue(np.isnan(grid.elevation[0, 2]))


class SimulationStateTests(unittest.TestCase):
    def test_dry_state_shapes_follow_grid(self) -> None:
        grid = RasterGrid(
            elevation=np.array(
                [
                    [10, 11, 12, 13],
                    [9, 10, 11, 12],
                    [8, 9, 10, 11],
                ],
                dtype=np.float64,
            ),
            dx=30,
            dy=30,
        )

        state = SimulationState.dry(
            grid,
            manning_n=0.08,
            runoff_coefficient=0.35,
        )

        self.assertEqual(state.hydraulic.depth.shape, grid.shape)
        self.assertEqual(state.hydraulic.qx.shape, grid.x_face_shape)
        self.assertEqual(state.hydraulic.qy.shape, grid.y_face_shape)
        self.assertEqual(state.diagnostics.arrival_time.shape, grid.shape)
        self.assertTrue(np.all(state.hydraulic.depth == 0))
        self.assertTrue(np.all(state.hydraulic.qx == 0))
        self.assertTrue(np.all(state.hydraulic.qy == 0))
        self.assertTrue(np.all(state.surface.manning_n == 0.08))
        self.assertTrue(np.all(state.surface.runoff_coefficient == 0.35))
        self.assertTrue(np.all(np.isnan(state.diagnostics.arrival_time)))
        np.testing.assert_array_equal(state.hydraulic.water_surface(grid), grid.elevation)


if __name__ == "__main__":
    unittest.main()
