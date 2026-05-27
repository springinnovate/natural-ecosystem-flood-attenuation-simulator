from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np

from nefas.config import load_config
from nefas.simulation import RasterGrid
from nefas.synthetic_cases import (
    all_cases,
    bowl_with_spillway,
    flat_plain,
    incised_floodplain,
    long_slope,
    open_boundary_drainage,
    ridge_with_gap,
    roughness_patch,
)


class SyntheticCaseTests(unittest.TestCase):
    def test_all_cases_have_consistent_arrays(self) -> None:
        for case in all_cases():
            with self.subTest(case=case.name):
                self.assertEqual(case.elevation.shape, case.valid_cells.shape)
                self.assertEqual(case.elevation.shape, case.storm_mask.shape)
                self.assertTrue(case.valid_cells.any())
                self.assertTrue(case.storm_mask.any())
                self.assertGreater(case.cell_size, 0)
                self.assertTrue(case.expected_behavior)

    def test_flat_plain_is_uniform(self) -> None:
        case = flat_plain(shape=(4, 5), elevation_meters=3)

        np.testing.assert_allclose(case.elevation, np.full((4, 5), 3))
        self.assertTrue(case.storm_mask.all())

    def test_long_slope_descends_from_west_to_east(self) -> None:
        case = long_slope(shape=(3, 6), cell_size=30, slope=0.001)

        self.assertGreater(case.elevation[0, 0], case.elevation[0, -1])
        np.testing.assert_allclose(case.elevation[0], case.elevation[1])

    def test_bowl_has_lower_center_than_edges(self) -> None:
        case = bowl_with_spillway(shape=(21, 31))
        center = case.elevation[case.shape[0] // 2, case.shape[1] // 2]

        self.assertLess(center, case.elevation[0, 0])
        self.assertLess(center, case.elevation[-1, -1])

    def test_ridge_with_gap_has_lower_breach_than_ridge(self) -> None:
        case = ridge_with_gap(shape=(20, 40))
        ridge_col = case.shape[1] // 2
        breach_row = case.shape[0] // 2

        self.assertLess(case.elevation[breach_row, ridge_col], case.elevation[0, ridge_col])

    def test_incised_floodplain_has_lower_center_path(self) -> None:
        case = incised_floodplain(shape=(40, 80))
        middle_col = case.shape[1] // 2
        center_row = case.shape[0] // 2

        self.assertLess(
            case.elevation[center_row, middle_col],
            case.elevation[0, middle_col],
        )

    def test_roughness_patch_includes_spatial_manning_values(self) -> None:
        case = roughness_patch(shape=(20, 30))

        self.assertIsNotNone(case.manning_n)
        assert case.manning_n is not None
        self.assertGreater(case.manning_n.max(), case.manning_n.min())

    def test_open_boundary_drainage_storm_is_near_eastern_edge(self) -> None:
        case = open_boundary_drainage(shape=(10, 20))

        self.assertFalse(case.storm_mask[:, :10].any())
        self.assertTrue(case.storm_mask[:, -1].all())

    def test_grid_returns_raster_grid(self) -> None:
        case = long_slope(shape=(3, 4), cell_size=30)
        grid = case.grid()

        self.assertIsInstance(grid, RasterGrid)
        self.assertEqual(grid.shape, (3, 4))
        self.assertEqual(grid.dx, 30)
        self.assertEqual(grid.dy, 30)

    def test_export_writes_pipeline_inputs_and_config(self) -> None:
        case = long_slope(shape=(6, 8))
        with tempfile.TemporaryDirectory() as workspace:
            export = case.export(Path(workspace))

            self.assertTrue(export.dem.exists())
            self.assertTrue(export.area_of_interest.exists())
            self.assertTrue(export.storm_footprint.exists())
            self.assertTrue(export.config.exists())

            grid = RasterGrid.from_dem(export.dem)
            self.assertEqual(grid.shape, case.shape)
            np.testing.assert_allclose(grid.elevation, case.elevation)

            area_of_interest = gpd.read_file(export.area_of_interest)
            storm = gpd.read_file(export.storm_footprint)
            self.assertEqual(area_of_interest.iloc[0]["name"], case.name)
            self.assertEqual(storm.iloc[0]["name"], case.name)

            config = load_config(export.config)
            self.assertEqual(config.inputs.dem, export.dem.resolve())
            self.assertEqual(config.inputs.area_of_interest, export.area_of_interest.resolve())
            self.assertEqual(config.inputs.storm_footprint, export.storm_footprint.resolve())


if __name__ == "__main__":
    unittest.main()
