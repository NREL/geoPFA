import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from geopfa.prob.data import (
    construct_alpha_c,
    load_labeled_wells,
    prepare_component_arrays,
    sample_evidence_at_wells,
)


class ProbDataTest(unittest.TestCase):
    def test_load_labeled_wells_csv_filters_component(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "wells.csv"
            pd.DataFrame(
                [
                    {
                        "well_id": "w1",
                        "longitude": -120.0,
                        "latitude": 40.0,
                        "depth_m": np.nan,
                        "heat_label": 1.0,
                        "reservoir_label": np.nan,
                        "barrier_label": np.nan,
                        "label_source": "test",
                        "label_quality": "proxy",
                    },
                    {
                        "well_id": "w2",
                        "longitude": -121.0,
                        "latitude": 41.0,
                        "depth_m": np.nan,
                        "heat_label": np.nan,
                        "reservoir_label": np.nan,
                        "barrier_label": np.nan,
                        "label_source": "test",
                        "label_quality": "proxy",
                    },
                ]
            ).to_csv(csv_path, index=False)

            gdf = load_labeled_wells(csv_path, component="heat")

            self.assertEqual(len(gdf), 1)
            self.assertEqual(gdf.crs.to_epsg(), 4326)
            self.assertTrue(all(gdf.geometry.geom_type == "Point"))

    def test_sample_evidence_at_wells_samples_expected_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = Path(tmpdir) / "feature.tif"
            arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            transform = from_origin(0.0, 2.0, 1.0, 1.0)
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
            ) as ds:
                ds.write(arr, 1)

            gdf = gpd.GeoDataFrame(
                {"well_id": ["a", "b"]},
                geometry=[Point(0.5, 1.5), Point(1.5, 0.5)],
                crs="EPSG:4326",
            )
            X_df = sample_evidence_at_wells(gdf, {"f1": str(raster_path)})

            np.testing.assert_allclose(X_df["f1"].to_numpy(), [1.0, 4.0])

    def test_construct_alpha_c_returns_finite_logit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = Path(tmpdir) / "heat_prob.tif"
            arr = np.array([[0.2, 0.8]], dtype=np.float32)
            transform = from_origin(0.0, 1.0, 1.0, 1.0)
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=1,
                width=2,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
            ) as ds:
                ds.write(arr, 1)

            gdf = gpd.GeoDataFrame(
                {"well_id": ["a", "b"]},
                geometry=[Point(0.5, 0.5), Point(1.5, 0.5)],
                crs="EPSG:4326",
            )
            alpha = construct_alpha_c(gdf, str(raster_path), transform="logit")

            self.assertEqual(alpha.shape, (2,))
            self.assertTrue(np.isfinite(alpha).all())
            self.assertLess(alpha[0], 0.0)
            self.assertGreater(alpha[1], 0.0)

    def test_prepare_component_arrays_filters_invalid_rows(self):
        gdf = gpd.GeoDataFrame(
            {
                "well_id": ["a", "b", "c", "d"],
                "heat_label": [1.0, 0.0, np.nan, 1.0],
                "label_quality": [
                    "confirmed",
                    "pseudo_absence",
                    "confirmed",
                    "confirmed",
                ],
            },
            geometry=[
                Point(-121.0, 40.0),
                Point(-121.1, 40.1),
                Point(-121.2, 40.2),
                Point(-121.3, 40.3),
            ],
            crs="EPSG:4326",
        )
        X_df = pd.DataFrame({"f1": [0.2, 0.3, 0.4, np.nan]}, index=gdf.index)
        alpha = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

        y, X, alpha_labeled, coords = prepare_component_arrays(
            gdf,
            X_df,
            alpha,
            component="heat",
            exclude_pseudo_absence=True,
        )

        np.testing.assert_array_equal(y, np.array([1]))
        self.assertEqual(X.shape, (1, 1))
        self.assertEqual(alpha_labeled.shape, (1,))
        self.assertEqual(coords.shape, (1, 2))
