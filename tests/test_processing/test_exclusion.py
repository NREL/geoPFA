import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from geopfa.processing import Exclusions
from geopandas.testing import assert_geodataframe_equal
import geopfa.geopfa2d.processing as processing_2D


def test_mask_exclusion_areas():
    """Minimal test for exclusion areas."""
    data = {
        "name": ["Location A", "Location B"],
        "geometry": [Point(1, 2), Point(3, 4)],
        "value": [10, 20],
    }
    points = gpd.GeoDataFrame(data, crs="EPSG:4326")

    exclusion = {
        "name": ["Exclusion B"],
        "geometry": [
            Polygon([(2.5, 3.5), (3.5, 3.5), (3.5, 4.5), (2.5, 4.5)])
        ],
    }
    exclusion_area = gpd.GeoDataFrame(exclusion, crs="EPSG:4326")

    exclusions = Exclusions()
    output = exclusions.mask_exclusion_areas(points, exclusion_area)

    assert output.to_dict()["value"] == {0: 10, 1: 0}


def test_validate_2D_mask_exclusion_areas():
    data = {
        "name": ["Location A", "Location B"],
        "geometry": [Point(1, 2), Point(3, 4)],
        "value": [10, 20],
    }
    points = gpd.GeoDataFrame(data, crs="EPSG:4326")

    exclusion = {
        "name": ["Exclusion B"],
        "geometry": [
            Polygon([(2.5, 3.5), (3.5, 3.5), (3.5, 4.5), (2.5, 4.5)])
        ],
    }
    exclusion_area = gpd.GeoDataFrame(exclusion, crs="EPSG:4326")

    exclusions = Exclusions()
    output = exclusions.mask_exclusion_areas(points, exclusion_area)

    with pytest.warns(DeprecationWarning):
        output_2D = processing_2D.Exclusions().mask_exclusion_areas(
            points, exclusion_area
        )
    assert_geodataframe_equal(output, output_2D)


def test_add_exclusions():
    """Minimal test for adding exclusions."""
    data = {
        "name": ["Location A", "Location B"],
        "geometry": [Point(1, 2), Point(3, 4)],
        "value": [10, 20],
    }
    points = gpd.GeoDataFrame(data, crs="EPSG:4326")

    exclusion = {
        "name": ["Exclusion B"],
        "geometry": [
            Polygon([(2.5, 3.5), (3.5, 3.5), (3.5, 4.5), (2.5, 4.5)])
        ],
    }
    exclusion_area = gpd.GeoDataFrame(exclusion, crs="EPSG:4326")

    pfa = {
        "pr": points,
        "exclusions": {
            "components": {
                "component-1": {
                    "set_to": "0",
                    "layers": {"layer-1": {"model": exclusion_area}},
                }
            }
        },
    }
    exclusions = Exclusions()
    # Sanity check if data is useful for this test
    assert np.allclose(pfa["pr"]["value"].values, [10, 20])
    output = exclusions.add_exclusions(pfa)
    assert np.allclose(output["pr"]["value"].values, [10, 20])

    assert "pr_excl" in output
    assert "favorability" in output["pr_excl"]
    assert np.allclose(
        output["pr_excl"]["favorability"].astype("f"),
        [np.nan, 0],
        equal_nan=True,
    )
