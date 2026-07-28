import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from geopfa.io.data_readers import GeospatialDataReaders
from geopfa.processing import Processing


def _pfa(data_col=...):
    layer = {
        "data": gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:3857"
        )
    }
    if data_col is not ...:
        layer["data_col"] = data_col
    return {
        "criteria": {
            "criterion": {
                "components": {
                    "component": {"layers": {"layer": layer}}
                }
            }
        }
    }


@pytest.mark.parametrize("data_col", [..., None, "", "  ", "None", "none", "NONE"])
def test_weighted_distance_accepts_no_data_column(data_col):
    pfa = _pfa(data_col)

    result = Processing.weighted_distance_from_points(
        pfa,
        "criterion",
        "component",
        "layer",
        extent=[0, 0, 1, 1],
        nx=2,
        ny=2,
    )

    layer = result["criteria"]["criterion"]["components"]["component"][
        "layers"
    ]["layer"]
    assert layer["data_col"] is None
    assert np.isfinite(layer["model"]["weighted_point_score"]).all()


@pytest.mark.parametrize("data_col", [..., None, "", "None", "NONE"])
def test_interpolation_requires_data_column(data_col):
    pfa = _pfa(data_col)

    with pytest.raises(
        ValueError,
        match="interpolate_points_2d.*criterion/component/layer",
    ):
        Processing.interpolate_points_2d(
            pfa, "criterion", "component", "layer", nx=2, ny=2
        )


def test_configured_missing_data_column_has_clear_error():
    pfa = _pfa("temperature")

    with pytest.raises(ValueError, match="temperature.*Available columns"):
        Processing.weighted_distance_from_points(
            pfa,
            "criterion",
            "component",
            "layer",
            extent=[0, 0, 1, 1],
            nx=2,
            ny=2,
        )


def test_weighted_distance_preserves_valid_data_column():
    pfa = _pfa("value")
    layer = pfa["criteria"]["criterion"]["components"]["component"][
        "layers"
    ]["layer"]
    layer["data"]["value"] = [1.0, 2.0]

    Processing.weighted_distance_from_points(
        pfa,
        "criterion",
        "component",
        "layer",
        extent=[0, 0, 1, 1],
        nx=2,
        ny=2,
    )

    assert layer["data_col"] == "value"


def test_weighted_distance_3d_accepts_no_data_column():
    pfa = _pfa("NONE")
    layer = pfa["criteria"]["criterion"]["components"]["component"][
        "layers"
    ]["layer"]
    layer["data"] = gpd.GeoDataFrame(
        geometry=[Point(0, 0, 0), Point(1, 1, 1)], crs="EPSG:3857"
    )

    Processing.weighted_distance_from_points_3d(
        pfa,
        "criterion",
        "component",
        "layer",
        extent=[0, 0, 0, 1, 1, 1],
        nx=1,
        ny=1,
        nz=1,
    )

    assert layer["data_col"] is None
    assert np.isfinite(layer["model"]["weighted_point_score"]).all()


@pytest.mark.parametrize("gather_method", ["gather_data", "gather_processed_data"])
def test_gather_normalizes_missing_data_column(tmp_path, gather_method):
    component_dir = tmp_path / "criterion" / "component"
    component_dir.mkdir(parents=True)
    pfa = _pfa("NoNe")

    if gather_method == "gather_data":
        GeospatialDataReaders.gather_data(tmp_path, pfa, file_types=[])
    else:
        GeospatialDataReaders.gather_processed_data(tmp_path, pfa, crs=None)

    layer = pfa["criteria"]["criterion"]["components"]["component"][
        "layers"
    ]["layer"]
    assert layer["data_col"] is None
