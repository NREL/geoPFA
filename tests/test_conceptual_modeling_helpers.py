import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from geopfa.conceptual_modeling import (
    _apply_slices,
    _build_well_pts,
    _camera_from_view,
    _coords3_from_point,
    _infer_spacing,
    _require_3d,
)


# ---------------------------------------------------------------------------
# _coords3_from_point
# ---------------------------------------------------------------------------


def test_coords3_from_point_3d():
    pt = Point(1.0, 2.0, 3.0)
    result = _coords3_from_point(pt)
    assert result == (1.0, 2.0, 3.0)


def test_coords3_from_point_2d_defaults_z_zero():
    pt = Point(4.0, 5.0)
    result = _coords3_from_point(pt)
    assert result == (4.0, 5.0, 0.0)


# ---------------------------------------------------------------------------
# _build_well_pts
# ---------------------------------------------------------------------------


def test_build_well_pts_none_returns_none():
    assert _build_well_pts(None) is None


def test_build_well_pts_empty_gdf_returns_none():
    gdf = gpd.GeoDataFrame({"geometry": []})
    assert _build_well_pts(gdf) is None


def test_build_well_pts_point_gdf():
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0, 1), Point(1, 1, 2)]})
    result = _build_well_pts(gdf)
    assert result is not None
    assert result.shape == (2, 3)
    assert np.allclose(result[:, 2], [1.0, 2.0])


def test_build_well_pts_linestring_geometry():
    ls = LineString([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    result = _build_well_pts(ls)
    assert result is not None
    assert result.shape == (3, 3)


def test_build_well_pts_2d_linestring_adds_zero_z():
    ls = LineString([(0, 0), (1, 1)])
    result = _build_well_pts(ls)
    assert result is not None
    assert result.shape[1] == 3
    assert np.all(result[:, 2] == 0.0)


def test_build_well_pts_multilinestring():
    mls = MultiLineString([[(0, 0, 0), (1, 0, 0)], [(2, 0, 1), (3, 0, 1)]])
    result = _build_well_pts(mls)
    assert result is not None
    assert result.shape[1] == 3


# ---------------------------------------------------------------------------
# _apply_slices
# ---------------------------------------------------------------------------


def test_apply_slices_none_input():
    assert _apply_slices(None) is None


def test_apply_slices_no_filters():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    result = _apply_slices(arr)
    assert np.array_equal(result, arr)


def test_apply_slices_x_filter():
    arr = np.array([[1.0, 0, 0], [3.0, 0, 0], [5.0, 0, 0]])
    result = _apply_slices(arr, x_slice=3.0)
    assert len(result) == 2
    assert np.all(result[:, 0] <= 3.0)


def test_apply_slices_y_filter():
    arr = np.array([[0, 1.0, 0], [0, 4.0, 0], [0, 7.0, 0]])
    result = _apply_slices(arr, y_slice=4.0)
    assert len(result) == 2


def test_apply_slices_z_filter():
    arr = np.array([[0, 0, 1.0], [0, 0, 5.0], [0, 0, 10.0]])
    result = _apply_slices(arr, z_slice=5.0)
    assert len(result) == 2


def test_apply_slices_combined_filters():
    arr = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]])
    result = _apply_slices(arr, x_slice=3.0, y_slice=3.0, z_slice=3.0)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _infer_spacing
# ---------------------------------------------------------------------------


def test_infer_spacing_single_value_returns_one():
    result = _infer_spacing(np.array([5.0, 5.0, 5.0]))
    assert result == 1.0


def test_infer_spacing_regular():
    arr = np.array([0.0, 100.0, 200.0, 300.0])
    result = _infer_spacing(arr)
    assert np.isclose(result, 100.0)


def test_infer_spacing_uses_median():
    arr = np.array([0.0, 10.0, 20.0, 30.0, 1000.0])
    result = _infer_spacing(arr)
    assert np.isclose(result, 10.0)


# ---------------------------------------------------------------------------
# _camera_from_view
# ---------------------------------------------------------------------------


def test_camera_from_view_returns_three_elements():
    bounds = (0, 10, 0, 10, 0, 10)
    result = _camera_from_view(bounds, elev_deg=45, azim_deg=45)
    assert len(result) == 3


def test_camera_from_view_viewup_is_z():
    bounds = (0, 10, 0, 10, 0, 10)
    result = _camera_from_view(bounds, elev_deg=30, azim_deg=90)
    assert result[2] == (0.0, 0.0, 1.0)


def test_camera_from_view_focal_is_center():
    bounds = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
    result = _camera_from_view(bounds, elev_deg=45, azim_deg=45)
    focal = result[1]
    assert np.allclose(focal, [5.0, 5.0, 5.0])


# ---------------------------------------------------------------------------
# _require_3d
# ---------------------------------------------------------------------------


def test_require_3d_passes_for_3d_points():
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0, 1), Point(1, 1, 2)]})
    _require_3d(gdf, "test")  # should not raise


def test_require_3d_raises_for_2d_points():
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
    with pytest.raises(ValueError, match="Z coordinates"):
        _require_3d(gdf, "test")


def test_require_3d_raises_for_mixed_2d_3d():
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0, 1), Point(1, 1)]})
    with pytest.raises(ValueError, match="Z coordinates"):
        _require_3d(gdf, "test")
