import geopandas as gpd
import pytest
from shapely.geometry import Point

from geopfa.processing import Cleaners


def test_convert_z_measurements_converts_meters_to_feet():
    gdf = gpd.GeoDataFrame(geometry=[Point(1, 2, 10)])

    converted = Cleaners.convert_z_measurements(gdf, "m-msl", "ft-msl")

    assert converted.geometry.iloc[0].z == pytest.approx(32.8084)


def test_convert_z_measurements_normalizes_reference_strings():
    gdf = gpd.GeoDataFrame(geometry=[Point(1, 2, 10)])

    converted = Cleaners.convert_z_measurements(gdf, " M-MSL ", "m-msl")

    assert converted.geometry.iloc[0].z == 10


@pytest.mark.parametrize(
    ("source", "target"),
    [("depth", "m-msl"), ("m-msl", "epsg:5703"), (None, "m-msl")],
)
def test_convert_z_measurements_rejects_unsupported_references(source, target):
    gdf = gpd.GeoDataFrame(geometry=[Point(1, 2, 10)])

    with pytest.raises(ValueError, match="Z measurement|Unsupported Z"):
        Cleaners.convert_z_measurements(gdf, source, target)
