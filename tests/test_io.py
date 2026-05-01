import geopandas
import pandas as pd
import pytest

from geopfa.io import GeospatialDataWriters


def test_write_shapefile(tmp_path):
    """Minimum test to write a shapefile."""
    filename = tmp_path / "empty.gpkg"

    df = pd.DataFrame(
        {
            "City": ["Golden"],
            "State": ["CO"],
            "Latitude": [39.755],
            "Longitude": [-105.221],
        }
    )
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326",
    )
    GeospatialDataWriters.write_shapefile(
        gdf, filename, target_crs="EPSG:4326"
    )

    assert filename.exists()


def test_write_csv(tmp_path):
    """Minimum test to write a CSV."""
    filename = tmp_path / "empty.gpkg"

    df = pd.DataFrame(
        {
            "City": ["Golden"],
            "State": ["CO"],
            "Latitude": [39.755],
            "Longitude": [-105.221],
        }
    )
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326",
    )
    GeospatialDataWriters.write_csv(gdf, filename, target_crs="EPSG:4326")

    assert filename.exists()
