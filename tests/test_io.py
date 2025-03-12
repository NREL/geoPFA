import geopandas
import pandas as pd
import pytest

from geoPFA.io import GeospatialDataWriters


def test_write(tmp_path):
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
    GeospatialDataWriters.write_shapefile(gdf, filename)

    assert filename.exists()
