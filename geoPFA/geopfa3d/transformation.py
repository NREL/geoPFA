# -*- coding: utf-8 -*-
"""
Set of methods to transform data from evidence layers into evidence layers, now supporting 3D data.
"""

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point

from geoPFA.transformation import normalize_gdf as _normalize_gdf
from geoPFA.transformation import normalize_array as _normalize_array
from geoPFA.transformation import transform as _transform


class VoterVetoTransformation3D:
    """Class of functions for transforming 3D data layers into evidence layers."""
    
    @staticmethod
    def normalize_gdf(gdf, col, norm_to=1):
        """
        Normalize the values in a specified column of a GeoDataFrame (3D compatible).
        """
        return _normalize_gdf(gdf, col, norm_to)

    @staticmethod
    def normalize_array(rasterized_array,method):
        """Normalize a 2D NumPy array.
        """
        return _normalize_array(rasterized_array, method)

    @staticmethod
    def transform(array, method):
        """Function to transform rasterized array to map data values to relative favorability values.
        """
        return _transform(array, method)

    @staticmethod
    def rasterize_map_3d(gdf, col):
        """
        Rasterize a GeoDataFrame with 3D geometries into a 3D NumPy array.
        """
        if len(gdf) == 0:
            raise ValueError("GeoDataFrame 'gdf' is empty.")

        unique_x = np.sort(gdf.geometry.x.unique())
        unique_y = np.sort(gdf.geometry.y.unique())
        unique_z = np.sort(gdf.geometry.apply(lambda p: p.z if hasattr(p, 'z') else 0).unique())

        num_cols = len(unique_x)
        num_rows = len(unique_y)
        num_depths = len(unique_z)

        rasterized_map = np.zeros((num_depths, num_rows, num_cols), dtype=np.float32)

        tolerance = 1e-6

        for _, row in gdf.iterrows():
            value = row[col]
            col_idx = np.where(np.abs(unique_x - row.geometry.x) < tolerance)[0]
            row_idx = np.where(np.abs(unique_y - row.geometry.y) < tolerance)[0]
            depth_idx = np.where(np.abs(unique_z - row.geometry.z) < tolerance)[0]

            if len(col_idx) == 0 or len(row_idx) == 0 or len(depth_idx) == 0:
                print(f"Warning: Point ({row.geometry.x}, {row.geometry.y}, {row.geometry.z}) not found.")
                continue

            col_idx, row_idx, depth_idx = col_idx[0], row_idx[0], depth_idx[0]
            rasterized_map[depth_idx, row_idx, col_idx] = value

        return rasterized_map

    @staticmethod
    def derasterize_map_3d(rasterized_map, gdf_geom):
        """
        Convert a 3D rasterized NumPy array back to a GeoDataFrame with 3D points.
        """
        if len(gdf_geom) == 0:
            raise ValueError("GeoDataFrame 'gdf_geom' is empty.")

        unique_x = gdf_geom.geometry.x.unique()
        unique_y = gdf_geom.geometry.y.unique()
        unique_z = gdf_geom.geometry.apply(lambda p: p.z if hasattr(p, 'z') else 0).unique()

        crs = gdf_geom.crs
        num_cols = len(unique_x)
        num_rows = len(unique_y)
        num_depths = len(unique_z)

        geometries = []
        values = []

        for depth_idx in range(num_depths):
            for row_idx in range(num_rows):
                for col_idx in range(num_cols):
                    x_coord = unique_x[col_idx]
                    y_coord = unique_y[row_idx]
                    z_coord = unique_z[depth_idx]
                    value = rasterized_map[depth_idx, row_idx, col_idx]

                    point = Point(x_coord, y_coord, z_coord)
                    geometries.append(point)
                    values.append(value)

        gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
        gdf['favorability'] = values

        return gdf
