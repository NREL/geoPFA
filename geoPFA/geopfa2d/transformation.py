# -*- coding: utf-8 -*-
"""
Set of methods to transform data from evidence layers into evidence layers.
"""

import geopandas as gpd
import numpy as np
import shapely

class VoterVetoTransformation:
    """Class of functions for use in transforming data layers into evidence layers
    i.e., data values to 'favorability' values.
    """
    @staticmethod
    def normalize_gdf(gdf, col, norm_to=1): 
        """
        Normalize the values in a specified column of a GeoDataFrame using min-max scaling,
        such that the minimum value becomes 0 and the maximum value becomes norm_to.
        
        Parameters:
        - gdf: The GeoDataFrame containing the column to normalize.
        - col: The name of the column in the GeoDataFrame to normalize.
        - norm_to: The value to which the maximum column value should be scaled (default is 1).
        
        Returns:
        - A new GeoDataFrame with the normalized column.
        """
        # Find the min and max of the column
        min_val = gdf[col].min()
        max_val = gdf[col].max()
        
        # Avoid division by zero if all values in the column are the same
        if min_val == max_val:
            gdf[col] = norm_to  # All values are the same, set them to norm_to
        else:
            # Perform min-max normalization
            gdf[col] = (gdf[col] - min_val) / (max_val - min_val) * norm_to
        
        return gdf

    @staticmethod
    def normalize_array(rasterized_array,method):
        """Normalize a 2D NumPy array.

        Parameters
        ----------
        rasterized_array : np.ndarray)
            Input 2D NumPy array to be normalized.
        method : str
            Method to use to normalize rasterized_array. Can be one of 
            ['minmax','mad']

        Returns
        -------
        normalized_array : np.ndarray
            Normalized 2D NumPy array.
        """
        if method == 'minmax':
            # Find the minimum and maximum values in the array
            min_val = np.nanmin(rasterized_array)
            max_val = np.nanmax(rasterized_array)
            
            # Normalize the array to the range [0, 1]
            normalized_array = (rasterized_array - min_val) / (max_val - min_val)
        if method == 'mad':
            num = rasterized_array - np.nanmedian(rasterized_array)
            den = 1.482*np.nanmedian(num)
            normalized_array = num/den
        if method == 'mad':
            print('Normalized a layer using '+method+' >:(')
        else:
            print('Normalized a layer using '+method)
        return normalized_array

    @staticmethod
    def transform(array, method):
        """Function to transform rasterized array to map data values to relative favorability values.
        Includes several types of transformation methods
        
        Parameters
        ----------
        array : np.ndarray
            Input 2D rasterized np.array to transform
        method : str
            Method to transform data to relative favorability. Can be one of 
            ['inverse', 'negate', 'ln']
        Returns
        -------
        transformed_array : np.ndarray
            Array with data values transformed to relative favorability values
        """
        if (method == 'inverse') | (method == 'Inverse'):
            transformed_array = 1/array
        elif (method == 'negate') | (method == 'Negate'):
            transformed_array = -array
        elif (method == 'ln') | (method == 'Ln'):
            transformed_array = np.log(array)
        elif (method == 'none') | (method == 'None'):
            transformed_array = array
        else:
            raise ValueError('Transformation method ',method,' not yet implemented.')
        print('Transformed a layer using '+method)
        return transformed_array

    @staticmethod
    def rasterize_map(gdf, col):
        """Function to go from a geodataframe to a rasterized 2D numpy array representation for 
        use in linear algebra functions. Maintains the resolution of the geodataframe

        Parameters
        ----------
        gdf : geodataframe
            GeoDataFrame with point geometry containing map to rasterize
        col : str
            Name of column in gdf where data values are stored

        Returns
        -------
        rasterized_map : np.ndarray
            Numpy array containing 2D rasterized version of gdf
        """
        if len(gdf) == 0:
            raise ValueError("GeoDataFrame 'gdf' is empty.")

        # Get the unique x and y coordinates from the GeoDataFrame
        unique_x = np.sort(gdf.geometry.x.unique())
        unique_y = np.sort(gdf.geometry.y.unique())

        # Debugging: Print unique coordinates
        # print("Unique X Coordinates:", unique_x)
        # print("Unique Y Coordinates:", unique_y)

        # Check if unique_y is empty
        if len(unique_y) == 0:
            raise ValueError("No y-coordinates found in 'gdf'.")

        # Determine the number of unique x and y coordinates
        num_cols = len(unique_x)
        num_rows = len(unique_y)

        # Create an empty 2D NumPy array representing the rasterized map
        rasterized_map = np.zeros((num_rows, num_cols), dtype=np.float32)  # Use float32 to support non-integer values

        # Invert the y-coordinates
        min_y = gdf.geometry.y.min()
        max_y = gdf.geometry.y.max()
        gdf['inverted_y'] = min_y + (max_y - gdf.geometry.y)

        # Debugging: Print inverted_y values
        # print("Inverted Y Values:", gdf['inverted_y'].unique())

        # Tolerance for floating point comparisons
        tolerance = 1e-6

        # Iterate over each point in the GeoDataFrame and rasterize onto the map
        for _, row in gdf.iterrows():
            # Extract the associated value or column at the point
            value = row[col]
            
            # Find the index of the point's coordinates in the unique x and y arrays
            col_idx = np.where(np.abs(unique_x - row.geometry.x) < tolerance)[0]
            row_idx = np.where(np.abs(unique_y - row.inverted_y) < tolerance)[0]

            # Debugging: Print coordinate indices
            # print(f"Point ({row.geometry.x}, {row.geometry.y}) -> Col Index: {col_idx}, Row Index: {row_idx}")

            if len(col_idx) == 0 or len(row_idx) == 0:
                print(f"Warning: Point ({row.geometry.x}, {row.geometry.y}) not found in unique coordinates")
                continue

            col_idx = col_idx[0]
            row_idx = row_idx[0]

            # Update the pixel value with the associated value
            rasterized_map[row_idx, col_idx] = value

        return rasterized_map

    @staticmethod
    def derasterize_map(rasterized_map, gdf_geom):
        """Function to go from a rasterized 2D numpy array representation back to a geodataframe.
        Retains geometry of the original geodataframe that was rasterized.

        Parameters
        ----------
        rasterized_map : np.ndarray
            Numpy array containing 2D rasterized version of gdf
        gdf_geom : Pandas GeoSeries
            GeoSeries representing geomtry from original GeoDataFrame to use to transform 
            rasterized array back into a GeoDataFrame.

        Returns
        -------
        gdf : geodataframe
            GeoDataFrame with point geometry containing map to rasterize
        """
        if len(gdf_geom) == 0:
            raise ValueError("GeoDataFrame 'gdf_geom' is empty.")

        # Get the unique x and y coordinates from the GeoDataFrame
        unique_x = gdf_geom.geometry.x.unique()
        unique_y = gdf_geom.geometry.y.unique()
        crs = gdf_geom.crs

        # Determine the number of unique x and y coordinates
        num_cols = len(unique_x)
        num_rows = len(unique_y)

        # Create an empty list to store Point geometries
        geometries = []
        rasterized_map = np.flipud(rasterized_map)

        # Iterate over each row and column in the rasterized map
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                # Calculate the x and y coordinates of the raster cell
                x_coord = unique_x[col_idx]
                y_coord = unique_y[row_idx]

                # Create a Point geometry using the x and y coordinates
                point = shapely.geometry.Point(x_coord, y_coord)

                # Append the Point geometry to the list
                geometries.append(point)

        # Create a GeoDataFrame from the geometries and rasterized values
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
        # Assign the values from the rasterized map to the 'col' column of the GeoDataFrame
        gdf['favorability'] = rasterized_map.flatten()
        return gdf

