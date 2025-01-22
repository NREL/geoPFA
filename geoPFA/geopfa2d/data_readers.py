# -*- coding: utf-8 -*-
"""
Set of methods to read in data in various formats.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt
import shapely
import rasterio
import os

class GeospatialDataReaders:
    """Class of functions to read in geospatial data in various formats"""
    @staticmethod
    def read_shapefile(path):
        """Reads in a shapefile and returns a geopandas dataframe.
    
        Parameters
        ----------
        path : 'str'
            Path to shapefile

        Returns
        -------
        data : Geopandas DataFrame
            Geopandas DataFrame containing contents of shapefile
        """
        data = gpd.read_file(path)
        return data

    @staticmethod
    def read_csv(path, crs, x_col=None, y_col=None, z_col=None, geometry_column_name='geometry'):
        """Reads in a CSV file and returns a geopandas dataframe.
    
        Parameters
        ----------
        path : 'str'
            Path to csv file
        crs : 'str'
            String version of coordinate reference system associated with the CSV file
        x_col : 'str'
            Name of x geometry column if no combined geometry column is provided.
        y_col : 'str'
            Name of y geometry column if no combined geometry column is provided.
        z_col : 'str'
            Name of z geometry column if 3d, and if no combined geometry column is provided.
        geometry_column_name : str
            Name of column containing the geometry information. Defaults to 'geometry.'

        Returns
        -------
        gdf : Geopandas DataFrame
            Geopandas DataFrame containing contents of CSV file
        """
        df = pd.read_csv(path)
        if sum([(x_col is None), (y_col is None)]) == 1:
            raise ValueError("Must specifiy both x_col and y_col, or a combined geometry column.")
        elif sum([(x_col is None),(y_col is None),(z_col is None)]) == 1:
            raise ValueError("Cannnot specify z_col without also specifying x_col and y_col.")
        
        if x_col is None and y_col is None and z_col is None:
            df = df.rename(columns={geometry_column_name:'geometry'})
            df['geometry'] = df['geometry'].apply(wkt.loads)
            gdf =  gpd.GeoDataFrame(df, crs=crs)
        elif z_col is None:
            gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(df[x_col], df[y_col])
            )
        else:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=[shapely.geometry.Point(x, y, z) for x, y, z in zip(df[x_col], df[y_col], df[z_col])]
            )
        
        gdf.set_crs(crs, inplace=True)
        return gdf

    @staticmethod
    def read_raster(path):
        """Reads in a raster file and returns a rasterio dataset object.
    
        Parameters
        ----------
        path : 'str'
            Path to raster file

        Returns
        -------
        data : Geopandas DataFrame
            Geopandas DataFrame containing contents of raster file
        """
        # TODO: Return geopandas dataframe instead of rasterio dataset object
        data = rasterio.open(path)
        return data
    
    @staticmethod
    def read_tif(tif_path):
        """Read a TIFF file and convert it to a GeoDataFrame with points.

        Parameters
        ----------
        tif_path : str
            Path to the TIFF file.

        Returns
        -------
        GeoDataFrame
            GeoDataFrame with points representing the raster data.
        """
        # Open the TIFF file
        with rasterio.open(tif_path) as src:
            # Read the raster data
            data = src.read(1)  # Read the first band
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

        # Get all indices where data is not equal to nodata
        rows, cols = np.where(data != nodata)

        # Convert raster indices to spatial coordinates
        x_coords, y_coords = rasterio.transform.xy(transform, rows, cols)

        # Create points and values arrays
        points = [shapely.geometry.Point(x, y) for x, y in zip(x_coords, y_coords)]
        values = data[rows, cols]

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': points, 'value': values})
        gdf.set_crs(crs, inplace=True)

        return gdf

    @classmethod
    def gather_data(cls,data_dir,pfa,file_types):
        """Function to read in data layers associated with each component of each criteria.
        Note that data must be stored in a directory with the following structure which matches
        the config: criteria/component/layers. Criteria directory, component directory, and
        data file names must match the critera, components, and layers specified in the pfa,
        and file extensions must match those specified in file_types.

        Parameters
        ----------
        data_dir : str
            Path to directory where data is stored
        pfa : dictionary
            Config specifying criteria, components, and data layers' relationship to one another.
            Read in from json file.
        file_types : list
            List of file types to look for when gathering data. File types excluded from list will
            be ignored.
        csv_crs : int
            Integer value associated with CRS associated with csv files. Should be set to None if not 
            reading csv files.

        Returns
        -------
        pfa : dictionary
            Updated pfa config which includes data
        """
        for criteria in pfa['criteria']:
            print('criteria: '+criteria)
            for component in pfa['criteria'][criteria]['components']:
                print('\t component: '+component)
                COMPONENT_DIR = os.path.join(data_dir,criteria,component)
                file_names = sorted(os.listdir(COMPONENT_DIR))
                if '.shp' in file_types:
                    shapefile_names = [x for x in file_names if (x.endswith('.shp'))]
                    for layer in pfa['criteria'][criteria]['components'][component]['layers']:
                        if layer+'.shp' in shapefile_names:
                            print('\t\t reading layer: '+layer)
                            pfa['criteria'][criteria]['components'][component]['layers'][layer]['data']\
                                = GeospatialDataReaders.read_shapefile(os.path.join(COMPONENT_DIR,layer+'.shp'))
                if '.csv' in file_types:
                    csv_file_names = [x for x in file_names if (x.endswith('.csv'))]
                    for layer in pfa['criteria'][criteria]['components'][component]['layers']:
                        if layer+'.csv' in csv_file_names:
                            print('\t\t reading layer: '+layer)
                            layer_config = pfa['criteria'][criteria]['components'][component]['layers'][layer]
                            csv_crs = layer_config['crs']
                            if 'x_col' in layer_config and 'y_col' in layer_config:
                                x_col = layer_config['x_col']
                                y_col = layer_config['y_col']
                                if 'z_col' in layer_config:
                                    z_col = layer_config['z_col']
                                    data = GeospatialDataReaders.read_csv(os.path.join(COMPONENT_DIR,layer+'.csv'),csv_crs, x_col, y_col, z_col)
                                else:
                                    data = GeospatialDataReaders.read_csv(os.path.join(COMPONENT_DIR,layer+'.csv'),csv_crs, x_col, y_col)
                            else:
                               data = GeospatialDataReaders.read_csv(os.path.join(COMPONENT_DIR,layer+'.csv'),csv_crs)
                            pfa['criteria'][criteria]['components'][component]['layers'][layer]['data'] = data
                for file_type in file_types:
                    if file_type not in ['.shp', '.csv']:
                        print('Warning: file type: '+file_type+' not currently compatible with geoPFA.')
        return pfa
    
    @classmethod
    def gather_processed_data(cls,data_dir,pfa,crs):
        """Function to read in processed data layers associated with each component of each criteria.
        Note that data must be stored in a directory with the following structure which matches
        the config: criteria/component/layers. Criteria directory, component directory, and
        data file names must match the critera, components, and layers specified in the pfa,
        and file extensions must match those specified in file_types.

        Parameters
        ----------
        data_dir : str
            Path to directory where data is stored
        pfa : dictionary
            Config specifying criteria, components, and data layers' relationship to one another.
            Read in from json file.

        Returns
        -------
        pfa : dictionary
            Updated pfa config which includes data
        """
        for criteria in pfa['criteria']:
            print('criteria: '+criteria)
            for component in pfa['criteria'][criteria]['components']:
                print('\t component: '+component)
                COMPONENT_DIR = os.path.join(data_dir,criteria,component)
                file_names = sorted(os.listdir(COMPONENT_DIR))
                csv_file_names = [x for x in file_names if (x.endswith('_processed.csv'))]
                for layer in pfa['criteria'][criteria]['components'][component]['layers']:
                    if layer+'_processed.csv' in csv_file_names:
                        print('\t\t reading layer: '+layer)
                        pfa['criteria'][criteria]['components'][component]['layers'][layer]['map']\
                            = GeospatialDataReaders.read_csv(os.path.join(COMPONENT_DIR,layer+'_processed.csv'),crs)
        return pfa

    @classmethod
    def gather_exclusion_areas(cls, data_dir, pfa, target_crs):
        """Gathers/reads in exclusion area shapefiles for a given set of exclusion components and layers, 
        transforming them into the target coordinate reference system (CRS) and storing them in the `pfa` dictionary.

        This function iterates over the exclusion components and their associated layers in the `pfa` dictionary, 
        reads the corresponding shapefiles from the specified directory, filters the shapefile data based on the 
        `DN` field (keeping only entries where `DN > 0`), and reprojects the geometries to the target CRS. 
        The processed shapefiles are stored back into the `pfa` dictionary under the relevant component and layer.

        Parameters:
        ----------
        cls : class
            The class that the method belongs to. This is typically passed automatically in class methods.
        data_dir : str
            The directory path where the exclusion shapefiles are stored. The shapefiles are expected to be located 
            under a subdirectory named 'exclusion' within this directory.
        pfa : dict
            A dictionary containing spatial data and exclusion components. The function reads exclusion areas for each 
            component and layer and updates the dictionary with the processed shapefiles
        target_crs : str or dict
            The target Coordinate Reference System (CRS) to which the exclusion shapefiles will be reprojected. This can 
            be a CRS string (e.g., 'EPSG:4326') or a CRS dictionary format.
        
        Returns:
        -------
        dict
            The updated `pfa` dictionary, where the processed exclusion areas are stored under 
            `pfa['exclusions']['components'][exclusion_component]['layers'][layer]['map']`.

        Notes:
        ------
        - The function assumes that the exclusion shapefiles are stored in the `data_dir` under a subdirectory 
        named 'exclusion' and that the filenames match the layer names.
        - Only shapefile records where the `DN` field has a value greater than 0 are retained for further processing.
        - The shapefile geometries are reprojected to the specified `target_crs` to ensure consistent spatial reference.
        - The processed shapefiles are stored in the `pfa` dictionary under their respective exclusion components and layers.
        """


        for exclusion_component in pfa['exclusions']['components']:
            for layer in pfa['exclusions']['components'][exclusion_component]['layers']:
                print('reading '+layer)
                path = os.path.join(data_dir,'exclusion',layer+'.shp')
                shp = GeospatialDataReaders.read_shapefile(path)
                shp = shp[shp.DN > 0]
                shp = shp.to_crs(target_crs)
                pfa['exclusions']['components'][exclusion_component]['layers'][layer]['map'] = shp
        return pfa