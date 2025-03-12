"""Set of methods to read in data in various formats."""

import os


class GenericFunctions:
    """Class of functions compatible with any data type."""

    def ensure_directory_exists(file_path):
        """Ensure that the directory structure for a given file path exists.

        If the directory or any intermediate directories do not exist, they
        are created.

        Parameters
        ----------
        file_path : str
            The full file path for which to ensure directory existence. This
            path includes the file name and its intended directories.

        Notes
        -----
        - If the directory structure already exists, no new directories will
          be created.
        - This function does not create the file itself, only the necessary
          directories.
        - If the directory structure already exists, a message indicating so
          will be printed.
        """
        # Extract the directory path from the file path
        directory = os.path.dirname(file_path)

        # Check if the directory exists
        if not os.path.exists(directory):
            # Create the directory if it does not exist
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")


class GeospatialDataWriters:
    """Write geopandas dataframes to various geospatial data formats"""

    @staticmethod
    def write_shapefile(gdf, path, target_crs="EPSG:4326"):
        """Writes geopandas dataframe to a shapefile.

        Parameters
        ----------
        path : 'str'
            Path to shapefile to write to
        gdf : Geopandas DataFrame
            Geopandas DataFrame containing data to write to the shapefile
        target_crs : 'int'
            Integer value associated with the CRS you with to write to.
            Defaults to 4326
        """
        GenericFunctions.ensure_directory_exists(path)
        if gdf.crs is None:
            gdf.set_crs(target_crs)
        gdf.to_crs(target_crs).to_file(path)
