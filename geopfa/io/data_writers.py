"""Set of methods to read in data in various formats."""

import os
import json
from pathlib import Path
import pandas as pd
import geopandas as gpd


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
            gdf = gdf.set_crs(target_crs)

        gdf.to_crs(target_crs).to_file(path)

    @staticmethod
    def write_csv(gdf, path, target_crs="EPSG:4326"):
        """Writes geopandas dataframe to a CSV file.

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
            gdf = gdf.set_crs(target_crs)

        gdf.to_crs(target_crs).to_csv(path, index=False)

    @staticmethod
    def save_processed_layers(pfa, data_dir):
        """
        Save processed PFA layers (GeoDataFrames) to CSV files.

        Each layer is written to:
            data_dir / criteria / component / "<layer>_processed.csv"

        Parameters
        ----------
        pfa : dict
            PFA dictionary containing processed data.
        data_dir : str or Path
            Root directory where processed data will be saved.

        Raises
        ------
        ValueError
            If required data is missing or invalid.
        """

        data_dir = Path(data_dir)

        if not data_dir.exists():
            raise ValueError(f"data_dir does not exist: {data_dir}")

        print("\nSaving processed layers...\n")

        for criteria, crit_data in pfa.get("criteria", {}).items():
            print(criteria)

            for component, comp_data in crit_data.get(
                "components", {}
            ).items():
                print(f"\t{component}")

                for layer, layer_data in comp_data.get("layers", {}).items():
                    if (
                        "model" not in layer_data
                        or layer_data["model"] is None
                    ):
                        raise ValueError(
                            f"Missing 'model' data for {criteria}/{component}/{layer}"
                        )

                    gdf = layer_data["model"]

                    if not isinstance(gdf, pd.DataFrame):
                        raise TypeError(
                            f"'model' is not a DataFrame for {criteria}/{component}/{layer}"
                        )

                    out_fp = (
                        data_dir
                        / criteria
                        / component
                        / f"{layer}_processed.csv"
                    )

                    GenericFunctions.ensure_directory_exists(str(out_fp))

                    try:
                        gdf.to_csv(out_fp, index=False)
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to write CSV for {criteria}/{component}/{layer} → {out_fp}"
                        ) from e

                    print(f"\t\tSaved: {out_fp}")

        print("\nFinished saving processed layers.\n")

    @staticmethod
    def _drop_geodataframes(data):
        """Recursively remove GeoDataFrame objects from a nested dictionary."""
        clean = {}

        for key, val in data.items():
            if isinstance(val, gpd.GeoDataFrame):
                continue
            if isinstance(val, dict):
                clean[key] = GeospatialDataWriters._drop_geodataframes(val)
            else:
                clean[key] = val

        return clean

    @staticmethod
    def save_clean_pfa_config(pfa, output_path):
        """
        Save a "clean" PFA configuration by removing GeoDataFrames and writing to JSON.

        Parameters
        ----------
        pfa : dict
            PFA dictionary.
        output_path : str or Path
            Output JSON file path.

        Raises
        ------
        ValueError
            If output path is invalid.
        RuntimeError
            If writing fails.
        """

        output_path = Path(output_path)

        if output_path.suffix.lower() != ".json":
            raise ValueError(
                f"Output path must be a .json file: {output_path}"
            )

        print("\nSaving clean PFA configuration...\n")

        try:
            pfa_clean = GeospatialDataWriters._drop_geodataframes(pfa)
        except Exception as e:
            raise RuntimeError("Failed while removing GeoDataFrames") from e

        GenericFunctions.ensure_directory_exists(str(output_path))

        try:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(pfa_clean, f, indent=4)
        except Exception as e:
            raise RuntimeError(
                f"Failed to write JSON config → {output_path}"
            ) from e

        print(f"Processed PFA configuration saved to: {output_path}\n")
