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

    @staticmethod
    def export_favorability_models(  # noqa: PLR0913, PLR0917
        pfa,
        output_dir,
        target_crs="EPSG:4326",
        fmt="shp",
        level="all",
        criteria=None,
        component=None,
    ):
        """
        Export favorability models (combined, criteria, component).

        Parameters
        ----------
        pfa : dict
            PFA dictionary after running do_voter_veto.
        output_dir : str or Path
            Directory to write outputs to.
        target_crs : str, optional
            CRS to export to.
        fmt : {"shp", "csv", "both"}, optional
            Output file format.
        level : {"all", "combined", "criteria", "component"}, optional
            Controls which levels of the PFA favorability hierarchy are exported.
            - "combined"
                Export only the final combined favorability model (pfa["pr_norm"] or pfa["pr"]).
            - "criteria"
                Export one or more criteria-level models (pfa["criteria"][...]["pr_norm"]).
                If ``criteria`` is provided, only that criterion is exported; otherwise,
                all criteria are exported.
            - "component"
                Export component-level models within a criterion
                (pfa["criteria"][...]["components"][...]["pr_norm"]).
                Requires ``criteria`` to be specified. If ``component`` is provided,
                only that component is exported; otherwise, all components within the
                specified criterion are exported.
            - "all"
                Export combined, all criteria-level, and all component-level models.
        criteria : str, optional
            If provided, only export this criterion.
        component : str, optional
            If provided, only export this component (requires criteria).

        Notes
        -----
        - Uses "pr_norm" if available, otherwise falls back to "pr".
        - Outputs are consistently named using "model" convention.
        """

        if component and not criteria:
            raise ValueError(
                "Must specify 'criteria' when filtering by component."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def _get_gdf(d):
            if "pr_norm" in d and d["pr_norm"] is not None:
                return d["pr_norm"]
            if "pr" in d and d["pr"] is not None:
                return d["pr"]
            return None

        def _write_all(gdf, base_path):
            if fmt in {"shp", "both"}:
                GeospatialDataWriters.write_shapefile(
                    gdf, base_path.with_suffix(".shp"), target_crs
                )
            if fmt in {"csv", "both"}:
                GeospatialDataWriters.write_csv(
                    gdf, base_path.with_suffix(".csv"), target_crs
                )
            if fmt not in {"shp", "csv", "both"}:
                raise ValueError(f"Invalid format: {fmt}")

        print("\nExporting favorability models...\n")

        # --- combined ---
        if level in {"all", "combined"}:
            gdf = _get_gdf(pfa)
            if gdf is None:
                raise ValueError("No combined favorability model found.")

            out_fp = output_dir / "combined_favorability_model"
            _write_all(gdf, out_fp)

            print(f"Combined model written to: {out_fp}")

        # --- criteria + components ---
        if level in {"all", "criteria", "component"}:
            for crit_name, crit_data in pfa.get("criteria", {}).items():
                if criteria and crit_name != criteria:
                    continue

                # ---- criteria-level ----
                if level in {"all", "criteria"}:
                    gdf = _get_gdf(crit_data)
                    if gdf is not None:
                        crit_out_dir = (
                            output_dir
                            / f"{crit_name}_criteria_favorability_models"
                        )
                        crit_out_dir.mkdir(exist_ok=True)

                        out_fp = (
                            crit_out_dir
                            / f"{crit_name}_criteria_favorability_model"
                        )
                        _write_all(gdf, out_fp)

                        print(f"\tWrote {crit_name} criteria model")

                # ---- component-level ----
                if level in {"all", "component"}:
                    for comp_name, comp_data in crit_data.get(
                        "components", {}
                    ).items():
                        if component and comp_name != component:
                            continue

                        gdf = _get_gdf(comp_data)
                        if gdf is not None:
                            comp_out_dir = (
                                output_dir
                                / f"{comp_name}_component_favorability_models"
                            )
                            comp_out_dir.mkdir(exist_ok=True)

                            out_fp = (
                                comp_out_dir
                                / f"{comp_name}_component_favorability_model"
                            )
                            _write_all(gdf, out_fp)

                            print(f"\t\tWrote {comp_name} component model")

        print("\nFinished exporting favorability models.\n")
