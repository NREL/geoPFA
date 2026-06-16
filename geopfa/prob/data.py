"""Stage-1 data utilities for labeled wells and evidence sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.special import expit

_LABEL_COLUMNS = ("heat_label", "reservoir_label", "barrier_label")
_REQUIRED_COLUMNS = (
    "well_id",
    "label_source",
    "label_quality",
    *_LABEL_COLUMNS,
)
_EPSG_WGS84 = 4326
_EPSG_NAD83_UTM11N = 26911
_EPSG_WGS84_UTM11N = 32611


def _validate_component(component: str | None) -> str | None:
    if component is None:
        return None
    if component not in {"heat", "reservoir", "barrier"}:
        raise ValueError("component must be one of: heat, reservoir, barrier")
    return component


def _ensure_required_columns(frame: pd.DataFrame) -> None:
    missing = [
        column for column in _REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"missing required labeled-well columns: {missing}")


def load_labeled_wells(
    path: str | Path, layer: str | None = None, component: str | None = None
):
    """Load labeled wells from GeoPackage or CSV and return EPSG:4326 GeoDataFrame."""

    component = _validate_component(component)
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".gpkg":
        gdf = (
            gpd.read_file(path, layer=layer)
            if layer is not None
            else gpd.read_file(path)
        )
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        if "longitude" not in frame.columns or "latitude" not in frame.columns:
            raise ValueError(
                "CSV input must include longitude and latitude columns"
            )
        gdf = gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
            crs="EPSG:4326",
        )
    else:
        raise ValueError(f"unsupported labeled-well format: {path.suffix}")

    if gdf.crs is None:
        raise ValueError("labeled-well input must have a CRS")
    gdf = gdf.to_crs(epsg=_EPSG_WGS84)
    if (
        not np.isfinite(gdf.total_bounds).all()
        and gdf.crs is not None
        and gdf.crs.to_epsg() == _EPSG_WGS84
    ):
        original = (
            gpd.read_file(path, layer=layer) if suffix == ".gpkg" else None
        )
        if (
            original is not None
            and original.crs is not None
            and original.crs.to_epsg() == _EPSG_NAD83_UTM11N
        ):
            original = original.set_crs(
                epsg=_EPSG_WGS84_UTM11N,
                allow_override=True,
            ).to_crs(epsg=_EPSG_WGS84)
            gdf = original

    _ensure_required_columns(gdf)

    if "longitude" not in gdf.columns:
        gdf["longitude"] = gdf.geometry.x
    if "latitude" not in gdf.columns:
        gdf["latitude"] = gdf.geometry.y
    if "depth_m" not in gdf.columns:
        gdf["depth_m"] = np.nan

    out = gdf[
        [
            "well_id",
            "longitude",
            "latitude",
            "depth_m",
            "heat_label",
            "reservoir_label",
            "barrier_label",
            "label_source",
            "label_quality",
            "geometry",
        ]
    ].copy()

    if component is not None:
        out = out[out[f"{component}_label"].notna()].copy()

    return out


def sample_evidence_at_wells(
    gdf: gpd.GeoDataFrame,
    raster_paths: dict[str, str | Path],
    band: int = 1,
    nodata_to_nan: bool = True,
) -> pd.DataFrame:
    """Sample rasters at well locations and return one feature column per raster."""

    if gdf.crs is None:
        raise ValueError("gdf must have a valid CRS")
    if not raster_paths:
        raise ValueError("raster_paths cannot be empty")

    out = pd.DataFrame(index=gdf.index)
    base = gdf.to_crs(epsg=_EPSG_WGS84)

    for feature_name, raster_path in raster_paths.items():
        raster_path_obj = Path(raster_path)
        with rasterio.open(raster_path_obj) as dataset:
            raster_epsg = (
                dataset.crs.to_epsg() if dataset.crs is not None else None
            )
            target_crs = dataset.crs
            if raster_epsg == _EPSG_NAD83_UTM11N:
                target_crs = "EPSG:32611"
            reproj = base.to_crs(target_crs)
            coords = list(
                zip(
                    reproj.geometry.x.to_numpy(),
                    reproj.geometry.y.to_numpy(),
                    strict=False,
                ),
            )
            values: list[float] = []
            for sample in dataset.sample(coords, indexes=band):
                value = float(sample[0])
                if (
                    nodata_to_nan
                    and dataset.nodata is not None
                    and np.isclose(value, dataset.nodata)
                ):
                    value = np.nan
                values.append(value)
            out[feature_name] = np.asarray(values, dtype=float)

    return out


def construct_alpha_c(  # noqa: PLR0913, PLR0917
    gdf: gpd.GeoDataFrame,
    heat_raster_path: str | Path,
    t_star_celsius: float = 150.0,
    depth_km: float = 3.0,
    transform: str = "logit",
    clip_eps: float = 1e-4,
) -> np.ndarray:
    """Construct heat-component offset alpha_c from sampled heat raster values."""

    del depth_km  # metadata for later stages

    sampled = sample_evidence_at_wells(gdf, {"heat": heat_raster_path})[
        "heat"
    ].to_numpy(dtype=float)

    if transform == "logit_threshold":
        probability = expit((sampled - t_star_celsius) / 10.0)
    elif transform == "logit":
        probability = sampled.copy()
        finite = np.isfinite(probability)
        if np.any(finite):
            finite_values = probability[finite]
            if (
                np.nanmin(finite_values) < 0.0
                or np.nanmax(finite_values) > 1.0
            ):
                ranks = (
                    pd.Series(finite_values)
                    .rank(method="average")
                    .to_numpy(dtype=float)
                )
                probability[finite] = (ranks - 0.5) / len(ranks)
    else:
        raise ValueError("transform must be 'logit' or 'logit_threshold'")

    alpha = np.full_like(probability, np.nan, dtype=float)
    finite = np.isfinite(probability)
    clipped = np.clip(probability[finite], clip_eps, 1.0 - clip_eps)
    alpha[finite] = np.log(clipped / (1.0 - clipped))
    return alpha


def prepare_component_arrays(  # noqa: PLR0913, PLR0917
    gdf: gpd.GeoDataFrame,
    X_df: pd.DataFrame,
    alpha_c: np.ndarray,
    component: str,
    coords_epsg: int = 32610,
    exclude_pseudo_absence: bool = True,
):
    """Prepare y, X, alpha, and projected coordinates for one component fit."""

    component = _validate_component(component)
    assert component is not None
    label_column = f"{component}_label"
    if label_column not in gdf.columns:
        raise ValueError(f"missing label column: {label_column}")

    if len(alpha_c) != len(gdf):
        raise ValueError("alpha_c must have the same length as gdf")
    if len(X_df) != len(gdf):
        raise ValueError("X_df must have the same length as gdf")

    mask = gdf[label_column].notna().to_numpy()
    if exclude_pseudo_absence:
        mask &= gdf["label_quality"].to_numpy() != "pseudo_absence"

    feature_values = X_df.to_numpy(dtype=float)
    mask &= np.isfinite(feature_values).all(axis=1)
    mask &= np.isfinite(alpha_c)

    gdf_masked = gdf.loc[mask].copy()
    y = gdf_masked[label_column].to_numpy(dtype=int)
    X = X_df.loc[mask].to_numpy(dtype=float)
    alpha = alpha_c[mask]

    coords = gdf_masked.to_crs(epsg=coords_epsg).geometry
    coords_proj = np.column_stack(
        [coords.x.to_numpy(dtype=float), coords.y.to_numpy(dtype=float)]
    )

    return y, X, alpha, coords_proj
