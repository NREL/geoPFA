"""Probabilistic component fits per geoPFA_prob.memo.20260612.pdf (near-term stage).

Implements:
  logit p_c(s,z) = alpha_c(s,z) + Sigma_k beta_ck x_k(s,z) + u_c(s)
                    physics-prior   evidence-layers          spatial-field

Near-term stage:
  - alpha_c: fixed offset derived from thermal model (pr0 in config)
  - beta_ck: learned from labeled wells via logistic regression
  - u_c: sequential approximation (GP-like smooth residual spatial field)

Evidence layers (beta_ck predictors) exclude spatial coordinates and sparse binary layers;
spatial structure is modeled separately via u_c, not mixed into the feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit


@dataclass(frozen=True)
class DemoComponentProbability:
    """Probability surface and fit metadata for one component."""

    probability: Any
    model: Any
    feature_names: tuple[str, ...]
    spatial_field: np.ndarray | None = None


def _numeric_columns(frame) -> list[str]:
    """Return numeric, non-geometry columns from a GeoDataFrame-like frame."""

    return [
        column
        for column in frame.columns
        if column != "geometry"
        and np.issubdtype(frame[column].dtype, np.number)
    ]


def _is_sparse_binary(values: np.ndarray, threshold: float = 0.1) -> bool:
    """Check if a column is sparse and binary-like (dominates spatial structure)."""
    # Remove NaNs
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return False

    # Check if mostly binary (only two distinct values)
    unique_vals = np.unique(valid)
    max_unique_for_binary = 3
    if len(unique_vals) > max_unique_for_binary:
        return False

    # Check if sparse (large regions with one value, small regions with another)
    counts = np.array([np.sum(valid == v) for v in unique_vals])
    max_frac = counts.max() / len(valid)
    return max_frac > (1.0 - threshold)  # > 90% same value = sparse


def _flatten_component_features(
    component_data: dict[str, Any],
    *,
    excluded_layer_names: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix from component's evidence layers (not including spatial).

    Per the near-term design, evidence layers are predictors β_ck; spatial structure
    is handled separately via u_c (spatial random effect). This excludes:
    - Sparse binary layers (e.g., quaternary_faults) which should be spatial, not predictors
    - Spatial coordinates (which are handled via sequential GP fit)
    - Any non-evidence numeric columns (e.g., `inverted_y` coordinate columns)
    """

    excluded_layer_names = excluded_layer_names or set()
    # Coordinate-like or non-evidence columns that must never enter the regression.
    non_evidence_columns = {
        "inverted_y",
        "x",
        "y",
        "X",
        "Y",
        "longitude",
        "latitude",
        "easting",
        "northing",
        "row",
        "col",
    }

    features: list[np.ndarray] = []
    names: list[str] = []

    for layer_name, layer_data in component_data["layers"].items():
        if layer_name in excluded_layer_names:
            continue
        model = layer_data["model"]
        # Use only the layer's declared evidence column when present; fall back to
        # `value_interpolated`, which is the standard processed column.
        evidence_col = layer_data.get("model_data_col") or "value_interpolated"
        if evidence_col not in model.columns:
            # As a last resort, pick the first numeric column that is not on the
            # non-evidence blocklist.
            evidence_col = next(
                (
                    col
                    for col in _numeric_columns(model)
                    if col not in non_evidence_columns
                ),
                None,
            )
            if evidence_col is None:
                continue

        if evidence_col in non_evidence_columns:
            continue

        values = model[evidence_col].to_numpy(dtype=float)

        # Skip sparse binary layers; they should be handled via spatial term, not predictors
        if _is_sparse_binary(values):
            continue

        features.append(values)
        names.append(f"{layer_name}:{evidence_col}")

    if not features:
        raise ValueError("No numeric layer features found for component demo.")
    return np.column_stack(features), names


def _standardize_features(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score features for logistic regression stability."""

    mean = np.nanmean(features, axis=0)
    scale = np.nanstd(features, axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (features - mean) / scale, mean, scale


def _fit_offset_logit(
    features: np.ndarray,
    labels: np.ndarray,
    offset: np.ndarray,
    regularization: float = 1e-2,
) -> Any:
    """Fit offset-logit model: logit p = alpha (offset) + X @ beta."""

    def objective(beta: np.ndarray) -> float:
        eta = offset + features @ beta
        prob = expit(eta)
        prob = np.clip(prob, 1e-9, 1.0 - 1e-9)
        nll = -np.sum(
            labels * np.log(prob) + (1.0 - labels) * np.log(1.0 - prob)
        )
        penalty = 0.5 * regularization * float(np.dot(beta, beta))
        return float(nll + penalty)

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = offset + features @ beta
        prob = expit(eta)
        grad = features.T @ (prob - labels)
        grad += regularization * beta
        return grad

    result = minimize(
        objective,
        x0=np.zeros(features.shape[1], dtype=float),
        jac=gradient,
        method="BFGS",
    )
    return result


def _fit_spatial_field(  # noqa: PLR0914
    x: np.ndarray,
    y: np.ndarray,
    residual: np.ndarray,
    *,
    max_points: int = 1500,
    seed: int = 13,
) -> np.ndarray:
    """Fit a smooth residual spatial field u_c(s) with sparse-RBF fallback."""

    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(residual))
    if not np.any(valid):
        return np.zeros_like(residual, dtype=float)

    xv = x[valid]
    yv = y[valid]
    rv = residual[valid]

    n = len(rv)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        xs = xv[idx]
        ys = yv[idx]
        rs = rv[idx]
    else:
        xs, ys, rs = xv, yv, rv

    # Distance-based epsilon heuristic from anchor sample.
    anchors = np.column_stack([xs, ys])
    if len(anchors) > 1:
        probe_n = min(len(anchors), 300)
        probe = anchors[:probe_n]
        d = cdist(probe, anchors)
        d[d == 0.0] = np.nan
        epsilon = float(np.nanmedian(np.nanmin(d, axis=1)))
        if not np.isfinite(epsilon) or epsilon <= 0:
            epsilon = 1.0
    else:
        epsilon = 1.0

    # Smoothness grows with residual variance.
    smooth = max(float(np.nanvar(rs) * 0.25), 1e-3)

    try:
        rbf = Rbf(
            xs,
            ys,
            rs,
            function="multiquadric",
            epsilon=epsilon,
            smooth=smooth,
        )
        u = np.zeros_like(residual, dtype=float)
        u_valid = rbf(xv, yv)
    except (
        Exception
    ):  # any RBF failure should fall back silently to a flat field
        return np.zeros_like(residual, dtype=float)
    else:
        u[valid] = np.clip(u_valid, -3.0, 3.0)
        return u


def _fit_spatial_field_to_targets(
    src_x: np.ndarray,
    src_y: np.ndarray,
    residual: np.ndarray,
    tgt_x: np.ndarray,
    tgt_y: np.ndarray,
) -> np.ndarray:
    """Fit an RBF residual field on source points and evaluate at target points."""

    valid = ~(np.isnan(src_x) | np.isnan(src_y) | np.isnan(residual))
    min_anchor_points = 4
    if valid.sum() < min_anchor_points:
        return np.zeros_like(tgt_x, dtype=float)

    xs = src_x[valid]
    ys = src_y[valid]
    rs = residual[valid]

    anchors = np.column_stack([xs, ys])
    if len(anchors) > 1:
        d = cdist(anchors, anchors)
        d[d == 0.0] = np.nan
        epsilon = float(np.nanmedian(np.nanmin(d, axis=1)))
        if not np.isfinite(epsilon) or epsilon <= 0:
            epsilon = 1.0
    else:
        epsilon = 1.0

    smooth = max(float(np.nanvar(rs) * 0.25), 1e-3)

    try:
        rbf = Rbf(
            xs, ys, rs, function="multiquadric", epsilon=epsilon, smooth=smooth
        )
        u = rbf(tgt_x, tgt_y)
        return np.clip(np.asarray(u, dtype=float), -3.0, 3.0)
    except Exception:
        return np.zeros_like(tgt_x, dtype=float)


def fit_component_probability(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    component_data: dict[str, Any],
    *,
    prior_probability: float,
    threshold_fraction: float = 1.0,
    include_spatial: bool = False,
    excluded_layer_names: tuple[str, ...] = (),
    labeled_wells: gpd.GeoDataFrame | None = None,
    label_column: str | None = None,
    prior_layer_name: str | None = None,
    prior_p_min: float = 0.2,
    prior_p_max: float = 0.8,
) -> DemoComponentProbability:
    """Fit near-term probabilistic component model.

    Structure: ``logit p_c = alpha_c + Sigma beta_ck x_k + u_c``

    The prior offset ``alpha_c`` has two modes. In the scalar mode it is
    ``logit(prior_probability)``, applied uniformly to every cell. In the
    spatial mode (Option 2 in the methodology memo) it is a per-cell
    offset built from a designated raster layer; the layer's interpolated
    values are min-max rescaled to ``[prior_p_min, prior_p_max]`` and then
    passed through ``logit`` to give ``alpha_c(s)``. The chosen layer is
    automatically excluded from the regression features to avoid
    double-counting. ``Sigma beta_ck x_k`` is the evidence-layer logistic
    regression. ``u_c`` is an optional spatial residual field fit by an
    RBF smoother.

    Training mode: if ``labeled_wells`` and ``label_column`` are provided,
    the regression is fit on evidence features sampled at well points
    using actual labels. Otherwise the legacy proxy-label path is used
    (favorability above a threshold on the grid); it is kept only for
    backward compatibility and is inconsistent with the labelled-well
    overlay.

    Parameters
    ----------
    component_data : dict
        Component config with layers and ``pr_norm`` surface.
    prior_probability : float
        Scalar ``P(component | no wells)``, used if no prior layer.
    threshold_fraction : float, optional
        Legacy proxy threshold; only used in fallback.
    include_spatial : bool, optional
        If True, fit spatial residual field ``u_c(s)``.
    excluded_layer_names : tuple of str, optional
        Layers to exclude from fit.
    labeled_wells : geopandas.GeoDataFrame, optional
        GeoDataFrame of labelled wells (preferred training data).
    label_column : str, optional
        Column on ``labeled_wells`` with 0/1 labels for this component.
    prior_layer_name : str, optional
        Optional layer name to use as a spatial prior offset. If provided
        AND the layer exists, it overrides the scalar
        ``prior_probability`` and is excluded from regression predictors.
    prior_p_min : float, optional
        Minimum probability bound for the spatial-prior rescale.
    prior_p_max : float, optional
        Maximum probability bound for the spatial-prior rescale.

    Returns
    -------
    DemoComponentProbability
        Probability surface and model metadata.
    """

    # Determine prior offset (scalar fallback or spatial from prior_layer_name).
    layers = component_data.get("layers", {})
    scalar_offset_value = float(
        np.log(prior_probability / (1.0 - prior_probability))
    )
    use_spatial_prior = (
        prior_layer_name is not None and prior_layer_name in layers
    )

    excluded_set = set(excluded_layer_names)
    if use_spatial_prior:
        excluded_set.add(prior_layer_name)

    # Build grid features and standardize using grid statistics.
    grid_gdf = component_data["pr_norm"].copy()
    X_grid, feature_names = _flatten_component_features(
        component_data,
        excluded_layer_names=excluded_set,
    )
    X_grid = np.where(
        np.isnan(X_grid), np.nanmean(X_grid, axis=0, keepdims=True), X_grid
    )
    X_grid_scaled, _, _ = _standardize_features(X_grid)

    # Compute grid-level prior offset.
    if use_spatial_prior:
        prior_layer = layers[prior_layer_name]["model"]
        prior_layer_aligned = grid_gdf[["geometry"]].copy()
        prior_layer_aligned = gpd.sjoin_nearest(
            prior_layer_aligned,
            prior_layer[["value_interpolated", "geometry"]].copy(),
            how="left",
            distance_col="__prior_dist__",
        )
        prior_vals = prior_layer_aligned["value_interpolated"].to_numpy(
            dtype=float
        )
        if np.all(~np.isfinite(prior_vals)):
            raise ValueError(
                f"prior layer '{prior_layer_name}' has no finite values; cannot build offset"
            )
        finite = np.isfinite(prior_vals)
        prior_vals[~finite] = np.nanmean(prior_vals[finite])
        v_min = float(np.nanmin(prior_vals))
        v_max = float(np.nanmax(prior_vals))
        if v_max <= v_min:
            prior_p_grid = np.full_like(
                prior_vals, (prior_p_min + prior_p_max) / 2.0
            )
        else:
            prior_p_grid = prior_p_min + (prior_vals - v_min) / (
                v_max - v_min
            ) * (prior_p_max - prior_p_min)
        prior_p_grid = np.clip(prior_p_grid, 1e-4, 1.0 - 1e-4)
        offset_grid = np.log(prior_p_grid / (1.0 - prior_p_grid))
        # Store the per-cell prior probability for downstream plotting.
        grid_gdf["prior_probability_spatial"] = prior_p_grid
    else:
        offset_grid = np.full(len(grid_gdf), scalar_offset_value)

    use_wells = (
        labeled_wells is not None
        and label_column is not None
        and label_column in labeled_wells.columns
    )

    feature_names_out = list(feature_names)
    spatial_u: np.ndarray | None = None

    if use_wells:
        # Sample standardized grid features (and prior offset) at well points.
        grid_feat = grid_gdf[["geometry"]].copy()
        for i, name in enumerate(feature_names):
            grid_feat[name] = X_grid_scaled[:, i]
        grid_feat["__prior_offset__"] = offset_grid

        wells_proj = labeled_wells.to_crs(grid_gdf.crs)
        wells_proj = wells_proj[wells_proj[label_column].notna()].copy()
        sampled = gpd.sjoin_nearest(
            wells_proj[[label_column, "geometry"]],
            grid_feat,
            how="inner",
            distance_col="_dist_m",
        )

        min_overlapping_wells = 4
        if len(sampled) < min_overlapping_wells:
            raise ValueError(
                "fewer than 4 labeled wells overlap component grid; cannot fit",
            )

        y_wells = sampled[label_column].astype(int).to_numpy()
        X_wells = sampled[list(feature_names)].to_numpy(dtype=float)
        offset_wells = sampled["__prior_offset__"].to_numpy(dtype=float)

        result = _fit_offset_logit(X_wells, y_wells, offset_wells)

        # Evaluate full grid using fitted coefficients.
        eta_grid = offset_grid + X_grid_scaled @ result.x

        if include_spatial:
            eta_wells = offset_wells + X_wells @ result.x
            p_wells = expit(eta_wells)
            residuals = y_wells - p_wells

            well_geom = sampled.geometry
            grid_geom = grid_gdf.geometry
            spatial_u = _fit_spatial_field_to_targets(
                well_geom.x.to_numpy(dtype=float),
                well_geom.y.to_numpy(dtype=float),
                residuals,
                grid_geom.x.to_numpy(dtype=float),
                grid_geom.y.to_numpy(dtype=float),
            )
            eta_grid += spatial_u
            feature_names_out.append("spatial:rbf_residual_at_wells")

        probabilities = expit(eta_grid)
        grid_gdf["probability"] = probabilities
        if spatial_u is not None:
            grid_gdf["spatial_u"] = spatial_u
        return DemoComponentProbability(
            probability=grid_gdf,
            model=result,
            feature_names=tuple(feature_names_out),
            spatial_field=spatial_u,
        )

    # ---- Fallback (legacy proxy-label fit; kept only for backward compatibility) ----
    favorability_values = grid_gdf["favorability"].to_numpy(dtype=float)
    threshold = threshold_fraction * 5.0 * prior_probability
    y = (favorability_values >= threshold).astype(int)
    if y.min() == y.max():
        threshold = float(np.nanmedian(favorability_values))
        y = (favorability_values >= threshold).astype(int)

    result = _fit_offset_logit(X_grid_scaled, y, offset_grid)
    eta_grid = offset_grid + X_grid_scaled @ result.x

    if include_spatial:
        coords = grid_gdf.geometry
        spatial_u = _fit_spatial_field(
            coords.x.to_numpy(dtype=float),
            coords.y.to_numpy(dtype=float),
            y - expit(eta_grid),
        )
        eta_grid += spatial_u
        feature_names_out.append("spatial:rbf_residual")

    probabilities = expit(eta_grid)
    grid_gdf["probability"] = probabilities
    if spatial_u is not None:
        grid_gdf["spatial_u"] = spatial_u
    return DemoComponentProbability(
        probability=grid_gdf,
        model=result,
        feature_names=tuple(feature_names_out),
        spatial_field=spatial_u,
    )


def combine_probability_surfaces(
    component_probabilities: list[np.ndarray],
) -> np.ndarray:
    """Combine component probabilities with the product rule."""

    if not component_probabilities:
        raise ValueError(
            "At least one component probability surface is required."
        )
    return np.prod(np.stack(component_probabilities, axis=0), axis=0)
