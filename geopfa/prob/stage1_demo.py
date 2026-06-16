"""Stage-1 probabilistic demo utilities for region-state scenario comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import warnings


@dataclass(frozen=True)
class Stage1ScenarioSpec:
    """Configuration for one region-state scenario."""

    name: str
    drop_features: tuple[str, ...] = ()
    include_priors: bool = True
    include_spatial: bool = True
    missing_fraction: float = 0.0


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    scale = np.nanstd(X, axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return (X - mean) / scale, mean, scale


def _fit_logit(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    regularization: float = 1e-2,
) -> np.ndarray:
    def objective(beta: np.ndarray) -> float:
        eta = offset + X @ beta
        p = np.clip(expit(eta), 1e-9, 1.0 - 1e-9)
        nll = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        return float(nll + 0.5 * regularization * np.dot(beta, beta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = offset + X @ beta
        p = expit(eta)
        return X.T @ (p - y) + regularization * beta

    x0 = np.zeros(X.shape[1])
    for method in ("BFGS", "L-BFGS-B"):
        result = minimize(objective, x0=x0, jac=gradient, method=method)
        if result.success and np.isfinite(result.x).all():
            return result.x
        x0 = result.x if np.isfinite(result.x).all() else x0
    warnings.warn(
        f"logit fit did not converge ({result.message}); using zero-coefficient fallback",
        RuntimeWarning,
        stacklevel=2,
    )
    return np.zeros(X.shape[1], dtype=float)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


_DEFAULT_THRESHOLD = 0.5
_MIN_CLASSES = 2
_MIN_PLATT_SAMPLES = 4
_MIN_DIMS = 2


def _accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= _DEFAULT_THRESHOLD) == (y == 1)))


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    if np.unique(y).size < _MIN_CLASSES:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    pos = y == 1
    n_pos = float(np.sum(pos))
    n_neg = float(np.sum(~pos))
    return float(
        (np.sum(ranks[pos]) - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg),
    )


def _fit_platt_scaler(
    scores: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Fit Platt scaling parameters a,b for p = sigmoid(a*s + b)."""
    if len(scores) < _MIN_PLATT_SAMPLES or np.unique(y).size < _MIN_CLASSES:
        return 1.0, 0.0

    def objective(theta: np.ndarray) -> float:
        a, b = float(theta[0]), float(theta[1])
        p = np.clip(expit(a * scores + b), 1e-9, 1.0 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    result = minimize(
        objective,
        x0=np.array([1.0, 0.0], dtype=float),
        method="BFGS",
    )
    if result.success and np.isfinite(result.x).all():
        return float(result.x[0]), float(result.x[1])
    return 1.0, 0.0


def _block_ids(coords: np.ndarray, grid_size: int = 4) -> np.ndarray:
    if coords.ndim != _MIN_DIMS or coords.shape[1] != _MIN_DIMS:
        raise ValueError("coords must have shape (n, 2)")
    x = coords[:, 0]
    y = coords[:, 1]
    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), grid_size + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), grid_size + 1)
    x_bin = np.clip(
        np.digitize(x, x_edges[1:-1], right=False),
        0,
        grid_size - 1,
    )
    y_bin = np.clip(
        np.digitize(y, y_edges[1:-1], right=False),
        0,
        grid_size - 1,
    )
    return x_bin + grid_size * y_bin


def spatial_block_holdout_mask(
    coords: np.ndarray,
    holdout_fraction: float = 0.2,
    grid_size: int = 4,
) -> np.ndarray:
    """Return boolean mask where True marks holdout samples by spatial block."""

    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be in (0, 1)")
    blocks = _block_ids(coords, grid_size=grid_size)
    unique = np.unique(blocks)
    n_holdout = max(1, int(np.ceil(len(unique) * holdout_fraction)))
    holdout_blocks = set(unique[:n_holdout].tolist())
    return np.array([block in holdout_blocks for block in blocks], dtype=bool)


def _fit_and_eval(  # noqa: PLR0913, PLR0917, PLR0914
    X_train: np.ndarray,
    y_train: np.ndarray,
    offset_train: np.ndarray,
    coords_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    offset_test: np.ndarray,
) -> dict[str, float]:
    calibration_mask = spatial_block_holdout_mask(
        coords_train,
        holdout_fraction=0.2,
        grid_size=3,
    )
    model_mask = ~calibration_mask
    if (
        model_mask.sum() < _MIN_CLASSES
        or calibration_mask.sum() < _MIN_CLASSES
    ):
        model_mask = np.ones(len(y_train), dtype=bool)
        calibration_mask = np.zeros(len(y_train), dtype=bool)

    X_model = X_train[model_mask]
    y_model = y_train[model_mask]
    offset_model = offset_train[model_mask]
    X_model_std, mean, scale = _standardize(X_model)
    X_test_std = (X_test - mean) / scale
    beta = _fit_logit(X_model_std, y_model, offset_model)
    test_scores = offset_test + X_test_std @ beta
    p_test = expit(test_scores)

    p_test_calibrated = p_test
    if calibration_mask.any():
        X_cal = X_train[calibration_mask]
        y_cal = y_train[calibration_mask]
        offset_cal = offset_train[calibration_mask]
        X_cal_std = (X_cal - mean) / scale
        cal_scores = offset_cal + X_cal_std @ beta
        a, b = _fit_platt_scaler(cal_scores, y_cal)
        p_test_calibrated = expit(a * test_scores + b)

    return {
        "log_loss": _log_loss(y_test, p_test),
        "brier": _brier(y_test, p_test),
        "accuracy": _accuracy(y_test, p_test),
        "auc": _roc_auc(y_test, p_test),
        "log_loss_calibrated": _log_loss(y_test, p_test_calibrated),
        "brier_calibrated": _brier(y_test, p_test_calibrated),
        "accuracy_calibrated": _accuracy(y_test, p_test_calibrated),
        "auc_calibrated": _roc_auc(y_test, p_test_calibrated),
    }


def _scenario_arrays(
    X_df: pd.DataFrame,
    y: np.ndarray,
    alpha: np.ndarray,
    coords: np.ndarray,
    spec: Stage1ScenarioSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    missing_features = [
        name for name in spec.drop_features if name not in X_df.columns
    ]
    if missing_features:
        raise ValueError(
            f"scenario {spec.name} references unknown features: {missing_features}"
        )

    selected = X_df.drop(columns=list(spec.drop_features)).copy()
    if selected.shape[1] == 0:
        raise ValueError(f"scenario {spec.name} dropped all features")

    work = selected.to_numpy(dtype=float)
    if spec.missing_fraction > 0:
        n_missing = int(np.floor(len(work) * spec.missing_fraction))
        if n_missing > 0:
            work = work.copy()
            work[:n_missing, 0] = np.nan

    if spec.include_spatial:
        work = np.column_stack([work, coords])

    offset = alpha.copy() if spec.include_priors else np.zeros_like(alpha)
    mask = np.isfinite(work).all(axis=1) & np.isfinite(offset) & np.isfinite(y)
    return (
        work[mask],
        y[mask],
        offset[mask],
        coords[mask],
        tuple(selected.columns.tolist()),
    )


def run_region_state_matrix(  # noqa: PLR0913, PLR0917, PLR0914
    X_df: pd.DataFrame,
    y: np.ndarray,
    alpha: np.ndarray,
    coords: np.ndarray,
    scenarios: Iterable[Stage1ScenarioSpec],
    n_splits: int = 4,
) -> pd.DataFrame:
    """Run holdout + block-CV metrics for each scenario."""
    rows: list[dict[str, float | int | str]] = []

    for spec in scenarios:
        X_s, y_s, alpha_s, coords_s, feature_names = _scenario_arrays(
            X_df,
            y,
            alpha,
            coords,
            spec,
        )
        holdout_mask = spatial_block_holdout_mask(
            coords_s,
            holdout_fraction=0.2,
            grid_size=4,
        )
        train_mask = ~holdout_mask
        if train_mask.sum() < _MIN_CLASSES or holdout_mask.sum() < 1:
            raise ValueError(
                f"scenario {spec.name} has insufficient train/test samples",
            )

        holdout = _fit_and_eval(
            X_s[train_mask],
            y_s[train_mask],
            alpha_s[train_mask],
            coords_s[train_mask],
            X_s[holdout_mask],
            y_s[holdout_mask],
            alpha_s[holdout_mask],
        )

        blocks = _block_ids(coords_s, grid_size=4)
        unique_blocks = np.unique(blocks)
        cv_metrics = {
            "log_loss": [],
            "brier": [],
            "accuracy": [],
            "auc": [],
            "log_loss_calibrated": [],
            "brier_calibrated": [],
            "accuracy_calibrated": [],
            "auc_calibrated": [],
        }
        for fold in range(max(1, min(n_splits, len(unique_blocks)))):
            test_blocks = set(unique_blocks[fold::n_splits].tolist())
            test = np.array(
                [block in test_blocks for block in blocks],
                dtype=bool,
            )
            train = ~test
            if train.sum() < _MIN_CLASSES or test.sum() < 1:
                continue
            fold_out = _fit_and_eval(
                X_s[train],
                y_s[train],
                alpha_s[train],
                coords_s[train],
                X_s[test],
                y_s[test],
                alpha_s[test],
            )
            for key, value in fold_out.items():
                cv_metrics[key].append(value)

        row: dict[str, float | int | str] = {
            "scenario": spec.name,
            "n_samples": len(y_s),
            "n_features": int(
                len(feature_names) + (2 if spec.include_spatial else 0)
            ),
            "priors_enabled": str(spec.include_priors),
            "spatial_enabled": str(spec.include_spatial),
            "dropped_features": ",".join(spec.drop_features),
            "holdout_log_loss": holdout["log_loss"],
            "holdout_brier": holdout["brier"],
            "holdout_accuracy": holdout["accuracy"],
            "holdout_auc": holdout["auc"],
            "holdout_log_loss_calibrated": holdout["log_loss_calibrated"],
            "holdout_brier_calibrated": holdout["brier_calibrated"],
            "holdout_accuracy_calibrated": holdout["accuracy_calibrated"],
            "holdout_auc_calibrated": holdout["auc_calibrated"],
            "cv_log_loss_mean": float(np.nanmean(cv_metrics["log_loss"]))
            if cv_metrics["log_loss"]
            else float("nan"),
            "cv_brier_mean": float(np.nanmean(cv_metrics["brier"]))
            if cv_metrics["brier"]
            else float("nan"),
            "cv_accuracy_mean": float(np.nanmean(cv_metrics["accuracy"]))
            if cv_metrics["accuracy"]
            else float("nan"),
            "cv_auc_mean": float(np.nanmean(cv_metrics["auc"]))
            if cv_metrics["auc"]
            else float("nan"),
            "cv_log_loss_calibrated_mean": float(
                np.nanmean(cv_metrics["log_loss_calibrated"])
            )
            if cv_metrics["log_loss_calibrated"]
            else float("nan"),
            "cv_brier_calibrated_mean": float(
                np.nanmean(cv_metrics["brier_calibrated"])
            )
            if cv_metrics["brier_calibrated"]
            else float("nan"),
            "cv_accuracy_calibrated_mean": float(
                np.nanmean(cv_metrics["accuracy_calibrated"])
            )
            if cv_metrics["accuracy_calibrated"]
            else float("nan"),
            "cv_auc_calibrated_mean": float(
                np.nanmean(cv_metrics["auc_calibrated"])
            )
            if cv_metrics["auc_calibrated"]
            else float("nan"),
        }
        rows.append(row)

    return pd.DataFrame(rows)
