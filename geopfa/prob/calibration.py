"""Probability calibration diagnostics for the Stage-1 probabilistic demo.

These helpers are framework-agnostic - they operate on `(y, p)` arrays of held-out
labels and predicted probabilities. They are used by the Stage-1 scenario runner
to produce reliability diagrams, ECE / MCE / Brier / log-loss metrics, and a
diagnostic temperature-scaling estimate.

Temperature scaling is reported as a *diagnostic*; it is not applied to map
outputs. A scalar temperature can sharpen or soften predictions around 0.5 but
cannot translate a systematic location bias. For datasets whose reliability
diagram is uniformly above or below the diagonal, treat T as a summary number
and use Platt or isotonic recalibration if a corrective transform is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm as _scipy_norm


@dataclass(frozen=True)
class CalibrationBin:
    """Per-bin reliability summary."""

    bin_index: int
    n: int
    bin_p_lo: float
    bin_p_hi: float
    mean_predicted_p: float
    observed_fraction: float
    wilson_ci_lo: float
    wilson_ci_hi: float

    def as_dict(self) -> dict:
        """Return a plain-dict representation of the bin row."""
        return {
            "bin": self.bin_index,
            "n": self.n,
            "bin_p_lo": self.bin_p_lo,
            "bin_p_hi": self.bin_p_hi,
            "mean_predicted_p": self.mean_predicted_p,
            "observed_fraction": self.observed_fraction,
            "wilson_ci_lo": self.wilson_ci_lo,
            "wilson_ci_hi": self.wilson_ci_hi,
        }


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% (default) Wilson score interval for a binomial proportion ``k / n``.

    Wilson is preferred over the normal-approximation interval at small ``n``
    and when ``p_hat`` is near 0 or 1.
    """
    if n <= 0:
        return (0.0, 1.0)
    z = float(_scipy_norm.ppf(1.0 - alpha / 2.0))
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = (
        z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    ) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def equal_frequency_reliability_table(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 5,
    alpha: float = 0.05,
) -> list[CalibrationBin]:
    """Equal-frequency (quantile) reliability table.

    Predictions are sorted by ``p`` and split into ``n_bins`` contiguous chunks
    of (nearly) equal size via ``numpy.array_split``. Each row carries the bin
    sample size, the predicted-probability range, mean predicted probability,
    observed positive fraction, and the (1 - alpha) Wilson CI on the observed
    fraction.
    """
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    if y_arr.shape != p_arr.shape:
        raise ValueError("y and p must have the same shape")
    if y_arr.size == 0:
        return []
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    order = np.argsort(p_arr)
    bin_groups = np.array_split(order, n_bins)
    rows: list[CalibrationBin] = []
    for b_idx, bin_idx in enumerate(bin_groups):
        if len(bin_idx) == 0:
            continue
        bin_p = p_arr[bin_idx]
        bin_y = y_arr[bin_idx]
        n_bin = len(bin_p)
        k_pos = int(bin_y.sum())
        ci_lo, ci_hi = wilson_ci(k_pos, n_bin, alpha=alpha)
        rows.append(
            CalibrationBin(
                bin_index=b_idx,
                n=n_bin,
                bin_p_lo=float(bin_p.min()),
                bin_p_hi=float(bin_p.max()),
                mean_predicted_p=float(bin_p.mean()),
                observed_fraction=k_pos / n_bin,
                wilson_ci_lo=ci_lo,
                wilson_ci_hi=ci_hi,
            )
        )
    return rows


def expected_calibration_error(rows: Iterable[CalibrationBin]) -> float:
    """Sample-weighted mean gap between predicted probability and observed fraction.

    Returns NaN when no rows are provided.
    """
    rows_list = list(rows)
    if not rows_list:
        return float("nan")
    n_total = sum(r.n for r in rows_list)
    if n_total <= 0:
        return float("nan")
    return float(
        sum(
            (r.n / n_total) * abs(r.mean_predicted_p - r.observed_fraction)
            for r in rows_list
        )
    )


def maximum_calibration_error(rows: Iterable[CalibrationBin]) -> float:
    """Worst-case absolute calibration gap across the supplied bins."""
    rows_list = list(rows)
    if not rows_list:
        return float("nan")
    return float(
        max(abs(r.mean_predicted_p - r.observed_fraction) for r in rows_list)
    )


def brier_score(
    y: Sequence[int] | np.ndarray, p: Sequence[float] | np.ndarray
) -> float:
    """Brier score: ``mean((y - p) ** 2)``."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if y_arr.size == 0:
        return float("nan")
    return float(np.mean((y_arr - p_arr) ** 2))


def log_loss(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    eps: float = 1e-9,
) -> float:
    """Binary cross-entropy / log-loss with predictions clipped to ``[eps, 1-eps]``."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    if y_arr.size == 0:
        return float("nan")
    return float(
        -np.mean(y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log(1.0 - p_arr))
    )


def fit_temperature(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 50.0),
    eps: float = 1e-9,
) -> tuple[float, np.ndarray]:
    """Fit the scalar temperature ``T`` that minimises held-out log-loss.

    The transform is ``p_T = sigmoid(logit(p) / T)``. ``T < 1`` sharpens
    predictions away from 0.5; ``T > 1`` softens them toward 0.5. Returns
    ``(T, p_T)``. Falls back to ``T = 1`` when the optimiser fails or when
    only one class is present in ``y``.
    """
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    n_classes_required = 2
    if y_arr.size == 0 or np.unique(y_arr).size < n_classes_required:
        return 1.0, p_arr.copy()

    logits = np.log(p_arr / (1.0 - p_arr))

    def _loss(temperature: float) -> float:
        if not np.isfinite(temperature) or temperature <= 0.0:
            return 1e9
        p_t = 1.0 / (1.0 + np.exp(-logits / float(temperature)))
        p_t = np.clip(p_t, eps, 1.0 - eps)
        return float(
            -np.mean(y_arr * np.log(p_t) + (1.0 - y_arr) * np.log(1.0 - p_t))
        )

    try:
        result = minimize_scalar(_loss, bounds=bounds, method="bounded")
        if result.success and np.isfinite(result.x):
            T = float(result.x)
        else:
            T = 1.0
    except Exception:
        T = 1.0
    p_T = 1.0 / (1.0 + np.exp(-logits / T))
    return T, p_T


def calibration_summary(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 5,
    alpha: float = 0.05,
) -> dict:
    """One-call calibration summary used by the Stage-1 scenario runner.

    Returns a dict with raw and post-temperature ECE / MCE / Brier / log-loss
    plus the fitted temperature, the raw bin table, and the post-temperature
    bin table.
    """
    rows = equal_frequency_reliability_table(y, p, n_bins=n_bins, alpha=alpha)
    n = int(np.asarray(y).size)
    out: dict = {
        "n": n,
        "n_bins": n_bins,
        "ECE": expected_calibration_error(rows),
        "MCE": maximum_calibration_error(rows),
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "bins_raw": [r.as_dict() for r in rows],
    }
    T, p_T = fit_temperature(y, p)
    rows_T = equal_frequency_reliability_table(
        y, p_T, n_bins=n_bins, alpha=alpha
    )
    out.update(
        {
            "temperature_T": T,
            "ECE_post_temperature": expected_calibration_error(rows_T),
            "MCE_post_temperature": maximum_calibration_error(rows_T),
            "brier_post_temperature": brier_score(y, p_T),
            "log_loss_post_temperature": log_loss(y, p_T),
            "bins_post_temperature": [r.as_dict() for r in rows_T],
        }
    )
    return out


__all__ = [
    "CalibrationBin",
    "brier_score",
    "calibration_summary",
    "equal_frequency_reliability_table",
    "expected_calibration_error",
    "fit_temperature",
    "log_loss",
    "maximum_calibration_error",
    "wilson_ci",
]
