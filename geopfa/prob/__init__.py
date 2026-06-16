"""Stage-1 probabilistic-demo helpers built on top of geoPFA favorability surfaces.

The package implements the near-term hierarchical-component approximation
``logit p_c(s) = alpha_c(s) + Sigma beta_ck x_k(s) + u_c(s)``:

* ``demo.fit_component_probability`` fits the per-component model on labelled
  wells with optional spatial-prior offset and RBF residual smoother.
* ``data`` provides labelled-well loading, raster sampling, and alpha-offset
  construction.
* ``stage1_demo`` runs scenario / spatial-block-CV matrices.
* ``calibration`` computes ECE / MCE / Brier / log-loss + a diagnostic
  temperature scaling on held-out predictions.
* ``decision_metrics`` produces decision-class tables, top-N targeting
  curves, and confusion matrices at fixed thresholds.

Nothing in this package is region-specific; the worked Nevada example lives
under ``examples/Nevada/`` (when present) and uses these modules as an
external user would.
"""

from .calibration import (
    CalibrationBin,
    brier_score,
    calibration_summary,
    equal_frequency_reliability_table,
    expected_calibration_error,
    fit_temperature,
    log_loss,
    maximum_calibration_error,
    wilson_ci,
)
from .data import (
    construct_alpha_c,
    load_labeled_wells,
    prepare_component_arrays,
    sample_evidence_at_wells,
)
from .decision_metrics import (
    ConfusionAtThreshold,
    DecisionClassRow,
    TopNRow,
    auc_tie_safe,
    confusion_at_threshold,
    decision_class_table,
    top_n_targeting,
)
from .demo import (
    DemoComponentProbability,
    combine_probability_surfaces,
    fit_component_probability,
)
from .stage1_demo import (
    Stage1ScenarioSpec,
    run_region_state_matrix,
    spatial_block_holdout_mask,
)

__all__ = [
    "CalibrationBin",
    "ConfusionAtThreshold",
    "DecisionClassRow",
    "DemoComponentProbability",
    "Stage1ScenarioSpec",
    "TopNRow",
    "auc_tie_safe",
    "brier_score",
    "calibration_summary",
    "combine_probability_surfaces",
    "confusion_at_threshold",
    "construct_alpha_c",
    "decision_class_table",
    "equal_frequency_reliability_table",
    "expected_calibration_error",
    "fit_component_probability",
    "fit_temperature",
    "load_labeled_wells",
    "log_loss",
    "maximum_calibration_error",
    "prepare_component_arrays",
    "run_region_state_matrix",
    "sample_evidence_at_wells",
    "spatial_block_holdout_mask",
    "top_n_targeting",
    "wilson_ci",
]
