# Stage-1 probabilistic geoPFA demo

This page describes the **Stage-1 probabilistic** workflow shipped in
`geopfa.prob`. The workflow turns the deterministic favorability surface
into a calibrated, decision-grade probability surface using the
hierarchical-component approximation

```
logit p_c(s) = α_c(s) + Σ_k β_ck x_k(s) + u_c(s)
```

for each component `c` (heat, reservoir, …) at every grid cell `s`. The
final probability surface is the product across components,

```
p(s) = ∏_c p_c(s)
```

The implementation is region-agnostic. The Nevada Great Basin worked
example that the model was originally calibrated against lives in
`examples/Nevada/` (when present) and uses these same modules as an
external user would.

## What is in `geopfa.prob`

| Module                    | Purpose                                                                                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `geopfa.prob.demo`        | Per-component fit (`fit_component_probability`) and product-rule combination (`combine_probability_surfaces`).                                                                                           |
| `geopfa.prob.data`        | Labelled-well loaders, raster sampling at well points, and α-offset construction (`load_labeled_wells`, `sample_evidence_at_wells`, `construct_alpha_c`, `prepare_component_arrays`).                    |
| `geopfa.prob.stage1_demo` | Region-state scenario / spatial-block-CV matrices (`run_region_state_matrix`, `Stage1ScenarioSpec`, `spatial_block_holdout_mask`).                                                                       |
| `geopfa.prob.calibration` | Calibration diagnostics on held-out predictions: Wilson CI, equal-frequency reliability tables, ECE / MCE / Brier / log-loss, temperature scaling, and the `calibration_summary` one-call helper.      |
| `geopfa.prob.decision_metrics` | Decision-relevant metrics: confusion matrices at fixed thresholds (`confusion_at_threshold`), tie-safe AUC (`auc_tie_safe`), decision-class tables driven by surface percentiles (`decision_class_table`), and top-N targeting curves (`top_n_targeting`). |

The public API surface lives in `geopfa/prob/__init__.py`.

## Model components

### Prior offset α_c(s)

Two modes are supported in `fit_component_probability`:

1. **Scalar prior** — `α_c = logit(prior_probability)`, applied uniformly to every cell.
2. **Spatial prior (Option 2 in the methodology memo)** — pass `prior_layer_name="<layer>"` and `fit_component_probability` will:
   - take the layer's `value_interpolated` column,
   - **min-max rescale** it to `[prior_p_min, prior_p_max]` (default `[0.2, 0.8]`),
   - apply `logit` to give a per-cell α_c(s),
   - **automatically exclude the layer from the regression features** so it is not double-counted.

The default `[0.2, 0.8]` rescale window keeps `logit(α)` well away from the
asymptotic blow-ups at 0 and 1 while still letting the prior layer move the
component probability several logit units. Users who want a wider prior
range can pass `prior_p_min` / `prior_p_max` explicitly.

### Evidence regression Σ β_ck x_k(s)

* All evidence layers in `component_data["layers"]` are flattened into a
  feature matrix, **standardised** (`mean=0, std=1` using grid-level
  statistics), and used as predictors in the per-cell logit.
* Coordinate columns (`X`, `Y`, and any `inverted_y`-type columns) are
  rejected to prevent location leakage.
* Sparse-binary layers (≤3 unique values, ≥90 % of cells the same value)
  are detected by `_is_sparse_binary` and dropped to avoid having one
  spike-pattern dominate the regression.

### Spatial residual u_c(s)

When `include_spatial=True`, an RBF residual smoother is fit to the model
residuals `y - p̂` to capture coherent spatial structure that the linear
regression cannot. The smoother uses a small set of representative grid
points (sub-sampled deterministically) so it remains tractable on large
grids.

### Training data

* If `labeled_wells` and `label_column` are provided, the regression is
  fit **on labelled wells** with evidence sampled at the well points. This
  is the consistent, label-driven path.
* Otherwise a legacy proxy-label path is used (favorability ≥ threshold on
  the grid). It is kept only for backward compatibility and is documented
  as inconsistent with the labelled-well overlay.
* When fewer than four labelled positives are available the proxy path is
  used and a `RuntimeWarning` is raised so callers know the fit was not
  data-driven.

## Calibration diagnostics

`calibration_summary(y, p, n_bins=5)` returns a dict with both raw and
post-temperature ECE / MCE / Brier / log-loss plus the fitted scalar
temperature `T`. **Temperature scaling is reported as a diagnostic only;
the demo map outputs always keep their raw probabilities** so users see the
model's native behaviour.

* `T < 1` → predictions were *under-confident* (true rate higher than predicted at the high bins).
* `T > 1` → predictions were *over-confident*.
* `T ≈ 1` → already well-calibrated.

The reliability table is built on **equal-frequency bins** so each bin has
roughly the same number of observations. Each bin reports the empirical
positive rate and a 95 % Wilson confidence interval. We deliberately
report the same `n=k` in the title rather than per-bin so the diagram
stays uncluttered.

## Decision diagnostics

* **`decision_class_table`** uses cuts computed on the **full surface
  distribution** (not on the well-sample distribution). This is what makes
  it a decision tool: each percentile bin is defined by where the surface
  sits, and the well-overlay tells you what the labelled wells say about
  each region.
* **`top_n_targeting`** answers "of the top-N highest-scoring labelled
  wells, how many are positive?". Compare against `random_expected_hits`
  and `max_possible_hits` to judge enrichment.
* **`confusion_at_threshold`** is the classical 0.5-threshold confusion
  matrix with accuracy, precision, recall, and specificity.

## Quick-start

The repository ships a synthetic two-component fixture and a runnable
demo script that exercise the full pipeline end-to-end without any
region-specific data:

```bash
pixi run python scripts/run_stage1_demo.py --output-dir outputs/stage1_demo
```

This will:

1. Build a deterministic `pfa` dictionary with two components and three
   layers each (one prior, one shared gradient, one sparse-binary
   indicator that is auto-excluded from the regression).
2. Generate a random sample of 80 labelled wells whose labels are drawn
   from a logistic model on the prior bumps.
3. Fit `fit_component_probability` for each component.
4. Combine via the product rule.
5. Print and write CSV / JSON artifacts for the calibration metrics,
   decision-class table, top-N targeting, and 0.5-threshold confusion.

The fixture lives at `tests/fixtures/synthetic_prob.py`. Use it as a
template for wiring real data into the same pipeline — replace the
`pfa` dictionary build and `wells_gdf` with your own sources.

## Tests

Synthetic-data unit tests cover every module in `geopfa.prob`:

* `tests/test_prob_calibration.py` — calibration helpers
* `tests/test_prob_decision_metrics.py` — decision-relevant metrics
* `tests/test_prob_demo.py` — `fit_component_probability`, sparse-binary
  rejection, coordinate-leak rejection, prior-layer rescaling and
  exclusion, label-driven vs proxy fallbacks
* `tests/test_prob_stage1_demo.py` — `run_region_state_matrix`,
  `Stage1ScenarioSpec`, spatial-block hold-out mask
* `tests/test_prob_data.py` — labelled-well loading, raster sampling,
  α-offset construction
* `tests/test_prob_integration.py` — end-to-end pipeline on the synthetic
  fixture (fit → combine → calibrate → decision table → top-N → confusion)

Run with:

```bash
pixi run python -m pytest tests/test_prob_*.py -q
```

## Limitations and roadmap

The Stage-1 implementation is intentionally a **near-term approximation**:

* Logistic regression is point-estimated (no posterior over β).
* The spatial residual smoother is RBF-based, not a Gaussian process.
* Calibration is reported as a diagnostic and not propagated as
  uncertainty bands on the map.
* The product-rule combination assumes conditional independence of the
  components given the location, which is only approximately true for
  geothermal play favourability.

The methodology memo lays out the full Stage-2 hierarchical Bayesian
upgrade path (joint inference over `β_c`, a proper GP for `u_c`,
explicit shared confounders across components, and posterior-predictive
calibration). Stage-1 is the bridge: it surfaces the same calibration
and decision diagnostics that the Stage-2 model will, so downstream
tooling and presentation pipelines stay stable across the upgrade.
