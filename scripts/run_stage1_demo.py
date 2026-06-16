#!/usr/bin/env python3
"""Generic Stage-1 probabilistic-demo runner.

This script exercises the full ``geopfa.prob`` workflow on a deterministic
synthetic two-component fixture so users can reproduce the pipeline end-to-end
without needing any region-specific data.

The reported outputs are:

* per-component fitted ``DemoComponentProbability`` surfaces,
* a combined ``p = ∏_c p_c`` surface,
* calibration diagnostics on well-overlay predictions (ECE / MCE / Brier /
  log-loss + a diagnostic temperature),
* a decision-class table built from the combined-surface percentiles, and
* a top-N targeting curve and a 0.5-threshold confusion matrix.

Use this script as a template for wiring real labelled wells and a real
gridded ``pfa`` object into the same pipeline.

Run with::

    pixi run python scripts/run_stage1_demo.py --output-dir outputs/stage1_demo

If ``--output-dir`` is omitted the script prints metrics to stdout only and
writes nothing to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geopfa.prob import (
    calibration_summary,
    combine_probability_surfaces,
    confusion_at_threshold,
    decision_class_table,
    fit_component_probability,
    top_n_targeting,
)
from tests.fixtures.synthetic_prob import make_synthetic_pfa


# Mapping from component name (in pfa["criteria"]["geologic"]["components"]) to
# the layer that should be used as a per-cell α_c(s) prior offset. When the
# named layer is present it is automatically excluded from the regression.
DEFAULT_COMPONENT_PRIORS: dict[str, str] = {
    "component_a": "prior_layer_a",
    "component_b": "prior_layer_b",
}

DEFAULT_LABEL_COLUMNS: dict[str, str] = {
    "component_a": "heat_label",
    "component_b": "reservoir_label",
}


def _fit_components(
    pfa: dict,
    wells: gpd.GeoDataFrame,
    *,
    include_priors: bool,
    include_spatial: bool,
    component_priors: dict[str, str],
    label_columns: dict[str, str],
) -> dict:
    """Fit ``fit_component_probability`` for each declared component."""
    components = pfa["criteria"]["geologic"]["components"]
    surfaces: dict = {}
    for component, comp_data in components.items():
        prior_layer = component_priors.get(component) if include_priors else None
        label_column = label_columns.get(component, "heat_label")
        surfaces[component] = fit_component_probability(
            comp_data,
            prior_probability=float(comp_data.get("pr0", 0.5)),
            include_spatial=include_spatial,
            labeled_wells=wells,
            label_column=label_column,
            prior_layer_name=prior_layer,
        )
    return surfaces


def _well_overlay(
    surface_gdf: gpd.GeoDataFrame,
    wells: gpd.GeoDataFrame,
    *,
    label_column: str,
    value_column: str = "probability",
) -> pd.DataFrame:
    """Sample a fitted-surface GeoDataFrame at each well via nearest-neighbour join."""
    if surface_gdf.crs is None or wells.crs is None:
        raise ValueError("surface and wells must both have a CRS")
    cols = ["geometry", value_column]
    base = wells.to_crs(surface_gdf.crs)
    joined = gpd.sjoin_nearest(
        base[["well_id", label_column, "geometry"]],
        surface_gdf[cols],
        how="left",
    )
    df = pd.DataFrame(
        {
            "well_id": joined["well_id"].to_numpy(),
            "label": joined[label_column].astype(float).to_numpy(),
            "probability": joined[value_column].astype(float).to_numpy(),
        }
    )
    return df.dropna(subset=["label", "probability"]).reset_index(drop=True)


def _combine_component_surfaces(component_surfaces: dict) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Combine per-component surfaces into a single GeoDataFrame.

    Returns ``(combined_gdf, combined_probabilities)`` where ``combined_gdf``
    has a ``probability`` column equal to the product across components.
    """
    component_arrays: list[np.ndarray] = []
    base_gdf: gpd.GeoDataFrame | None = None
    for component, surface in component_surfaces.items():
        gdf = surface.probability
        if base_gdf is None:
            base_gdf = gdf[["geometry"]].copy()
        component_arrays.append(gdf["probability"].astype(float).to_numpy())

    combined = combine_probability_surfaces(component_arrays)
    assert base_gdf is not None
    base_gdf["probability"] = combined
    return base_gdf, combined


def _calibration_to_dict(summary: dict) -> dict:
    return {
        "n": int(summary.get("n", 0)),
        "n_bins": int(summary.get("n_bins", 0)),
        "ECE": float(summary.get("ECE", float("nan"))),
        "MCE": float(summary.get("MCE", float("nan"))),
        "brier": float(summary.get("brier", float("nan"))),
        "log_loss": float(summary.get("log_loss", float("nan"))),
        "temperature_T": float(summary.get("temperature_T", 1.0)),
        "ECE_post_temperature": float(summary.get("ECE_post_temperature", float("nan"))),
        "MCE_post_temperature": float(summary.get("MCE_post_temperature", float("nan"))),
        "brier_post_temperature": float(summary.get("brier_post_temperature", float("nan"))),
        "log_loss_post_temperature": float(summary.get("log_loss_post_temperature", float("nan"))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write CSV / JSON artifacts to.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Synthetic-data RNG seed.")
    parser.add_argument("--grid-n", type=int, default=18, help="Grid side length.")
    parser.add_argument("--n-wells", type=int, default=80, help="Number of synthetic wells.")
    parser.add_argument(
        "--no-priors",
        action="store_true",
        help="Disable component-specific spatial priors (use scalar pr0 only).",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="Disable the RBF residual smoother u_c(s).",
    )
    args = parser.parse_args(argv)

    fixture = make_synthetic_pfa(
        grid_n=args.grid_n, n_wells=args.n_wells, seed=args.seed
    )
    pfa = fixture.pfa
    wells = fixture.wells

    component_surfaces = _fit_components(
        pfa,
        wells,
        include_priors=not args.no_priors,
        include_spatial=not args.no_spatial,
        component_priors=DEFAULT_COMPONENT_PRIORS,
        label_columns=DEFAULT_LABEL_COLUMNS,
    )

    per_component_metrics: list[dict] = []
    for component, surface in component_surfaces.items():
        label_column = DEFAULT_LABEL_COLUMNS.get(component, "heat_label")
        well_df = _well_overlay(surface.probability, wells, label_column=label_column)
        cs = calibration_summary(
            well_df["label"].to_numpy(),
            well_df["probability"].to_numpy(),
            n_bins=5,
        )
        per_component_metrics.append({"component": component, **_calibration_to_dict(cs)})

    combined_gdf, combined_array = _combine_component_surfaces(component_surfaces)
    # Use heat_label for the combined view (synthetic fixture sets heat == reservoir).
    combined_well_df = _well_overlay(combined_gdf, wells, label_column="heat_label")
    combined_metrics = calibration_summary(
        combined_well_df["label"].to_numpy(),
        combined_well_df["probability"].to_numpy(),
        n_bins=5,
    )
    per_component_metrics.append(
        {"component": "combined", **_calibration_to_dict(combined_metrics)}
    )

    decision_table = decision_class_table(
        surface_name="combined",
        surface_values=combined_array,
        well_scores=combined_well_df["probability"].to_numpy(),
        well_labels=combined_well_df["label"].to_numpy(),
    )
    top_n = top_n_targeting(
        combined_well_df["label"].to_numpy(),
        combined_well_df["probability"].to_numpy(),
        ns=(5, 10, 20),
    )
    cm = confusion_at_threshold(
        combined_well_df["label"].to_numpy(),
        combined_well_df["probability"].to_numpy(),
        threshold=0.5,
    )

    metrics_df = pd.DataFrame(per_component_metrics)
    decision_df = pd.DataFrame([row.as_dict() for row in decision_table])
    top_n_df = pd.DataFrame([row.as_dict() for row in top_n])

    print("\n=== Calibration metrics ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Decision-class table ===")
    print(decision_df.to_string(index=False))
    print("\n=== Top-N targeting ===")
    print(top_n_df.to_string(index=False))
    print("\n=== Combined confusion matrix at threshold 0.5 ===")
    print(json.dumps(cm.as_dict(), indent=2))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(args.output_dir / "calibration_metrics.csv", index=False)
        decision_df.to_csv(args.output_dir / "decision_class_table.csv", index=False)
        top_n_df.to_csv(args.output_dir / "top_n_targeting.csv", index=False)
        (args.output_dir / "confusion_at_0_5.json").write_text(
            json.dumps(cm.as_dict(), indent=2)
        )
        print(f"\nWrote artifacts to {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
