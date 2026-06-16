import unittest

import numpy as np
import pandas as pd

from geopfa.prob.stage1_demo import (
    Stage1ScenarioSpec,
    run_region_state_matrix,
    spatial_block_holdout_mask,
)


class ProbStage1DemoTest(unittest.TestCase):
    def test_spatial_block_holdout_mask_keeps_train_and_test(self):
        coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ]
        )
        holdout = spatial_block_holdout_mask(
            coords, holdout_fraction=0.33, grid_size=2
        )

        self.assertEqual(holdout.dtype, bool)
        self.assertGreater(holdout.sum(), 0)
        self.assertLess(holdout.sum(), len(holdout))

    def test_run_region_state_matrix_returns_expected_rows(self):
        X_df = pd.DataFrame(
            {
                "heat": [0.1, 0.2, 0.8, 0.7, 0.3, 0.9],
                "fault": [0.2, 0.1, 0.9, 0.8, 0.4, 0.95],
            }
        )
        y = np.array([0, 0, 1, 1, 0, 1], dtype=int)
        alpha = np.array([-1.0, -0.8, 0.8, 1.0, -0.5, 1.2], dtype=float)
        coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ]
        )
        scenarios = [
            Stage1ScenarioSpec(name="all_data"),
            Stage1ScenarioSpec(name="no_priors", include_priors=False),
            Stage1ScenarioSpec(name="drop_fault", drop_features=("fault",)),
        ]

        out = run_region_state_matrix(
            X_df, y, alpha, coords, scenarios, n_splits=2
        )

        self.assertEqual(len(out), 3)
        self.assertEqual(
            set(out["scenario"]), {"all_data", "no_priors", "drop_fault"}
        )
        self.assertTrue((out["n_samples"] > 0).all())
        self.assertIn("holdout_log_loss_calibrated", out.columns)
        self.assertIn("holdout_brier_calibrated", out.columns)
        self.assertIn("cv_log_loss_calibrated_mean", out.columns)
