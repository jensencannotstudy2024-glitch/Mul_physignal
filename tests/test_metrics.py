from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from deception.data import TrialDataset
from deception.metrics import summarize_predictions, write_run


def predictions():
    return pd.DataFrame({"Subject_ID": ["A", "A", "B", "B"], "Trial_ID": ["0", "1", "0", "1"],
                         "Label": [0, 1, 0, 1], "Model": "test", "Feature_Set": "eeg",
                         "Prediction": pd.array([0, 1, pd.NA, 0], dtype="Int64"),
                         "Probability": [.1, .8, np.nan, .2]})


class MetricsTests(unittest.TestCase):
    def test_failures_remain_visible_in_coverage(self):
        model = summarize_predictions(predictions())["models"][0]
        self.assertEqual(model["coverage"], .75)
        self.assertAlmostEqual(model["accuracy"], 2 / 3)
        self.assertEqual(model["subjects"], 2)

    def test_categorical_llm_has_no_invented_auc(self):
        frame = predictions()
        frame.Probability = np.nan
        self.assertIsNone(summarize_predictions(frame)["models"][0]["roc_auc"])

    def test_complete_failure_returns_null_scores(self):
        frame = predictions()
        frame.Prediction = pd.NA
        frame.Probability = np.nan
        model = summarize_predictions(frame)["models"][0]
        self.assertEqual(model["coverage"], 0)
        self.assertIsNone(model["accuracy"])

    def test_missing_group_fields_fail_instead_of_disappearing(self):
        for column in ("Model", "Feature_Set", "Subject_ID", "Trial_ID"):
            frame = predictions()
            frame.loc[0, column] = None
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                summarize_predictions(frame)

    def test_duplicate_trials_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_predictions(pd.concat([predictions(), predictions()]))

    def test_truth_must_match_across_models(self):
        other = predictions()
        other.Model = "other"
        other.Label = 1 - other.Label
        with self.assertRaisesRegex(ValueError, "disagree"):
            summarize_predictions(pd.concat([predictions(), other]))

    def test_different_attempted_cohorts_cannot_be_compared(self):
        other = predictions().iloc[:2].copy()
        other.Model = "other"
        with self.assertRaisesRegex(ValueError, "same trial"):
            summarize_predictions(pd.concat([predictions(), other]))

    def test_probabilities_are_checked(self):
        for value in (np.inf, -.1, 1.1):
            frame = predictions()
            frame.loc[0, "Probability"] = value
            with self.assertRaises(ValueError):
                summarize_predictions(frame)

    def test_writer_rejects_wrong_truth_or_undeclared_omissions(self):
        frame = predictions()
        data = TrialDataset(frame[["Subject_ID", "Trial_ID", "Label"]].copy())
        with tempfile.TemporaryDirectory() as temporary:
            bad = frame.iloc[:2].copy()
            with self.assertRaisesRegex(ValueError, "omitted"):
                write_run({"predictions": bad}, data, temporary, configuration={}, run_status="test")
            bad = frame.copy()
            bad.loc[0, "Label"] = 1
            with self.assertRaisesRegex(ValueError, "disagree"):
                write_run({"predictions": bad}, data, temporary, configuration={}, run_status="test")
            self.assertFalse((Path(temporary) / "predictions.csv").exists())

    def test_explicit_sampling_is_reported_and_outputs_not_overwritten(self):
        frame = predictions()
        data = TrialDataset(frame[["Subject_ID", "Trial_ID", "Label"]].copy())
        with tempfile.TemporaryDirectory() as temporary:
            result = {"predictions": frame.iloc[:2].copy(), "diagnostic": True}
            write_run(result, data, temporary, configuration={}, run_status="sampled_diagnostic")
            self.assertIn('"sampling_fraction": 0.5', (Path(temporary) / "metrics.json").read_text())
            with self.assertRaisesRegex(ValueError, "not empty"):
                write_run(result, data, temporary, configuration={}, run_status="sampled_diagnostic")


if __name__ == "__main__":
    unittest.main()
