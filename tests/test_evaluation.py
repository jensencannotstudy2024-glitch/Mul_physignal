"""Regression tests for subject isolation, schema handling, and fusion."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from deception.data import TrialDataset
from deception.evaluation import evaluate_classical, evaluate_fusion, _inner_cv, _model_spec


def make_dataset(subjects=5, trials=12):
    rng = np.random.default_rng(41)
    rows = []
    for subject in range(subjects):
        labels = np.tile([0, 1], trials // 2)
        rng.shuffle(labels)
        for trial, label in enumerate(labels):
            rows.append({
                "Subject_ID": f"S{subject}", "Trial_ID": f"T{trial}", "Label": label,
                "EEG_z": rng.normal() + label, "EEG_a": rng.normal() - label,
                "EEG_constant": 3.0, "ECG_rate": rng.normal() + label / 2,
                "Beh_Answer_Self_RT": 500 + 80 * label + rng.normal(0, 20),
                "Beh_self_irrel_agree": label, "Beh_Trial_Balance": 100 * label,
                "Beh_trial_order": trial,
            })
    table = pd.DataFrame(rows)
    # Missing values must be imputed independently by the training pipeline.
    table.loc[::13, "EEG_z"] = np.nan
    return TrialDataset(table, provenance={"status": "synthetic"}).validate()


FAST_GRID = {"logreg": {"classifier__C": [1.0]}}


class EvaluationTests(unittest.TestCase):
    def test_group_folds_have_no_subject_overlap(self):
        dataset = make_dataset()
        y = dataset.table.Label.to_numpy()
        groups = dataset.table.Subject_ID.to_numpy()
        for train, test in _inner_cv(y, groups, 3):
            self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_inner_fold_with_one_training_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inner training fold contains only one"):
            _inner_cv(np.array([0, 0, 1, 1]), np.array(["A", "A", "B", "B"]), 2)

    def test_outer_predictions_and_training_selection_ignore_held_out_labels(self):
        dataset = make_dataset()
        options = dict(models=("logreg",), feature_sets=("eeg",), inner_splits=2,
                       model_grids=FAST_GRID, threshold_grid=(0.35, 0.5, 0.65))
        original = evaluate_classical(dataset, **options)
        changed = make_dataset()
        changed.table.loc[changed.table.Subject_ID == "S0", "Label"] ^= 1
        altered = evaluate_classical(changed, **options)
        left = original["predictions"].query("Subject_ID == 'S0'").sort_values(["Model", "Trial_ID"])
        right = altered["predictions"].query("Subject_ID == 'S0'").sort_values(["Model", "Trial_ID"])
        np.testing.assert_allclose(left.Probability, right.Probability, rtol=0, atol=0)
        np.testing.assert_array_equal(left.Prediction, right.Prediction)
        a = original["selection"].query("Test_Subject == 'S0'")
        b = altered["selection"].query("Test_Subject == 'S0'")
        np.testing.assert_array_equal(a.Threshold, b.Threshold)
        np.testing.assert_array_equal(a.Parameters, b.Parameters)
        self.assertTrue(all("S0" not in subjects for subjects in a.Train_Subjects))

    def test_feature_column_order_does_not_change_predictions(self):
        dataset = make_dataset()
        options = dict(models=("logreg",), feature_sets=("fusion",), inner_splits=2,
                       model_grids=FAST_GRID, pca_components=0.95)
        first = evaluate_classical(dataset, **options)
        reordered = TrialDataset(dataset.table.loc[:, list(reversed(dataset.table.columns))])
        second = evaluate_classical(reordered, **options)
        np.testing.assert_allclose(first["predictions"].Probability, second["predictions"].Probability)
        selected = set(first["selection"].Columns.iloc[0])
        self.assertNotIn("Beh_self_irrel_agree", selected)
        self.assertNotIn("Beh_Trial_Balance", selected)
        self.assertNotIn("Beh_trial_order", selected)
        self.assertIn("Beh_Answer_Self_RT", selected)

    def test_held_out_only_feature_is_discarded_without_changing_prediction(self):
        dataset = make_dataset(subjects=4)
        held_out = dataset.table.Subject_ID == "S0"
        dataset.table["EEG_test_only"] = np.nan
        dataset.table.loc[held_out, "EEG_test_only"] = np.arange(held_out.sum()) * 1e5
        options = dict(models=("logreg",), feature_sets=("eeg",), inner_splits=2,
                       model_grids=FAST_GRID, pca_components=0.9)
        first = evaluate_classical(dataset, **options)
        dataset.table.loc[held_out, "EEG_test_only"] *= -100
        second = evaluate_classical(dataset, **options)
        a = first["predictions"].query("Subject_ID == 'S0'").sort_values(["Model", "Trial_ID"])
        b = second["predictions"].query("Subject_ID == 'S0'").sort_values(["Model", "Trial_ID"])
        np.testing.assert_allclose(a.Probability, b.Probability, rtol=0, atol=0)

    def test_missing_modality_cannot_be_replaced_by_an_imputed_block(self):
        dataset = make_dataset()
        dataset.table["ECG_rate"] = np.nan
        with self.assertRaisesRegex(ValueError, "No observed values for modality 'ECG'"):
            evaluate_classical(dataset, models=("logreg",), feature_sets=("fusion",),
                               model_grids=FAST_GRID)
        with self.assertRaisesRegex(ValueError, "No observed values for modality 'ECG'"):
            evaluate_fusion(dataset, feature_sets=("eeg", "physiology"), model_grids=FAST_GRID)
        # A subset with genuinely observed EEG is still a valid evaluation.
        result = evaluate_classical(dataset, models=("logreg",), feature_sets=("eeg",),
                                    model_grids=FAST_GRID)
        self.assertEqual(len(result["predictions"]), 2 * len(dataset.table))

    def test_modality_missing_only_in_an_inner_training_fold_is_rejected(self):
        dataset = make_dataset()
        dataset.table.loc[dataset.table.Subject_ID != "S0", "ECG_rate"] = np.nan
        with self.assertRaisesRegex(ValueError, "No observed values for modality 'ECG'"):
            evaluate_fusion(dataset, method="stacking", feature_sets=("eeg", "physiology"),
                            inner_splits=3, model_grids=FAST_GRID)

    def test_all_constant_training_features_fail_clearly(self):
        dataset = make_dataset()
        dataset.table.loc[:, ["EEG_a", "EEG_z"]] = 1.0
        with self.assertRaisesRegex(ValueError, "No varying features remain in a training fold"):
            evaluate_classical(dataset, models=("logreg",), feature_sets=("eeg",),
                               model_grids=FAST_GRID)

    def test_whole_constant_modality_cannot_support_a_fusion_claim(self):
        dataset = make_dataset()
        dataset.table["ECG_rate"] = 0.0
        with self.assertRaisesRegex(ValueError, "modality 'ECG'"):
            evaluate_classical(dataset, models=("logreg",), feature_sets=("fusion",),
                               model_grids=FAST_GRID)
        with self.assertRaisesRegex(ValueError, "modality 'ECG'"):
            evaluate_fusion(dataset, feature_sets=("eeg", "physiology"), model_grids=FAST_GRID)
        result = evaluate_classical(dataset, models=("dummy",), feature_sets=("fusion",))
        self.assertEqual(len(result["predictions"]), len(dataset.table))

    def test_optional_boosters_are_lazy_and_report_missing_dependencies(self):
        for name in ("catboost", "xgboost"):
            with patch.dict("sys.modules", {name: None}):
                with self.assertRaisesRegex(ImportError, f"Optional model '{name}'"):
                    _model_spec(name, 2026)

    def test_dummy_works_with_one_class_but_learned_models_fail_explicitly(self):
        dataset = make_dataset()
        dataset.table["Label"] = 0
        baseline = evaluate_classical(dataset, models=("dummy",), feature_sets=("eeg",))
        self.assertTrue((baseline["predictions"].Probability == 0).all())
        with self.assertRaisesRegex(ValueError, "one label class"):
            evaluate_classical(dataset, models=("logreg",), feature_sets=("eeg",))

    def test_within_subject_folds_never_mix_participants(self):
        dataset = make_dataset()
        result = evaluate_classical(
            dataset, models=("logreg",), feature_sets=("eeg",), inner_splits=2,
            model_grids=FAST_GRID, grouping="within", per_modality_scaling=True,
        )
        predictions = result["predictions"].query("Model == 'logreg'")
        for subject, block in predictions.groupby("Subject_ID"):
            self.assertEqual(len(block), len(dataset.table.query("Subject_ID == @subject")))
            self.assertEqual(block.Trial_ID.nunique(), len(block))
            self.assertLessEqual(block.Fold.nunique(), 5)
        self.assertEqual(set(result["selection"].Grouping), {"within"})
        self.assertTrue((result["selection"].Train_Subjects.map(len) == 1).all())

    def test_cli_tuning_defaults_to_five_inner_splits(self):
        from deception.cli import build_parser
        args = build_parser().parse_args(["benchmark", "--input", "x", "--output", "y"])
        self.assertEqual(args.inner_splits, 5)
        self.assertEqual(args.grouping, "loso")

    def test_supported_classifiers_use_the_same_outer_trials(self):
        dataset = make_dataset(subjects=4)
        result = evaluate_classical(
            dataset, models=("logreg", "hist_gb", "random_forest"),
            feature_sets=("eeg",), inner_splits=2,
            model_grids={
                **FAST_GRID,
                "hist_gb": {"classifier__max_iter": [5]},
                "random_forest": {"classifier__n_estimators": [10], "classifier__max_depth": [3]},
            },
        )
        self.assertEqual(set(result["predictions"].Model), {"dummy", "logreg", "hist_gb", "random_forest"})
        self.assertTrue((result["predictions"].groupby("Model").size() == len(dataset.table)).all())

    def test_late_fusion_and_stacking_keep_same_held_out_rows(self):
        dataset = make_dataset()
        for method in ("late", "stacking"):
            result = evaluate_fusion(
                dataset, method=method, feature_sets=("eeg", "physiology"),
                inner_splits=2, model_grids=FAST_GRID,
            )
            predictions = result["predictions"]
            self.assertEqual(set(predictions.Model), {"dummy", f"{method}:logreg"})
            for _, block in predictions.groupby("Model"):
                self.assertEqual(len(block), len(dataset.table))
                self.assertEqual(block[["Subject_ID", "Trial_ID"]].duplicated().sum(), 0)
                self.assertTrue(block.Probability.between(0, 1).all())

    def test_stacking_does_not_use_outer_test_labels(self):
        dataset = make_dataset(subjects=4)
        options = dict(method="stacking", feature_sets=("eeg", "physiology"),
                       inner_splits=3, model_grids=FAST_GRID)
        first = evaluate_fusion(dataset, **options)
        dataset.table.loc[dataset.table.Subject_ID == "S0", "Label"] ^= 1
        second = evaluate_fusion(dataset, **options)
        a = first["predictions"].query("Subject_ID == 'S0'").sort_values(["Model", "Trial_ID"])
        b = second["predictions"].query("Subject_ID == 'S0'").sort_values(["Model", "Trial_ID"])
        np.testing.assert_allclose(a.Probability, b.Probability, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
