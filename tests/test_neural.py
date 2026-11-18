import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from deception.neural import AttentionNet, _feature_columns, _fit_preprocessor, evaluate_neural


class MiniDataset:
    def __init__(self, table):
        self.table = table

    def validate(self):
        if not {"Subject_ID", "Trial_ID", "Label"}.issubset(self.table):
            raise ValueError("Missing identifiers")


def tiny_dataset():
    rows = []
    for person in range(3):
        for trial in range(4):
            rows.append({
                "Subject_ID": f"P{person}", "Trial_ID": trial, "Label": trial % 2,
                "EEG_a": person + trial / 4, "EEG_b": np.sin(person + trial),
                "Beh_Prediction_RT": 500 + 25 * trial + person,
                "Beh_self_irrel_agree": trial % 2, "EMG_power": 0.2 * person + trial,
            })
    return MiniDataset(pd.DataFrame(rows))


class NeuralTests(unittest.TestCase):
    def test_preprocessing_uses_training_statistics_and_preserves_columns(self):
        train = np.array([[0.0, np.nan], [2.0, np.nan]])
        a, b, fitted = _fit_preprocessor(train, np.array([[100.0, 900.0]]))
        self.assertEqual(a.shape, (2, 2))
        self.assertEqual(b.shape, (1, 2))
        np.testing.assert_allclose(a[:, 0], [-1, 1])
        self.assertAlmostEqual(b[0, 0], 99)
        self.assertEqual(b[0, 1], 0)
        self.assertEqual(fitted["support"].supported_features_.tolist(), [True, False])
        _, _, other_fit = _fit_preprocessor(train, np.array([[-99999.0, np.nan]]))
        np.testing.assert_allclose(fitted[-1].mean_, other_fit[-1].mean_)

    def test_modality_and_outcome_selection(self):
        table = tiny_dataset().table
        _, secondary = _feature_columns(table)
        self.assertEqual(secondary, ["Beh_Prediction_RT"])
        _, secondary = _feature_columns(table, "physiology")
        self.assertEqual(secondary, ["EMG_power"])
        with self.assertRaises(ValueError):
            _feature_columns(table.drop(columns="EMG_power"), "physiology")
        with self.assertRaises(ValueError):
            _feature_columns(table, secondary_features=["Beh_self_irrel_agree"])
        eeg, _ = _feature_columns(table, eeg_features=["EEG_b", "EEG_a"])
        self.assertEqual(eeg, ["EEG_b", "EEG_a"])

    def test_attention_has_one_key_per_secondary_feature(self):
        model = AttentionNet(2, 3, embed_dim=8, num_heads=2).eval()
        with torch.no_grad():
            logits, weights = model(torch.randn(4, 2), torch.randn(4, 3))
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertEqual(tuple(weights.shape), (4, 2, 1, 3))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4, 2, 1))

    def test_wholly_missing_training_modality_is_not_imputed_as_a_sensor(self):
        dataset = tiny_dataset()
        dataset.table.loc[dataset.table.Subject_ID != "P0", "EMG_power"] = np.nan
        with self.assertRaisesRegex(ValueError, "no observed EMG"):
            evaluate_neural(dataset, secondary="physiology", epochs=1,
                            embed_dim=8, num_heads=2)

    def test_constant_training_modality_is_rejected_but_one_constant_feature_is_allowed(self):
        dataset = tiny_dataset()
        dataset.table.loc[dataset.table.Subject_ID != "P0", "EMG_power"] = 5.0
        dataset.table["EMG_offset"] = 8.0
        with self.assertRaisesRegex(ValueError, "only constant EMG"):
            evaluate_neural(dataset, secondary="physiology", epochs=1,
                            embed_dim=8, num_heads=2)
        dataset.table["EMG_power"] = np.arange(len(dataset.table), dtype=float)
        result = evaluate_neural(dataset, secondary="physiology", epochs=1,
                                 embed_dim=8, num_heads=2)
        self.assertEqual(len(result["predictions"]), len(dataset.table))

    def test_nonfinite_logits_fail_instead_of_becoming_predictions(self):
        def invalid_forward(model, eeg, secondary):
            return torch.full((len(eeg), 2), float("nan")), None
        with patch.object(AttentionNet, "forward", invalid_forward):
            with self.assertRaisesRegex(FloatingPointError, "non-finite logits"):
                evaluate_neural(tiny_dataset(), epochs=1, embed_dim=8, num_heads=2)

    def test_nonfinite_hyperparameters_are_rejected(self):
        for options in ({"learning_rate": np.nan}, {"temperature": np.inf},
                        {"epochs": 0}, {"alignment_weight": np.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluate_neural(tiny_dataset(), **options)

    def test_attention_covers_every_person_and_is_reproducible(self):
        dataset = tiny_dataset()
        options = dict(epochs=1, batch_size=3, embed_dim=8, num_heads=2, seed=18)
        first = evaluate_neural(dataset, **options)
        second = evaluate_neural(dataset, **options)
        pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
        self.assertEqual(len(first["predictions"]), len(dataset.table))
        self.assertTrue(first["predictions"]["Probability"].between(0, 1).all())
        self.assertEqual(first["attention_trials"]["Subject_ID"].nunique(), 3)
        self.assertTrue((first["attention"]["Participants"] == 3).all())
        self.assertEqual(first["configuration"]["eeg_features"], ["EEG_a", "EEG_b"])

    def test_input_column_order_does_not_change_feature_alignment(self):
        dataset = tiny_dataset()
        reordered = MiniDataset(dataset.table[dataset.table.columns[::-1]].copy())
        options = dict(epochs=1, batch_size=3, embed_dim=8, num_heads=2, seed=42)
        original = evaluate_neural(dataset, **options)
        changed = evaluate_neural(reordered, **options)
        pd.testing.assert_frame_equal(original["predictions"], changed["predictions"])
        self.assertEqual(original["configuration"]["secondary_features"],
                         changed["configuration"]["secondary_features"])

    def test_test_labels_do_not_change_that_persons_predictions(self):
        original = tiny_dataset()
        changed = MiniDataset(original.table.copy())
        mask = changed.table.Subject_ID == "P0"
        changed.table.loc[mask, "Label"] = 1 - changed.table.loc[mask, "Label"]
        options = dict(epochs=1, batch_size=3, embed_dim=8, num_heads=2, seed=9)
        a = evaluate_neural(original, **options)["predictions"]
        b = evaluate_neural(changed, **options)["predictions"]
        np.testing.assert_allclose(
            a.loc[a.Subject_ID == "P0", "Probability"],
            b.loc[b.Subject_ID == "P0", "Probability"],
        )

    def test_unsupported_test_features_cannot_drive_untrained_weights(self):
        original = tiny_dataset()
        held_out = original.table.Subject_ID == "P0"
        original.table["EEG_unobserved"] = np.nan
        original.table["EEG_train_constant"] = 7.0
        original.table["Beh_RT_train_constant"] = 5.0
        for column in ("EEG_unobserved", "EEG_train_constant", "Beh_RT_train_constant"):
            original.table.loc[held_out, column] = np.arange(4, dtype=float)
        changed = MiniDataset(original.table.copy())
        for column in ("EEG_unobserved", "EEG_train_constant", "Beh_RT_train_constant"):
            changed.table.loc[held_out, column] = 1e12 + np.arange(4, dtype=float)
        options = dict(epochs=1, batch_size=3, embed_dim=8, num_heads=2, seed=19)
        first = evaluate_neural(original, **options)
        second = evaluate_neural(changed, **options)
        np.testing.assert_array_equal(
            first["predictions"].loc[lambda frame: frame.Subject_ID == "P0", "Probability"],
            second["predictions"].loc[lambda frame: frame.Subject_ID == "P0", "Probability"],
        )
        unsupported = first["feature_qc"].query("Test_Subject == 'P0' and Feature in ['EEG_unobserved', 'EEG_train_constant', 'Beh_RT_train_constant']")
        self.assertEqual(len(unsupported), 3)
        self.assertFalse(unsupported.Supported_In_Training.any())
        attention = first["attention_trials"].query("Subject_ID == 'P0' and Feature == 'Beh_RT_train_constant'")
        self.assertFalse(attention.Supported_In_Training.any())

    def test_small_contrastive_and_distillation_have_fair_comparators(self):
        dataset = tiny_dataset()
        for method, names in (
            ("contrastive", {"Contrastive", "Contrastive_No_Alignment"}),
            ("distillation", {"Attention_Teacher", "Secondary_Student", "Distilled_Student"}),
        ):
            with self.subTest(method=method):
                result = evaluate_neural(dataset, method=method, epochs=1, batch_size=3,
                                         embed_dim=8, num_heads=2)
                predictions = result["predictions"]
                self.assertEqual(set(predictions.Model), names)
                self.assertEqual(len(predictions), len(dataset.table) * len(names))
                self.assertFalse(predictions.Probability.isna().any())
                self.assertTrue((predictions.groupby("Model").size() == len(dataset.table)).all())


if __name__ == "__main__":
    unittest.main()
