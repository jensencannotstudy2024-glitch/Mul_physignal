"""Synthetic checks of feature meaning, names, and predictor availability."""
import unittest

import numpy as np
import pandas as pd

from deception.data import TrialDataset
from deception.features import behavior_features, eeg_features, extract_features, physiology_window_features


class FeatureTests(unittest.TestCase):
    def test_batched_eeg_matches_single_batch_with_identical_columns(self):
        rng = np.random.default_rng(52)
        epochs = rng.normal(size=(67, 3, 436)).astype(np.float32)
        names = ["Fz", "Cz", "Pz"]
        batched = eeg_features(epochs, 256, names, batch_size=7)
        single = eeg_features(epochs, 256, names, batch_size=100)
        self.assertEqual(list(batched), list(single))
        np.testing.assert_allclose(batched.to_numpy(), single.to_numpy(), rtol=1e-12, atol=1e-12)
        self.assertEqual(epochs.dtype, np.float32)

    def test_alpha_sine_has_alpha_power_in_named_channel(self):
        fs = 256
        seconds = np.arange(512) / fs
        x = np.sin(2 * np.pi * 10 * seconds)[None, None, :]
        table = eeg_features(x, fs, ["Fz"])
        self.assertGreater(table.loc[0, "EEG_Fz_alpha_LogPower"],
                           table.loc[0, "EEG_Fz_theta_LogPower"] + 4)
        self.assertAlmostEqual(table.loc[0, "EEG_Fz_Std"], np.sqrt(.5), places=5)
        self.assertFalse(any("frontal" in name for name in table))

    def test_nonfinite_eeg_is_rejected(self):
        x = np.ones((2, 1, 512))
        x[1, 0, 5] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            eeg_features(x, 256, ["Cz"], batch_size=1)

    def test_channel_names_cannot_silently_collide(self):
        with self.assertRaisesRegex(ValueError, "collide"):
            eeg_features(np.ones((1, 2, 512)), 256, ["F-z", "F z"])

    def test_reextraction_preserves_recording_onsets_and_replaces_features(self):
        dataset = TrialDataset(table=pd.DataFrame({"Subject_ID": ["sub-A"], "Trial_ID": ["9"],
            "Label": [1], "EEG_Onset": [12.345], "EEG_Timestamp": [678.], "EEG_Fz_Std": [999.]}),
            epochs=np.zeros((1, 1, 512)), sfreq=256., channel_names=("Fz",))
        result = extract_features(dataset)
        self.assertEqual(result.table.loc[0, "EEG_Onset"], 12.345)
        self.assertEqual(result.table.loc[0, "EEG_Timestamp"], 678.)
        self.assertEqual(result.table.loc[0, "EEG_Fz_Std"], 0.)

    def test_physiology_slope_uses_seconds_and_never_claims_hrv_or_rate(self):
        fs = 100
        t = np.arange(170) / fs
        window = np.column_stack([3 + 2 * t, np.sin(2 * np.pi * .2 * t), np.ones_like(t)])
        features = physiology_window_features(window, fs, ["ECG", "RSP", "Trigger"])
        self.assertAlmostEqual(features["ECG_ECG_SlopePerSecond"], 2., places=10)
        self.assertAlmostEqual(features["ECG_ECG_Mean"], np.mean(3 + 2 * t))
        self.assertFalse(any(term in name for name in features for term in ("HRV", "Rate", "SCR", "Trigger")))

    def test_missing_channel_samples_are_nan_with_qc(self):
        x = np.column_stack([np.arange(20, dtype=float), np.ones(20)])
        x[7, 0] = np.nan
        features = physiology_window_features(x, 100, ["ECG", "EDA"])
        self.assertTrue(np.isnan(features["ECG_ECG_Mean"]))
        self.assertEqual(features["QC_ECG_ECG_FiniteFraction"], .95)
        self.assertEqual(features["EDA_EDA_Mean"], 1.)

    def test_outcome_and_label_columns_never_become_behavior_features(self):
        table = pd.DataFrame({"Answer_Self_RT": [120., -2.], "Trial_Balance_Effect": [1, 0],
                              "Prediction_Result": [1, 6], "Answer_Prediction_Correct": ["LEFT", "RIGHT"],
                              "Label": [0, 1]})
        result = behavior_features(table)
        self.assertEqual(list(result), ["Beh_RT_Answer_Self_RT"])
        self.assertTrue(np.isnan(result.iloc[1, 0]))

    def test_behavior_requires_explicit_opt_in_and_preserves_trial_keys(self):
        source = TrialDataset(table=pd.DataFrame({"Subject_ID": ["sub-A", "sub-A"],
                              "Trial_ID": ["7", "19"], "Label": [0, 1],
                              "Answer_Self_RT": [500., 700.]}))
        default = extract_features(source)
        opted = extract_features(source, include_behavior=True)
        self.assertFalse(any(name.startswith("Beh_") for name in default.table))
        self.assertIn("Beh_RT_Answer_Self_RT", opted.table)
        self.assertEqual(opted.table["Trial_ID"].tolist(), ["7", "19"])
        self.assertIn("after all trial responses", opted.provenance["feature_extraction"]["behavior_availability"])
        self.assertNotIn("Beh_RT_Answer_Self_RT", source.table)


if __name__ == "__main__":
    unittest.main()
