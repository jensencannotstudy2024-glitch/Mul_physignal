import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from deception.data import (TrialDataset, audit_table, feature_columns, load_bundle,
                            load_legacy_table, require_evaluable, save_bundle)


def dataset():
    return TrialDataset(pd.DataFrame({
        "Subject_ID": ["A", "A", "B", "B"], "Trial_ID": ["0", "2", "0", "1"],
        "Label": [0, 1, 1, 0], "EEG_Cz_Std": [1., 2., 3., 4.],
        "EEG_Onset": [7., 20., 9., 31.], "ECG_Recorded_Std": [2., 1., 4., 3.],
        "Beh_RT_Answer_Self_RT": [1., 2., 3., np.nan],
        "Beh_pred_correct": [0, 1, 1, 0], "Beh_Trial_Balance": [1, 2, 3, 4],
    }), np.ones((4, 2, 20), dtype=np.float32), 100., -.2, ("Cz", "Pz"), {"status": "synthetic"})


class DataTests(unittest.TestCase):
    def test_roundtrip_preserves_epoch_and_trial_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = dataset()
            save_bundle(original, temporary)
            loaded = load_bundle(temporary)
            self.assertEqual(loaded.table.Trial_ID.tolist(), ["0", "2", "0", "1"])
            np.testing.assert_array_equal(original.epochs, loaded.epochs)
            self.assertEqual(loaded.provenance, original.provenance)

    def test_bundle_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            save_bundle(dataset(), temporary)
            with (Path(temporary) / "trials.csv").open("a") as stream:
                stream.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                load_bundle(temporary)

    def test_refused_save_does_not_change_existing_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            save_bundle(dataset(), temporary)
            root = Path(temporary)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            altered = dataset()
            altered.epochs = None
            altered.table.Label = 1
            with self.assertRaisesRegex(ValueError, "already contains"):
                save_bundle(altered, temporary)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            load_bundle(root)

    def test_failed_serialization_does_not_publish_partial_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = dataset()
            data.provenance["invalid"] = np.nan
            with self.assertRaises(ValueError):
                save_bundle(data, temporary)
            self.assertFalse((Path(temporary) / "trials.csv").exists())

    def test_mixed_identifier_collisions_are_rejected(self):
        data = dataset()
        data.table.loc[0, ["Subject_ID", "Trial_ID"]] = [1, 0]
        data.table.loc[1, ["Subject_ID", "Trial_ID"]] = ["1", "0"]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            data.validate()

    def test_numeric_identifiers_are_canonicalized(self):
        data = dataset()
        data.table.Trial_ID = [0, 2, 0, 1]
        data.validate()
        self.assertTrue(all(isinstance(value, str) for value in data.table.Trial_ID))

    def test_numeric_string_labels_have_the_same_downstream_contract(self):
        data = dataset()
        data.table.Label = data.table.Label.astype(str)
        data.validate()
        self.assertTrue(pd.api.types.is_integer_dtype(data.table.Label.dtype))

    def test_empty_ids_wrong_labels_and_epoch_counts_fail(self):
        for column, value in [("Trial_ID", " "), ("Label", 2), ("Subject_ID", None)]:
            data = dataset()
            data.table.loc[0, column] = value
            with self.assertRaises(ValueError):
                data.validate()
        data = dataset()
        data.epochs = data.epochs[:-1]
        with self.assertRaises(ValueError):
            data.validate()

    def test_nonfinite_epochs_fail_without_zero_filling(self):
        data = dataset()
        data.epochs[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            data.validate()

    def test_boolean_subset_keeps_labels_and_epochs_aligned(self):
        data = dataset()
        selected = data.subset(np.array([True, False, False, True]))
        self.assertEqual(selected.table.Label.tolist(), [0, 0])
        self.assertEqual(selected.table.Trial_ID.tolist(), ["0", "1"])
        np.testing.assert_array_equal(selected.epochs, data.epochs[[0, 3]])

    def test_clock_metadata_and_task_outcomes_are_not_predictors(self):
        data = dataset()
        self.assertEqual(feature_columns(data.table, "eeg"), ["EEG_Cz_Std"])
        self.assertEqual(feature_columns(data.table, "behavior"), ["Beh_RT_Answer_Self_RT"])
        self.assertNotIn("Beh_pred_correct", feature_columns(data.table))

    def test_fusion_requires_every_named_modality(self):
        table = dataset().table[["Subject_ID", "Trial_ID", "Label", "EEG_Cz_Std"]]
        for mode in ("eeg_behavior", "eeg_physiology", "fusion"):
            with self.assertRaisesRegex(ValueError, "No eligible"):
                feature_columns(table, mode)

    def test_unverified_and_synthetic_data_need_explicit_diagnostic_mode(self):
        data = dataset()
        with self.assertRaises(ValueError):
            require_evaluable(data)
        self.assertEqual(require_evaluable(data, allow_synthetic=True), "synthetic_check")
        data.provenance = {"status": "legacy_unverified"}
        with self.assertRaises(ValueError):
            require_evaluable(data)
        self.assertEqual(require_evaluable(data, allow_unverified=True), "diagnostic_unverified")

    def test_legacy_import_reports_unknown_row_identity_and_csp(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.csv"
            pd.DataFrame({"Subject_ID": ["A", "B"], "Label": [0, 1], "CSP_feat0": [1, 2]}).to_csv(path, index=False)
            data = load_legacy_table(path)
            self.assertTrue(data.provenance["has_prefitted_supervised_features"])
            self.assertTrue(data.table.Trial_ID.str.startswith("legacy-row-").all())
            report = audit_table(data)
            self.assertEqual(report["subjects"], 2)
            self.assertIn("CSP_feat0", report["excluded_feature_columns"])

    def test_manifest_paths_cannot_escape_the_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            save_bundle(dataset(), temporary)
            path = Path(temporary) / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["files"]["../secret"] = "fake"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                load_bundle(temporary, verify_hashes=False)


if __name__ == "__main__":
    unittest.main()
