"""Test keyed alignment, recording clocks, and explicit acquisition failures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from deception.preparation import (AlignmentRequiredError, _eeg_metadata_channels,
    _read_physio_metadata, epoch_from_recording, prepare_dataset,
    read_physiology_features, validate_alignment, write_alignment_template)
from deception.data import file_sha256


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.beh = self.root / "sub-A" / "ses-01" / "beh"
        self.beh.mkdir(parents=True)
        self.stem = "sub-A_ses-01_task-GG_run-01"

    def tearDown(self):
        self.temp.cleanup()

    def write_behavior(self, conditions=None):
        conditions = ["Probable_Truth", "Other", "Probable_Lie"] if conditions is None else conditions
        table = pd.DataFrame({"Trial_Order": [0, 1, 2], "Trial_Condition": conditions,
                              "Answer_Self_RT": [110., 220., 330.]})
        table.to_csv(self.beh / f"{self.stem}_beh.tsv", sep="\t", index=False)

    def test_template_preserves_actual_noncontiguous_trial_ids_without_fake_onsets(self):
        self.write_behavior()
        table = write_alignment_template(self.root, self.root / "template.csv")
        self.assertEqual(table["Trial_ID"].tolist(), ["0", "2"])
        self.assertTrue(table["EEG_Onset"].isna().all())
        self.assertTrue(table["Physio_Onset"].isna().all())

    def test_missing_alignment_stops_before_opening_raw_data(self):
        self.write_behavior()
        with self.assertRaisesRegex(AlignmentRequiredError, "no Sync-to-trial mapping"):
            prepare_dataset(self.root, self.root / "output", None)
        self.assertTrue((self.root / "output" / "alignment_template.csv").is_file())
        self.assertFalse((self.root / "output" / "epochs.npy").exists())

    def test_template_omits_subject_without_eeg_when_raw_recordings_exist(self):
        self.write_behavior()
        eeg_dir = self.beh.parent / "eeg"
        eeg_dir.mkdir()
        (eeg_dir / f"{self.stem}_eeg.edf").touch()
        missing_beh = self.root / "sub-B" / "ses-01" / "beh"
        missing_beh.mkdir(parents=True)
        pd.DataFrame({"Trial_Order": [91], "Trial_Condition": ["Probable_Lie"]}).to_csv(
            missing_beh / "sub-B_ses-01_task-GG_run-01_beh.tsv", sep="\t", index=False)
        template = write_alignment_template(self.root, self.root / "template.csv")
        self.assertEqual(template["Subject_ID"].unique().tolist(), ["sub-A"])

    def test_event_phase_must_be_explicit_and_consistent(self):
        trials = pd.DataFrame({"Subject_ID": ["sub-A", "sub-A"], "Trial_ID": ["0", "2"]})
        supplied = pd.DataFrame({"Subject_ID": "sub-A", "Trial_ID": ["0", "2"],
                                 "EEG_Onset": [6., 30.], "Physio_Onset": [8., 32.],
                                 "Event_Phase": ["trial_start", "answer_onset"]})
        with self.assertRaisesRegex(ValueError, "Event_Phase"):
            validate_alignment(supplied, trials)

    def test_keyed_join_reorders_alignment_instead_of_truncating(self):
        trials = pd.DataFrame({"Subject_ID": ["sub-A", "sub-A"], "Trial_ID": ["0", "2"]})
        supplied = pd.DataFrame({"Subject_ID": ["sub-A", "sub-A"], "Trial_ID": ["2", "0"],
                                 "EEG_Onset": [30., 6.], "Physio_Onset": [32., 8.],
                                 "Event_Phase": "answer_onset"})
        joined = validate_alignment(supplied, trials)
        self.assertEqual(joined["EEG_Onset"].tolist(), [6., 30.])
        for broken in [supplied.iloc[:1], pd.concat([supplied, supplied.iloc[:1]])]:
            with self.assertRaises(ValueError):
                validate_alignment(broken, trials)
        supplied.loc[0, "EEG_Onset"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_alignment(supplied, trials)

    def test_physiology_starttime_and_chunk_boundaries_use_correct_samples(self):
        fs, starttime = 100, 5.
        sample_time = np.arange(1000) / fs
        table = np.column_stack([2 * sample_time + 3, np.ones(1000)])
        path = self.beh / "physio.tsv.gz"
        pd.DataFrame(table).to_csv(path, sep="\t", header=False, index=False)
        metadata = {"Columns": ["ECG", "Trigger"], "SamplingFrequency": fs, "StartTime": starttime}
        features = read_physiology_features(path, metadata, [6., 14.8], -.2, 1.5, chunksize=73)
        wanted = table[80:251, 0]
        self.assertAlmostEqual(features.loc[0, "ECG_ECG_Mean"], wanted.mean())
        self.assertEqual(features.loc[0, "QC_Physio_Status"], "ok")
        self.assertTrue(np.isnan(features.loc[1, "ECG_ECG_Mean"]))
        self.assertEqual(features.loc[1, "QC_Physio_Status"], "window_outside_recording")

    def test_metadata_is_required_instead_of_assuming_column_positions(self):
        path = self.beh / "physio.json"
        path.write_text(json.dumps({"SamplingFrequency": 100, "Columns": ["ECG"]}))
        with self.assertRaisesRegex(ValueError, "StartTime"):
            _read_physio_metadata(path)

    def test_physiology_endpoint_grid_errors_are_bounded_and_empty_selection_is_safe(self):
        path = self.beh / "physio.tsv"
        pd.DataFrame(np.arange(1000) / 100).to_csv(path, sep="\t", header=False, index=False)
        metadata = {"Columns": ["ECG"], "SamplingFrequency": 100, "StartTime": 2.}
        features = read_physiology_features(path, metadata, [4.003125, 5.008], -.203125, 1.5)
        self.assertTrue((features["QC_Physio_WindowStartGridError_s"].abs() <= .00500001).all())
        self.assertTrue((features["QC_Physio_WindowEndGridError_s"].abs() <= .00500001).all())
        empty = read_physiology_features(path, metadata, [], -.203125, 1.5)
        self.assertTrue(empty.empty)
        self.assertIn("QC_Physio_Status", empty)


    def test_eeg_selection_uses_types_and_removes_documented_eye_channels(self):
        channels = pd.DataFrame({"name": [f"EEG{i}" for i in range(1, 129)], "type": "EEG"})
        chosen, reasons = _eeg_metadata_channels(channels)
        self.assertNotIn("EEG124", chosen)
        self.assertNotIn("EEG125", chosen)
        self.assertEqual(len(chosen), 126)
        self.assertTrue(all(r["reason"] == "documented_EOG_124_125" for r in reasons))

    def test_auxiliary_acquisition_channels_are_excluded_from_eeg(self):
        channels = pd.DataFrame({"name": ["1", "2", "Sync", "Gnd-Ref", "Gnd-Ref1",
                                         "Batt_0", "Foto", "Ref"], "type": "EEG"})
        chosen, reasons = _eeg_metadata_channels(channels)
        self.assertEqual(chosen, ["1", "2"])
        self.assertEqual({r["reason"] for r in reasons}, {"non_EEG_acquisition_channel"})

    def test_buffered_antialiased_epochs_and_explicit_boundary_drop(self):
        fs = 200
        t = np.arange(fs * 30) / fs
        continuous = np.sin(2 * np.pi * 10 * t)[None, :]
        reader = lambda start, stop: continuous[:, start:stop]
        epoch, reason, error = epoch_from_recording(reader, len(t), fs, 10., -.2, 1.5, 100.)
        self.assertEqual(reason, "ok")
        self.assertEqual(epoch.shape, (1, 171))
        self.assertEqual(error, 0.)
        self.assertAlmostEqual(float(epoch.std()), np.sqrt(.5), delta=.03)
        dropped, reason, _ = epoch_from_recording(reader, len(t), fs, 1., -.2, 1.5, 100.)
        self.assertIsNone(dropped)
        self.assertEqual(reason, "insufficient_continuous_filter_context")

    def test_eeg_grid_error_includes_output_resampling_grid(self):
        fs, target, onset = 200, 128, 10.001
        t = np.arange(fs * 30) / fs
        signal = np.sin(2 * np.pi * 10 * t)[None, :]
        _, reason, error = epoch_from_recording(lambda start, stop: signal[:, start:stop],
                                                len(t), fs, onset, -.2, 1.5, target)
        self.assertEqual(reason, "ok")
        self.assertLessEqual(abs(error), .5 / fs + .5 / target + 1e-12)
        self.assertNotAlmostEqual(error, round(onset * fs) / fs - onset, places=6)

    def test_middle_eeg_drop_preserves_original_labels_and_physiology_rows(self):
        self.write_behavior(["Probable_Truth", "Probable_Lie", "Probable_Lie"])
        eeg_dir = self.beh.parent / "eeg"
        eeg_dir.mkdir()
        (eeg_dir / f"{self.stem}_eeg.edf").touch()
        (eeg_dir / f"{self.stem}_eeg.json").write_text(json.dumps({"SamplingFrequency": 200}))
        pd.DataFrame({"name": ["Fz"], "type": ["EEG"]}).to_csv(
            eeg_dir / f"{self.stem}_channels.tsv", sep="\t", index=False)
        pd.DataFrame({"onset": [6., 15., 25.], "duration": 0., "trial_type": "Sync(1)"}).to_csv(
            eeg_dir / f"{self.stem}_events.tsv", sep="\t", index=False)
        metadata = {"SamplingFrequency": 100, "StartTime": 2., "Columns": ["ECG"]}
        (self.beh / f"{self.stem}_physio.json").write_text(json.dumps(metadata))
        pd.DataFrame(np.arange(4000) / 100).to_csv(self.beh / f"{self.stem}_physio.tsv.gz",
                                                  sep="\t", header=False, index=False)
        alignment = pd.DataFrame({"Subject_ID": "sub-A", "Trial_ID": ["0", "1", "2"],
                                  "EEG_Onset": [6., 15., 25.], "Physio_Onset": [8., 17., 27.],
                                  "Event_Phase": "trial_start"})
        alignment_path = self.root / "alignment.csv"
        alignment.to_csv(alignment_path, index=False)

        class FakeRaw:
            info = {"sfreq": 200.}
            ch_names = ["Fz"]
            n_times = 8000
            annotations = []
            def get_data(self, picks, start, stop):
                x = np.sin(2 * np.pi * 10 * np.arange(start, stop) / 200)[None, :]
                if start <= 3000 < stop:
                    x[0, 3000 - start] = np.nan
                return x
            def close(self):
                pass

        with patch("deception.preparation._read_raw", return_value=FakeRaw()):
            result = prepare_dataset(self.root, self.root / "bundle", alignment_path, target_sfreq=128.)
        self.assertEqual(result.table["Trial_ID"].tolist(), ["0", "2"])
        self.assertEqual(result.table["Label"].tolist(), [0, 1])
        self.assertEqual(result.epochs.shape[0], 2)
        self.assertAlmostEqual(result.table.loc[0, "ECG_ECG_Mean"], (5.8 + 7.5) / 2)
        self.assertAlmostEqual(result.table.loc[1, "ECG_ECG_Mean"], (24.8 + 26.5) / 2)
        dropped = pd.read_csv(self.root / "bundle" / "dropped_trials.csv", dtype={"Trial_ID": str})
        self.assertEqual(dropped["Trial_ID"].tolist(), ["1"])
        self.assertFalse(any(c.startswith("Beh_") for c in result.table))
        self.assertTrue((self.root / "bundle" / "metadata.json").is_file())
        self.assertEqual(result.provenance["alignment_sha256"], file_sha256(alignment_path))
        recording = result.provenance["recordings"]["sub-A"]
        self.assertEqual(recording["behavior_sha256"], file_sha256(self.beh / f"{self.stem}_beh.tsv"))
        self.assertEqual(recording["source_metadata_sha256"]["events"],
                         file_sha256(eeg_dir / f"{self.stem}_events.tsv"))
        self.assertEqual(recording["raw_recordings"]["eeg"]["size_bytes"], 0)
        self.assertIsNone(recording["raw_recordings"]["eeg"]["sha256"])
        eeg_start_relative = result.table["QC_EEG_ActualWindowStart_s"] - result.table["EEG_Onset"]
        phys_start_relative = result.table["QC_Physio_ActualWindowStart_s"] - result.table["Physio_Onset"]
        self.assertTrue(((eeg_start_relative - phys_start_relative).abs() <= .00500001).all())


if __name__ == "__main__":
    unittest.main()
