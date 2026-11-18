import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from deception.cli import _available_sets, main
import pandas as pd


class CliTests(unittest.TestCase):
    def test_default_feature_sets_include_three_way_fusion(self):
        sets = _available_sets(pd.DataFrame({"EEG_Cz_Std": [1, 2], "ECG_Std": [2, 3],
                                             "Beh_RT_Answer_Self_RT": [1, 2]}))
        self.assertIn("fusion", sets)
        self.assertIn("eeg_physiology", sets)
        self.assertIn("eeg_behavior", sets)

    def test_help_lists_functional_entry_points(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as result:
            main(["--help"])
        self.assertEqual(result.exception.code, 0)
        for command in ("prepare", "benchmark", "spatial", "fusion", "neural", "llm"):
            self.assertIn(command, stream.getvalue())

    def test_control_audit_and_majority_benchmark_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            bundle, results, audit = root / "bundle", root / "results", root / "audit.json"
            main(["mock", "--output", str(bundle), "--subjects", "3", "--trials", "4"])
            main(["audit", "--input", str(bundle), "--output", str(audit)])
            main(["benchmark", "--input", str(bundle), "--output", str(results),
                  "--allow-synthetic", "--models", "dummy", "--feature-sets", "eeg"])
            self.assertEqual(json.loads(audit.read_text())["subjects"], 3)
            report = json.loads((results / "metrics.json").read_text())
            self.assertEqual(report["run_status"], "synthetic_check")
            self.assertFalse(report["state_of_the_art_established"])
            self.assertEqual(report["models"][0]["attempted"], 12)
            self.assertTrue((results / "run.json").exists())

    def test_synthetic_control_cannot_silently_become_real_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            main(["mock", "--output", str(root / "bundle")])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                main(["benchmark", "--input", str(root / "bundle"), "--output", str(root / "scores")])
            self.assertEqual(result.exception.code, 2)
            self.assertFalse((root / "scores").exists())

    def test_original_raw_data_are_required_for_prepare(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(["prepare", "--dataset-root", str(Path(temporary) / "missing"),
                      "--output", str(Path(temporary) / "results")])
            self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
