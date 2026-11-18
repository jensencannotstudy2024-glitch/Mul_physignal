"""Regression checks for supervised spatial fitting and the empty baseline bug."""
import os
from pathlib import Path
import tempfile
import unittest

# MNE/numba require writable caches in restricted environments. This does not
# change HOME or touch the user's MNE configuration.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "deception-numba-tests"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "deception-matplotlib-tests"))

import numpy as np
import pandas as pd

from deception.data import TrialDataset
from deception.spatial import CSPFeatures, XdawnERPFeatures, evaluate_spatial


def make_epochs(subjects=4, trials=10):
    rng = np.random.default_rng(19)
    rows, epochs = [], []
    time = -0.2 + np.arange(171) / 100
    p300 = np.exp(-((time - 0.4) / 0.07) ** 2)
    for subject in range(subjects):
        for trial in range(trials):
            label = trial % 2
            signal = rng.normal(size=(6, len(time))) * 0.3
            signal[0] *= 1 + label
            signal[1] += (2 * label - 1) * p300
            signal[2] -= (2 * label - 1) * p300 / 2
            epochs.append(signal)
            rows.append({"Subject_ID": f"S{subject}", "Trial_ID": f"T{trial}", "Label": label})
    return TrialDataset(pd.DataFrame(rows), np.stack(epochs), 100.0, -0.2,
                        tuple(f"C{i}" for i in range(6)), {"status": "synthetic"}).validate()


class SpatialTests(unittest.TestCase):
    def test_csp_inference_uses_frozen_training_filters(self):
        dataset = make_epochs()
        train = dataset.table.Subject_ID != "S0"
        transformer = CSPFeatures(n_components=4, sfreq=dataset.sfreq).fit(
            dataset.epochs[train], dataset.table.Label.to_numpy()[train]
        )
        original_filters = transformer.filters_.copy()
        result = transformer.transform(dataset.epochs[~train])
        self.assertEqual(result.shape, (10, 4))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(original_filters, transformer.filters_)
        with self.assertRaisesRegex(ValueError, "Channel count"):
            transformer.transform(dataset.epochs[~train, :4])

    def test_car_reference_does_not_create_a_zero_variance_component(self):
        dataset = make_epochs()
        transformer = CSPFeatures(n_components=6).fit(dataset.epochs, dataset.table.Label.to_numpy())
        self.assertEqual(transformer.effective_rank_, 5)
        self.assertEqual(transformer.transform(dataset.epochs).shape[1], 5)
        np.testing.assert_allclose(transformer.filters_.sum(axis=1), 0, atol=1e-10)
        single = CSPFeatures(n_components=1).fit(dataset.epochs, dataset.table.Label.to_numpy())
        self.assertGreater(np.std(single.transform(dataset.epochs)), 0.01)

    def test_spatial_outer_predictions_ignore_held_out_labels(self):
        dataset = make_epochs()
        options = dict(representation="csp", models=("logreg",),
                       inner_splits=2, n_components=4,
                       model_grids={"logreg": {"classifier__C": [1.0]}})
        first = evaluate_spatial(dataset, **options)
        dataset.table.loc[dataset.table.Subject_ID == "S0", "Label"] ^= 1
        second = evaluate_spatial(dataset, **options)
        a = first["predictions"].query("Subject_ID == 'S0'").sort_values(["Model", "Trial_ID"])
        b = second["predictions"].query("Subject_ID == 'S0'").sort_values(["Model", "Trial_ID"])
        np.testing.assert_allclose(a.Probability, b.Probability, rtol=0, atol=0)

    def test_random_noise_cannot_get_near_perfect_scores_through_csp_labels(self):
        rng = np.random.default_rng(99)
        epochs, rows = [], []
        for subject in range(8):
            labels = np.tile([0, 1], 15)
            rng.shuffle(labels)
            epochs.append(rng.normal(size=(30, 20, 100)))
            rows.extend({"Subject_ID": f"S{subject}", "Trial_ID": f"T{trial}", "Label": int(label)}
                        for trial, label in enumerate(labels))
        dataset = TrialDataset(pd.DataFrame(rows), np.concatenate(epochs), 100., -.2,
                               tuple(f"C{i}" for i in range(20)), {"status": "synthetic"})
        result = evaluate_spatial(
            dataset, representation="csp", models=("logreg",),
            band=None, common_average=False, inner_splits=2,
            model_grids={"logreg": {"classifier__C": [1.0]}},
        )
        predicted = result["predictions"].query("Model == 'logreg'")
        accuracy = np.mean(predicted.Label.to_numpy() == predicted.Prediction.to_numpy())
        # A deterministic regression for the original bug, which could score
        # >99% on pure random noise by fitting each test subject using its labels.
        self.assertGreater(accuracy, 0.25)
        self.assertLess(accuracy, 0.75)

    def test_xdawn_erp_features_are_finite_nonconstant_and_baseline_corrected(self):
        dataset = make_epochs(subjects=2)
        transformer = XdawnERPFeatures(n_components=2, sfreq=100, tmin=-0.2,
                                       band=None, common_average=False).fit(
            dataset.epochs[:10], dataset.table.Label.to_numpy()[:10]
        )
        features = transformer.transform(dataset.epochs[10:])
        self.assertEqual(features.shape, (10, 32))  # two components per class
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(np.std(features[:, 0]), 0.01)
        np.testing.assert_allclose(
            features, transformer.transform(dataset.epochs[10:] + 500), atol=1e-10
        )
        latencies = features.reshape(10, 4, 8)[..., 7]
        self.assertTrue(((latencies >= 0.3) & (latencies < 0.5)).all())

    def test_invalid_baselines_and_nonfinite_epochs_fail_loudly(self):
        dataset = make_epochs()
        labels = dataset.table.Label.to_numpy()
        with self.assertRaisesRegex(ValueError, "baseline"):
            XdawnERPFeatures(tmin=0).fit(dataset.epochs, labels)
        bad = dataset.epochs.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            CSPFeatures().fit(bad, labels)
        with self.assertRaisesRegex(ValueError, "both 0 and 1"):
            CSPFeatures().fit(dataset.epochs, np.zeros(len(labels)))

    def test_standard_xdawn_runs_inside_nested_grouped_pipeline(self):
        dataset = make_epochs(subjects=3, trials=6)
        result = evaluate_spatial(
            dataset, representation="xdawn", models=("logreg",),
            inner_splits=2, n_components=2,
            model_grids={"logreg": {"classifier__C": [1.0]}},
        )
        self.assertEqual(len(result["predictions"]), 2 * len(dataset.table))
        self.assertTrue(result["predictions"].Probability.between(0, 1).all())

    def test_spatial_dummy_does_not_need_two_classes_or_fit_filters(self):
        dataset = make_epochs()
        dataset.table["Label"] = 0
        result = evaluate_spatial(dataset, models=("dummy",))
        self.assertTrue((result["predictions"].Probability == 0).all())


if __name__ == "__main__":
    unittest.main()
