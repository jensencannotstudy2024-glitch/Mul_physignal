import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from deception.llm import _parse_prediction, evaluate_llm


class MiniDataset:
    def __init__(self):
        self.table = pd.DataFrame([
            {"Subject_ID": f"P{person}", "Trial_ID": trial, "Label": trial % 2,
             "EEG_value": 100 * person + trial, "Beh_Prediction_RT": 400 + trial,
             "Beh_self_irrel_agree": trial % 2}
            for person in range(3) for trial in range(4)
        ])

    def validate(self):
        pass


class FakeClient:
    def __init__(self, responses=None, default='{"prediction": 0}'):
        self.responses = list(responses or [])
        self.default = default
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0) if self.responses else self.default
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])


class LLMTests(unittest.TestCase):
    def test_strict_json_labels(self):
        self.assertEqual(_parse_prediction('{"prediction": 1}'), 1)
        self.assertEqual(_parse_prediction(' {"prediction": 0}\n'), 0)
        for content in (
            '```json\n{"prediction": 1}\n```', 'The answer is 1',
            '{"prediction": true}', '{"prediction": 1.0}', '{"prediction": "1"}',
            '{"prediction": 2}', '{"prediction": 1, "reason": "x"}',
            '{"prediction": 0, "prediction": 1}', None,
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                _parse_prediction(content)

    def test_key_is_required_without_injected_client(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            evaluate_llm(MiniDataset(), model="offline-test")

    def test_api_failure_is_missing_and_never_random(self):
        client = FakeClient(default=RuntimeError("This secret-bearing message must not escape"))
        result = evaluate_llm(MiniDataset(), model="offline-test", client=client,
                              max_trials=2, max_retries=1, retry_delay=0)
        self.assertEqual(len(client.calls), 4)
        self.assertTrue(result["predictions"].Prediction.isna().all())
        self.assertTrue(result["predictions"].Probability.isna().all())
        self.assertEqual(result["coverage"], 0)
        self.assertEqual(set(result["predictions"].Error), {"RuntimeError"})
        self.assertEqual(set(result["predictions"].Attempts), {2})

    def test_retry_after_malformed_response(self):
        client = FakeClient(["not JSON", '{"prediction": 1}'])
        result = evaluate_llm(MiniDataset(), model="offline-test", client=client,
                              max_trials=1, max_retries=1, retry_delay=0)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result["predictions"].Prediction.iloc[0], 1)
        self.assertEqual(result["predictions"].Attempts.iloc[0], 2)
        self.assertEqual(result["coverage"], 1)

    def test_provider_specific_token_option_and_optional_temperature(self):
        default_client = FakeClient()
        evaluate_llm(MiniDataset(), model="offline-test", client=default_client,
                     max_trials=1, max_retries=0)
        self.assertNotIn("temperature", default_client.calls[0])
        self.assertEqual(default_client.calls[0]["max_tokens"], 128)
        selected_client = FakeClient()
        evaluate_llm(MiniDataset(), model="offline-test", client=selected_client,
                     max_trials=1, max_retries=0, max_output_tokens=256,
                     token_limit_parameter="max_completion_tokens", temperature=0.5)
        self.assertNotIn("max_tokens", selected_client.calls[0])
        self.assertEqual(selected_client.calls[0]["max_completion_tokens"], 256)
        self.assertEqual(selected_client.calls[0]["temperature"], 0.5)

    def test_nonfinite_request_options_are_rejected_before_requests(self):
        client = FakeClient()
        for options in ({"timeout": np.nan}, {"timeout": np.inf},
                        {"retry_delay": np.nan}, {"max_retries": np.nan},
                        {"temperature": np.nan}, {"max_output_tokens": 0},
                        {"token_limit_parameter": "unsupported"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluate_llm(MiniDataset(), model="offline-test", client=client, **options)
        self.assertEqual(client.calls, [])

    def test_wholly_missing_modality_is_not_sent_as_imputed_evidence(self):
        dataset = MiniDataset()
        dataset.table["EMG_power"] = np.nan
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "no observed EMG"):
            evaluate_llm(dataset, model="offline-test", client=client, max_retries=0)
        self.assertEqual(client.calls, [])

    def test_numeric_baseline_can_use_one_actual_modality(self):
        dataset = MiniDataset()
        dataset.table = dataset.table.drop(columns=["Beh_Prediction_RT", "Beh_self_irrel_agree"])
        client = FakeClient()
        result = evaluate_llm(dataset, model="offline-test", client=client,
                              max_trials=1, max_retries=0)
        self.assertEqual(result["configuration"]["features"], ["EEG_value"])
        self.assertEqual(result["coverage"], 1)

    def test_constant_modality_is_rejected_before_any_request(self):
        dataset = MiniDataset()
        dataset.table["EMG_power"] = 5.0
        dataset.table["EMG_offset"] = 8.0
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "only constant EMG"):
            evaluate_llm(dataset, model="offline-test", client=client, max_retries=0)
        self.assertEqual(client.calls, [])

    def test_held_out_only_features_are_omitted_from_that_folds_prompts(self):
        original = MiniDataset()
        mask = original.table.Subject_ID == "P0"
        original.table["EEG_unobserved"] = np.nan
        original.table["EEG_train_constant"] = 7.0
        for column in ("EEG_unobserved", "EEG_train_constant"):
            original.table.loc[mask, column] = np.arange(4, dtype=float)
        changed = MiniDataset()
        changed.table = original.table.copy()
        for column in ("EEG_unobserved", "EEG_train_constant"):
            changed.table.loc[mask, column] = 1e12 + np.arange(4, dtype=float)
        a, b = FakeClient(), FakeClient()
        first = evaluate_llm(original, model="offline-test", client=a, max_retries=0)
        second = evaluate_llm(changed, model="offline-test", client=b, max_retries=0)
        for before, after in zip(a.calls[:4], b.calls[:4]):
            self.assertEqual(before["messages"], after["messages"])
            target = json.loads(before["messages"][0]["content"])["target"]
            self.assertNotIn("EEG_unobserved", target)
            self.assertNotIn("EEG_train_constant", target)
        qc = first["feature_qc"].query("Test_Subject == 'P0' and Feature in ['EEG_unobserved', 'EEG_train_constant']")
        self.assertEqual(len(qc), 2)
        self.assertFalse(qc.Supported_In_Training.any())

    def test_partial_coverage_is_explicit_and_sampling_is_diagnostic(self):
        client = FakeClient(['{"prediction": 1}', RuntimeError("offline"), '{"prediction": 0}'])
        result = evaluate_llm(MiniDataset(), model="offline-test", client=client,
                              max_trials=3, max_retries=0)
        self.assertEqual(result["successful"], 2)
        self.assertEqual(result["attempted"], 3)
        self.assertEqual(result["total_available"], 12)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertTrue(result["diagnostic"])
        self.assertTrue(result["predictions"].Probability.isna().all())

    def test_few_shot_examples_are_from_other_people_and_no_outcome_features(self):
        dataset = MiniDataset()
        client = FakeClient()
        result = evaluate_llm(dataset, model="offline-test", client=client,
                              mode="few_shot", max_retries=0)
        self.assertFalse(result["diagnostic"])
        self.assertEqual(len(client.calls), 12)
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        for call, (_, output) in zip(client.calls, result["predictions"].iterrows()):
            prompt = json.loads(call["messages"][0]["content"])
            self.assertEqual({example["prediction"] for example in prompt["examples"]}, {0, 1})
            self.assertNotIn("Beh_self_irrel_agree", prompt["target"])
            columns = list(prompt["target"])
            pool = dataset.table[dataset.table.Subject_ID != output.Subject_ID]
            pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
            transformed = pipeline.fit_transform(pool[columns].to_numpy(dtype=float))
            for example in prompt["examples"]:
                target = np.array([example["features"][column] for column in columns])
                matches = np.all(np.isclose(transformed, target, atol=1e-5), axis=1)
                self.assertTrue(matches.any())
                self.assertIn(example["prediction"], set(pool.loc[matches, "Label"]))


if __name__ == "__main__":
    unittest.main()
