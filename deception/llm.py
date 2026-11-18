"""Exploratory LLM classification with explicit failures and coverage."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import feature_columns as audited_feature_columns


def _parse_prediction(content):
    """Accept exactly one JSON integer label, with no repairs or guessing."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    if not isinstance(content, str):
        raise ValueError("Missing text response")
    try:
        result = json.loads(content, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        raise ValueError("Response is not a single valid JSON label") from None
    if not isinstance(result, dict) or set(result) != {"prediction"}:
        raise ValueError("Response must contain only prediction")
    value = result["prediction"]
    if type(value) is not int or value not in {0, 1}:
        raise ValueError("prediction must be integer 0 or 1")
    return value


def _feature_columns(table, requested=None):
    allowed = []
    for modality in ("eeg", "behavior", "physiology"):
        try:
            allowed.extend(audited_feature_columns(table, modality))
        except ValueError as error:
            if not str(error).startswith("No eligible"):
                raise
    allowed = sorted(allowed)
    columns = list(requested) if requested is not None else allowed
    if not columns:
        raise ValueError("No admissible numerical feature columns found")
    if len(set(columns)) != len(columns):
        raise ValueError("Feature names must be unique")
    if not set(columns).issubset(allowed):
        raise ValueError("Selected columns must belong to the audited feature schema")
    return columns


def _numeric_features(row, columns):
    result = {}
    for name, value in zip(columns, row):
        number = float(value)
        result[name] = round(number, 6) if np.isfinite(number) else None
    return result


def _build_prompt(test_values, columns, examples=()):
    body = {
        "task": "Predict the recorded binary task label: 0 = Probable_Truth, 1 = Probable_Lie. "
                "These are recorded task conditions, not independently verified truth/lie ground truth.",
        "features": "Numerical features are standardized using training participants only.",
        "examples": [
            {"features": _numeric_features(values, columns), "prediction": int(label)}
            for values, label in examples
        ],
        "target": _numeric_features(test_values, columns),
        "output": 'Return only a JSON object with one integer field: {"prediction": 0} or {"prediction": 1}.',
    }
    return json.dumps(body, allow_nan=False, ensure_ascii=True)


def _request_prediction(client, prompt, model, max_retries, retry_delay, completion_options):
    last_error = "No successful response"
    for attempt in range(1, max_retries + 2):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                **completion_options,
            )
            content = response.choices[0].message.content
            return _parse_prediction(content), "ok", None, attempt
        except Exception as error:
            # Exception messages can contain HTTP headers or credentials.
            last_error = (
                "InvalidResponse" if isinstance(error, (ValueError, AttributeError, IndexError, TypeError))
                else type(error).__name__
            )
            if attempt <= max_retries and retry_delay:
                time.sleep(min(retry_delay * (2 ** (attempt - 1)), 5.0))
    return pd.NA, "error", last_error, max_retries + 1


def evaluate_llm(dataset, *, model: str, base_url: str | None = None,
                 mode="direct", client=None, api_key_env="OPENAI_API_KEY",
                 max_trials=None, seed=2026, max_retries=2, retry_delay=0.5,
                 timeout=30.0, feature_columns=None, max_output_tokens=128,
                 token_limit_parameter="max_tokens", temperature=None):
    """Evaluate direct or few-shot prompts; failed trials stay missing.

    Injecting a client is intended for offline tests. Otherwise, credentials come
    only from the named environment variable. This function sends selected
    numerical features to the configured service when called with a live client.
    No generated label is a calibrated probability, so Probability remains NaN.
    Temperature is omitted by default. Select the token-limit parameter accepted
    by the chosen service/model; unsupported options are reported as failures.
    """
    if mode not in {"direct", "few_shot"}:
        raise ValueError("mode must be 'direct' or 'few_shot'")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("An explicit model name is required")
    if type(max_retries) is not int or not 0 <= max_retries <= 5:
        raise ValueError("max_retries must be an integer between zero and five")
    if not isinstance(retry_delay, (int, float)) or not isinstance(timeout, (int, float)):
        raise ValueError("retry_delay and timeout must be finite numbers")
    if not np.isfinite(retry_delay) or not np.isfinite(timeout) or retry_delay < 0 or timeout <= 0:
        raise ValueError("retry_delay must be nonnegative and timeout positive")
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be a positive integer")
    if token_limit_parameter not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("token_limit_parameter must be max_tokens or max_completion_tokens")
    if temperature is not None and (
        not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
        or not np.isfinite(temperature) or temperature < 0
    ):
        raise ValueError("temperature must be a finite nonnegative number or None")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise ValueError("api_key_env must name an environment variable")
    if max_trials is not None and (type(max_trials) is not int or max_trials < 1):
        raise ValueError("max_trials must be a positive integer or None")
    owned_client = client is None
    if owned_client:
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"Set {api_key_env} before running the LLM evaluator")
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Install the optional openai dependency to use a live LLM service") from None
        options = {"api_key": key, "timeout": timeout, "max_retries": 0}
        if base_url is not None:
            options["base_url"] = base_url
        client = OpenAI(**options)
    records, feature_qc = [], []
    completion_options = {token_limit_parameter: max_output_tokens, "timeout": timeout}
    if temperature is not None:
        completion_options["temperature"] = temperature
    try:
        dataset.validate()
        table = dataset.table.reset_index(drop=True)
        columns = _feature_columns(table, feature_columns)
        labels = table["Label"].to_numpy(dtype=np.int64)
        groups = table["Subject_ID"].to_numpy()
        if len(np.unique(groups)) < 2 or set(np.unique(labels)) != {0, 1}:
            raise ValueError("LOSO evaluation requires at least two participants and both binary classes")
        values = table[columns].to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        selected = np.arange(len(table))
        if max_trials is not None and max_trials < len(table):
            selected = np.sort(np.random.default_rng(seed).choice(selected, size=max_trials, replace=False))
        selected = set(selected.tolist())
        for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(values, labels, groups)):
            test_idx = np.array([position for position in test_idx if position in selected], dtype=int)
            if not len(test_idx):
                continue
            preprocessing = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
            )
            for prefix in ("EEG_", "ECG_", "RSP_", "EMG_", "EDA_", "Beh_"):
                indices = [i for i, name in enumerate(columns) if name.startswith(prefix)]
                if not indices:
                    continue
                block = values[train_idx][:, indices]
                if not np.isfinite(block).any():
                    raise ValueError(f"The training fold has no observed {prefix.rstrip('_')} modality values")
                if not any(np.unique(column[np.isfinite(column)]).size > 1 for column in block.T):
                    raise ValueError(f"The training fold has only constant {prefix.rstrip('_')} modality features")
            supported = np.array([
                np.unique(column[np.isfinite(column)]).size > 1
                for column in values[train_idx].T
            ], dtype=bool)
            for column, values_column, active in zip(columns, values[train_idx].T, supported):
                finite = values_column[np.isfinite(values_column)]
                feature_qc.append({
                    "Fold": fold, "Test_Subject": str(groups[test_idx][0]),
                    "Feature": column, "Supported_In_Training": bool(active),
                    "Observed_Training_Rows": len(finite),
                    "Distinct_Training_Values": len(np.unique(finite)),
                })
            active_columns = [column for column, active in zip(columns, supported) if active]
            train_values = preprocessing.fit_transform(values[train_idx][:, supported])
            test_values = preprocessing.transform(values[test_idx][:, supported])
            examples = []
            if mode == "few_shot":
                rng = np.random.default_rng(seed + fold)
                for label in (0, 1):
                    candidates = np.flatnonzero(labels[train_idx] == label)
                    if not len(candidates):
                        raise ValueError("Every few-shot training fold must contain both classes")
                    position = rng.choice(candidates)
                    examples.append((train_values[position], label))
            for position, features in zip(test_idx, test_values):
                prediction, status, error, attempts = _request_prediction(
                    client, _build_prompt(features, active_columns, examples), model, max_retries,
                    retry_delay, completion_options,
                )
                row = table.iloc[position]
                records.append({
                    "Subject_ID": row["Subject_ID"], "Trial_ID": row["Trial_ID"],
                    "Label": int(row["Label"]), "Model": f"LLM_{mode}:{model}",
                    "Feature_Set": "numeric_features", "Probability": np.nan,
                    "Prediction": prediction, "Status": status, "Error": error,
                    "Attempts": attempts,
                })
        predictions = pd.DataFrame(records)
        predictions["Prediction"] = predictions["Prediction"].astype("Int64")
        successful = int(predictions["Prediction"].notna().sum())
        attempted = len(predictions)
        diagnostic = max_trials is not None and max_trials < len(table)
        return {
            "predictions": predictions,
            "feature_qc": pd.DataFrame(feature_qc),
            "coverage": successful / attempted if attempted else 0.0,
            "successful": successful, "attempted": attempted, "total_available": len(table),
            "diagnostic": diagnostic,
            "configuration": {
                "mode": mode, "model": model, "seed": seed, "features": columns,
                "completion_options": completion_options, "max_retries": max_retries,
                "retry_delay": retry_delay,
                "preprocessing": "training-fold support selection, median imputation and standardization; unsupported features omitted",
            },
            "note": "Scores apply to successful responses only; failures are missing, and AUC is unavailable without probabilities.",
        }
    finally:
        if owned_client and client is not None:
            client.close()
