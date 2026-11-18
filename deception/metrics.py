"""Comparable scores with explicit prediction and probability coverage."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from .data import ID_COLUMNS, TrialDataset, file_sha256


def _scores(frame: pd.DataFrame) -> dict[str, Any]:
    successful = frame.Prediction.notna()
    y = frame.loc[successful, "Label"].to_numpy(dtype=int)
    predicted = frame.loc[successful, "Prediction"].to_numpy(dtype=int)
    result = {
        "attempted": int(len(frame)), "successful": int(successful.sum()),
        "coverage": float(successful.mean()), "accuracy": None,
        "balanced_accuracy": None, "f1_macro": None, "roc_auc": None,
        "probability_coverage": 0.0,
    }
    if len(y):
        result.update(accuracy=float(accuracy_score(y, predicted)),
                      balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                      f1_macro=float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)))
    if "Probability" in frame:
        probabilities = pd.to_numeric(frame.Probability, errors="raise")
        usable = successful & probabilities.notna()
        result["probability_coverage"] = float(usable.mean())
        if usable.any() and frame.loc[usable, "Label"].nunique() == 2:
            result["roc_auc"] = float(roc_auc_score(frame.loc[usable, "Label"], probabilities[usable]))
    return result


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    required = {*ID_COLUMNS, "Label", "Model", "Feature_Set", "Prediction"}
    if not required.issubset(predictions):
        raise ValueError(f"Prediction table is missing {sorted(required - set(predictions))}.")
    if predictions.empty:
        raise ValueError("No predictions were produced.")
    predictions = predictions.copy()
    for column in (*ID_COLUMNS, "Model", "Feature_Set"):
        if predictions[column].isna().any() or predictions[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Prediction grouping column {column} contains missing or empty identifiers.")
        predictions[column] = predictions[column].astype(str)
    if predictions.duplicated([*ID_COLUMNS, "Model", "Feature_Set"]).any():
        raise ValueError("A model has duplicate predictions for the same trial.")
    if not predictions.Label.isin([0, 1]).all():
        raise ValueError("Invalid ground-truth labels in the prediction table.")
    if (predictions.groupby(list(ID_COLUMNS)).Label.nunique() != 1).any():
        raise ValueError("Ground-truth labels disagree across models for the same trial.")
    universes = [set(map(tuple, frame[list(ID_COLUMNS)].astype(str).to_numpy()))
                 for _, frame in predictions.groupby(["Model", "Feature_Set"], sort=True)]
    if any(universe != universes[0] for universe in universes[1:]):
        raise ValueError("Models in one comparison must attempt exactly the same trial identities.")
    present = predictions.Prediction.dropna()
    if not present.isin([0, 1]).all():
        raise ValueError("Predictions must be 0, 1, or missing on failure.")
    if "Probability" in predictions:
        p = pd.to_numeric(predictions.Probability, errors="raise").dropna()
        if not np.isfinite(p).all() or not p.between(0, 1).all():
            raise ValueError("Predicted probabilities must be finite values between 0 and 1.")
    models = []
    for (model, features), frame in predictions.groupby(["Model", "Feature_Set"], sort=True):
        summary = {"Model": model, "Feature_Set": features, **_scores(frame)}
        by_subject = [{"Subject_ID": str(subject), **_scores(part)}
                      for subject, part in frame.groupby("Subject_ID", sort=True)]
        summary["subjects"] = len(by_subject)
        summary["per_subject"] = by_subject
        accuracies = [row["accuracy"] for row in by_subject if row["accuracy"] is not None]
        summary["mean_subject_accuracy"] = float(np.mean(accuracies)) if accuracies else None
        summary["std_subject_accuracy"] = float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else None
        models.append(summary)
    return {
        "metric_definitions": {
            "accuracy": "Pooled fraction of successful predictions that are correct.",
            "balanced_accuracy": "Mean recall of the classes present in the evaluated rows.",
            "f1_macro": "Pooled macro F1 over labels 0 and 1, zero_division=0.",
            "roc_auc": "Pooled AUC on successful rows with actual predicted probabilities; null if unavailable.",
            "coverage": "Successful predictions divided by attempted predictions; always report with scores.",
            "mean_subject_accuracy": "Unweighted mean of each held-out participant's accuracy.",
        },
        "models": models,
    }


def write_run(result: dict[str, Any], dataset: TrialDataset, output_dir: str | Path,
              *, configuration: dict[str, Any], run_status: str) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("The results directory is not empty; use a new directory for each run.")
    summary = summarize_predictions(result["predictions"])
    predictions = result["predictions"]
    truth = dataset.table[[*ID_COLUMNS, "Label"]].copy()
    for column in ID_COLUMNS:
        truth[column] = truth[column].astype(str)
    checked = predictions[[*ID_COLUMNS, "Label"]].copy()
    for column in ID_COLUMNS:
        checked[column] = checked[column].astype(str)
    checked = checked.merge(truth, on=list(ID_COLUMNS), how="left", suffixes=("", "_expected"), validate="many_to_one")
    if checked.Label_expected.isna().any() or not checked.Label.eq(checked.Label_expected).all():
        raise ValueError("Prediction labels or trial identities disagree with the input dataset.")
    attempted = predictions[[*ID_COLUMNS]].drop_duplicates()
    if len(attempted) != len(dataset.table) and not result.get("diagnostic", False):
        raise ValueError("The run omitted trials without declaring an explicit diagnostic sample.")
    summary["sampling_fraction"] = len(attempted) / len(dataset.table)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output / f"{name}.csv", index=False)
            artifacts[f"{name}.csv"] = file_sha256(output / f"{name}.csv")
    summary["run_status"] = run_status
    summary["state_of_the_art_established"] = False
    summary["limitations"] = [
        "Correct execution does not establish state-of-the-art performance.",
        "Comparison requires identical dataset versions, trial identities, event phases, splits and metrics.",
        "Scores from legacy tables cannot verify original trial alignment or supervised feature fitting.",
    ]
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "mne", "torch", "openai", "catboost", "xgboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    code_files = {p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")}
    metadata = {
        "configuration": configuration, "run_status": run_status,
        "dataset_provenance": dataset.provenance, "versions": versions,
        "source_sha256": code_files, "artifacts_sha256": artifacts,
        "trial_count": len(dataset.table), "subject_count": int(dataset.table.Subject_ID.nunique()),
        "result_notes": {k: v for k, v in result.items()
                         if not isinstance(v, pd.DataFrame) and isinstance(v, (str, bool, int, float, dict, list, type(None)))},
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False, default=str), encoding="utf-8")
    return output
