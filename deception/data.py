"""Aligned trials, portable bundles, and explicit feature selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd


ID_COLUMNS = ("Subject_ID", "Trial_ID")
REQUIRED_COLUMNS = (*ID_COLUMNS, "Label")
PHYSIOLOGY_PREFIXES = ("ECG_", "RSP_", "EMG_", "EDA_")
NONFEATURE_COLUMNS = {"EEG_Onset", "EEG_Timestamp", "EEG_Datetime", "EEG_Sample", "EEG_Trial_ID"}
BEHAVIOR_RT_COLUMNS = {
    "Beh_Answer_Self_RT", "Beh_Answer_Irrelevant_RT", "Beh_Prediction_RT",
    "Beh_Answer_Prediction_Agent_RT", "Beh_Answer_Prediction_Correct_RT",
    "Beh_log_self_rt", "Beh_log_irrel_rt", "Beh_log_pred_rt", "Beh_rt_diff",
    "Beh_rt_ratio", "Beh_rt_sum", "Beh_rt_max", "Beh_rt_min", "Beh_rt_product",
    "Beh_total_rt", "Beh_mean_rt", "Beh_max_rt", "Beh_min_rt", "Beh_std_rt",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_columns(table: pd.DataFrame, feature_set: str = "fusion") -> list[str]:
    """Select a fixed schema, excluding labels, identities, and task outcomes.

    Behavioral reaction times describe post-response analysis. They are not
    evidence of a model that can detect deception before a response occurs.
    """
    eeg = [c for c in table if c.startswith("EEG_") and c not in NONFEATURE_COLUMNS]
    physiology = [c for c in table if c.startswith(PHYSIOLOGY_PREFIXES)]
    behavior = [c for c in table if c.startswith("Beh_RT_") or c in BEHAVIOR_RT_COLUMNS]
    sets = {
        "eeg": eeg, "behavior": behavior, "physiology": physiology,
        "eeg_behavior": eeg + behavior, "eeg_physiology": eeg + physiology,
        "fusion": eeg + physiology + behavior,
    }
    if feature_set not in sets:
        raise ValueError(f"Unknown feature set {feature_set!r}; choose {', '.join(sets)}.")
    components = {"eeg_behavior": (eeg, behavior), "eeg_physiology": (eeg, physiology)}
    if feature_set in components and any(not group for group in components[feature_set]):
        raise ValueError(f"No eligible complete {feature_set} combination: each named modality is required.")
    if feature_set == "fusion" and sum(bool(group) for group in (eeg, physiology, behavior)) < 2:
        raise ValueError("No eligible fusion combination: at least two distinct modalities are required.")
    columns = sorted(sets[feature_set])
    if not columns:
        raise ValueError(f"No eligible {feature_set} features are available.")
    for column in columns:
        values = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"Infinite values in feature {column}; repair extraction before evaluation.")
    return columns


@dataclass
class TrialDataset:
    table: pd.DataFrame
    epochs: np.ndarray | None = None
    sfreq: float | None = None
    tmin: float = -0.2
    channel_names: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "TrialDataset":
        missing = set(REQUIRED_COLUMNS) - set(self.table)
        if missing:
            raise ValueError(f"Missing trial columns: {sorted(missing)}")
        if self.table.empty:
            raise ValueError("The trial table is empty.")
        if self.table.columns.duplicated().any():
            raise ValueError("Duplicate column names are not allowed.")
        for column in ID_COLUMNS:
            values = self.table[column]
            if values.isna().any() or values.astype(str).str.strip().eq("").any():
                raise ValueError(f"{column} contains missing or empty identifiers.")
            if not values.astype(str).eq(values.astype(str).str.strip()).all():
                raise ValueError(f"{column} contains surrounding whitespace; normalize source identifiers first.")
        canonical_ids = self.table[list(ID_COLUMNS)].astype(str)
        if canonical_ids.duplicated().any():
            raise ValueError("Each Subject_ID/Trial_ID pair must identify exactly one trial.")
        for column in ID_COLUMNS:
            if not self.table[column].map(lambda value: isinstance(value, str)).all():
                self.table = self.table.copy()
                self.table[column] = canonical_ids[column]
        labels = pd.to_numeric(self.table["Label"], errors="raise")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("Label must contain only 0 (Probable_Truth) or 1 (Probable_Lie).")
        if not pd.api.types.is_integer_dtype(self.table["Label"].dtype):
            self.table = self.table.copy()
            self.table["Label"] = labels.astype("int64")
        if not np.isfinite(self.tmin):
            raise ValueError("tmin must be finite.")
        if self.epochs is not None:
            if self.epochs.ndim != 3 or len(self.epochs) != len(self.table):
                raise ValueError("Epochs must have shape (trials, channels, samples) and match the table.")
            if not np.issubdtype(self.epochs.dtype, np.number) or any(
                not np.isfinite(self.epochs[start:start + 32]).all()
                for start in range(0, len(self.epochs), 32)
            ):
                raise ValueError("EEG epochs must contain finite numeric samples.")
            if not self.sfreq or not np.isfinite(self.sfreq) or self.sfreq <= 0:
                raise ValueError("EEG epochs require a positive sampling frequency.")
            if len(self.channel_names) != self.epochs.shape[1] or len(set(self.channel_names)) != len(self.channel_names):
                raise ValueError("EEG channel names must be unique and match the epoch channel dimension.")
            if self.epochs.shape[2] < 3:
                raise ValueError("EEG epochs are too short.")
        return self

    def subset(self, indices: Any) -> "TrialDataset":
        indices = np.asarray(indices)
        if indices.dtype == bool:
            if len(indices) != len(self.table):
                raise ValueError("A trial mask must have the same length as the table.")
            indices = np.flatnonzero(indices)
        return TrialDataset(
            table=self.table.iloc[indices].reset_index(drop=True).copy(),
            epochs=None if self.epochs is None else self.epochs[indices],
            sfreq=self.sfreq, tmin=self.tmin, channel_names=tuple(self.channel_names),
            provenance=dict(self.provenance),
        ).validate()


def save_bundle(dataset: TrialDataset, output_dir: str | Path) -> Path:
    dataset.validate()
    output = Path(output_dir)
    payload_names = ("trials.csv", "epochs.npy", "metadata.json")
    if any((output / name).exists() for name in payload_names):
        raise ValueError("The output already contains a bundle; use a new directory to preserve existing data.")
    output.mkdir(parents=True, exist_ok=True)
    # Stage serialization first, then publish metadata last as the commit marker.
    with tempfile.TemporaryDirectory(prefix=".bundle-", dir=output.parent) as temporary:
        stage = Path(temporary)
        dataset.table.to_csv(stage / "trials.csv", index=False)
        files = {"trials.csv": file_sha256(stage / "trials.csv")}
        if dataset.epochs is not None:
            np.save(stage / "epochs.npy", dataset.epochs, allow_pickle=False)
            files["epochs.npy"] = file_sha256(stage / "epochs.npy")
        metadata = {
            "schema_version": 1, "dataset_id": "ds004511", "sfreq": dataset.sfreq,
            "tmin": dataset.tmin, "channel_names": list(dataset.channel_names),
            "provenance": dataset.provenance, "files": files,
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        for name in (*files, "metadata.json"):
            os.replace(stage / name, output / name)
    return output


def load_bundle(path: str | Path, *, verify_hashes: bool = True) -> TrialDataset:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("dataset_id") != "ds004511":
        raise ValueError("Unsupported bundle schema or dataset identifier.")
    files = metadata.get("files", {})
    if "trials.csv" not in files:
        raise ValueError("Bundle metadata must include a checksum for trials.csv.")
    if not set(files).issubset({"trials.csv", "epochs.npy"}):
        raise ValueError("Unexpected file name in bundle checksum manifest.")
    if verify_hashes:
        for name, expected in files.items():
            if name not in {"trials.csv", "epochs.npy"}:
                raise ValueError("Unexpected file name in bundle checksum manifest.")
            if file_sha256(root / name) != expected:
                raise ValueError(f"Checksum mismatch for {name}; regenerate the bundle.")
    table = pd.read_csv(root / "trials.csv", dtype={"Subject_ID": str, "Trial_ID": str})
    epochs = np.load(root / "epochs.npy", allow_pickle=False, mmap_mode="r") if "epochs.npy" in files else None
    return TrialDataset(table, epochs, metadata.get("sfreq"), metadata.get("tmin", -.2),
                        tuple(metadata.get("channel_names", [])), metadata.get("provenance", {})).validate()


def load_legacy_table(path: str | Path) -> TrialDataset:
    """Import for diagnostics only; original row identities cannot be recovered."""
    path = Path(path)
    table = pd.read_csv(path, dtype={"Subject_ID": str})
    if "Subject_ID" not in table or "Label" not in table:
        raise ValueError("A legacy table must contain Subject_ID and Label.")
    if "Trial_ID" not in table:
        table.insert(1, "Trial_ID", table.groupby("Subject_ID", sort=False).cumcount().map(lambda i: f"legacy-row-{i:04d}"))
    return TrialDataset(table, provenance={
        "status": "legacy_unverified", "source": path.name, "sha256": file_sha256(path),
        "warning": "Original trial alignment and feature-extraction provenance are unknown. Diagnostic use only.",
        "has_prefitted_supervised_features": any(c.startswith(("CSP_", "xDawn_")) for c in table),
    }).validate()


def require_evaluable(dataset: TrialDataset, *, allow_unverified: bool = False,
                      allow_synthetic: bool = False) -> str:
    """Return the run status without treating a checksum as scientific validation."""
    dataset.validate()
    status = dataset.provenance.get("status", "unknown")
    if status == "synthetic" and allow_synthetic:
        return "synthetic_check"
    if status in {"prepared_raw", "verified"}:
        return "prepared_raw_requires_scientific_validation"
    if allow_unverified:
        return "diagnostic_unverified"
    raise ValueError("Unverified or synthetic inputs cannot be reported as a dataset benchmark. "
                     "Rebuild from raw data, or explicitly select diagnostic mode.")


def audit_table(dataset: TrialDataset) -> dict[str, Any]:
    dataset.validate()
    table = dataset.table
    numeric = table.select_dtypes(include="number")
    available = {}
    for name in ("eeg", "behavior", "physiology"):
        try:
            available[name] = len(feature_columns(table, name))
        except ValueError:
            available[name] = 0
    allowed = {c for name in ("eeg", "behavior", "physiology") for c in (
        feature_columns(table, name) if available[name] else [])}
    return {
        "trials": len(table), "subjects": int(table.Subject_ID.nunique()),
        "class_counts": {str(k): int(v) for k, v in table.Label.value_counts().items()},
        "missing_values": int(table.isna().sum().sum()),
        "infinite_numeric_values": int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
        "constant_columns": [c for c in table if table[c].nunique(dropna=False) <= 1],
        "feature_counts": available,
        "excluded_feature_columns": [c for c in table if c.startswith(("Beh_", "CSP_", "xDawn_")) and c not in allowed],
        "provenance": dataset.provenance,
    }
