"""Honest short-window features with no outcome-derived predictors.

EEG features use recorded channel names, not guessed anatomical regions. The
default omits behavioral features: reaction times are only available after the
corresponding responses, and cannot support detection early in a trial.
Physiology statistics describe amplitudes; they do not claim HRV, breathing
rate, or detected skin-conductance responses from a 1.7-second window.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid

from .data import TrialDataset, NONFEATURE_COLUMNS


RT_COLUMNS = (
    "Answer_Self_RT", "Answer_Irrelevant_RT", "Prediction_RT",
    "Answer_Prediction_Agent_RT", "Answer_Prediction_Correct_RT",
)
EEG_BANDS = {"theta": (4., 8.), "alpha": (8., 13.),
             "beta": (13., 30.), "gamma": (30., 45.)}
# 文档第 1 节第 2 步：EEG 按 4 个时间子窗 × 4 个频带拆成 16 组向量。
EEG_SUBWINDOWS = {"1": (0.1, 0.4), "2": (0.4, 0.8),
                  "3": (0.8, 1.2), "4": (1.2, 1.5)}


def safe_name(value: str) -> str:
    """Create a stable column token without changing its physiological meaning."""
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    if not token:
        raise ValueError(f"An empty channel name cannot identify a feature: {value!r}")
    return token


def behavior_features(table: pd.DataFrame) -> pd.DataFrame:
    """Return only whitelisted RTs in source units, after-response use only.

    Invalid, negative, and missing RTs remain NaN. No correctness, balance,
    outcomes, answer choices, task probabilities, trial order, or labels enter
    these predictors. Availability is after completion of the entire trial.
    """
    result = pd.DataFrame(index=table.index)
    for name in RT_COLUMNS:
        if name in table:
            values = pd.to_numeric(table[name], errors="coerce").astype(float)
            result[f"Beh_RT_{name}"] = values.where(np.isfinite(values) & (values >= 0))
    return result


def _band_masks(frequencies: np.ndarray, bands: Mapping[str, tuple[float, float]],
                sfreq: float) -> dict[str, np.ndarray]:
    masks = {}
    for band_name, (low, high) in bands.items():
        if not (0 < low < high < sfreq / 2):
            raise ValueError(f"Band {band_name!r} is outside the EEG Nyquist limit.")
        mask = (frequencies >= low) & (frequencies < high)
        if mask.sum() < 1:
            raise ValueError(f"Epoch is too short to resolve band {band_name!r}.")
        token = safe_name(band_name)
        if token in masks:
            raise ValueError("EEG band names collide after sanitization.")
        masks[token] = mask
    return masks


def eeg_features(epochs: np.ndarray, sfreq: float, channel_names: Sequence[str],
                 bands: Mapping[str, tuple[float, float]] | None = None, *,
                 batch_size: int = 32, tmin: float | None = None,
                 subwindows: Mapping[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """Extract per-channel amplitude and integrated spectral features.

    Epoch amplitudes remain in their input units (MNE preparation uses volts).
    Spectral features are log10(integrated PSD), with units inherited from input.
    The whole epoch supplies Welch's segment; no sub-hertz estimates are made.
    """
    # Preserve a memory-mapped/float32 source. Cast and FFT only a small batch.
    x = np.asarray(epochs)
    if x.ndim != 3 or x.shape[1] != len(channel_names):
        raise ValueError("EEG must have shape (trials, named channels, samples).")
    if not np.isfinite(sfreq) or sfreq <= 0 or x.shape[-1] < 8:
        raise ValueError("EEG needs a positive sampling rate and at least 8 samples.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("EEG feature batch_size must be a positive integer.")
    names = [safe_name(c) for c in channel_names]
    if len(set(names)) != len(names):
        raise ValueError("EEG channel names collide after sanitization.")
    bands = EEG_BANDS if bands is None else bands
    frequencies = np.fft.rfftfreq(x.shape[-1], d=1. / sfreq)
    band_masks = {}
    for band_name, (low, high) in bands.items():
        if not (0 < low < high < sfreq / 2):
            raise ValueError(f"Band {band_name!r} is outside the EEG Nyquist limit.")
        mask = (frequencies >= low) & (frequencies < high)
        if mask.sum() < 2:
            raise ValueError(f"Epoch is too short to resolve band {band_name!r}.")
        token = safe_name(band_name)
        if token in band_masks:
            raise ValueError("EEG band names collide after sanitization.")
        band_masks[token] = mask
    # 子窗特征：位置相对于事件零点，因此需要 tmin。
    sub_specs = []
    if subwindows is not None:
        if tmin is None:
            raise ValueError("Sub-window EEG features require the epoch tmin.")
        for label, (low, high) in subwindows.items():
            first = int(round((low - tmin) * sfreq))
            last = int(round((high - tmin) * sfreq))
            if first < 0 or last > x.shape[-1] or last - first < 8:
                raise ValueError(f"Sub-window {label!r} falls outside the epoch.")
            sub_freqs = np.fft.rfftfreq(last - first, d=1. / sfreq)
            sub_specs.append((safe_name(label), first, last, sub_freqs,
                              _band_masks(sub_freqs, bands, sfreq)))
    output = {}
    for first in range(0, len(x), batch_size):
        last = min(first + batch_size, len(x))
        batch = np.asarray(x[first:last], dtype=float)
        if not np.isfinite(batch).all():
            raise ValueError("EEG contains non-finite samples; review acquisition QC first.")
        _, psd = welch(batch, fs=sfreq, nperseg=x.shape[-1], axis=-1,
                       detrend="constant", scaling="density")
        for j, name in enumerate(names):
            batch_features = {f"EEG_{name}_Std": batch[:, j].std(axis=-1),
                              f"EEG_{name}_PeakToPeak": np.ptp(batch[:, j], axis=-1)}
            for band_name, mask in band_masks.items():
                power = trapezoid(psd[:, j, mask], frequencies[mask], axis=-1)
                batch_features[f"EEG_{name}_{band_name}_LogPower"] = np.log10(
                    np.maximum(power, np.finfo(float).tiny))
            for key, values in batch_features.items():
                if key not in output:
                    output[key] = np.empty(len(x), dtype=float)
                output[key][first:last] = values
        for label, first_s, last_s, sub_freqs, masks in sub_specs:
            segment = batch[:, :, first_s:last_s]
            _, sub_psd = welch(segment, fs=sfreq, nperseg=segment.shape[-1], axis=-1,
                               detrend="constant", scaling="density")
            for j, name in enumerate(names):
                for band_name, mask in masks.items():
                    power = trapezoid(sub_psd[:, j, mask], sub_freqs[mask], axis=-1)
                    key = f"EEG_{name}_SW{label}_{band_name}_LogPower"
                    if key not in output:
                        output[key] = np.empty(len(x), dtype=float)
                    output[key][first:last] = np.log10(np.maximum(power, np.finfo(float).tiny))
    return pd.DataFrame(output)


def physiology_feature_prefix(column: str) -> str | None:
    """Identify only modalities explicitly named in physiology JSON Columns."""
    token = safe_name(column)
    upper = token.upper()
    patterns = (("ECG", r"^(ECG|EKG)(?:_|\d|$)"),
                ("RSP", r"^(RSP|RESP|RESPIRATION)(?:_|\d|$)"),
                ("EDA", r"^(EDA|GSR|SKIN_CONDUCTANCE)(?:_|\d|$)"),
                ("EMG", r"^EMG(?:_|\d|$)"))
    for modality, pattern in patterns:
        if re.match(pattern, upper):
            return f"{modality}_{token}"
    return None


def physiology_window_features(window: np.ndarray, sfreq: float,
                                columns: Sequence[str]) -> dict[str, float]:
    """Summarize recorded physiology channels without inventing missing values.

    Any missing/non-finite channel samples produce NaN features and a QC finite
    fraction. Time, synchronization, and unrecognized channels are not predictors.
    Slopes use actual seconds, rather than sample differences.
    """
    data = np.asarray(window, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError("Physiology window and JSON Columns disagree.")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Physiology SamplingFrequency must be positive.")
    result = {}
    seen = set()
    for j, column in enumerate(columns):
        prefix = physiology_feature_prefix(column)
        if prefix is None:
            continue
        if prefix in seen:
            raise ValueError("Physiology column names collide after sanitization.")
        seen.add(prefix)
        signal = data[:, j]
        finite = np.isfinite(signal)
        result[f"QC_{prefix}_FiniteFraction"] = float(finite.mean()) if len(signal) else 0.
        keys = [f"{prefix}_{stat}" for stat in ("Mean", "Std", "RMS", "Min", "Max", "SlopePerSecond")]
        if len(signal) < 2 or not finite.all():
            result.update({key: np.nan for key in keys})
            continue
        seconds = np.arange(len(signal)) / sfreq
        centered_t = seconds - seconds.mean()
        slope = np.dot(centered_t, signal - signal.mean()) / np.dot(centered_t, centered_t)
        stats = (signal.mean(), signal.std(), np.sqrt(np.mean(signal ** 2)),
                 signal.min(), signal.max(), slope)
        result.update(zip(keys, map(float, stats)))
    if not seen:
        raise ValueError("JSON Columns identify no supported ECG/RSP/EDA/EMG channels; supply accurate metadata.")
    return result


def extract_features(dataset: TrialDataset, *, include_behavior: bool = False) -> TrialDataset:
    """Return the same keyed trials with safe features and explicit availability."""
    dataset.validate()
    table = dataset.table.copy().reset_index(drop=True)
    # Re-extraction replaces the feature family instead of creating duplicate columns.
    if dataset.epochs is not None:
        table = table.drop(columns=[c for c in table if c.startswith("EEG_") and c not in NONFEATURE_COLUMNS], errors="ignore")
        if dataset.sfreq is None or not dataset.channel_names:
            raise ValueError("EEG features require recorded sampling rate and channel names.")
        eeg = eeg_features(dataset.epochs, dataset.sfreq, dataset.channel_names,
                           tmin=dataset.tmin, subwindows=EEG_SUBWINDOWS)
        if len(eeg) != len(table):
            raise ValueError("EEG trial count does not match keyed trial rows.")
        table = pd.concat([table, eeg], axis=1)
    if include_behavior:
        table = table.drop(columns=[c for c in table if c.startswith("Beh_")], errors="ignore")
        table = pd.concat([table, behavior_features(table)], axis=1)
    else:
        table = table.drop(columns=[c for c in table if c.startswith("Beh_")], errors="ignore")
    provenance = dict(dataset.provenance)
    provenance["feature_extraction"] = {
        "version": 2, "eeg": "named-channel amplitude, whole-window PSD, and 4 sub-window PSD statistics",
        "eeg_subwindows_s": {key: list(value) for key, value in EEG_SUBWINDOWS.items()},
        "physiology": "short-window amplitude statistics; no HRV/breathing-rate/SCR claims",
        "behavior_included": bool(include_behavior),
        "behavior_availability": "after all trial responses; not an early-detection predictor",
        "reaction_time_units": "unchanged source units; consult acquisition metadata",
        "eeg_feature_availability": "after full epoch; zero-phase filtering also uses future context",
    }
    return replace(dataset, table=table, provenance=provenance)
