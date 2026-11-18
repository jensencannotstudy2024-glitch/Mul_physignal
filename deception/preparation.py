"""Prepare ds004511 without guessing which Sync marker belongs to a trial.

The dataset has many indistinguishably named Sync events per gambling trial.
Trial_Datetime plus the first Sync is not sufficient to establish trial phase.
Consequently preparation requires an externally checked alignment CSV containing
Subject_ID, Trial_ID, EEG_Onset, Physio_Onset, and Event_Phase. EEG_Onset is seconds from the
first EEG sample. Physio_Onset is seconds on the BIDS reference clock, where
physiology row zero occurs at the JSON StartTime. The two onsets must refer to
the same scientifically chosen trial phase. No timestamps are inferred or snapped.

Labels reproduce task-defined Probable_Truth/Probable_Lie; they are not claimed
to be independently verified lies. Only session 01/task GG/run 01 is selected.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from .data import TrialDataset, save_bundle, file_sha256
from .features import extract_features, physiology_window_features, physiology_feature_prefix, RT_COLUMNS


LABELS = {"Probable_Truth": 0, "Probable_Lie": 1}
ALIGNMENT_COLUMNS = ["Subject_ID", "Trial_ID", "EEG_Onset", "Physio_Onset", "Event_Phase"]
FILTER_CONTEXT_SECONDS = 5.0
LINE_NOISE_HZ = 50.0
LINE_NOISE_Q = 30.0
# 采集设备的辅助通道：Sync 触发、地/参考、电池电压、光敏。它们在 channels.tsv 里
# 被标成 EEG，但在数据里是常数或与脑电无关，必须显式剔除而不是靠折内方差阈值兜底。
NON_EEG_ACQUISITION = re.compile(r"^(?:Sync|Ref|Gnd-Ref\d*|Batt_\d+|Foto)$", re.IGNORECASE)


class AlignmentRequiredError(ValueError):
    """An explicit per-trial alignment is required before extracting samples."""


def _recording_files(dataset_root: str | Path, subjects: Sequence[str] | None = None):
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw BIDS dataset directory does not exist: {root}")
    chosen = None if subjects is None else {s if str(s).startswith("sub-") else f"sub-{s}" for s in subjects}
    files = sorted(root.glob("sub-*/ses-01/beh/*_ses-01_task-GG_run-01_beh.tsv"))
    files = [f for f in files if chosen is None or f.parents[2].name in chosen]
    if not files:
        raise FileNotFoundError("No session-01/task-GG/run-01 behavior files found. Original raw BIDS files are required.")
    found = {f.parents[2].name for f in files}
    if chosen and chosen - found:
        raise FileNotFoundError(f"No GG behavior recording for subjects: {sorted(chosen - found)}")
    if len(found) != len(files):
        raise ValueError("More than one behavior file matched a subject recording.")
    return files


def _behavior_table(path: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    path = Path(path)
    source = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    if "Trial_Order" not in source or "Trial_Condition" not in source:
        raise ValueError(f"{path.name} must contain original Trial_Order and Trial_Condition.")
    ids = pd.to_numeric(source["Trial_Order"], errors="coerce")
    if ids.isna().any() or not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError(f"{path.name}: Trial_Order must contain unique integer identifiers.")
    if ids.duplicated().any():
        raise ValueError(f"{path.name}: Trial_Order identifiers are duplicated.")
    subject = path.parents[2].name
    table = pd.DataFrame({"Subject_ID": subject, "Trial_ID": ids.astype("int64").astype(str),
                          "Label": source["Trial_Condition"].map(LABELS),
                          "Trial_Condition": source["Trial_Condition"],
                          "QC_Source_Row": np.arange(len(source))})
    # Raw RT columns are retained as metadata, and become predictors only on opt-in.
    for column in (*RT_COLUMNS, "Trial_Datetime"):
        if column in source:
            table[column] = source[column]
    invalid = table["Label"].isna()
    dropped = [{"Subject_ID": subject, "Trial_ID": row.Trial_ID,
                "Reason": "unmapped_task_condition", "Detail": str(row.Trial_Condition)}
               for row in table.loc[invalid].itertuples()]
    table = table.loc[~invalid].copy()
    table["Label"] = table["Label"].astype(int)
    return table.reset_index(drop=True), dropped


def write_alignment_template(dataset_root: str | Path, output_csv: str | Path,
                             subjects: Sequence[str] | None = None) -> pd.DataFrame:
    """Write original trial IDs and blank onset fields; never fabricate timing."""
    files = _recording_files(dataset_root, subjects)
    available = [f for f in files if _eeg_candidates(f)]
    # A behavior-only export can supply trial IDs before raw recordings arrive.
    # In an actual raw tree, omit subjects with no corresponding EEG recording.
    files = available if available else files
    tables = [_behavior_table(f)[0] for f in files]
    template = pd.concat(tables, ignore_index=True)[["Subject_ID", "Trial_ID"]]
    template["EEG_Onset"] = np.nan
    template["Physio_Onset"] = np.nan
    template["Event_Phase"] = ""
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(target, index=False)
    return template


def validate_alignment(alignment: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Validate a one-to-one join by original keys; keep behavior row order."""
    missing = set(ALIGNMENT_COLUMNS) - set(alignment)
    if missing:
        raise ValueError(f"Alignment CSV lacks columns: {sorted(missing)}")
    source = alignment[ALIGNMENT_COLUMNS].copy()
    for name in ("Subject_ID", "Trial_ID"):
        if source[name].isna().any():
            raise ValueError(f"Alignment {name} contains blank identifiers.")
        source[name] = source[name].astype(str).str.strip()
    if source["Event_Phase"].isna().any():
        raise ValueError("Alignment Event_Phase must identify the chosen event, such as trial_start or answer_onset.")
    source["Event_Phase"] = source["Event_Phase"].astype(str).str.strip()
    if (source["Event_Phase"] == "").any() or source["Event_Phase"].nunique() != 1:
        raise ValueError("Use one explicit Event_Phase across trials and subjects; do not mix trial-start and answer onsets.")
    if source.duplicated(["Subject_ID", "Trial_ID"]).any():
        raise ValueError("Alignment contains duplicate Subject_ID/Trial_ID pairs.")
    for name in ("EEG_Onset", "Physio_Onset"):
        source[name] = pd.to_numeric(source[name], errors="coerce")
        if not np.isfinite(source[name]).all():
            raise ValueError(f"Alignment {name} must contain verified finite onsets, in seconds.")
    # Trials logged before acquisition started keep their real (negative) onset: epoching
    # reports them as insufficient_continuous_filter_context instead of silently shifting them.
    expected = trials[["Subject_ID", "Trial_ID"]].copy().astype(str)
    actual_keys = set(map(tuple, source[["Subject_ID", "Trial_ID"]].to_numpy()))
    expected_keys = set(map(tuple, expected.to_numpy()))
    if actual_keys != expected_keys:
        raise ValueError(f"Alignment trial IDs do not match labeled trials: "
                         f"missing={sorted(expected_keys - actual_keys)[:5]}, "
                         f"extra={sorted(actual_keys - expected_keys)[:5]}.")
    merged = expected.merge(source, on=["Subject_ID", "Trial_ID"], how="left", validate="one_to_one", sort=False)
    for subject, group in merged.groupby("Subject_ID", sort=False):
        for name in ("EEG_Onset", "Physio_Onset"):
            if group[name].duplicated().any() or (np.diff(group[name]) <= 0).any():
                raise ValueError(f"{subject}: {name} must increase in original trial order; review alignment.")
    return merged


def _read_physio_metadata(path: str | Path) -> dict:
    metadata = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required = {"Columns", "SamplingFrequency", "StartTime"}
    if not required.issubset(metadata):
        raise ValueError(f"Physiology JSON lacks {sorted(required - set(metadata))}; timing/channels cannot be guessed.")
    columns = metadata["Columns"]
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) and c.strip() for c in columns):
        raise ValueError("Physiology JSON Columns must be a nonempty list of recorded column names.")
    if len(set(columns)) != len(columns):
        raise ValueError("Physiology JSON Columns must be unique.")
    fs, start = float(metadata["SamplingFrequency"]), float(metadata["StartTime"])
    if not np.isfinite([fs, start]).all() or fs <= 0:
        raise ValueError("Physiology JSON has invalid SamplingFrequency or StartTime.")
    if not any(physiology_feature_prefix(c) for c in columns):
        raise ValueError("Physiology metadata identifies no ECG/RSP/EDA/EMG channels.")
    return {**metadata, "SamplingFrequency": fs, "StartTime": start}


def read_physiology_features(path: str | Path, metadata: dict, onsets: Sequence[float],
                             tmin: float, tmax: float, *, chunksize: int = 200_000) -> pd.DataFrame:
    """Read only requested windows in chunks; return NaNs plus explicit QC on truncation."""
    columns = metadata["Columns"]
    sfreq, starttime = float(metadata["SamplingFrequency"]), float(metadata["StartTime"])
    onsets = np.asarray(onsets, dtype=float)
    if (not np.isfinite([sfreq, starttime, tmin, tmax]).all() or sfreq <= 0
            or tmax <= tmin or not np.isfinite(onsets).all()):
        raise ValueError("Physiology windows need finite recording clocks/onsets and tmax>tmin.")
    if not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError("Physiology chunksize must be a positive integer.")
    starts = np.rint((onsets + tmin - starttime) * sfreq).astype(np.int64)
    ends = np.rint((onsets + tmax - starttime) * sfreq).astype(np.int64)
    lengths = ends - starts + 1
    if (lengths < 2).any():
        raise ValueError("Physiology window is too short on the recorded sampling grid.")
    if not len(onsets):
        schema = list(physiology_window_features(np.empty((0, len(columns))), sfreq, columns))
        schema += ["QC_Physio_Status", "QC_Physio_WindowSamples", "QC_Physio_ExpectedSamples",
                   "QC_Physio_ActualWindowStart_s", "QC_Physio_ActualWindowEnd_s",
                   "QC_Physio_WindowStartGridError_s", "QC_Physio_WindowEndGridError_s"]
        return pd.DataFrame(columns=schema)
    pieces = [[] for _ in starts]
    total = 0
    for chunk in pd.read_csv(path, sep="\t", header=None, encoding="utf-8-sig", chunksize=chunksize):
        if chunk.shape[1] != len(columns):
            raise ValueError(f"{Path(path).name}: {chunk.shape[1]} columns disagree with JSON's {len(columns)}.")
        values = chunk.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        end = total + len(values)
        for j, (first, length) in enumerate(zip(starts, lengths)):
            left, right = max(int(first), total), min(int(first) + int(length), end)
            if right > left:
                pieces[j].append(values[left - total:right - total])
        total = end
    rows = []
    for first, length, onset, part in zip(starts, lengths, onsets, pieces):
        window = np.vstack(part) if part else np.empty((0, len(columns)))
        in_bounds = first >= 0 and first + length <= total and len(window) == length
        # Partial recordings cannot silently become complete-trial features.
        row = physiology_window_features(window if in_bounds else np.empty((0, len(columns))), sfreq, columns)
        row["QC_Physio_Status"] = "ok" if in_bounds and np.isfinite(window).all() else (
            "non_finite_samples" if in_bounds else "window_outside_recording")
        row["QC_Physio_WindowSamples"] = int(len(window))
        row["QC_Physio_ExpectedSamples"] = length
        row["QC_Physio_ActualWindowStart_s"] = starttime + first / sfreq
        row["QC_Physio_ActualWindowEnd_s"] = starttime + (first + length - 1) / sfreq
        row["QC_Physio_WindowStartGridError_s"] = row["QC_Physio_ActualWindowStart_s"] - (onset + tmin)
        row["QC_Physio_WindowEndGridError_s"] = row["QC_Physio_ActualWindowEnd_s"] - (onset + tmax)
        rows.append(row)
    return pd.DataFrame(rows)


def _eeg_metadata_channels(channels: pd.DataFrame) -> tuple[list[str], list[dict]]:
    if not {"name", "type"}.issubset(channels):
        raise ValueError("BIDS channels.tsv must supply channel names and types.")
    if channels["name"].isna().any() or channels["name"].duplicated().any():
        raise ValueError("BIDS channel names must be present and unique.")
    chosen, exclusions = [], []
    for row in channels.to_dict("records"):
        name = str(row["name"]).strip()
        kind = str(row["type"]).upper()
        numbered = re.fullmatch(r"(?:EEG|CH|E)?[ _-]*(124|125)", name, flags=re.I)
        # Identify the documented EOG acquisition channel numbers by name; a
        # channel's position in a reordered table is not an anatomical identity.
        eye_channel = bool(numbered)
        auxiliary = bool(NON_EEG_ACQUISITION.fullmatch(name))
        reason = ("non_EEG_type" if kind != "EEG" else "documented_EOG_124_125" if eye_channel else
                  "non_EEG_acquisition_channel" if auxiliary else
                  "marked_bad" if str(row.get("status", "good")).lower() == "bad" else None)
        if reason:
            exclusions.append({"channel": name, "reason": reason})
        else:
            chosen.append(name)
    if not chosen:
        raise ValueError("No usable EEG channels remain after channel-type/EOG/bad-channel exclusions.")
    return chosen, exclusions


def epoch_from_recording(get_data, n_samples: int, sfreq: float, onset: float,
                         tmin: float, tmax: float, target_sfreq: float,
                         *, context_seconds: float = FILTER_CONTEXT_SECONDS) -> tuple[np.ndarray | None, str, float]:
    """Filter buffered continuous samples, antialias-resample, then crop one epoch.

    get_data(start, stop) must return (channels, samples), using original sample
    indices. Failure returns an explicit reason; it never changes another trial's
    ID or fills samples with zeros. Endpoint times use the nearest target grid.
    """
    values = np.array([sfreq, onset, tmin, tmax, target_sfreq, context_seconds], dtype=float)
    if not np.isfinite(values).all() or sfreq <= 90 or target_sfreq <= 90 or tmax <= tmin or context_seconds < 5:
        raise ValueError("Epoching requires finite values, sfreq/target_sfreq >90 Hz, tmax>tmin, and >=5s filter context.")
    event = int(round(onset * sfreq))
    first = int(np.floor(event + (tmin - context_seconds) * sfreq))
    last = int(np.ceil(event + (tmax + context_seconds) * sfreq)) + 1
    error = event / sfreq - onset
    if first < 0 or last > n_samples:
        return None, "insufficient_continuous_filter_context", error
    signal = np.asarray(get_data(first, last), dtype=float)
    if signal.ndim != 2 or signal.shape[1] != last - first:
        raise ValueError("EEG reader did not return the requested named-channel samples.")
    if not np.isfinite(signal).all():
        return None, "non_finite_EEG_samples", error
    sos = butter(4, [1., 45.], btype="bandpass", fs=sfreq, output="sos")
    if LINE_NOISE_HZ < sfreq / 2:
        # 1–45 Hz 带通在 50 Hz 只剩约 9 dB 衰减，实测工频残留是邻域的 287 倍；
        # 先在原始采样率上做陷波，再做带通。
        notch_b, notch_a = iirnotch(LINE_NOISE_HZ, LINE_NOISE_Q, fs=sfreq)
        signal = filtfilt(notch_b, notch_a, signal, axis=-1)
    filtered = sosfiltfilt(sos, signal, axis=-1)
    ratio = (Fraction(str(target_sfreq)) / Fraction(str(sfreq))).limit_denominator(1_000_000)
    if not np.isclose(sfreq * float(ratio), target_sfreq, rtol=1e-9, atol=1e-9):
        raise ValueError("Target sampling rate cannot be represented accurately by a bounded resampling ratio.")
    filtered = resample_poly(filtered, ratio.numerator, ratio.denominator, axis=-1)
    zero = int(round((event - first) * target_sfreq / sfreq))
    # Include both original-sample and output-grid quantization in alignment QC.
    error = first / sfreq + zero / target_sfreq - onset
    left, right = zero + int(round(tmin * target_sfreq)), zero + int(round(tmax * target_sfreq)) + 1
    epoch = filtered[:, left:right]
    expected = int(round(tmax * target_sfreq)) - int(round(tmin * target_sfreq)) + 1
    if left < 0 or right > filtered.shape[-1] or epoch.shape[-1] != expected:
        raise ValueError("Resampled EEG epoch does not match the requested target time grid.")
    return epoch.astype(np.float32), "ok", error


def _read_raw(path: Path):
    try:
        import mne
    except Exception as exc:
        raise RuntimeError("MNE could not be imported; install the declared dependencies in a clean environment.") from exc
    readers = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf,
               ".vhdr": mne.io.read_raw_brainvision, ".fif": mne.io.read_raw_fif}
    return readers[path.suffix.lower()](path, preload=False, verbose="ERROR")


def _eeg_candidates(behavior_path: Path) -> list[Path]:
    stem = behavior_path.name.removesuffix("_beh.tsv")
    eeg_dir = behavior_path.parent.parent / "eeg"
    return [path for ext in (".edf", ".bdf", ".vhdr", ".fif")
            if (path := eeg_dir / f"{stem}_eeg{ext}").exists()]


def _recording_paths(behavior_path: Path) -> dict:
    stem = behavior_path.name.removesuffix("_beh.tsv")
    eeg_dir = behavior_path.parent.parent / "eeg"
    eeg_files = _eeg_candidates(behavior_path)
    if len(eeg_files) != 1:
        raise FileNotFoundError(f"{stem}: expected one raw EDF/BDF/VHDR/FIF EEG recording, found {len(eeg_files)}.")
    paths = {"eeg": eeg_files[0], "channels": eeg_dir / f"{stem}_channels.tsv",
             "eeg_json": eeg_dir / f"{stem}_eeg.json", "events": eeg_dir / f"{stem}_events.tsv",
             "physio": behavior_path.parent / f"{stem}_physio.tsv.gz",
             "physio_json": behavior_path.parent / f"{stem}_physio.json"}
    if not paths["physio"].exists():
        paths["physio"] = behavior_path.parent / f"{stem}_physio.tsv"
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required raw recording/metadata file missing ({key}): {path}")
    return paths


def _event_distances(path: Path, onsets: Sequence[float]) -> np.ndarray:
    events = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    if "onset" not in events:
        raise ValueError(f"{path.name} lacks BIDS event onsets.")
    event_onsets = pd.to_numeric(events["onset"], errors="coerce").to_numpy(float)
    event_onsets = np.sort(event_onsets[np.isfinite(event_onsets)])
    if not len(event_onsets):
        raise ValueError(f"{path.name} contains no usable event onsets.")
    return np.array([np.min(np.abs(event_onsets - t)) for t in onsets])


def prepare_dataset(dataset_root: str | Path, output_dir: str | Path,
                    alignment_csv: str | Path | None, *, tmin: float = -.2,
                    tmax: float = 1.5, target_sfreq: float = 256.,
                    subjects: Sequence[str] | None = None,
                    include_behavior: bool = False) -> TrialDataset:
    """Prepare keyed EEG/physiology data, extract safe features, and save a bundle.

    An alignment file is mandatory. Its onsets are treated as user-supplied,
    externally verified mappings, not automatically verified ground truth.
    Boundary/bad-annotation/non-finite EEG drops are recorded by original trial ID.
    Missing physiological windows remain explicit NaNs with QC, never zero rows.
    """
    if (not np.isfinite([tmin, tmax, target_sfreq]).all() or tmax <= tmin
            or target_sfreq <= 90):
        raise ValueError("Preparation requires finite tmax>tmin and target_sfreq>90 Hz.")
    actual_tmin = round(tmin * target_sfreq) / target_sfreq
    actual_tmax = round(tmax * target_sfreq) / target_sfreq
    behavior_files = _recording_files(dataset_root, subjects)
    behavior_by_subject = {f.parents[2].name: f for f in behavior_files}
    tables, dropped = [], []
    for path in behavior_files:
        table, exclusions = _behavior_table(path)
        tables.append(table)
        dropped.extend(exclusions)
    trials = pd.concat(tables, ignore_index=True)
    if trials.empty:
        raise ValueError("No task-defined Probable_Truth/Probable_Lie trials remain.")
    output = Path(output_dir)
    available = [f for f in behavior_files if _eeg_candidates(f)]
    excluded_subjects = []
    if available:
        active_subjects = {f.parents[2].name for f in available}
        absent = ~trials["Subject_ID"].isin(active_subjects)
        excluded_subjects = sorted(trials.loc[absent, "Subject_ID"].unique())
        dropped.extend({"Subject_ID": row.Subject_ID, "Trial_ID": row.Trial_ID,
                        "Reason": "missing_EEG_recording"}
                       for row in trials.loc[absent].itertuples())
        trials = trials.loc[~absent].reset_index(drop=True)
        behavior_files = available
    if alignment_csv is None:
        template_path = output / "alignment_template.csv"
        write_alignment_template(dataset_root, template_path, subjects)
        raise AlignmentRequiredError(f"Supply checked per-trial onsets using {template_path}; no Sync-to-trial mapping was assumed.")
    alignment = pd.read_csv(alignment_csv, dtype={"Subject_ID": str, "Trial_ID": str})
    # Rows for known absent recordings cannot be meaningfully aligned. All other
    # keyed rows are checked strictly, including blank/missing onsets.
    if excluded_subjects:
        alignment = alignment.loc[~alignment["Subject_ID"].isin(excluded_subjects)].copy()
    if subjects is not None:
        selected_ids = set(trials["Subject_ID"])
        alignment = alignment.loc[alignment["Subject_ID"].isin(selected_ids)].copy()
    alignment = validate_alignment(alignment, trials)
    paths_by_subject, names_by_subject, excluded_channels = {}, {}, {}
    for behavior in behavior_files:
        subject = behavior.parents[2].name
        paths = _recording_paths(behavior)
        channels = pd.read_csv(paths["channels"], sep="\t", encoding="utf-8-sig")
        names, exclusions = _eeg_metadata_channels(channels)
        paths_by_subject[subject], names_by_subject[subject] = paths, names
        excluded_channels[subject] = exclusions
    common = set.intersection(*(set(names) for names in names_by_subject.values()))
    channel_names = tuple(name for name in next(iter(names_by_subject.values())) if name in common)
    if not channel_names:
        raise ValueError("Recordings have no common usable named EEG channels.")
    for subject, names in names_by_subject.items():
        excluded_channels[subject].extend({"channel": name, "reason": "not_usable_in_all_subjects"}
                                           for name in names if name not in common)
    retained_tables, all_epochs, recording_qc = [], [], {}
    for subject in trials["Subject_ID"].drop_duplicates():
        paths = paths_by_subject[subject]
        row_ids = np.flatnonzero(trials["Subject_ID"].to_numpy() == subject)
        subject_trials = trials.iloc[row_ids].reset_index(drop=True)
        subject_alignment = alignment.iloc[row_ids].reset_index(drop=True)
        physio_metadata = _read_physio_metadata(paths["physio_json"])
        distances = _event_distances(paths["events"], subject_alignment["EEG_Onset"])
        eeg_metadata = json.loads(paths["eeg_json"].read_text(encoding="utf-8-sig"))
        raw = _read_raw(paths["eeg"])
        try:
            sfreq = float(raw.info["sfreq"])
            if "SamplingFrequency" in eeg_metadata and not np.isclose(float(eeg_metadata["SamplingFrequency"]), sfreq):
                raise ValueError(f"{subject}: EDF/header sampling rate disagrees with EEG JSON.")
            if not set(channel_names).issubset(raw.ch_names):
                raise ValueError(f"{subject}: raw EEG names disagree with channels.tsv; no positional remapping was guessed.")
            picks = [raw.ch_names.index(name) for name in channel_names]
            retained, errors = [], []
            for j, row in subject_alignment.iterrows():
                bad_reason = None
                for annotation in raw.annotations:
                    if str(annotation["description"]).lower().startswith("bad"):
                        left = float(annotation["onset"]) - float(getattr(raw, "first_time", 0.))
                        right = left + float(annotation["duration"])
                        if left < row.EEG_Onset + tmax + FILTER_CONTEXT_SECONDS and right > row.EEG_Onset + tmin - FILTER_CONTEXT_SECONDS:
                            bad_reason = "bad_annotation_in_EEG_or_filter_context"
                            break
                if bad_reason:
                    epoch, reason, error = None, bad_reason, np.nan
                else:
                    epoch, reason, error = epoch_from_recording(
                        lambda first, last: raw.get_data(picks=picks, start=first, stop=last),
                        raw.n_times, sfreq, float(row.EEG_Onset), tmin, tmax, target_sfreq)
                if epoch is None:
                    dropped.append({"Subject_ID": subject, "Trial_ID": str(row.Trial_ID), "Reason": reason})
                    continue
                retained.append(j)
                errors.append(error)
                all_epochs.append(epoch)
            kept = subject_trials.iloc[retained].reset_index(drop=True)
            kept["EEG_Onset"] = subject_alignment.iloc[retained]["EEG_Onset"].to_numpy()
            kept["Physio_Onset"] = subject_alignment.iloc[retained]["Physio_Onset"].to_numpy()
            kept["QC_EEG_OnsetQuantization_s"] = errors
            kept["QC_EEG_ActualEventOnset_s"] = kept["EEG_Onset"] + np.asarray(errors)
            kept["QC_EEG_ActualWindowStart_s"] = kept["QC_EEG_ActualEventOnset_s"] + actual_tmin
            kept["QC_EEG_ActualWindowEnd_s"] = kept["QC_EEG_ActualEventOnset_s"] + actual_tmax
            kept["QC_EEG_NearestEventDistance_s"] = distances[retained]
            # Match the actual EEG relative window, including its tiny event-grid
            # shift, then quantize physiology endpoints on their own sample grid.
            phys_features = read_physiology_features(paths["physio"], physio_metadata,
                kept["Physio_Onset"].to_numpy() + np.asarray(errors), actual_tmin, actual_tmax)
            kept = pd.concat([kept, phys_features], axis=1)
            retained_tables.append(kept)
            recording_qc[subject] = {
                "original_eeg_sfreq": sfreq, "physiology_sfreq": physio_metadata["SamplingFrequency"],
                "physiology_start_time": physio_metadata["StartTime"],
                "physiology_columns": physio_metadata["Columns"],
                "physiology_unmodeled_columns": [c for c in physio_metadata["Columns"] if physiology_feature_prefix(c) is None],
                "retained_trials": len(retained), "source_trials": len(subject_trials),
                "physiology_status": dict(Counter(kept["QC_Physio_Status"])),
                "source_files": {key: str(value) for key, value in paths.items()},
                "source_metadata_sha256": {key: file_sha256(paths[key])
                    for key in ("channels", "events", "eeg_json", "physio_json")},
                "behavior_sha256": file_sha256(behavior_by_subject[subject]),
                "behavior_file": str(behavior_by_subject[subject].resolve()),
                "raw_recordings": {key: {"path": str(paths[key].resolve()),
                    "size_bytes": paths[key].stat().st_size, "sha256": None,
                    "hash_status": "not computed; large raw recordings are identified by path/size only"}
                    for key in ("eeg", "physio")},
            }
        finally:
            raw.close()
    output.mkdir(parents=True, exist_ok=True)
    drop_table = pd.DataFrame(dropped, columns=["Subject_ID", "Trial_ID", "Reason", "Detail"])
    drop_table.to_csv(output / "dropped_trials.csv", index=False)
    if not all_epochs:
        raise ValueError(f"No EEG trials passed QC. Drop reasons: {dict(Counter(d['Reason'] for d in dropped))}; see {output / 'dropped_trials.csv'}")
    provenance = {
        "status": "prepared_raw",
        "source": "ds004511 raw BIDS session-01/task-GG/run-01",
        "label_definition": "task-defined Probable_Truth/Probable_Lie; not independently verified ground truth",
        "alignment_csv": str(Path(alignment_csv).resolve()),
        "alignment_sha256": file_sha256(alignment_csv),
        "behavior_files_sha256": {str(path.resolve()): file_sha256(path)
                                  for path in behavior_by_subject.values()},
        "alignment_verification": "explicit externally checked user mappings; not inferred from Sync markers",
        "event_phase": str(alignment["Event_Phase"].iloc[0]),
        "onset_units": "seconds; EEG relative sample0; physiology reference clock uses JSON StartTime",
        "eeg_filter": {"kind": "4th-order Butterworth zero-phase", "passband_hz": [1., 45.],
                       "context_seconds_each_side": FILTER_CONTEXT_SECONDS,
                       "resampling": "polyphase FIR antialiasing"},
        "requested_window": [tmin, tmax], "actual_eeg_window": [actual_tmin, actual_tmax],
        "window_alignment": "physiology endpoints use actual EEG relative window and event-grid shift; endpoint grid errors are recorded per trial",
        "channel_units": "volts as read by MNE; physiology source units from JSON",
        "excluded_channels": excluded_channels,
        "subjects_excluded_missing_EEG": excluded_subjects,
        "dropped_trials": dropped, "recordings": recording_qc,
        "behavior_features_default": "excluded; opt-in features are available only after trial responses",
        "sources": ["https://github.com/OpenNeuroDatasets/ds004511/blob/main/README",
                    "https://github.com/OpenNeuroDatasets/ds004511/tree/main/sub-S200116/ses-01/eeg"],
    }
    dataset = TrialDataset(table=pd.concat(retained_tables, ignore_index=True), epochs=np.stack(all_epochs),
                           sfreq=float(target_sfreq), tmin=float(actual_tmin),
                           channel_names=channel_names, provenance=provenance)
    dataset = extract_features(dataset, include_behavior=include_behavior)
    save_bundle(dataset, output)
    return dataset
