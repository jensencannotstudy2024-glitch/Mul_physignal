"""Explicit synthetic controls for testing software, never scientific results."""

import numpy as np
import pandas as pd

from .data import TrialDataset
from .features import extract_features


def make_control(*, subjects=6, trials=16, seed=2026, signal=True):
    if subjects < 2 or trials < 4 or trials % 2:
        raise ValueError("Controls need at least two subjects and an even number of trials >=4.")
    rng = np.random.default_rng(seed)
    sfreq, tmin = 128., -.2
    time = np.arange(218) / sfreq + tmin
    records, epochs = [], []
    for subject in range(subjects):
        labels = np.tile([0, 1], trials // 2)
        rng.shuffle(labels)
        for trial, label in enumerate(labels):
            x = rng.normal(0, 2e-6, (4, len(time)))
            if signal:
                x[0] += label * (8e-6 * np.exp(-((time - .4) / .08) ** 2)
                                 + 5e-6 * np.sin(2 * np.pi * 10 * time))
            epochs.append(x.astype(np.float32))
            records.append({"Subject_ID": f"synthetic-{subject:02d}", "Trial_ID": str(trial),
                            "Label": label, "ECG_Synthetic_Std": float(rng.normal() + (label if signal else 0)),
                            "Beh_RT_Synthetic": float(rng.normal() + (label if signal else 0))})
    data = TrialDataset(pd.DataFrame(records), np.stack(epochs), sfreq, tmin,
                        ("Fz", "Cz", "Pz", "Oz"), {
                            "status": "synthetic", "seed": seed, "signal": bool(signal),
                            "warning": "Synthetic software control; these are not ds004511 recordings.",
                        })
    # RT control is a synthetic scalar, not an acquisition reaction time.
    behavior = data.table["Beh_RT_Synthetic"].copy()
    data = extract_features(data)
    data.table["Beh_RT_Synthetic"] = behavior
    return data.validate()
