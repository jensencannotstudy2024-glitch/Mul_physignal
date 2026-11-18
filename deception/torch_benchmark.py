"""AGFL 五组配置的 torch 对照评测（encoding 折内拟合，折划分复用 evaluation）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from .agfl import AGFLModel
from .data import feature_columns
from .evaluation import _group_scaler, _outer_splits, _vector_groups

CONFIGS = ("none", "k0", "intra", "inter", "combined")


def _fit_encoder(groups, width, X_train):
    return _group_scaler(groups, width, equal_width=True).fit(X_train)


def _class_weights(target: np.ndarray) -> torch.Tensor:
    counts = np.bincount(target, minlength=2).astype(float)
    weights = len(target) / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_agfl(dataset, *, grouping="within", configs=CONFIGS, seeds=(2026,),
                  d=16, a=16, hops=2, kappa=5, epochs=60, lr=1e-3,
                  weight_decay=1e-2, shuffle_labels=False, fold_seed=2026,
                  n_threads=4):
    """每一折独立拟合编码器与 PCA 基；AGFL 参数只在该折训练集上学习。"""
    dataset.validate()
    columns = tuple(sorted(set(feature_columns(dataset.table, "eeg"))
                           | set(feature_columns(dataset.table, "physiology"))))
    if not columns:
        raise ValueError("AGFL needs the EEG and physiology feature blocks.")
    groups = _vector_groups(columns)
    modality_sizes = [16, len(groups) - 16]
    table = dataset.table
    X = table.loc[:, list(columns)].to_numpy(dtype=float)
    y = table.Label.to_numpy(dtype=int)
    subjects = table.Subject_ID.astype(str).to_numpy()

    try:
        torch.use_deterministic_algorithms(True)
        deterministic = True
    except Exception:
        deterministic = False
    torch.set_num_threads(int(n_threads))

    rows, folds, selections = [], [], []
    for fold, held_out, train, test in _outer_splits(y, subjects, grouping=grouping, seed=fold_seed):
        folds.append({"Fold": fold, "Test_Subject": held_out,
                      "Train_Subjects": tuple(sorted(set(subjects[train]))),
                      "Train_Rows": len(train), "Test_Rows": len(test)})
        counts = np.bincount(y[train], minlength=2)
        skipped = counts.min() < 2
        for config in configs:
            for seed in seeds:
                block = table.iloc[test][["Subject_ID", "Trial_ID", "Label"]].copy()
                block["Model"] = f"{config}::seed{seed}"
                block["Feature_Set"] = "agfl20"
                block["Fold"] = fold
                if skipped:
                    block["Probability"] = np.nan
                    block["Prediction"] = np.nan
                    rows.append(block)
                    selections.append({"Fold": fold, "Test_Subject": held_out,
                                       "Model": f"{config}::seed{seed}", "Status": "skipped",
                                       "Reason": "fewer than two training trials in a class"})
                    continue
                encoder = _fit_encoder(groups, d, X[train])
                h_train = encoder.transform(X[train]).reshape(len(train), len(groups), d)
                h_test = encoder.transform(X[test]).reshape(len(test), len(groups), d)
                pca = PCA(n_components=0.95, svd_solver="full").fit(
                    h_train.reshape(len(train), -1))
                mean = torch.tensor(pca.mean_, dtype=torch.float32)
                components = torch.tensor(pca.components_, dtype=torch.float32)
                h_train_t = torch.tensor(h_train, dtype=torch.float32)
                h_test_t = torch.tensor(h_test, dtype=torch.float32)

                target = y[train].copy()
                if shuffle_labels:
                    target = np.random.default_rng(seed + fold).permutation(target)
                target_t = torch.tensor(target, dtype=torch.long)
                loss_fn = nn.CrossEntropyLoss(weight=_class_weights(target))

                torch.manual_seed(int(seed))
                model = AGFLModel(config, modality_sizes, d, a, hops, kappa, mean, components)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                model.train()
                for _ in range(int(epochs)):
                    optimizer.zero_grad()
                    loss = loss_fn(model(h_train_t), target_t)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    probability = torch.softmax(model(h_test_t), dim=-1)[:, 1].numpy()
                block["Probability"] = probability
                block["Prediction"] = (probability >= 0.5).astype(int)
                rows.append(block)
                selections.append({"Fold": fold, "Test_Subject": held_out,
                                   "Model": f"{config}::seed{seed}", "Status": "ok",
                                   "Reason": "", "PCA_Components": int(pca.n_components_),
                                   "Deterministic": deterministic})
        print(f"fold {fold} {held_out} done", flush=True)

    return {"predictions": pd.concat(rows, ignore_index=True),
            "folds": pd.DataFrame(folds),
            "selection": pd.DataFrame(selections)}
