"""CPU neural baselines evaluated on held-out participants."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import feature_columns as audited_feature_columns


def _feature_columns(table, secondary="behavior", eeg_features=None, secondary_features=None):
    if secondary not in {"behavior", "physiology"}:
        raise ValueError("secondary must be 'behavior' or 'physiology'")
    allowed_eeg = audited_feature_columns(table, "eeg")
    allowed_secondary = audited_feature_columns(table, secondary)
    eeg = list(eeg_features) if eeg_features is not None else allowed_eeg
    other = list(secondary_features) if secondary_features is not None else allowed_secondary
    if not eeg or not other:
        raise ValueError(f"EEG features and actual {secondary} features are required")
    if len(set(eeg + other)) != len(eeg + other):
        raise ValueError("Feature lists must not contain duplicates or overlap")
    if not set(eeg).issubset(allowed_eeg):
        raise ValueError("EEG features must belong to the audited EEG schema")
    if not set(other).issubset(allowed_secondary):
        raise ValueError(f"Selected columns are not admissible {secondary} features")
    return eeg, other


class TrainingSupportMask(TransformerMixin, BaseEstimator):
    """Keep a fixed width while neutralizing coordinates unsupported by training."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.supported_features_ = np.array([
            np.unique(column[np.isfinite(column)]).size > 1 for column in X.T
        ], dtype=bool)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Feature order and width must match the fitted training schema")
        X[:, ~self.supported_features_] = 0.0
        return X


def _fit_preprocessor(train, test):
    """Estimate imputation and scaling using training rows only."""
    if np.isinf(train).any() or np.isinf(test).any():
        raise ValueError("Infinite features must be repaired before preprocessing")
    pipeline = Pipeline([
        ("support", TrainingSupportMask()),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    clean_train = np.where(np.isfinite(train), train, np.nan)
    clean_test = np.where(np.isfinite(test), test, np.nan)
    return (
        pipeline.fit_transform(clean_train).astype(np.float32),
        pipeline.transform(clean_test).astype(np.float32),
        pipeline,
    )


def _require_observed_modalities(values, columns):
    for prefix in ("EEG_", "ECG_", "RSP_", "EMG_", "EDA_", "Beh_"):
        indices = [i for i, name in enumerate(columns) if name.startswith(prefix)]
        if not indices:
            continue
        block = values[:, indices]
        if not np.isfinite(block).any():
            raise ValueError(f"The training fold has no observed {prefix.rstrip('_')} modality values")
        if not any(np.unique(column[np.isfinite(column)]).size > 1 for column in block.T):
            raise ValueError(f"The training fold has only constant {prefix.rstrip('_')} modality features")


def _require_finite_logits(logits):
    if not torch.isfinite(logits).all():
        raise FloatingPointError("The neural model produced non-finite logits; no predictions were scored")


class AttentionNet(nn.Module):
    """One EEG query attends to individually identified secondary features."""

    def __init__(self, eeg_dim, secondary_dim, embed_dim=32, num_heads=4):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.eeg_projection = nn.Linear(eeg_dim, embed_dim)
        self.secondary_projection = nn.Linear(1, embed_dim)
        self.feature_embedding = nn.Parameter(torch.randn(1, secondary_dim, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, 2)
        )

    def forward(self, eeg, secondary):
        query = self.eeg_projection(eeg).unsqueeze(1)
        tokens = self.secondary_projection(secondary.unsqueeze(-1)) + self.feature_embedding
        context, weights = self.attention(
            query, tokens, tokens, need_weights=True, average_attn_weights=False
        )
        logits = self.classifier(torch.cat([query[:, 0], context[:, 0]], dim=1))
        return logits, weights


class ContrastiveNet(nn.Module):
    def __init__(self, eeg_dim, secondary_dim, embed_dim=32):
        super().__init__()
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, 64), nn.ReLU(), nn.Linear(64, embed_dim)
        )
        self.secondary_encoder = nn.Sequential(
            nn.Linear(secondary_dim, 32), nn.ReLU(), nn.Linear(32, embed_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, 16), nn.ReLU(), nn.Dropout(0.4), nn.Linear(16, 2)
        )

    def forward(self, eeg, secondary):
        z_eeg = self.eeg_encoder(eeg)
        z_secondary = self.secondary_encoder(secondary)
        return self.classifier(torch.cat([z_eeg, z_secondary], dim=1)), z_eeg, z_secondary


class StudentNet(nn.Module):
    def __init__(self, secondary_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(secondary_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2),
        )

    def forward(self, secondary):
        return self.network(secondary)


def _contrastive_loss(z_eeg, z_secondary, temperature):
    # A final singleton batch still contributes its classification loss.
    if len(z_eeg) < 2:
        return z_eeg.sum() * 0
    a = nn.functional.normalize(z_eeg, dim=-1)
    b = nn.functional.normalize(z_secondary, dim=-1)
    similarities = a @ b.T / temperature
    targets = torch.arange(len(a), device=a.device)
    return (
        nn.functional.cross_entropy(similarities, targets)
        + nn.functional.cross_entropy(similarities.T, targets)
    ) / 2


def _loader(eeg, secondary, labels, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(eeg), torch.from_numpy(secondary), torch.as_tensor(labels)),
        batch_size=batch_size, shuffle=True, generator=generator, drop_last=False,
    )


def _train(model, loader, epochs, learning_rate, *, alignment_weight=0.0,
           temperature=0.1, teacher=None, kd_temperature=3.0, hard_label_weight=0.3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    if teacher is not None:
        teacher.eval()
    for _ in range(epochs):
        for eeg, secondary, labels in loader:
            optimizer.zero_grad()
            if isinstance(model, StudentNet):
                logits = model(secondary)
            else:
                result = model(eeg, secondary)
                logits = result[0]
            _require_finite_logits(logits)
            loss = nn.functional.cross_entropy(logits, labels)
            if isinstance(model, ContrastiveNet) and alignment_weight:
                loss = loss + alignment_weight * _contrastive_loss(result[1], result[2], temperature)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(eeg, secondary)[0]
                soft_loss = nn.functional.kl_div(
                    nn.functional.log_softmax(logits / kd_temperature, dim=1),
                    nn.functional.softmax(teacher_logits / kd_temperature, dim=1),
                    reduction="batchmean",
                ) * kd_temperature**2
                loss = hard_label_weight * loss + (1 - hard_label_weight) * soft_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Neural training produced a non-finite loss")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise FloatingPointError("Neural training produced non-finite gradients")
            optimizer.step()
    model.eval()
    return model


def evaluate_neural(dataset, *, method="attention", secondary="behavior", epochs=40,
                    batch_size=64, learning_rate=0.001, seed=2026, embed_dim=32,
                    num_heads=4, alignment_weight=0.5, temperature=0.1,
                    teacher_epochs=None, student_epochs=None, kd_temperature=3.0,
                    hard_label_weight=0.3, eeg_features: Sequence[str] | None = None,
                    secondary_features: Sequence[str] | None = None):
    """Return out-of-participant predictions; hyperparameters are fixed in advance.

    Attention weights describe the fitted model's routing, not causal importance.
    Labels and the test participant never influence preprocessing or training.
    """
    if method not in {"attention", "contrastive", "distillation"}:
        raise ValueError("Unknown neural method")
    if type(epochs) is not int or type(batch_size) is not int or epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if not all(np.isfinite(value) for value in (temperature, kd_temperature, alignment_weight)):
        raise ValueError("Temperatures and alignment_weight must be finite")
    if temperature <= 0 or kd_temperature <= 0 or alignment_weight < 0:
        raise ValueError("Temperatures must be positive and alignment_weight nonnegative")
    if not np.isfinite(hard_label_weight) or not 0 <= hard_label_weight <= 1:
        raise ValueError("hard_label_weight must be between zero and one")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    teacher_epochs = epochs if teacher_epochs is None else teacher_epochs
    student_epochs = epochs if student_epochs is None else student_epochs
    if type(teacher_epochs) is not int or type(student_epochs) is not int or teacher_epochs < 1 or student_epochs < 1:
        raise ValueError("Teacher and student epochs must be positive integers")
    dataset.validate()
    table = dataset.table.reset_index(drop=True)
    eeg_cols, secondary_cols = _feature_columns(table, secondary, eeg_features, secondary_features)
    labels = table["Label"].to_numpy(dtype=np.int64)
    groups = table["Subject_ID"].to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("LOSO evaluation requires at least two participants")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Binary labels 0 and 1 are required")
    eeg_all = table[eeg_cols].to_numpy(dtype=float)
    secondary_all = table[secondary_cols].to_numpy(dtype=float)
    predictions, attention_rows, feature_qc = [], [], []
    prior_threads = torch.get_num_threads()
    # Small tabular CPU workloads avoid excessive thread overhead.
    torch.set_num_threads(1)
    try:
        for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(eeg_all, labels, groups)):
            if len(np.unique(labels[train_idx])) != 2:
                raise ValueError("Every training fold must contain both classes")
            _require_observed_modalities(eeg_all[train_idx], eeg_cols)
            _require_observed_modalities(secondary_all[train_idx], secondary_cols)
            eeg_train, eeg_test, eeg_preprocessing = _fit_preprocessor(eeg_all[train_idx], eeg_all[test_idx])
            other_train, other_test, other_preprocessing = _fit_preprocessor(secondary_all[train_idx], secondary_all[test_idx])
            secondary_support = other_preprocessing["support"].supported_features_
            for columns, values, preprocessing in (
                (eeg_cols, eeg_all[train_idx], eeg_preprocessing),
                (secondary_cols, secondary_all[train_idx], other_preprocessing),
            ):
                for column, values_column, supported in zip(
                    columns, values.T, preprocessing["support"].supported_features_
                ):
                    finite = values_column[np.isfinite(values_column)]
                    feature_qc.append({
                        "Fold": fold, "Test_Subject": str(groups[test_idx][0]),
                        "Feature": column, "Supported_In_Training": bool(supported),
                        "Observed_Training_Rows": len(finite),
                        "Distinct_Training_Values": len(np.unique(finite)),
                    })
            fold_seed = seed + fold
            trained = []
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(fold_seed)
                def get_loader():
                    return _loader(eeg_train, other_train, labels[train_idx], batch_size, fold_seed)
                if method in {"attention", "distillation"}:
                    teacher = AttentionNet(len(eeg_cols), len(secondary_cols), embed_dim, num_heads)
                    _train(teacher, get_loader(), teacher_epochs if method == "distillation" else epochs, learning_rate)
                    trained.append(("Attention" if method == "attention" else "Attention_Teacher", teacher))
                if method == "contrastive":
                    for name, weight in (("Contrastive", alignment_weight), ("Contrastive_No_Alignment", 0.0)):
                        torch.manual_seed(fold_seed)
                        model = ContrastiveNet(len(eeg_cols), len(secondary_cols), embed_dim)
                        _train(model, get_loader(), epochs, learning_rate,
                               alignment_weight=weight, temperature=temperature)
                        trained.append((name, model))
                if method == "distillation":
                    for name, teacher_for_student in (("Secondary_Student", None), ("Distilled_Student", teacher)):
                        torch.manual_seed(fold_seed + 100_000)
                        student = StudentNet(len(secondary_cols))
                        _train(student, get_loader(), student_epochs, learning_rate,
                               teacher=teacher_for_student, kd_temperature=kd_temperature,
                               hard_label_weight=hard_label_weight)
                        trained.append((name, student))
                eeg_tensor, other_tensor = torch.from_numpy(eeg_test), torch.from_numpy(other_test)
                with torch.no_grad():
                    for name, model in trained:
                        if isinstance(model, StudentNet):
                            logits = model(other_tensor)
                        else:
                            result = model(eeg_tensor, other_tensor)
                            logits = result[0]
                        _require_finite_logits(logits)
                        probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
                        for position, probability in zip(test_idx, probabilities):
                            row = table.iloc[position]
                            predictions.append({
                                "Subject_ID": row["Subject_ID"], "Trial_ID": row["Trial_ID"],
                                "Label": int(row["Label"]), "Model": name,
                                "Feature_Set": secondary if isinstance(model, StudentNet) else f"EEG+{secondary}",
                                "Probability": float(probability), "Prediction": int(probability >= 0.5),
                            })
                        if isinstance(model, AttentionNet):
                            # Store every trial so the aggregate includes every participant.
                            weights = result[1].numpy()[:, :, 0, :]
                            for sample, position in enumerate(test_idx):
                                row = table.iloc[position]
                                for head in range(weights.shape[1]):
                                    for feature, weight, supported in zip(
                                        secondary_cols, weights[sample, head], secondary_support
                                    ):
                                        attention_rows.append({
                                            "Subject_ID": row["Subject_ID"], "Trial_ID": row["Trial_ID"],
                                            "Model": name, "Head": head + 1, "Feature": feature,
                                            "Weight": float(weight),
                                            "Supported_In_Training": bool(supported),
                                        })
    finally:
        torch.set_num_threads(prior_threads)
    result = {
        "predictions": pd.DataFrame(predictions),
        "feature_qc": pd.DataFrame(feature_qc),
        "configuration": {
            "method": method, "secondary": secondary, "seed": seed, "epochs": epochs,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "embed_dim": embed_dim, "num_heads": num_heads,
            "alignment_weight": alignment_weight, "temperature": temperature,
            "teacher_epochs": teacher_epochs, "student_epochs": student_epochs,
            "kd_temperature": kd_temperature, "hard_label_weight": hard_label_weight,
            "eeg_features": eeg_cols, "secondary_features": secondary_cols,
            "preprocessing": "training-fold support mask, median imputation and standardization; unsupported coordinates fixed to zero",
        },
    }
    if attention_rows:
        attention = pd.DataFrame(attention_rows)
        by_subject = attention.groupby(["Model", "Subject_ID", "Head", "Feature"], as_index=False).agg(
            Weight=("Weight", "mean"), Supported_In_Training=("Supported_In_Training", "first")
        )
        result["attention"] = by_subject.groupby(["Model", "Head", "Feature"], as_index=False).agg(
            Mean_Weight=("Weight", "mean"), Participants=("Subject_ID", "nunique"),
            Supported_Participants=("Supported_In_Training", "sum"),
        )
        result["attention_trials"] = attention
        result["attention_note"] = (
            "Equal-participant mean of model attention weights; not causal feature importance. "
            "Unsupported coordinates have fixed zero input and can retain token-embedding attention; "
            "inspect Supported_Participants and feature_qc before interpreting a feature."
        )
    return result
