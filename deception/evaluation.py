"""Subject-held-out evaluation with all learned steps fitted on training rows.

The outer leave-one-subject-out predictions are an evaluation, not a model
selection set. Hyperparameters and optional thresholds are selected using only
grouped splits of that outer fold's training subjects.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, LeaveOneGroupOut,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


def _positive_probability(estimator, X):
    """Return P(Label=1), including a well-defined one-class dummy model."""
    probability = estimator.predict_proba(X)
    classes = np.asarray(estimator.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive):
        return probability[:, positive[0]]
    return np.zeros(len(X), dtype=float)


def _model_spec(name, seed):
    if name == "logreg":
        return LogisticRegression(
            solver="liblinear", class_weight="balanced", max_iter=3000,
            random_state=seed,
        ), {"classifier__C": [0.1, 1.0, 10.0]}
    if name == "hist_gb":
        return HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.05, max_depth=3,
            min_samples_leaf=10, early_stopping=False, random_state=seed,
        ), {"classifier__max_leaf_nodes": [7, 15],
            "classifier__l2_regularization": [1.0, 10.0]}
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=150, class_weight="balanced_subsample",
            random_state=seed, n_jobs=1,
        ), {"classifier__max_depth": [4, 8],
            "classifier__min_samples_leaf": [3, 8]}
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent"), {}
    if name == "lda":
        # ERP 单试次分类的经典基线：高维小样本下比未正则的线性模型稳。
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), {}
    if name == "svm":
        return SVC(kernel="rbf", probability=True, class_weight="balanced",
                   random_state=seed), {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", 0.01, 0.001],
        }
    if name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except (ImportError, OSError) as exc:
            raise ImportError(
                "Optional model 'catboost' requires a working catboost installation. "
                "Install it explicitly or choose a built-in model; nothing is downloaded automatically."
            ) from exc
        return CatBoostClassifier(
            iterations=250, depth=4, learning_rate=0.05, l2_leaf_reg=3.0,
            loss_function="Logloss", auto_class_weights="Balanced",
            random_seed=seed, thread_count=1, task_type="CPU",
            verbose=False, allow_writing_files=False,
        ), {"classifier__depth": [3, 5], "classifier__l2_leaf_reg": [3.0, 10.0]}
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError) as exc:
            raise ImportError(
                "Optional model 'xgboost' requires a working xgboost installation compatible with scikit-learn. "
                "Install it explicitly or choose a built-in model; nothing is downloaded automatically."
            ) from exc
        return XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device="cpu", n_jobs=1,
            random_state=seed, verbosity=0,
        ), {"classifier__max_depth": [3, 5], "classifier__min_child_weight": [3, 10]}
    raise ValueError(
        f"Unknown model {name!r}; use logreg, lda, svm, hist_gb, random_forest, dummy, "
        "or an explicitly installed optional catboost/xgboost."
    )


def _feature_groups(columns):
    """Group columns by modality prefix so each block is scaled on its own."""
    groups = []
    for prefix in ("EEG_", "ECG_", "RSP_", "EDA_", "EMG_"):
        indices = [i for i, name in enumerate(columns) if name.startswith(prefix)]
        if indices:
            groups.append((prefix.rstrip("_").lower(), indices))
    return groups


def _vector_groups(columns):
    """文档第 2 步：EEG 按 4 子窗 × 4 频带拆成 16 组，生理按模态拆组。"""
    groups = []
    for window in ("1", "2", "3", "4"):
        for band in ("theta", "alpha", "beta", "gamma"):
            suffix = f"_SW{window}_{band}_LogPower"
            indices = [i for i, name in enumerate(columns)
                       if name.startswith("EEG_") and name.endswith(suffix)]
            if indices:
                groups.append((f"eeg_sw{window}_{band}", indices))
    for prefix in ("ECG_", "RSP_", "EDA_", "EMG_"):
        indices = [i for i, name in enumerate(columns) if name.startswith(prefix)]
        if indices:
            groups.append((prefix.rstrip("_").lower(), indices))
    return groups


def _report_vector_widths(feature_set, columns, column_groups, width):
    """打印每个特征组的 原始宽度 → 输出宽度，便于核对等宽改造。"""
    print(f"[vector widths] feature_set={feature_set}  common width d={width}", flush=True)
    total = 0
    for name, indices in column_groups:
        raw = len(indices)
        components = min(width, raw) if raw > width else raw
        pad = max(0, width - raw) if raw < width else 0
        total += width
        print(f"  {name:20s} raw={raw:4d} -> out={width:3d}  (PCA={components:3d}, pad={pad:2d})",
              flush=True)
    print(f"  concatenated width = {total}", flush=True)


class _GroupProjector(BaseEstimator, TransformerMixin):
    """把一个特征组投影到共同宽度；样本或特征不足时保持原宽度。"""

    def __init__(self, width=16):
        self.width = width

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        limit = min(int(self.width), max(1, X.shape[0] - 1))
        self.n_components_ = int(min(limit, X.shape[1]))
        self.pca_ = PCA(n_components=self.n_components_, svd_solver="full") \
            if X.shape[1] > self.n_components_ else None
        if self.pca_ is not None:
            self.pca_.fit(X)
        return self

    def transform(self, X):
        if self.pca_ is None:
            return np.asarray(X, dtype=float)
        return self.pca_.transform(X)


def _group_scaler(column_groups, project_width=None, equal_width=False):
    """Per-modality imputation and scaling; every fit stays inside the fold."""
    branches = []
    for name, indices in column_groups:
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
        if project_width:
            projector = (_EqualWidthProjector(project_width) if equal_width
                         else _GroupProjector(project_width))
            steps.append(("project", projector))
        branches.append((name, Pipeline(steps), list(indices)))
    return ColumnTransformer(branches, remainder="drop", sparse_threshold=0.0)


class _EqualWidthProjector(BaseEstimator, TransformerMixin):
    """投影到固定的 common width：宽了用 PCA，窄了零填充。

    零填充列在后续 VarianceThreshold 前方差为 0，会被剔除；即使不剔除也不
    对 PCA 子空间产生贡献。因此等宽改造不应改变任何指标。
    """

    def __init__(self, width=16):
        self.width = width

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.pca_ = None
        self.n_components_ = 0
        if X.shape[1] > int(self.width):
            limit = min(int(self.width), max(1, X.shape[0] - 1), X.shape[1])
            self.n_components_ = int(limit)
            # 全常数输入会让 PCA 的方差比为 0/0；变换本身仍然有效，只静音该提示。
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.pca_ = PCA(n_components=self.n_components_, svd_solver="full").fit(X)
            self.pad_ = int(self.width) - self.n_components_
        else:
            self.pad_ = int(self.width) - X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = self.pca_.transform(X) if self.pca_ is not None else X
        if self.pad_ > 0:
            out = np.pad(out, ((0, 0), (0, self.pad_)), mode="constant")
        return out


def _dummy_inputs(X):
    return np.zeros((len(X), 1), dtype=float)


def _pipeline(name, *, seed, transformer=None, pca_components=None,
              column_groups=None, project_width=None, equal_width=False):
    classifier, grid = _model_spec(name, seed)
    steps = []
    if name == "dummy":
        # A true majority control neither needs nor fits a supervised spatial
        # representation; it must also support a one-class training fold.
        steps.append(("representation", FunctionTransformer(_dummy_inputs, validate=False)))
    elif transformer is not None:
        steps.append(("representation", clone(transformer)))
    elif column_groups:
        # 文档第 2 步的第 1 段：各模态分开标准化后再拼接，避免 822 维 EEG
        # 与 30 维生理在同一个尺度块里竞争。
        steps.append(("modality_scaling", _group_scaler(column_groups, project_width, equal_width)))
    steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
    # The majority baseline must also work when every feature is constant.
    if name != "dummy":
        steps.extend([
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
        ])
        if pca_components is not None:
            steps.append(("pca", PCA(n_components=pca_components, svd_solver="full")))
    steps.append(("classifier", classifier))
    return Pipeline(steps), grid


def _inner_cv(y, groups, inner_splits, *, grouped=True, seed=2026):
    """Tuning splits: by subject when the outer protocol is cross-subject,
    by trial when a single subject is being evaluated in isolation."""
    if not isinstance(inner_splits, int) or inner_splits < 2:
        raise ValueError("inner_splits must be an integer of at least 2.")
    if grouped:
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError("Grouped tuning needs at least two training subjects.")
        splits = list(GroupKFold(n_splits=min(inner_splits, n_groups)).split(
            np.zeros((len(y), 1)), y, groups
        ))
    else:
        # 被试内：只有一个"组"，按试次分层切分；标签只在训练被试内部使用。
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
        n_splits = min(inner_splits, int(counts.min()))
        if n_splits < 2:
            raise ValueError(
                "Trial-level tuning needs at least two trials of each class inside the fold."
            )
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            np.zeros((len(y), 1)), y
        ))
    for train, valid in splits:
        if len(np.unique(y[train])) != 2:
            raise ValueError(
                "An inner training fold contains only one label class. "
                "Add suitable subjects, change the prespecified grouping, or run only the dummy baseline."
            )
        if grouped and set(groups[train]) & set(groups[valid]):
            raise AssertionError("Subject overlap in inner CV.")
    return splits


def _feature_blocks(columns):
    """Keep modality identity available for training-partition QC."""
    return {
        prefix.rstrip("_"): tuple(i for i, name in enumerate(columns) if name.startswith(prefix))
        for prefix in ("EEG_", "ECG_", "RSP_", "EMG_", "EDA_", "Beh_")
        if any(name.startswith(prefix) for name in columns)
    }


def _validate_training_observations(X, feature_blocks):
    for modality, columns in (feature_blocks or {}).items():
        if not np.isfinite(X[:, list(columns)]).any():
            raise ValueError(
                f"No observed values for modality {modality!r} in a training fold. "
                "Missing modalities cannot be represented by imputed zeros; "
                "repair the recordings or prespecify an available feature set."
            )
        varying = False
        for column in columns:
            observed = X[np.isfinite(X[:, column]), column]
            if len(observed) > 1 and np.ptp(observed) > 0:
                varying = True
                break
        if not varying:
            raise ValueError(
                f"No varying features remain in a training fold for modality {modality!r}. "
                "A wholly constant modality cannot support a claimed multimodal evaluation; "
                "use the majority baseline or prespecify an informative feature set."
            )


def _fit_training_only(X, y, groups, *, model, transformer, inner_splits,
                       seed, n_jobs, pca_components, model_grids,
                       feature_blocks=None, column_groups=None,
                       grouped_inner=True, project_width=None, equal_width=False):
    """Fit/tune using this training partition alone; no held-out inputs."""
    estimator, grid = _pipeline(
        model, seed=seed, transformer=transformer, pca_components=pca_components,
        column_groups=column_groups, project_width=project_width,
        equal_width=equal_width,
    )
    if model_grids is not None and model in model_grids:
        grid = model_grids[model]
    if model == "dummy":
        return estimator.fit(X, y), {}, float("nan"), None
    _validate_training_observations(X, feature_blocks)
    if len(np.unique(y)) != 2:
        raise ValueError("An outer training fold contains only one label class; use the dummy baseline only.")
    cv = _inner_cv(y, groups, inner_splits, grouped=grouped_inner, seed=seed)
    for train, _ in cv:
        _validate_training_observations(X[train], feature_blocks)
    search = GridSearchCV(
        estimator, param_grid=grid, scoring="balanced_accuracy", cv=cv,
        n_jobs=n_jobs, refit=True, error_score="raise",
    )
    try:
        search.fit(X, y)
    except ValueError as exc:
        if "No feature in X meets the variance threshold" in str(exc):
            raise ValueError(
                "No varying features remain in a training fold after training-only imputation. "
                "Use the majority baseline or repair the prespecified feature extraction."
            ) from exc
        raise
    return search.best_estimator_, search.best_params_, float(search.best_score_), cv


def _select_threshold(estimator, X, y, cv, thresholds):
    """Choose a threshold from inner OOF probabilities, never outer predictions."""
    candidates = tuple(float(t) for t in thresholds)
    if not candidates or any(not 0 < t < 1 for t in candidates):
        raise ValueError("threshold_grid must contain numbers strictly between zero and one.")
    if len(candidates) == 1:
        return candidates[0]
    oof = np.full(len(y), np.nan)
    for train, valid in cv:
        fitted = clone(estimator).fit(X[train], y[train])
        oof[valid] = _positive_probability(fitted, X[valid])
    if not np.isfinite(oof).all():
        raise AssertionError("Each inner validation row must receive exactly one probability.")
    # Prefer the conventional threshold when inner scores tie.
    return max(candidates, key=lambda t: (
        balanced_accuracy_score(y, oof >= t), -abs(t - 0.5), -t
    ))


def _validate_evaluation(dataset, models, *, grouping="loso"):
    dataset.validate()
    table = dataset.table
    y = table["Label"].to_numpy(dtype=int)
    groups = table["Subject_ID"].astype(str).to_numpy()
    names = tuple(dict.fromkeys(("dummy",) + tuple(models)))
    for name in names:
        _model_spec(name, 0)
    if grouping == "within":
        for subject, part in table.groupby("Subject_ID", sort=True):
            if part.Label.nunique() < 2:
                raise ValueError(
                    f"Within-subject evaluation needs both labels for {subject!r}."
                )
        return table, y, groups, names
    if len(np.unique(groups)) < 2:
        raise ValueError("Leave-one-subject-out evaluation needs at least two subjects.")
    if any(name != "dummy" for name in names) and len(np.unique(groups)) < 3:
        raise ValueError("Nested grouped evaluation needs at least three subjects, or models=('dummy',).")
    return table, y, groups, names


def _outer_splits(y, groups, *, grouping, seed, n_splits=5):
    """Yield (fold, held_out_name, train_rows, test_rows) for the chosen protocol."""
    if grouping == "loso":
        for fold, (train, test) in enumerate(
                LeaveOneGroupOut().split(np.zeros((len(y), 1)), y, groups)):
            yield fold, str(groups[test][0]), train, test
        return
    if grouping != "within":
        raise ValueError("grouping must be 'loso' or 'within'.")
    fold = 0
    for subject in sorted(set(groups)):
        rows = np.flatnonzero(groups == subject)
        counts = np.bincount(y[rows], minlength=2)
        k = min(n_splits, int(counts.min()))
        if k < 2:
            raise ValueError(
                f"Within-subject evaluation needs at least two trials of each class for {subject!r}."
            )
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for train, test in splitter.split(rows, y[rows]):
            yield fold, subject, rows[train], rows[test]
            fold += 1


def _evaluate_backend(dataset, X, *, feature_set, models, inner_splits, seed,
                      n_jobs, transformer=None, pca_components=None,
                      threshold_grid=(0.5,), model_grids=None,
                      feature_blocks=None, column_groups=None, grouping="loso",
                      project_width=None, equal_width=False):
    table, y, groups, names = _validate_evaluation(dataset, models, grouping=grouping)
    X = np.asarray(X, dtype=float)
    if X.shape[0] != len(table):
        raise ValueError("Features and trial table must have exactly the same rows.")
    if np.isinf(X).any():
        raise ValueError("Infinite inputs are invalid; repair their source rather than replacing them with zero.")
    predictions, selections, folds = [], [], []
    for fold, held_out, train, test in _outer_splits(
            y, groups, grouping=grouping, seed=seed):
        folds.append({"Fold": fold, "Test_Subject": held_out,
                      "Train_Subjects": tuple(sorted(set(groups[train]))),
                      "Train_Rows": len(train), "Test_Rows": len(test)})
        for model in names:
            fitted, parameters, score, cv = _fit_training_only(
                X[train], y[train], groups[train], model=model,
                transformer=transformer, inner_splits=inner_splits, seed=seed,
                n_jobs=n_jobs, pca_components=pca_components, model_grids=model_grids,
                feature_blocks=feature_blocks, column_groups=column_groups,
                grouped_inner=(grouping == "loso"), project_width=project_width,
                equal_width=equal_width,
            )
            threshold = 0.5 if model == "dummy" else _select_threshold(
                fitted, X[train], y[train], cv, threshold_grid
            )
            probability = _positive_probability(fitted, X[test])
            result = table.iloc[test][["Subject_ID", "Trial_ID", "Label"]].copy()
            result["Model"] = model
            result["Feature_Set"] = feature_set
            result["Probability"] = probability
            result["Prediction"] = (probability >= threshold).astype(int)
            result["Fold"] = fold
            predictions.append(result)
            selections.append({
                "Fold": fold, "Test_Subject": held_out, "Model": model,
                "Feature_Set": feature_set, "Grouping": grouping,
                "Parameters": json.dumps(parameters, sort_keys=True),
                "Threshold": threshold, "Inner_Balanced_Accuracy": score,
                "Train_Subjects": tuple(sorted(set(groups[train]))),
            })
    return {"predictions": pd.concat(predictions, ignore_index=True),
            "selection": pd.DataFrame(selections), "folds": pd.DataFrame(folds)}


def evaluate_classical(dataset, *, models=("logreg", "hist_gb", "random_forest"),
                       feature_sets=("eeg", "behavior", "fusion"), inner_splits=5,
                       seed=2026, n_jobs=1, pca_components=None,
                       threshold_grid=(0.5,), model_grids=None,
                       grouping="loso", per_modality_scaling=False,
                       vector_groups=False, vector_width=16,
                       equal_width=False):
    """Evaluate prespecified tabular feature sets using nested subject groups.

    Feature selection is delegated to data.feature_columns so acquisition/task
    metadata cannot silently become physiological predictors. PCA, if requested,
    is learned independently in every training fold. A fixed 0.5 decision
    threshold is the default; another prespecified grid is tuned only inside CV.

    ``grouping='within'`` evaluates each participant with stratified folds over
    that participant's own trials (the protocol where the 0.70 target is
    defined); ``'loso'`` keeps the cross-subject comparison.
    """
    from .data import feature_columns
    dataset.validate()
    if not feature_sets:
        raise ValueError("Provide at least one feature set.")
    results = []
    for feature_set in feature_sets:
        columns = tuple(sorted(feature_columns(dataset.table, feature_set)))
        if not columns:
            raise ValueError(f"No eligible columns for feature set {feature_set!r}.")
        X = dataset.table.loc[:, list(columns)].to_numpy(dtype=float)
        if vector_groups:
            column_groups = _vector_groups(columns)
            width = int(vector_width)
        elif per_modality_scaling:
            column_groups, width = _feature_groups(columns), None
        else:
            column_groups, width = None, None
        if equal_width and column_groups:
            _report_vector_widths(feature_set, columns, column_groups, width)
        result = _evaluate_backend(
            dataset, X, feature_set=feature_set, models=models,
            inner_splits=inner_splits, seed=seed, n_jobs=n_jobs,
            pca_components=pca_components, threshold_grid=threshold_grid,
            model_grids=model_grids, feature_blocks=_feature_blocks(columns),
            column_groups=column_groups, grouping=grouping, project_width=width,
            equal_width=equal_width,
        )
        result["selection"]["Columns"] = [columns] * len(result["selection"])
        results.append(result)
    return {"predictions": pd.concat([r["predictions"] for r in results], ignore_index=True),
            "selection": pd.concat([r["selection"] for r in results], ignore_index=True),
            "folds": results[0]["folds"]}


def evaluate_spatial(dataset, **kwargs):
    """Convenience import; the supervised transforms live in spatial.py."""
    from .spatial import evaluate_spatial as _spatial
    return _spatial(dataset, **kwargs)


def evaluate_fusion(dataset, *, method="late", feature_sets=("eeg", "physiology"),
                    base_model="logreg", inner_splits=3, seed=2026,
                    n_jobs=1, model_grids=None):
    """Equal-probability late fusion or grouped-OOF logistic stacking.

    Each modality's entire imputation/scaling/classifier pipeline is tuned on
    training subjects. Stacking's meta-model sees grouped out-of-fold base
    predictions, never predictions from models fitted on those same rows.
    The regularized meta-model and 0.5 decision threshold are prespecified.
    Late-fusion weights are equal and are not chosen using training/test AUC.
    """
    from .data import feature_columns
    if method not in {"late", "stacking"}:
        raise ValueError("method must be 'late' or 'stacking'.")
    if base_model == "dummy":
        raise ValueError("Use a learned base_model for fusion; the majority baseline is added separately.")
    if len(feature_sets) < 2 or len(set(feature_sets)) != len(feature_sets):
        raise ValueError("Fusion requires at least two distinct, prespecified feature sets.")
    table, y, groups, _ = _validate_evaluation(dataset, (base_model,))
    arrays, columns_used = [], []
    for name in feature_sets:
        columns = tuple(sorted(feature_columns(table, name)))
        if not columns:
            raise ValueError(f"No eligible columns for feature set {name!r}.")
        if any(set(columns) & set(previous) for previous in columns_used):
            raise ValueError("Late fusion branches must use disjoint feature columns.")
        columns_used.append(columns)
        array = table.loc[:, list(columns)].to_numpy(dtype=float)
        if np.isinf(array).any():
            raise ValueError("Infinite feature values are invalid.")
        arrays.append(array)
    predictions, selections = [], []
    combined_name = "+".join(feature_sets)
    for fold, (train, test) in enumerate(LeaveOneGroupOut().split(arrays[0], y, groups)):
        inner = _inner_cv(y[train], groups[train], inner_splits)
        if method == "stacking" and len(np.unique(groups[train])) < 3:
            raise ValueError("Nested grouped stacking needs at least four subjects in total.")
        test_probability = np.zeros((len(test), len(arrays)))
        oof = np.full((len(train), len(arrays)), np.nan)
        for branch, (name, X) in enumerate(zip(feature_sets, arrays)):
            fitted, parameters, score, _ = _fit_training_only(
                X[train], y[train], groups[train], model=base_model,
                transformer=None, inner_splits=inner_splits, seed=seed,
                n_jobs=n_jobs, pca_components=None, model_grids=model_grids,
                feature_blocks=_feature_blocks(columns_used[branch]),
            )
            test_probability[:, branch] = _positive_probability(fitted, X[test])
            if method == "stacking":
                for subtrain, valid in inner:
                    # Hyperparameter selection itself is nested inside this
                    # OOF fit: even the OOF labels cannot tune their predictor.
                    submodel, _, _, _ = _fit_training_only(
                        X[train][subtrain], y[train][subtrain], groups[train][subtrain],
                        model=base_model, transformer=None, inner_splits=inner_splits,
                        seed=seed, n_jobs=n_jobs, pca_components=None, model_grids=model_grids,
                        feature_blocks=_feature_blocks(columns_used[branch]),
                    )
                    oof[valid, branch] = _positive_probability(submodel, X[train][valid])
            selections.append({
                "Fold": fold, "Test_Subject": str(groups[test][0]),
                "Model": f"{method}:{base_model}", "Feature_Set": name,
                "Parameters": json.dumps(parameters, sort_keys=True),
                "Threshold": 0.5, "Inner_Balanced_Accuracy": score,
                "Train_Subjects": tuple(sorted(set(groups[train]))),
                "Columns": columns_used[branch],
            })
        if method == "late":
            probability = test_probability.mean(axis=1)
        else:
            if not np.isfinite(oof).all():
                raise AssertionError("Incomplete grouped out-of-fold stacking predictions.")
            meta = LogisticRegression(
                C=1.0, solver="liblinear", class_weight="balanced",
                max_iter=3000, random_state=seed,
            ).fit(oof, y[train])
            probability = _positive_probability(meta, test_probability)
        result = table.iloc[test][["Subject_ID", "Trial_ID", "Label"]].copy()
        result["Model"] = f"{method}:{base_model}"
        result["Feature_Set"] = combined_name
        result["Probability"] = probability
        result["Prediction"] = (probability >= 0.5).astype(int)
        result["Fold"] = fold
        predictions.append(result)
    # Same outer subjects/rows and combined input scope for the control model.
    dummy = _evaluate_backend(
        dataset, np.column_stack(arrays), feature_set=combined_name,
        models=("dummy",), inner_splits=inner_splits, seed=seed, n_jobs=n_jobs,
    )
    return {"predictions": pd.concat(predictions + [dummy["predictions"]], ignore_index=True),
            "selection": pd.concat([pd.DataFrame(selections), dummy["selection"]], ignore_index=True),
            "folds": dummy["folds"]}
