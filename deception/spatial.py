"""Training-only CSP and standard MNE xDAWN representations.

Inputs are aligned epoch arrays with a single documented channel order shared
by every subject. Their transform methods take no labels. A pipeline must fit the
representation anew in every outer and inner training fold.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, null_space
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _check_epochs(X, expected_channels=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[1] < 2 or X.shape[2] < 5 or len(X) == 0:
        raise ValueError("Epochs must have shape (trials, channels>=2, samples>=5).")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite EEG samples must be repaired or rejected before spatial filtering.")
    if expected_channels is not None and X.shape[1] != expected_channels:
        raise ValueError("Channel count/order must match the spatial filter's training representation.")
    return X


def _prepare_epochs(X, *, sfreq, band, common_average):
    X = X.copy()
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be a positive, measured sampling frequency.")
    if band is not None:
        low, high = band
        if not 0 < low < high < sfreq / 2:
            raise ValueError("The prespecified filter band must lie strictly below Nyquist.")
        sos = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
        X = sosfiltfilt(sos, X, axis=-1)
    if common_average:
        X -= X.mean(axis=1, keepdims=True)
    return X


def _check_binary_labels(y, n_rows):
    y = np.asarray(y)
    if y.shape != (n_rows,) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Spatial filters require aligned binary training labels containing both 0 and 1.")
    return y


def _orient_filters(filters):
    filters = np.asarray(filters).copy()
    for row in filters:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return filters


class CSPFeatures(TransformerMixin, BaseEstimator):
    """Regularized, trace-normalized binary CSP followed by log-variance.

    Class covariance estimates and component ordering come from fit(X_train,
    y_train) only. scipy.linalg.eigh solves the symmetric generalized problem,
    avoiding a non-symmetric matrix inverse/eigendecomposition. Class-0 and
    class-1 extremes alternate so coordinates retain their training meaning.
    """
    def __init__(self, n_components=6, sfreq=100.0, band=(0.5, 30.0),
                 regularization=0.1, common_average=True):
        self.n_components = n_components
        self.sfreq = sfreq
        self.band = band
        self.regularization = regularization
        self.common_average = common_average

    def fit(self, X, y):
        X = _check_epochs(X)
        y = _check_binary_labels(y, len(X))
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if not 0 < self.regularization <= 1:
            raise ValueError("regularization must be greater than zero and at most one.")
        self.n_channels_in_ = X.shape[1]
        X = _prepare_epochs(X, sfreq=self.sfreq, band=self.band,
                            common_average=self.common_average)
        # CAR removes one spatial degree of freedom. Solve in that known
        # subspace, rather than returning an arbitrary zero-variance component
        # created by regularizing the reference direction.
        basis = (null_space(np.ones((1, self.n_channels_in_)))
                 if self.common_average else np.eye(self.n_channels_in_))
        self.effective_rank_ = basis.shape[1]
        X = np.einsum("cr,nct->nrt", basis, X)
        X -= X.mean(axis=-1, keepdims=True)
        covariance = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
        traces = np.trace(covariance, axis1=1, axis2=2)
        if (traces <= np.finfo(float).tiny).any():
            raise ValueError("A constant/zero EEG epoch has no valid spatial covariance.")
        covariance /= traces[:, None, None]
        class_covariances = []
        for label in (0, 1):
            cov = covariance[y == label].mean(axis=0)
            target = np.trace(cov) / self.effective_rank_ * np.eye(self.effective_rank_)
            class_covariances.append((1 - self.regularization) * cov + self.regularization * target)
        cov0, cov1 = class_covariances
        values, vectors = eigh(cov0, cov0 + cov1, check_finite=True)
        descending = list(np.argsort(values)[::-1])
        ascending = list(np.argsort(values))
        selected = []
        for rank in range(len(values)):
            for index in (descending[rank], ascending[rank]):
                if index not in selected:
                    selected.append(index)
            if len(selected) >= min(self.n_components, self.effective_rank_):
                break
        selected = selected[:min(self.n_components, self.effective_rank_)]
        self.all_filters_ = _orient_filters((basis @ vectors).T)
        self.filters_ = self.all_filters_[selected].copy()
        self.eigenvalues_ = values[selected]
        self.classes_ = np.array([0, 1])
        return self

    def transform(self, X):
        check_is_fitted(self, "filters_")
        X = _prepare_epochs(
            _check_epochs(X, self.n_channels_in_), sfreq=self.sfreq,
            band=self.band, common_average=self.common_average,
        )
        filtered = np.einsum("fc,nct->nft", self.filters_, X)
        variance = np.var(filtered, axis=-1)
        # Normalize against the complete fitted CSP basis. Using just the
        # selected components would make n_components=1 identically log(1).
        all_filtered = np.einsum("fc,nct->nft", self.all_filters_, X)
        total = np.var(all_filtered, axis=-1).sum(axis=1, keepdims=True)
        if (total <= np.finfo(float).tiny).any():
            raise ValueError("A held-out epoch has no valid projected EEG variance.")
        # Floating-point flooring is defined only for zero component variance;
        # invalid raw samples are rejected above, never silently zero-filled.
        return np.log(np.maximum(variance / total, np.finfo(float).tiny))


class XdawnERPFeatures(TransformerMixin, BaseEstimator):
    """MNE's public xDAWN implementation with finite, baseline-corrected ERP features.

    MNE is imported lazily because tabular/CSP evaluation does not require it.
    Trials are assumed to have already been cut using verified event timing;
    overlap correction cannot be reconstructed from isolated saved epochs.
    """
    def __init__(self, n_components=4, sfreq=100.0, tmin=-0.2,
                 band=(0.5, 30.0), regularization=0.1,
                 common_average=True, channel_names=None):
        self.n_components = n_components
        self.sfreq = sfreq
        self.tmin = tmin
        self.band = band
        self.regularization = regularization
        self.common_average = common_average
        self.channel_names = channel_names

    def fit(self, X, y):
        X = _check_epochs(X)
        y = _check_binary_labels(y, len(X))
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if not 0 < self.regularization <= 1:
            raise ValueError("regularization must be greater than zero and at most one.")
        self.n_channels_in_ = X.shape[1]
        prepared = _prepare_epochs(X, sfreq=self.sfreq, band=self.band,
                                   common_average=self.common_average)
        # Validate ERP coverage before fitting an expensive representation.
        self._features(prepared[:, :1, :])
        import mne
        from mne.preprocessing import Xdawn
        names = tuple(self.channel_names or tuple(f"EEG{i:03d}" for i in range(X.shape[1])))
        if len(names) != X.shape[1] or len(set(names)) != len(names):
            raise ValueError("channel_names must identify every channel exactly once.")
        self.channel_names_ = names
        info = mne.create_info(list(names), sfreq=self.sfreq, ch_types="eeg")
        events = np.column_stack([
            np.arange(len(X)) * (X.shape[2] + 1), np.zeros(len(X), dtype=int), y + 1
        ]).astype(int)
        epochs = mne.EpochsArray(
            prepared, info, events=events, event_id={"truth": 1, "lie": 2},
            tmin=self.tmin, baseline=None, verbose=False,
        )
        with mne.use_log_level("ERROR"):
            self.xdawn_ = Xdawn(
                n_components=min(self.n_components, self.n_channels_in_),
                correct_overlap=False, reg=self.regularization,
            ).fit(epochs, y=y + 1)
        return self

    def _features(self, filtered):
        times = self.tmin + np.arange(filtered.shape[-1]) / self.sfreq
        baseline = times < 0
        if not baseline.any() or not (times >= 0).any():
            raise ValueError("ERP features require a nonempty pre-event baseline and post-event samples.")
        if times[-1] < 1.2 - 1 / self.sfreq - 1e-9:
            raise ValueError("ERP features require complete epoch coverage through 1.2 seconds.")
        corrected = filtered - filtered[..., baseline].mean(axis=-1, keepdims=True)
        windows = {"early": (0.05, 0.15), "n200": (0.15, 0.3),
                   "p300": (0.3, 0.5), "lpp": (0.5, 0.8), "late": (0.8, 1.2)}
        slices = {}
        for name, (start, stop) in windows.items():
            mask = (times >= start) & (times < stop)
            if not mask.any():
                raise ValueError(f"No samples in ERP window {name}; check timing and sampling frequency.")
            slices[name] = (corrected[..., mask], times[mask])
        p300, p300_times = slices["p300"]
        latency = p300_times[np.argmax(np.abs(p300), axis=-1)]
        features = np.stack([
            p300.mean(axis=-1), p300.max(axis=-1), slices["n200"][0].min(axis=-1),
            slices["lpp"][0].mean(axis=-1), np.abs(slices["early"][0]).max(axis=-1),
            slices["late"][0].mean(axis=-1),
            np.sqrt(np.mean(corrected[..., times >= 0] ** 2, axis=-1)), latency,
        ], axis=-1)
        if not np.isfinite(features).all():
            raise ValueError("ERP extraction produced non-finite values; inspect the EEG rather than zero-filling.")
        return features.reshape(len(filtered), -1)

    def transform(self, X):
        check_is_fitted(self, "xdawn_")
        prepared = _prepare_epochs(
            _check_epochs(X, self.n_channels_in_), sfreq=self.sfreq,
            band=self.band, common_average=self.common_average,
        )
        return self._features(self.xdawn_.transform(prepared))


def evaluate_spatial(dataset, *, representation="csp",
                     models=("logreg", "hist_gb", "random_forest"),
                     inner_splits=3, seed=2026, n_jobs=1,
                     n_components=None, band=(0.5, 30.0), regularization=0.1,
                     common_average=True, pca_components=None,
                     threshold_grid=(0.5,), model_grids=None):
    """Nested grouped evaluation with a supervised spatial step inside Pipeline."""
    from .evaluation import _evaluate_backend
    dataset.validate()
    if dataset.epochs is None or dataset.sfreq is None:
        raise ValueError("Spatial evaluation requires aligned raw epochs and measured sfreq, not a precomputed CSV.")
    if not dataset.channel_names or len(dataset.channel_names) != dataset.epochs.shape[1]:
        raise ValueError("Spatial evaluation requires a documented common channel order.")
    if representation == "csp":
        transformer = CSPFeatures(
            n_components=6 if n_components is None else n_components,
            sfreq=dataset.sfreq, band=band, regularization=regularization,
            common_average=common_average,
        )
    elif representation == "xdawn":
        transformer = XdawnERPFeatures(
            n_components=4 if n_components is None else n_components,
            sfreq=dataset.sfreq, tmin=dataset.tmin, band=band,
            regularization=regularization, common_average=common_average,
            channel_names=dataset.channel_names,
        )
    else:
        raise ValueError("representation must be 'csp' or 'xdawn'.")
    return _evaluate_backend(
        dataset, dataset.epochs, feature_set=representation,
        models=models, inner_splits=inner_splits, seed=seed, n_jobs=n_jobs,
        transformer=transformer, pca_components=pca_components,
        threshold_grid=threshold_grid, model_grids=model_grids,
    )
