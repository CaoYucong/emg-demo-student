"""Classifiers, splits, metrics. No claims beyond the numbers returned."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_linear_clf(seed: int = 0, max_iter: int = 400) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def fit_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 0,
    max_iter: int = 400,
) -> dict:
    clf = make_linear_clf(seed=seed, max_iter=max_iter)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    labels = np.unique(np.concatenate([y_train, y_test]))
    cm = confusion_matrix(y_test, pred, labels=labels)
    n_probe = int(clf.named_steps["clf"].coef_.size + clf.named_steps["clf"].intercept_.size)
    return {
        "accuracy": acc,
        "pred": pred,
        "confusion": cm,
        "labels": labels,
        "n_probe_params": n_probe,
        "clf": clf,
    }


def subsample_train(
    X: np.ndarray,
    y: np.ndarray,
    fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if fraction >= 1.0:
        return X, y
    rng = np.random.RandomState(seed)
    idx_keep = []
    for cls in np.unique(y):
        cidx = np.where(y == cls)[0]
        n = max(1, int(round(len(cidx) * fraction)))
        n = min(n, len(cidx))
        idx_keep.append(rng.choice(cidx, size=n, replace=False))
    idx = np.concatenate(idx_keep)
    rng.shuffle(idx)
    return X[idx], y[idx]


def few_label_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fractions: Sequence[float],
    seeds: Sequence[int],
    max_iter: int = 400,
) -> dict:
    out = {float(f): [] for f in fractions}
    for frac in fractions:
        for seed in seeds:
            Xt, yt = subsample_train(X_train, y_train, frac, seed)
            result = fit_eval(Xt, yt, X_test, y_test, seed=seed, max_iter=max_iter)
            out[float(frac)].append(result["accuracy"])
    summary = {}
    for frac, vals in out.items():
        arr = np.asarray(vals, dtype=np.float64)
        summary[frac] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
            "values": arr.tolist(),
        }
    return summary


def pca_2d(X: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=seed)
    z = pca.fit_transform(X)
    return z, pca.explained_variance_ratio_


def overlapping_subjects(
    subject: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> set:
    return set(subject[train_idx]).intersection(set(subject[test_idx]))


def threshold_predict(envelope: np.ndarray, threshold: float) -> np.ndarray:
    return (envelope >= threshold).astype(np.int32)


def binary_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(pred == truth))
