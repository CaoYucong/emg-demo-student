"""Small interpretable EMG feature set."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES_SINGLE = ["RMS", "MAV", "WL", "ZC"]


def _zero_crossings(x: np.ndarray, eps: float = 1e-6) -> float:
    signs = np.sign(x)
    signs[signs == 0] = 1
    crossings = np.abs(np.diff(signs)) > 0
    # ignore tiny noise around zero
    large_enough = (np.abs(x[:-1]) > eps) | (np.abs(x[1:]) > eps)
    return float(np.sum(crossings & large_enough))


def features_one_channel(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x**2)))
    mav = float(np.mean(np.abs(x)))
    wl = float(np.sum(np.abs(np.diff(x))))
    zc = _zero_crossings(x)
    return np.array([rms, mav, wl, zc], dtype=np.float64)


def extract_window_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """X: (N, C, T) -> (N, C*4) and feature names."""
    n, c, _t = X.shape
    feats = np.zeros((n, c * 4), dtype=np.float32)
    names = []
    for ch in range(c):
        names.extend([f"ch{ch+1}_{n_}" for n_ in FEATURE_NAMES_SINGLE])
    for i in range(n):
        vecs = [features_one_channel(X[i, ch]) for ch in range(c)]
        feats[i] = np.concatenate(vecs)
    return feats, names
