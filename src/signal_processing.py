"""Rectification, envelopes, FFT/PSD, and simple filters."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import signal


def rectify(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def moving_average(x: np.ndarray, win: int, axis: int = -1) -> np.ndarray:
    """Causal-looking uniform moving average via convolution."""
    win = max(int(win), 1)
    kernel = np.ones(win, dtype=np.float64) / win
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return np.convolve(x, kernel, mode="same")
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), axis, x
    )


def rms_envelope(x: np.ndarray, win: int, axis: int = -1) -> np.ndarray:
    return np.sqrt(moving_average(np.square(x), win, axis=axis))


def mean_channel_envelope(
    emg: np.ndarray,
    win: int,
    method: str = "mav",
) -> np.ndarray:
    """emg shape (T, C) -> envelope shape (T,)."""
    if method == "mav":
        env = moving_average(rectify(emg), win, axis=0)
    elif method == "rms":
        env = rms_envelope(emg, win, axis=0)
    else:
        raise ValueError(method)
    return env.mean(axis=1)


def fft_magnitude(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    spec = np.fft.rfft(x)
    mag = np.abs(spec) / max(n, 1)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return freq, mag


def welch_psd(
    x: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if nperseg is None:
        nperseg = min(256, len(x))
    nperseg = max(8, int(nperseg))
    freq, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    return freq, psd


def add_synthetic_mains(
    x: np.ndarray,
    fs: float,
    freq: float = 50.0,
    amplitude: Optional[float] = None,
) -> np.ndarray:
    """Add a labelled synthetic sinusoid. This is NOT observed dataset artefact."""
    x = np.asarray(x, dtype=np.float64)
    if amplitude is None:
        amplitude = 0.5 * float(np.std(x) + 1e-8)
    t = np.arange(len(x)) / fs
    return x + amplitude * np.sin(2 * np.pi * freq * t)


def butter_filter(
    x: np.ndarray,
    fs: float,
    cutoff,
    btype: str,
    order: int = 4,
    axis: int = 0,
) -> np.ndarray:
    nyq = 0.5 * fs
    if np.isscalar(cutoff):
        wn = float(cutoff) / nyq
    else:
        wn = [float(c) / nyq for c in cutoff]
    b, a = signal.butter(order, wn, btype=btype)
    return signal.filtfilt(b, a, x, axis=axis)


def notch_filter(
    x: np.ndarray,
    fs: float,
    freq: float = 50.0,
    q: float = 30.0,
    axis: int = 0,
) -> np.ndarray:
    b, a = signal.iirnotch(freq / (0.5 * fs), q)
    return signal.filtfilt(b, a, x, axis=axis)


def peak_near(
    freq: np.ndarray,
    mag: np.ndarray,
    target_hz: float,
    band: float = 2.0,
) -> float:
    mask = (freq >= target_hz - band) & (freq <= target_hz + band)
    if not np.any(mask):
        return 0.0
    return float(np.max(mag[mask]))
