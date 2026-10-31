"""UCI EMG dataset helpers for the teaching demo.

Label semantics are taken from the official UCI documentation for dataset 481
("EMG Data for Gestures"), NOT from variable names inside TinyMyo scripts.

Official raw-file class column:
  0 unmarked data
  1 hand at rest
  2 hand clenched in a fist
  3 wrist flexion
  4 wrist extension
  5 radial deviations
  6 ulnar deviations
  7 extended palm (not performed by all subjects)

TinyMyo ``scripts/uci.py`` drops rows with class < 1. Its docstring calls class 0
"rest", which disagrees with the UCI codebook. The *code* keeps class 1 (true
rest) and drops unmarked samples. After ``labs - 1`` the six classification
labels are rest, fist, flexion, extension, radial, ulnar.

Subject IDs are a teaching extension: the official HDF5 writer stores only
``data`` and ``label``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import CACHE_DIR, PROCESSED_DIR, RAW_DIR, VENDOR_DIR

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from uci import (  # type: ignore  # noqa: E402
    find_label_runs,
    preprocess_emg,
    read_emg_txt,
    sliding_window_majority,
    users_with_gesture,
)

UCI_FS = 200.0
N_CHANNELS = 8

UCI_RAW_CLASS_NAMES: Dict[int, str] = {
    0: "unmarked",
    1: "hand at rest",
    2: "hand clenched in a fist",
    3: "wrist flexion",
    4: "wrist extension",
    5: "radial deviations",
    6: "ulnar deviations",
    7: "extended palm",
}

# After official drop of class 0 and label shift labs-1.
CLASSIFICATION_CLASS_NAMES: Dict[int, str] = {
    0: "hand at rest",
    1: "hand clenched in a fist",
    2: "wrist flexion",
    3: "wrist extension",
    4: "radial deviations",
    5: "ulnar deviations",
}

OFFICIAL_SPLIT = {
    "train": list(range(1, 25)),
    "val": list(range(25, 31)),
    "test": list(range(31, 37)),
}

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"


@dataclass
class WindowedEMG:
    """Windowed EMG array plus labels and subject IDs."""

    X: np.ndarray  # (N, C, T)
    y: np.ndarray  # (N,) classification labels 0..5
    subject: np.ndarray  # (N,)
    fs: float
    window_samples: int
    stride: int
    class_names: Dict[int, str]
    preprocessed: bool
    split_name: str = "all"

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y)))


@dataclass
class RawRecording:
    """One continuous UCI text file, unfiltered."""

    time_ms: np.ndarray
    emg: np.ndarray  # (T, C)
    raw_label: np.ndarray  # (T,) official 0..7
    fs: float
    subject: int
    source_path: str


def uci_data_root() -> Path:
    """Return the extracted UCI master folder if present."""
    direct = RAW_DIR / "EMG_data_for_gestures-master"
    if direct.is_dir():
        return direct
    nested = RAW_DIR / "emg+data+for+gestures" / "EMG_data_for_gestures-master"
    if nested.is_dir():
        return nested
    matches = list(RAW_DIR.glob("**/EMG_data_for_gestures-master"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"UCI EMG folder not found under {RAW_DIR}. Run: python scripts/prepare.py"
    )


def list_subject_files(data_root: Path, subject: int) -> List[Path]:
    subj_dir = data_root / f"{subject:02d}"
    if not subj_dir.is_dir():
        # some copies use unpadded folder names
        alt = data_root / str(subject)
        subj_dir = alt if alt.is_dir() else subj_dir
    if not subj_dir.is_dir():
        return []
    return sorted(subj_dir.glob("*.txt"))


def load_raw_recording(txt_path: Path, subject: int) -> RawRecording:
    arr = read_emg_txt(str(txt_path))
    if arr.size == 0:
        raise ValueError(f"Empty EMG file: {txt_path}")
    return RawRecording(
        time_ms=arr[:, 0],
        emg=arr[:, 1:9],
        raw_label=arr[:, -1].astype(np.int32),
        fs=UCI_FS,
        subject=subject,
        source_path=str(txt_path),
    )


def iter_raw_recordings(
    data_root: Optional[Path] = None,
    subjects: Optional[Sequence[int]] = None,
) -> Iterable[RawRecording]:
    data_root = data_root or uci_data_root()
    if subjects is None:
        subjects = range(1, 37)
    for subj in subjects:
        for path in list_subject_files(data_root, subj):
            try:
                yield load_raw_recording(path, subj)
            except ValueError:
                continue


def label_run_lengths(raw_label: np.ndarray, fs: float = UCI_FS) -> List[dict]:
    """Return consecutive label-run durations for window-size decisions."""
    runs = []
    if raw_label.size == 0:
        return runs
    start = 0
    current = int(raw_label[0])
    for i in range(1, len(raw_label)):
        lab = int(raw_label[i])
        if lab != current:
            n = i - start
            runs.append(
                {
                    "raw_label": current,
                    "name": UCI_RAW_CLASS_NAMES.get(current, str(current)),
                    "n_samples": n,
                    "seconds": n / fs,
                }
            )
            current, start = lab, i
    n = len(raw_label) - start
    runs.append(
        {
            "raw_label": current,
            "name": UCI_RAW_CLASS_NAMES.get(current, str(current)),
            "n_samples": n,
            "seconds": n / fs,
        }
    )
    return runs


def class_rms_table(recordings: Sequence[RawRecording]) -> Dict[int, float]:
    """Mean RMS across channels and time for each raw class, from RAW EMG."""
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for rec in recordings:
        rms_t = np.sqrt(np.mean(rec.emg**2, axis=1))
        for lab in np.unique(rec.raw_label):
            lab = int(lab)
            mask = rec.raw_label == lab
            if not np.any(mask):
                continue
            sums[lab] = sums.get(lab, 0.0) + float(rms_t[mask].mean())
            counts[lab] = counts.get(lab, 0) + 1
    return {k: sums[k] / counts[k] for k in sorted(sums) if counts[k] > 0}


def choose_binary_amplitude_pair(rms_by_class: Dict[int, float]) -> dict:
    """Choose a binary pair for the threshold demo without assuming labels.

    Preference, after verifying UCI semantics:
      rest (1) vs fist (2) IF rest has clearly lower RMS than fist.
    Otherwise the two non-unmarked classes with the largest RMS ratio,
    labelled as binary amplitude-based classification.
    """
    rest, fist = 1, 2
    if rest in rms_by_class and fist in rms_by_class:
        r, f = rms_by_class[rest], rms_by_class[fist]
        if f > 0 and r <= 0.85 * f:
            return {
                "kind": "rest_vs_fist",
                "low_raw_label": rest,
                "high_raw_label": fist,
                "low_name": UCI_RAW_CLASS_NAMES[rest],
                "high_name": UCI_RAW_CLASS_NAMES[fist],
                "low_rms": r,
                "high_rms": f,
                "title": "hand at rest vs fist",
                "used_synthetic_rest": False,
            }
    candidates = {
        k: v for k, v in rms_by_class.items() if k not in (0, 7) and v > 0
    }
    if len(candidates) < 2:
        raise RuntimeError("Not enough labelled classes to build a binary amplitude pair.")
    ordered = sorted(candidates.items(), key=lambda kv: kv[1])
    low_lab, low_rms = ordered[0]
    high_lab, high_rms = ordered[-1]
    return {
        "kind": "binary_amplitude",
        "low_raw_label": int(low_lab),
        "high_raw_label": int(high_lab),
        "low_name": UCI_RAW_CLASS_NAMES[int(low_lab)],
        "high_name": UCI_RAW_CLASS_NAMES[int(high_lab)],
        "low_rms": float(low_rms),
        "high_rms": float(high_rms),
        "title": (
            f"binary amplitude-based classification: "
            f"{UCI_RAW_CLASS_NAMES[int(low_lab)]} vs {UCI_RAW_CLASS_NAMES[int(high_lab)]}"
        ),
        "used_synthetic_rest": False,
    }


def gesture7_subjects(data_root: Path) -> List[int]:
    return list(users_with_gesture(str(data_root), 7))


def official_keep_splits(data_root: Path) -> Dict[str, List[int]]:
    g7 = set(gesture7_subjects(data_root))
    return {k: [u for u in v if u not in g7] for k, v in OFFICIAL_SPLIT.items()}


def _window_one_file(
    txt_path: Path,
    subject: int,
    window_size: int,
    stride: int,
    preprocessed: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = read_emg_txt(str(txt_path))
    if arr.size == 0:
        return (
            np.empty((0, window_size, 8), np.float32),
            np.empty((0,), np.int32),
            np.empty((0,), np.int32),
        )
    if preprocessed:
        arr = preprocess_emg(arr, fs=UCI_FS, remove_class0=True)
    else:
        arr = arr[arr[:, -1] >= 1]
    xs, ys, ss = [], [], []
    for _lbl, seg_arr in find_label_runs(arr):
        segs, labs = sliding_window_majority(seg_arr, window_size, stride)
        if segs.size:
            xs.append(segs)
            ys.append(labs - 1)
            ss.append(np.full((len(labs),), subject, dtype=np.int32))
    if not xs:
        return (
            np.empty((0, window_size, 8), np.float32),
            np.empty((0,), np.int32),
            np.empty((0,), np.int32),
        )
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(ss, axis=0),
    )


def build_windows(
    data_root: Path,
    subjects: Sequence[int],
    window_size: int,
    stride: int,
    preprocessed: bool,
) -> WindowedEMG:
    xs, ys, ss = [], [], []
    for subj in subjects:
        for path in list_subject_files(data_root, subj):
            segs, labs, subj_ids = _window_one_file(
                path, subj, window_size, stride, preprocessed
            )
            if segs.size:
                xs.append(segs)
                ys.append(labs)
                ss.append(subj_ids)
    if not xs:
        X = np.empty((0, 8, window_size), np.float32)
        y = np.empty((0,), np.int32)
        subject = np.empty((0,), np.int32)
    else:
        X = np.concatenate(xs, axis=0).transpose(0, 2, 1).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.int32)
        subject = np.concatenate(ss, axis=0).astype(np.int32)
    return WindowedEMG(
        X=X,
        y=y,
        subject=subject,
        fs=UCI_FS,
        window_samples=window_size,
        stride=stride,
        class_names=CLASSIFICATION_CLASS_NAMES,
        preprocessed=preprocessed,
    )


def save_windowed(path: Path, data: WindowedEMG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=data.X,
        y=data.y,
        subject=data.subject,
        fs=np.array(data.fs),
        window_samples=np.array(data.window_samples),
        stride=np.array(data.stride),
        preprocessed=np.array(data.preprocessed),
        class_names=np.array(
            [CLASSIFICATION_CLASS_NAMES[i] for i in range(6)], dtype=object
        ),
    )


def load_windowed(path: Path) -> WindowedEMG:
    z = np.load(path, allow_pickle=True)
    names = CLASSIFICATION_CLASS_NAMES
    if "class_names" in z:
        raw_names = list(z["class_names"])
        names = {i: str(n) for i, n in enumerate(raw_names)}
    return WindowedEMG(
        X=z["X"],
        y=z["y"].astype(np.int32),
        subject=z["subject"].astype(np.int32),
        fs=float(z["fs"]),
        window_samples=int(z["window_samples"]),
        stride=int(z["stride"]),
        class_names=names,
        preprocessed=bool(z["preprocessed"]),
    )


def save_raw_example(path: Path, rec: RawRecording) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_ms=rec.time_ms,
        emg=rec.emg.astype(np.float32),
        raw_label=rec.raw_label.astype(np.int32),
        fs=np.array(rec.fs),
        subject=np.array(rec.subject),
        source_path=np.array(rec.source_path),
    )


def load_raw_example(path: Path) -> RawRecording:
    z = np.load(path, allow_pickle=True)
    return RawRecording(
        time_ms=z["time_ms"],
        emg=z["emg"],
        raw_label=z["raw_label"].astype(np.int32),
        fs=float(z["fs"]),
        subject=int(z["subject"]),
        source_path=str(z["source_path"]),
    )


def processed_paths() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "windows": PROCESSED_DIR / "classification_windows.npz",
        "raw_windows": PROCESSED_DIR / "raw_windows.npz",
        "example": PROCESSED_DIR / "example_recording.npz",
        "meta": PROCESSED_DIR / "meta.json",
        "fast_demo": CACHE_DIR / "fast_demo.npz",
        "embeddings": CACHE_DIR / "embeddings.npy",
        "embedding_index": CACHE_DIR / "embedding_index.npz",
    }


def load_meta() -> dict:
    path = processed_paths()["meta"]
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/prepare.py")
    return json.loads(path.read_text())


def load_fast_demo() -> dict:
    path = processed_paths()["fast_demo"]
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/prepare.py")
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def subset_windows(
    data: WindowedEMG,
    max_per_class_per_subject: int,
    subjects: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> WindowedEMG:
    rng = np.random.RandomState(seed)
    mask = np.zeros(len(data.y), dtype=bool)
    use_subjects = subjects if subjects is not None else np.unique(data.subject)
    for subj in use_subjects:
        for cls in np.unique(data.y):
            idx = np.where((data.subject == subj) & (data.y == cls))[0]
            if len(idx) == 0:
                continue
            take = min(max_per_class_per_subject, len(idx))
            chosen = rng.choice(idx, size=take, replace=False)
            mask[chosen] = True
    return WindowedEMG(
        X=data.X[mask],
        y=data.y[mask],
        subject=data.subject[mask],
        fs=data.fs,
        window_samples=data.window_samples,
        stride=data.stride,
        class_names=data.class_names,
        preprocessed=data.preprocessed,
    )


def longest_label_run(raw_label: np.ndarray, lab: int) -> Optional[tuple[int, int]]:
    """Return ``[start, stop)`` of the longest contiguous run of ``lab``."""
    mask = raw_label.astype(np.int32) == int(lab)
    if not np.any(mask):
        return None
    best = None
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif (not on) and start is not None:
            stop = i
            if best is None or (stop - start) > (best[1] - best[0]):
                best = (start, stop)
            start = None
    if start is not None:
        stop = len(mask)
        if best is None or (stop - start) > (best[1] - best[0]):
            best = (start, stop)
    return best


def snippet_from_recording(
    rec: RawRecording,
    lab: int,
    max_seconds: float = 3.0,
) -> Optional[dict]:
    """Middle of the longest hold for one official raw label."""
    span = longest_label_run(rec.raw_label, lab)
    if span is None:
        return None
    start, stop = span
    max_n = max(1, int(max_seconds * rec.fs))
    if stop - start > max_n:
        mid = (start + stop) // 2
        start = max(start, mid - max_n // 2)
        stop = min(stop, start + max_n)
    return {
        "label": int(lab),
        "emg": rec.emg[start:stop],
        "time_s": np.arange(stop - start) / rec.fs,
        "n_samples": int(stop - start),
        "subject": rec.subject,
        "source_path": rec.source_path,
        "from_primary": True,
    }


def collect_raw_class_snippets(
    primary: RawRecording,
    labels: Sequence[int] = range(8),
    max_seconds: float = 3.0,
    search_subjects: Sequence[int] = (11, 30),
) -> dict[int, dict]:
    """One snippet per official raw code. Search other subjects if missing.

    Class 7 (extended palm) is absent for most people. Official TinyMyo
    conversion drops subjects 11 and 30 for that reason; we still look
    there when we only want a picture of the gesture.
    """
    snippets: dict[int, dict] = {}
    missing = []
    for lab in labels:
        snip = snippet_from_recording(primary, lab, max_seconds=max_seconds)
        if snip is not None:
            snippets[int(lab)] = snip
        else:
            missing.append(int(lab))
    if not missing:
        return snippets
    try:
        root = uci_data_root()
    except FileNotFoundError:
        return snippets
    for lab in missing:
        for subj in search_subjects:
            found = None
            for rec in iter_raw_recordings(root, subjects=[int(subj)]):
                snip = snippet_from_recording(rec, lab, max_seconds=max_seconds)
                if snip is None:
                    continue
                snip["from_primary"] = False
                found = snip
                break
            if found is not None:
                snippets[lab] = found
                break
    return snippets


def extract_binary_span(
    rec: RawRecording,
    low_label: int,
    high_label: int,
    max_seconds: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return time, emg, raw_label, binary_truth for a span containing both classes."""
    wanted = np.isin(rec.raw_label, [low_label, high_label])
    if not np.any(wanted):
        raise ValueError("Recording does not contain the requested labels.")
    idx = np.where(wanted)[0]
    start, stop = int(idx[0]), int(idx[-1]) + 1
    max_n = int(max_seconds * rec.fs)
    if stop - start > max_n:
        # prefer a window that still contains both labels
        stop = min(len(rec.raw_label), start + max_n)
        span_labs = set(int(x) for x in np.unique(rec.raw_label[start:stop]))
        if not ({low_label, high_label} <= span_labs):
            high_idx = np.where(rec.raw_label == high_label)[0]
            centre = int(high_idx[len(high_idx) // 2])
            start = max(0, centre - max_n // 2)
            stop = min(len(rec.raw_label), start + max_n)
    time_s = np.arange(stop - start) / rec.fs
    emg = rec.emg[start:stop]
    raw_label = rec.raw_label[start:stop]
    truth = (raw_label == high_label).astype(np.int32)
    return time_s, emg, raw_label, truth


def random_window_split(
    y: np.ndarray,
    test_size: float = 0.25,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random split of *windows*. Same subject can appear in train and test."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    return train_idx.astype(np.int32), test_idx.astype(np.int32)


def subject_split_indices(
    subject: np.ndarray,
    train_subjects: Sequence[int],
    test_subjects: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = np.where(np.isin(subject, list(train_subjects)))[0].astype(np.int32)
    test_idx = np.where(np.isin(subject, list(test_subjects)))[0].astype(np.int32)
    return train_idx, test_idx
