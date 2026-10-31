#!/usr/bin/env python3
"""One-time setup: download UCI EMG + TinyMyo pretrained weights, window data, cache embeddings.

Classroom runs after this should not need the internet.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import CACHE_DIR, CHECKPOINT_DIR, OUTPUT_DIR, PROCESSED_DIR, RAW_DIR
from src.dataset import (
    CLASSIFICATION_CLASS_NAMES,
    UCI_FS,
    UCI_RAW_CLASS_NAMES,
    UCI_ZIP_URL,
    build_windows,
    choose_binary_amplitude_pair,
    class_rms_table,
    iter_raw_recordings,
    label_run_lengths,
    load_raw_recording,
    official_keep_splits,
    processed_paths,
    save_raw_example,
    save_windowed,
    subset_windows,
    uci_data_root,
)
from src.handcrafted_features import extract_window_features
from src.plotting import plot_summary_progression
from src.pretrained_encoder import (
    HF_PRETRAINED_FILE,
    build_encoder,
    count_parameters,
    download_pretrained_checkpoint,
    encode_windows,
    inspect_checkpoint,
    load_pretrained_weights,
    recommended_batch_size,
    select_device,
)


def download_uci_zip() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "emg_data_for_gestures.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1000:
        print(f"UCI zip already present: {zip_path}")
        return zip_path
    print(f"Downloading UCI EMG from {UCI_ZIP_URL}")
    urlretrieve(UCI_ZIP_URL, zip_path)
    return zip_path


def extract_uci(zip_path: Path) -> Path:
    try:
        root = uci_data_root()
        print(f"UCI already extracted: {root}")
        return root
    except FileNotFoundError:
        pass
    print(f"Extracting {zip_path} -> {RAW_DIR}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    return uci_data_root()


def summarise_runs(data_root: Path, n_files: int = 8) -> dict:
    durations = {i: [] for i in range(8)}
    n_seen = 0
    recordings = []
    for rec in iter_raw_recordings(data_root):
        recordings.append(rec)
        for run in label_run_lengths(rec.raw_label, rec.fs):
            durations[run["raw_label"]].append(run["n_samples"])
        n_seen += 1
        if n_seen >= n_files:
            break
    stats = {}
    for lab, ns in durations.items():
        if not ns:
            continue
        arr = np.asarray(ns)
        stats[str(lab)] = {
            "name": UCI_RAW_CLASS_NAMES[lab],
            "n_runs": int(len(arr)),
            "median_samples": float(np.median(arr)),
            "median_seconds": float(np.median(arr) / UCI_FS),
            "max_samples": int(arr.max()),
        }
    rms = class_rms_table(recordings)
    return {"run_length_stats_first_files": stats, "rms_by_raw_class": {str(k): v for k, v in rms.items()}}


def pick_window_protocol(data_root: Path, keep_subjects: list[int]) -> dict:
    """Prefer official 5 s / 1000-sample windows if every class has windows.

    Gesture holds in the UCI description are ~3 s, so 1000-sample windows can
    be empty. The official TinyMyo scripts also document 1 s / 200-sample
    windows. We choose using observed counts, not a preferred story.
    """
    candidates = [
        {"window_samples": 1000, "stride": 250, "name": "5s_official_uci"},
        {"window_samples": 200, "stride": 50, "name": "1s_official_uci"},
    ]
    reports = []
    chosen = None
    for cand in candidates:
        data = build_windows(
            data_root,
            keep_subjects,
            cand["window_samples"],
            cand["stride"],
            preprocessed=True,
        )
        uniq, cnt = np.unique(data.y, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
        report = {
            **cand,
            "n_windows": int(len(data.y)),
            "label_dist": dist,
            "n_classes_present": int(len(uniq)),
            "min_class_count": int(cnt.min()) if len(cnt) else 0,
        }
        reports.append(report)
        print(
            f"Window protocol {cand['name']}: N={report['n_windows']} "
            f"classes={report['n_classes_present']} min_class={report['min_class_count']} dist={dist}"
        )
        if (
            chosen is None
            and report["n_windows"] >= 80
            and report["n_classes_present"] >= 5
            and report["min_class_count"] >= 5
        ):
            chosen = cand
            chosen_data = data
    if chosen is None:
        # fall back to the protocol with more classes, then more windows
        best = max(reports, key=lambda r: (r["n_classes_present"], r["n_windows"]))
        chosen = {k: best[k] for k in ("window_samples", "stride", "name")}
        chosen_data = build_windows(
            data_root,
            keep_subjects,
            chosen["window_samples"],
            chosen["stride"],
            preprocessed=True,
        )
        print(f"Falling back to {chosen['name']}")
    return {"chosen": chosen, "reports": reports, "data": chosen_data}


def pick_example_recording(data_root: Path, pair: dict):
    low, high = pair["low_raw_label"], pair["high_raw_label"]
    for rec in iter_raw_recordings(data_root, subjects=range(1, 37)):
        labs = set(int(x) for x in np.unique(rec.raw_label))
        if low in labs and high in labs:
            return rec
    raise RuntimeError("Could not find a recording containing both binary-demo classes.")


def build_fast_demo(windows, embeddings, features, meta: dict) -> None:
    train_subj = meta["splits"]["train"]
    test_subj = meta["splits"]["test"]
    # Keep the classroom cache small but still multi-subject.
    train_keep = train_subj[:6] if len(train_subj) >= 6 else train_subj
    test_keep = test_subj[:3] if len(test_subj) >= 3 else test_subj
    keep = list(train_keep) + list(test_keep)
    fast = subset_windows(windows, max_per_class_per_subject=12, subjects=keep, seed=0)
    # embeddings/features were computed on the full window array; map by reconstructing mask
    # Safer: re-index via a hash of (subject, y, row). We stored windows in the same order,
    # so rebuild the subset from the same function on indices.
    # Recompute features/embeddings on the subset for alignment.
    idx_mask = _subset_mask(windows, 12, keep, seed=0)
    Z = embeddings[idx_mask]
    F = features[idx_mask]
    paths = processed_paths()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["fast_demo"],
        X=fast.X,
        y=fast.y,
        subject=fast.subject,
        embeddings=Z,
        features=F,
        fs=np.array(fast.fs),
        window_samples=np.array(fast.window_samples),
        stride=np.array(fast.stride),
        train_subjects=np.array(train_keep),
        test_subjects=np.array(test_keep),
        class_names=np.array([CLASSIFICATION_CLASS_NAMES[i] for i in range(6)], dtype=object),
    )
    print(f"FAST_DEMO cache: {fast.X.shape} embeddings {Z.shape} -> {paths['fast_demo']}")


def _subset_mask(data, max_per_class_per_subject, subjects, seed=0):
    rng = np.random.RandomState(seed)
    mask = np.zeros(len(data.y), dtype=bool)
    for subj in subjects:
        for cls in np.unique(data.y):
            idx = np.where((data.subject == subj) & (data.y == cls))[0]
            if len(idx) == 0:
                continue
            take = min(max_per_class_per_subject, len(idx))
            chosen = rng.choice(idx, size=take, replace=False)
            mask[chosen] = True
    return mask


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = download_uci_zip()
    data_root = extract_uci(zip_path)
    print("UCI root:", data_root)

    splits = official_keep_splits(data_root)
    keep = splits["train"] + splits["val"] + splits["test"]
    print("Official TinyMyo subject splits after dropping gesture-7 users:")
    for k, v in splits.items():
        print(f"  {k}: {v}")

    run_summary = summarise_runs(data_root, n_files=12)
    print("Observed label-run stats (first files):")
    print(json.dumps(run_summary["run_length_stats_first_files"], indent=2))

    # RMS for binary-demo decision uses a slightly larger raw sample.
    recs = []
    for rec in iter_raw_recordings(data_root):
        recs.append(rec)
        if len(recs) >= 16:
            break
    rms = class_rms_table(recs)
    print("Observed raw-class RMS:", {UCI_RAW_CLASS_NAMES[k]: v for k, v in rms.items()})
    pair = choose_binary_amplitude_pair(rms)
    print("Binary amplitude pair:", pair)

    example = pick_example_recording(data_root, pair)
    paths = processed_paths()
    save_raw_example(paths["example"], example)
    print("Saved example recording", example.source_path, "subject", example.subject)

    protocol = pick_window_protocol(data_root, keep)
    windows = protocol["data"]
    save_windowed(paths["windows"], windows)
    raw_windows = build_windows(
        data_root,
        keep,
        protocol["chosen"]["window_samples"],
        protocol["chosen"]["stride"],
        preprocessed=False,
    )
    save_windowed(paths["raw_windows"], raw_windows)
    print("Saved windows", windows.X.shape, "raw windows", raw_windows.X.shape)

    from src.machine_info import format_machine_report

    print(format_machine_report(lang="zh"))

    ckpt = download_pretrained_checkpoint()
    ckpt_info = inspect_checkpoint(ckpt)
    print("Checkpoint:", json.dumps({k: ckpt_info[k] for k in ckpt_info if k != "shapes"}, indent=2))

    img_size = int(windows.window_samples)
    device = select_device()
    print("Compute device:", device)
    if device == "cpu":
        print("Intel iGPU is not used; embeddings are computed on CPU. This is expected.")
    model = build_encoder(img_size=img_size, in_chans=8, device=device)
    load_report = load_pretrained_weights(model, ckpt)
    model.to(device)
    params = count_parameters(model)
    print("Weight load:", {k: load_report[k] for k in ("n_loaded",)})
    print(" missing", load_report["missing_keys"][:12], "...")
    print(" unexpected", load_report["unexpected_keys"][:12], "...")
    print("Encoder params:", params)

    print(f"Extracting frozen embeddings on {device} (one-time)...")
    embeddings = encode_windows(
        model,
        windows.X,
        batch_size=recommended_batch_size(device),
        device=device,
    )
    np.save(paths["embeddings"], embeddings)
    np.savez_compressed(
        paths["embedding_index"],
        y=windows.y,
        subject=windows.subject,
        window_samples=np.array(windows.window_samples),
    )
    features, feat_names = extract_window_features(windows.X)
    np.savez_compressed(CACHE_DIR / "handcrafted_features.npz", features=features, names=np.array(feat_names))
    print("Embeddings", embeddings.shape, "features", features.shape)

    meta = {
        "uci_fs": UCI_FS,
        "n_channels": 8,
        "window_protocol": protocol["chosen"],
        "window_protocol_candidates": protocol["reports"],
        "label_run_summary": run_summary,
        "rms_by_raw_class": {str(k): float(v) for k, v in rms.items()},
        "binary_pair": pair,
        "splits": splits,
        "n_windows": int(len(windows.y)),
        "window_shape": list(windows.X.shape),
        "embedding_shape": list(embeddings.shape),
        "feature_shape": list(features.shape),
        "checkpoint": {
            "repo": "PulpBio/TinyMyo",
            "file": HF_PRETRAINED_FILE,
            "path": str(ckpt),
            "inspect": {k: ckpt_info[k] for k in ckpt_info if k != "shapes"},
            "load_report": {
                "n_loaded": load_report["n_loaded"],
                "n_missing": len(load_report["missing_keys"]),
                "n_unexpected": len(load_report["unexpected_keys"]),
                "missing_keys": load_report["missing_keys"],
                "unexpected_keys": load_report["unexpected_keys"],
            },
            "params": params,
        },
        "example_recording": {
            "path": example.source_path,
            "subject": example.subject,
        },
        "notes": [
            "Do not use UCI_EMG/UCI_finetune_5sec.safetensors as a foundation encoder.",
            "Pooling is EMGClassificationHead mean-over-patches after channel concat.",
            "UCI 200 Hz / 8 ch / 20-90 Hz z-score is a domain shift vs pretraining 2 kHz / 16 ch / 20-450 Hz min-max.",
        ],
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))
    print("Wrote", paths["meta"])

    build_fast_demo(windows, embeddings, features, meta)
    plot_summary_progression(OUTPUT_DIR / "emg_to_foundation_model.png")
    print("Wrote", OUTPUT_DIR / "emg_to_foundation_model.png")
    print("Setup complete.")


if __name__ == "__main__":
    main()
