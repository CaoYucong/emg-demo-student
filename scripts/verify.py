#!/usr/bin/env python3
"""Headless run of the classroom experiments. Requires scripts/prepare.py first."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import OUTPUT_DIR
from src.dataset import (
    CLASSIFICATION_CLASS_NAMES,
    UCI_RAW_CLASS_NAMES,
    collect_raw_class_snippets,
    extract_binary_span,
    load_fast_demo,
    load_meta,
    load_raw_example,
    load_windowed,
    processed_paths,
    random_window_split,
    subject_split_indices,
)
from src.evaluation import (
    binary_accuracy,
    few_label_curve,
    fit_eval,
    overlapping_subjects,
    pca_2d,
    threshold_predict,
)
from src.handcrafted_features import extract_window_features
from src.plotting import (
    plot_confusion,
    plot_crosstalk_diagram,
    plot_few_label_curves,
    plot_grouped_bars,
    plot_model_ab_table,
    plot_notch_before_after,
    plot_pca_scatter,
    plot_raw_channels,
    plot_raw_class_gallery,
    plot_summary_progression,
    plot_three_pipelines,
    plot_threshold_controller,
    plot_threshold_sweep,
    plot_time_fft_psd,
    plot_window_length_sweep,
    plot_moving_average_as_lowpass,
)
from src.pretrained_encoder import (
    build_encoder,
    count_parameters,
    default_checkpoint_path,
    load_pretrained_weights,
    select_device,
)
from src.signal_processing import (
    add_synthetic_mains,
    fft_magnitude,
    mean_channel_envelope,
    notch_filter,
    peak_near,
    welch_psd,
)

FAST_DEMO = True
SEED = 0


def load_bundle(fast: bool = True):
    meta = load_meta()
    rec = load_raw_example(processed_paths()["example"])
    if fast:
        bundle = load_fast_demo()
        X, y, subject = bundle["X"], bundle["y"].astype(np.int32), bundle["subject"].astype(np.int32)
        Z, F = bundle["embeddings"], bundle["features"]
        train_subjects = bundle["train_subjects"].tolist()
        test_subjects = bundle["test_subjects"].tolist()
        window_samples = int(bundle["window_samples"])
    else:
        windows = load_windowed(processed_paths()["windows"])
        X, y, subject = windows.X, windows.y, windows.subject
        Z = np.load(processed_paths()["embeddings"])
        F = np.load(ROOT / "cache" / "handcrafted_features.npz")["features"]
        train_subjects = meta["splits"]["train"]
        test_subjects = meta["splits"]["test"]
        window_samples = windows.window_samples
    return {
        "meta": meta,
        "rec": rec,
        "X": X,
        "y": y,
        "subject": subject,
        "Z": Z,
        "F": F,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "window_samples": window_samples,
        "fs": float(meta["uci_fs"]),
        "class_names": CLASSIFICATION_CLASS_NAMES,
    }


def run(fast: bool = True) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    b = load_bundle(fast=fast)
    meta, rec = b["meta"], b["rec"]
    pair = meta["binary_pair"]
    fs = b["fs"]
    out = {}

    # Exp 1
    t = rec.time_ms / 1000.0
    sl = slice(0, int(min(len(t), 8 * fs)))
    plot_raw_channels(
        t[sl],
        rec.emg[sl],
        rec.raw_label[sl],
        channels=(0, 1, 2, 3),
        title=f"Raw EMG, subject {rec.subject}, first seconds",
        path=OUTPUT_DIR / "exp1_raw.png",
    )
    snippets = collect_raw_class_snippets(rec, labels=range(8), max_seconds=3.0)
    plot_raw_class_gallery(
        snippets,
        UCI_RAW_CLASS_NAMES,
        channels=(0, 1, 2, 3),
        title="Official raw codes 0–7, one hold each",
        path=OUTPUT_DIR / "exp1_by_class.png",
    )
    out["exp1"] = {
        "recording_shape": list(rec.emg.shape),
        "n_channels": rec.emg.shape[1],
        "fs": rec.fs,
        "unique_raw_labels": [int(x) for x in np.unique(rec.raw_label)],
        "gallery_codes": sorted(int(k) for k in snippets),
        "window_shape": list(b["X"].shape),
        "n_classes_windows": int(len(np.unique(b["y"]))),
    }

    # Exp 2
    time_s, emg, raw_lab, truth = extract_binary_span(
        rec, pair["low_raw_label"], pair["high_raw_label"]
    )
    env = mean_channel_envelope(emg, win=20, method="mav")
    # threshold between class-conditional envelope means
    low_m = float(env[truth == 0].mean()) if np.any(truth == 0) else float(np.percentile(env, 25))
    high_m = float(env[truth == 1].mean()) if np.any(truth == 1) else float(np.percentile(env, 75))
    thr = 0.5 * (low_m + high_m)
    pred = threshold_predict(env, thr)
    acc = binary_accuracy(pred, truth)
    plot_threshold_controller(
        time_s,
        emg[:, 0],
        env,
        thr,
        pred,
        truth,
        title=f"Threshold controller — {pair['title']} (acc={acc:.2f})",
        path=OUTPUT_DIR / "exp2_threshold.png",
    )
    thrs = np.linspace(env.min(), env.max(), 25)
    accs = [binary_accuracy(threshold_predict(env, t_), truth) for t_ in thrs]
    plot_threshold_sweep(thrs, accs, path=OUTPUT_DIR / "exp2_threshold_sweep.png")
    wins = [5, 10, 20, 40, 80]
    wacc = []
    for w in wins:
        e = mean_channel_envelope(emg, win=w, method="mav")
        t_ = 0.5 * (e[truth == 0].mean() + e[truth == 1].mean()) if (truth.min() == 0 and truth.max() == 1) else np.median(e)
        wacc.append(binary_accuracy(threshold_predict(e, t_), truth))
    plot_window_length_sweep(wins, wacc, path=OUTPUT_DIR / "exp2_window_sweep.png")
    plot_moving_average_as_lowpass(
        time_s,
        emg[:, 0],
        windows=[5, 20, 80],
        fs=fs,
        truth=truth,
        title="Moving average as a low-pass",
        path=OUTPUT_DIR / "exp2_ma_lowpass.png",
    )
    out["exp2"] = {
        "pair": pair,
        "threshold": float(thr),
        "accuracy": acc,
        "best_sweep_acc": float(np.max(accs)),
    }

    # Exp 3
    # pick a high-activation snippet
    high_idx = np.where(raw_lab == pair["high_raw_label"])[0]
    a = int(high_idx[0])
    z = min(len(emg), a + int(2 * fs))
    x = emg[a:z, 0]
    tt = np.arange(len(x)) / fs
    freq, mag = fft_magnitude(x, fs)
    pf, psd = welch_psd(x, fs)
    real_50 = peak_near(freq, mag, 50.0)
    plot_time_fft_psd(
        tt, x, freq, mag, pf, psd,
        title="OBSERVED window (no claim of mains interference)",
        path=OUTPUT_DIR / "exp3_observed.png",
    )
    x_syn = add_synthetic_mains(x, fs, freq=50.0)
    f2, m2 = fft_magnitude(x_syn, fs)
    p2f, p2 = welch_psd(x_syn, fs)
    plot_time_fft_psd(
        tt, x_syn, f2, m2, p2f, p2,
        title="SYNTHETIC teaching example: observed EMG + A sin(2π·50·t)",
        mark_hz=50.0,
        mark_label="SYNTHETIC 50 Hz",
        path=OUTPUT_DIR / "exp3_synthetic50.png",
    )
    out["exp3"] = {"observed_peak_near_50Hz": real_50}

    # Exp 4
    x_notch = notch_filter(x_syn, fs, freq=50.0)
    pf_b, psd_b = welch_psd(x_syn, fs)
    pf_a, psd_a = welch_psd(x_notch, fs)
    plot_notch_before_after(
        tt, x_syn, x_notch, pf_b, psd_b, pf_a, psd_a,
        title="SYNTHETIC 50 Hz — notch can remove a narrow-band line",
        path=OUTPUT_DIR / "exp4_notch.png",
    )
    plot_crosstalk_diagram(OUTPUT_DIR / "exp4_crosstalk.png")

    # Shared ML splits
    y, subject = b["y"], b["subject"]
    F, Z = b["F"], b["Z"]
    names = [b["class_names"][i] for i in sorted(np.unique(y))]
    max_iter = 250 if fast else 800
    rand_tr, rand_te = random_window_split(y, test_size=0.25, seed=SEED)
    leak = overlapping_subjects(subject, rand_tr, rand_te)
    rand_f = fit_eval(F[rand_tr], y[rand_tr], F[rand_te], y[rand_te], seed=SEED, max_iter=max_iter)
    plot_confusion(rand_f["confusion"], names, f"Handcrafted, random split acc={rand_f['accuracy']:.3f}", OUTPUT_DIR / "exp5_cm_random.png")

    sub_tr, sub_te = subject_split_indices(subject, b["train_subjects"], b["test_subjects"])
    if len(sub_tr) == 0 or len(sub_te) == 0:
        out["exp9_limitation"] = "Held-out subject split is empty in this subset."
        sub_f = sub_z = None
    else:
        sub_f = fit_eval(F[sub_tr], y[sub_tr], F[sub_te], y[sub_te], seed=SEED, max_iter=max_iter)
        sub_z = fit_eval(Z[sub_tr], y[sub_tr], Z[sub_te], y[sub_te], seed=SEED, max_iter=max_iter)
        plot_confusion(sub_f["confusion"], names, f"Handcrafted, subject split acc={sub_f['accuracy']:.3f}", OUTPUT_DIR / "exp5_cm_subject.png")
        plot_confusion(sub_z["confusion"], names, f"TinyMyo probe, subject split acc={sub_z['accuracy']:.3f}", OUTPUT_DIR / "exp6_cm_subject.png")

    pca_f, ev_f = pca_2d(F, seed=SEED)
    pca_z, ev_z = pca_2d(Z, seed=SEED)
    plot_pca_scatter(pca_f, y, b["class_names"], subject, "PCA of handcrafted features", path=OUTPUT_DIR / "exp7_pca_handcrafted.png")
    plot_pca_scatter(pca_z, y, b["class_names"], subject, "PCA of frozen TinyMyo embeddings", path=OUTPUT_DIR / "exp7_pca_tinymyo.png")

    # Exp 6 model card
    ckpt = default_checkpoint_path()
    device = select_device()
    model = build_encoder(img_size=b["window_samples"], in_chans=8, device=device)
    load_report = load_pretrained_weights(model, ckpt)
    model.to(device)
    params = count_parameters(model)
    print("encoder device:", device)
    probe_on_rand = fit_eval(Z[rand_tr], y[rand_tr], Z[rand_te], y[rand_te], seed=SEED, max_iter=max_iter)

    # Exp 8 few-label on within-subject random split of TRAIN subjects only
    train_mask = np.isin(subject, b["train_subjects"])
    test_mask = np.isin(subject, b["test_subjects"])
    if test_mask.sum() == 0:
        # FAST subset may still have train-only; use random split among train subjects
        y_tr_all, idx_local = y[train_mask], np.where(train_mask)[0]
        tr_i, te_i = random_window_split(y_tr_all, test_size=0.3, seed=SEED)
        tr_idx, te_idx = idx_local[tr_i], idx_local[te_i]
        few_note = "test windows from the same training subjects (no held-out person)"
    else:
        tr_idx, te_idx = np.where(train_mask)[0], np.where(test_mask)[0]
        few_note = "train subjects vs held-out subjects"
    fractions = [1.0, 0.25, 0.10]
    if tr_idx.size >= 80:
        fractions.append(0.05)
    seeds = [0, 1] if fast else [0, 1, 2]
    curve_f = few_label_curve(F[tr_idx], y[tr_idx], F[te_idx], y[te_idx], fractions, seeds, max_iter=max_iter)
    curve_z = few_label_curve(Z[tr_idx], y[tr_idx], Z[te_idx], y[te_idx], fractions, seeds, max_iter=max_iter)
    plot_few_label_curves(
        {"handcrafted + linear": curve_f, "frozen TinyMyo + linear probe": curve_z},
        title=f"Limited labelled data ({few_note})",
        path=OUTPUT_DIR / "exp8_few_label.png",
    )

    if sub_f is not None:
        plot_grouped_bars(
            ["held-out subjects"],
            {"handcrafted": [sub_f["accuracy"]], "TinyMyo probe": [sub_z["accuracy"]]},
            ylabel="test accuracy",
            title="Experiment 9 — train on seen people, test on unseen people",
            path=OUTPUT_DIR / "exp9_subject.png",
        )

    plot_three_pipelines(OUTPUT_DIR / "exp10_pipelines.png")
    plot_model_ab_table(OUTPUT_DIR / "exp10_model_ab.png")
    plot_summary_progression(OUTPUT_DIR / "emg_to_foundation_model.png")

    out.update(
        {
            "window_samples": b["window_samples"],
            "device": device,
            "input_shape": [None, 8, b["window_samples"]],
            "embedding_shape": [None, int(Z.shape[1])],
            "encoder_params": params,
            "checkpoint_missing": load_report["missing_keys"],
            "random_split_handcrafted": rand_f["accuracy"],
            "random_split_probe": probe_on_rand["accuracy"],
            "random_split_subject_overlap": sorted(int(s) for s in leak),
            "subject_split_handcrafted": None if sub_f is None else sub_f["accuracy"],
            "subject_split_probe": None if sub_z is None else sub_z["accuracy"],
            "few_label_handcrafted": curve_f,
            "few_label_probe": curve_z,
            "few_label_note": few_note,
            "pca_var_handcrafted": ev_f.tolist(),
            "pca_var_probe": ev_z.tolist(),
        }
    )
    (OUTPUT_DIR / "verify_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in out if k not in ("few_label_handcrafted", "few_label_probe", "checkpoint_missing")}, indent=2))
    plt.close("all")
    return out


if __name__ == "__main__":
    fast = "--full" not in sys.argv
    run(fast=fast)
