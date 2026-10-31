"""Simple teaching plots. Readable titles, axes, legends. No ornate styling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

from . import OUTPUT_DIR

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
from .dataset import UCI_RAW_CLASS_NAMES
from .signal_processing import moving_average, rectify

PLOT_LANG = "en"


def configure_chinese_matplotlib() -> str:
    """Use a macOS CJK font so Chinese titles do not become boxes."""
    from matplotlib import font_manager

    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for font_path in candidates:
        if not Path(font_path).exists():
            continue
        try:
            font_manager.fontManager.addfont(font_path)
        except Exception:
            pass
        name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return name
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def set_plot_language(lang: str = "en") -> str:
    """lang='zh' also switches matplotlib to a Chinese-capable font."""
    global PLOT_LANG
    PLOT_LANG = "zh" if lang.startswith("zh") else "en"
    if PLOT_LANG == "zh":
        return configure_chinese_matplotlib()
    return "default"


def T(en: str, zh: str) -> str:
    return zh if PLOT_LANG == "zh" else en


def _save(fig: plt.Figure, path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return path


def plot_raw_channels(
    time_s: np.ndarray,
    emg: np.ndarray,
    raw_label: Optional[np.ndarray] = None,
    channels: Sequence[int] = (0, 1, 2, 3),
    title: str = "Raw multichannel EMG",
    path: Optional[Path] = None,
) -> plt.Figure:
    chans = list(channels)
    n = len(chans) + (1 if raw_label is not None else 0)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, ch in zip(axes, chans):
        ax.plot(time_s, emg[:, ch], lw=0.8, color="0.15")
        ax.set_ylabel(f"{T('ch', '通道')} {ch+1}")
    if raw_label is not None:
        axes[-1].plot(time_s, raw_label, lw=1.2, color="C3", drawstyle="steps-post")
        axes[-1].set_ylabel(T("label", "标签"))
        axes[-1].set_yticks(sorted(np.unique(raw_label)))
    axes[-1].set_xlabel(T("time (s)", "时间 (s)"))
    axes[0].set_title(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_raw_class_gallery(
    snippets: Dict[int, dict],
    names: Dict[int, str],
    channels: Sequence[int] = (0, 1, 2, 3),
    codes: Sequence[int] = tuple(range(8)),
    title: str = "Each official raw class, one hold",
    path: Optional[Path] = None,
) -> plt.Figure:
    """One panel per official UCI raw code (0–7). Missing codes stay empty."""
    codes = list(codes)
    ncols = 2
    nrows = int(np.ceil(len(codes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 2.15 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()
    for i, code in enumerate(codes):
        ax = axes[i]
        name = names.get(int(code), str(code))
        snip = snippets.get(int(code))
        if snip is None:
            ax.text(
                0.5,
                0.5,
                T("not in the loaded files", "这份已加载的数据里没有"),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.set_title(f"{code}  {name}")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        t = snip["time_s"]
        emg = snip["emg"]
        for ch in channels:
            ax.plot(t, emg[:, ch], lw=0.7, alpha=0.9)
        extra = ""
        if not snip.get("from_primary", True):
            extra = T(
                f"  (subject {snip['subject']})",
                f"  （受试者 {snip['subject']}）",
            )
        ax.set_title(f"{code}  {name}{extra}")
        ax.set_xlabel(T("time (s)", "时间 (s)"))
        if i % ncols == 0:
            ax.set_ylabel(T("amp.", "幅度"))
    for j in range(len(codes), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_threshold_controller(
    time_s: np.ndarray,
    emg_one: np.ndarray,
    envelope: np.ndarray,
    threshold: float,
    pred: np.ndarray,
    truth: np.ndarray,
    title: str,
    path: Optional[Path] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
    axes[0].plot(time_s, emg_one, lw=0.7, color="0.4", label=T("raw (one channel)", "原始信号（单通道）"))
    axes[0].plot(time_s, envelope, lw=1.4, color="C0", label=T("envelope", "包络"))
    axes[0].axhline(threshold, color="C3", ls="--", lw=1.2, label=T("threshold", "阈值"))
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel(T("amplitude", "幅度"))
    axes[0].set_title(title)
    axes[1].plot(time_s, pred, drawstyle="steps-post", color="C1", label=T("predicted", "预测"))
    axes[1].set_ylabel(T("pred", "预测"))
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(time_s, truth, drawstyle="steps-post", color="C2", label=T("ground truth", "真实标签"))
    axes[2].set_ylabel(T("truth", "真值"))
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_xlabel(T("time (s)", "时间 (s)"))
    axes[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_threshold_sweep(
    thresholds: Sequence[float],
    accuracies: Sequence[float],
    title: str = "Accuracy vs threshold",
    path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(thresholds, accuracies, marker="o", color="C0")
    ax.set_xlabel(T("threshold", "阈值"))
    ax.set_ylabel(T("binary accuracy", "二分类准确率"))
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_window_length_sweep(
    windows: Sequence[int],
    accuracies: Sequence[float],
    title: str = "Accuracy vs averaging-window length",
    path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(windows, accuracies, marker="o", color="C0")
    ax.set_xlabel(T("moving-average length (samples)", "滑动平均窗口（采样点）"))
    ax.set_ylabel(T("binary accuracy", "二分类准确率"))
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_moving_average_as_lowpass(
    time_s: np.ndarray,
    x_one: np.ndarray,
    windows: Sequence[int],
    fs: float,
    truth: Optional[np.ndarray] = None,
    title: str = "Moving average is a low-pass filter",
    path: Optional[Path] = None,
) -> plt.Figure:
    """Rectified raw vs MA envelopes, plus spectra so the low-pass is visible."""
    t = np.asarray(time_s)
    x = np.asarray(x_one, dtype=np.float64)
    rect = rectify(x)
    sl = slice(0, len(t))
    if truth is not None and len(truth) == len(t):
        rise = np.where(np.diff(np.asarray(truth).astype(int)) == 1)[0]
        if len(rise):
            start = max(0, int(rise[0]) - int(2.0 * fs))
            stop = min(len(t), start + int(8.0 * fs))
            sl = slice(start, stop)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0))
    axes[0].plot(
        t[sl],
        rect[sl],
        color="0.72",
        lw=0.7,
        label=T("|raw| (rectified)", "整流后的原始 |x|"),
    )
    cmap = plt.get_cmap("viridis")
    wins = list(windows)
    for i, w in enumerate(wins):
        env = moving_average(rect, int(w))
        ms = 1000.0 * int(w) / float(fs)
        axes[0].plot(
            t[sl],
            env[sl],
            lw=1.8,
            color=cmap(0.15 + 0.75 * i / max(len(wins) - 1, 1)),
            label=T(f"MA {w} ({ms:.0f} ms)", f"滑动平均 {w} 点（{ms:.0f} ms）"),
        )
    axes[0].set_ylabel(T("amplitude", "幅度"))
    axes[0].set_xlabel(T("time (s)", "时间 (s)"))
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(
        T(
            "Time: longer boxcar = slower, more lag (low-pass)",
            "时域：窗口越长越平滑、边沿越滞后（低通）",
        )
    )

    fmax = min(80.0, 0.5 * float(fs) - 1.0)
    freq = np.linspace(0.0, fmax, 800)
    omega = np.pi * freq / float(fs)
    for i, w in enumerate(wins):
        n = max(int(w), 1)
        den = n * np.sin(omega)
        num = np.sin(n * omega)
        h = np.ones_like(freq)
        nz = np.abs(den) > 1e-12
        h[nz] = np.abs(num[nz] / den[nz])
        color = cmap(0.15 + 0.75 * i / max(len(wins) - 1, 1))
        axes[1].plot(
            freq,
            h,
            lw=1.8,
            color=color,
            label=T(f"|H| MA {n}  (null ~ {fs/n:.0f} Hz)", f"|H| 窗口 {n}  （第一零点约 {fs/n:.0f} Hz）"),
        )
        axes[1].axvline(float(fs) / n, color=color, ls=":", lw=0.9, alpha=0.7)
    axes[1].set_xlim(0, fmax)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel(T("frequency (Hz)", "频率 (Hz)"))
    axes[1].set_ylabel(T("|H(f)|", "|H(f)|"))
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title(
        T(
            "Frequency response of a uniform MA: a sinc low-pass",
            "均匀滑动平均的幅频响应：sinc 低通",
        )
    )
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_time_fft_psd(
    time_s: np.ndarray,
    x: np.ndarray,
    freq: np.ndarray,
    mag: np.ndarray,
    psd_f: np.ndarray,
    psd: np.ndarray,
    title: str,
    mark_hz: Optional[float] = None,
    mark_label: Optional[str] = None,
    path: Optional[Path] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.2))
    axes[0].plot(time_s, x, lw=0.8, color="0.15")
    axes[0].set_xlabel(T("time (s)", "时间 (s)"))
    axes[0].set_ylabel(T("amplitude", "幅度"))
    axes[0].set_title(title)
    axes[1].plot(freq, mag, lw=1.0, color="C0")
    axes[1].set_xlabel(T("frequency (Hz)", "频率 (Hz)"))
    axes[1].set_ylabel("|FFT|")
    axes[1].set_xlim(0, 100)
    axes[2].plot(psd_f, psd, lw=1.0, color="C1")
    axes[2].set_xlabel(T("frequency (Hz)", "频率 (Hz)"))
    axes[2].set_ylabel("PSD")
    axes[2].set_xlim(0, 100)
    if mark_hz is not None:
        for ax in axes[1:]:
            ax.axvline(mark_hz, color="C3", ls="--", lw=1.1, label=mark_label or f"{mark_hz} Hz")
            ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_notch_before_after(
    time_s: np.ndarray,
    x_before: np.ndarray,
    x_after: np.ndarray,
    f_b: np.ndarray,
    psd_b: np.ndarray,
    f_a: np.ndarray,
    psd_a: np.ndarray,
    title: str,
    path: Optional[Path] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.6))
    axes[0, 0].plot(time_s, x_before, lw=0.7)
    axes[0, 0].set_title(T("before notch (time)", "陷波前（时域）"))
    axes[0, 1].plot(time_s, x_after, lw=0.7, color="C2")
    axes[0, 1].set_title(T("after notch (time)", "陷波后（时域）"))
    axes[1, 0].plot(f_b, psd_b)
    axes[1, 0].axvline(50, color="C3", ls="--", lw=1)
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].set_title(T("before notch (PSD)", "陷波前（PSD）"))
    axes[1, 1].plot(f_a, psd_a, color="C2")
    axes[1, 1].axvline(50, color="C3", ls="--", lw=1)
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_title(T("after notch (PSD)", "陷波后（PSD）"))
    for ax in axes[1]:
        ax.set_xlabel("Hz")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_crosstalk_diagram(path: Optional[Path] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    boxes = [
        (0.3, 3.3, T("target muscle\nactivity", "目标肌肉活动")),
        (0.3, 2.2, T("neighbouring muscle\n(crosstalk)", "邻近肌肉\n（串扰）")),
        (0.3, 1.1, T("motion artefact", "运动伪迹")),
        (0.3, 0.1, T("electrical /\ninstrument interference", "电气 / 仪器干扰")),
        (5.6, 1.6, T("measured electrode\nsignal", "电极测到的信号")),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                3.4,
                0.95,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor="white",
                edgecolor="0.2",
                lw=1.2,
            )
        )
        ax.text(x + 1.7, y + 0.48, text, ha="center", va="center", fontsize=9)
    for y in (3.75, 2.65, 1.55, 0.55):
        ax.annotate(
            "",
            xy=(5.6, 2.05),
            xytext=(3.75, y),
            arrowprops=dict(arrowstyle="->", color="0.3"),
        )
    ax.set_title(T("What is on the electrode (not all of it is 'noise')", "电极上有什么（并不都是“噪声”）"))
    _save(fig, path)
    return fig


def plot_confusion(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    short = [n if len(n) < 16 else n[:14] + "…" for n in class_names]
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    ax.set_xlabel(T("predicted", "预测"))
    ax.set_ylabel(T("true", "真实"))
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_pca_scatter(
    z: np.ndarray,
    y: np.ndarray,
    class_names: Dict[int, str],
    subject: Optional[np.ndarray] = None,
    title: str = "PCA (2D)",
    note: str = "Visual intuition only; metrics are the comparison.",
    path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    classes = sorted(np.unique(y))
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    if subject is None:
        for i, cls in enumerate(classes):
            m = y == cls
            ax.scatter(
                z[m, 0],
                z[m, 1],
                s=18,
                color=cmap(i % 10),
                label=class_names.get(int(cls), str(cls)),
                alpha=0.75,
            )
    else:
        subjects = sorted(np.unique(subject))[:8]
        for i, cls in enumerate(classes):
            for j, subj in enumerate(subjects):
                m = (y == cls) & (subject == subj)
                if not np.any(m):
                    continue
                ax.scatter(
                    z[m, 0],
                    z[m, 1],
                    s=18,
                    color=cmap(i % 10),
                    marker=markers[j % len(markers)],
                    alpha=0.75,
                    label=class_names.get(int(cls), str(cls)) if j == 0 else None,
                )
        ax.set_xlabel(T("PC1  (marker ~ subject)", "PC1（标记形状 ≈ 受试者）"))
    if subject is None:
        ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", frameon=True)
    ax.text(
        0.01,
        -0.18,
        note,
        transform=ax.transAxes,
        fontsize=8,
        color="0.3",
    )
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_few_label_curves(
    curves: Dict[str, dict],
    title: str,
    path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (name, summary) in enumerate(curves.items()):
        fracs = sorted(summary.keys())
        means = [summary[f]["mean"] for f in fracs]
        stds = [summary[f]["std"] for f in fracs]
        xs = [100 * f for f in fracs]
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=name)
    ax.set_xlabel(T("labelled training data (%)", "有标签训练数据 (%)"))
    ax.set_ylabel(T("test accuracy", "测试准确率"))
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_grouped_bars(
    names: Sequence[str],
    series: Dict[str, Sequence[float]],
    ylabel: str,
    title: str,
    path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(names))
    width = 0.35
    keys = list(series.keys())
    for i, k in enumerate(keys):
        ax.bar(x + (i - 0.5) * width, series[k], width, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
    return fig


def _forearm_ring(ax, n: int, r: float = 0.82, offset: float = np.pi / 2) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset
    ax.add_patch(Circle((0.0, 0.0), 0.42, fill=False, lw=1.2, color="0.35"))
    ax.plot([0, 0], [0.15, 0.95], color="0.75", lw=6, solid_capstyle="round", zorder=0)
    return th


def plot_four_dataset_channels(path: Optional[Path] = None) -> plt.Figure:
    """Schematic channel layouts. Not a muscle atlas unless the official docs name one."""
    if PLOT_LANG == "zh":
        configure_chinese_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.4))

    ax = axes[0, 0]
    th = _forearm_ring(ax, 8)
    for i, a in enumerate(th, start=1):
        ax.scatter(0.82 * np.cos(a), 0.82 * np.sin(a), s=70, c="C0", zorder=2)
        ax.text(1.05 * np.cos(a), 1.05 * np.sin(a), str(i), ha="center", va="center", fontsize=8)
    ax.set_title(T("UCI 481 / Myo — 8 equal pods", "UCI 481 / Myo — 8 个等间距电极"))
    ax.text(
        0,
        -1.25,
        T("No official muscle-per-column map", "官方没有“第几通道=哪块肌肉”"),
        ha="center",
        fontsize=8,
        color="0.3",
    )

    ax = axes[0, 1]
    th8 = _forearm_ring(ax, 8)
    for a in th8:
        ax.scatter(0.82 * np.cos(a), 0.82 * np.sin(a), s=55, c="C3", zorder=2)
    th6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 2
    for a in th6:
        ax.scatter(0.58 * np.cos(a), 0.58 * np.sin(a) - 0.08, s=45, c="C1", zorder=2)
    ax.set_title(T("NinaPro DB6 — 14 Delsys", "NinaPro DB6 — 14 个 Delsys"))
    ax.text(
        0,
        -1.25,
        T("8 at radio-humeral joint + 6 distal", "近端桡肱关节一圈 8 个 + 远端再一圈 6 个"),
        ha="center",
        fontsize=8,
        color="0.3",
    )

    ax = axes[1, 0]
    th = _forearm_ring(ax, 8)
    for a in th:
        ax.scatter(0.82 * np.cos(a), 0.82 * np.sin(a), s=50, c="C3", zorder=2)
    ax.scatter([-0.35, 0.35], [-0.15, -0.15], s=70, c="C0", zorder=2)
    ax.text(-0.35, -0.32, "FDS", ha="center", fontsize=7)
    ax.text(0.35, -0.32, "EDC", ha="center", fontsize=7)
    ax.scatter([-0.22, 0.22], [0.55, 0.55], s=70, c="C2", zorder=2)
    ax.text(-0.22, 0.72, T("biceps", "肱二头"), ha="center", fontsize=7)
    ax.text(0.22, 0.72, T("triceps", "肱三头"), ha="center", fontsize=7)
    ax.set_title(T("NinaPro DB7 — 12 Delsys", "NinaPro DB7 — 12 个 Delsys"))
    ax.text(
        0,
        -1.25,
        T("8 ring + FDS/EDC + biceps/triceps", "8 个一圈 + 指浅屈/指伸 + 肱二/肱三"),
        ha="center",
        fontsize=8,
        color="0.3",
    )

    ax = axes[1, 1]
    n = 16
    th = np.linspace(0.15 * np.pi, 2 * np.pi - 0.15 * np.pi, n)
    ax.add_patch(Circle((0.0, 0.0), 0.42, fill=False, lw=1.2, color="0.35"))
    ax.plot([0, 0], [0.12, 0.7], color="0.75", lw=5, solid_capstyle="round", zorder=0)
    for i, a in enumerate(th):
        ax.scatter(0.8 * np.cos(a), 0.8 * np.sin(a), s=36, c="C4", zorder=2)
        if i in (0, 15, 7):
            ax.text(1.02 * np.cos(a), 1.02 * np.sin(a), str(i), ha="center", va="center", fontsize=7)
    ax.text(0, -0.15, T("ulnar gap", "尺骨空隙"), ha="center", fontsize=8, color="0.35")
    ax.set_title(T("EMG2Pose — 16 bipolar wrist pods", "EMG2Pose — 腕上一圈 16 个双极"))
    ax.text(
        0,
        -1.25,
        T("ch 0 and 15 sit beside the ulnar gap", "0 与 15 贴在尺骨空隙两侧"),
        ha="center",
        fontsize=8,
        color="0.3",
    )

    for ax in axes.ravel():
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.45, 1.25)
        ax.axis("off")
    fig.suptitle(
        T(
            "Where the channels sit — official geometry, not a muscle dictionary",
            "通道大概贴在哪 — 官方几何，不是肌肉词典",
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)
    return fig


def plot_collection_photos(path: Optional[Path] = None) -> plt.Figure:
    """Show downloaded official photos. Skip missing files."""
    if PLOT_LANG == "zh":
        configure_chinese_matplotlib()
    candidates = [
        (
            FIGURES_DIR / "SData_Protocol.png",
            T(
                "NinaPro official protocol figure (DB6/DB7 family). Source: ninapro.hevs.ch",
                "NinaPro 官方采集流程（DB6/DB7 同一套协议）。来源：ninapro.hevs.ch",
            ),
        ),
        (
            FIGURES_DIR / "DB6_movements1.jpg",
            T(
                "NinaPro DB6: a real hand performing the seven grasps. Source: ninapro.hevs.ch",
                "NinaPro DB6：真人手做那 7 种抓握。来源：ninapro.hevs.ch",
            ),
        ),
        (
            FIGURES_DIR / "SData_Movements.png",
            T(
                "NinaPro official movement-photo panel. Source: ninapro.hevs.ch",
                "NinaPro 官方动作照片。来源：ninapro.hevs.ch",
            ),
        ),
    ]
    found = []
    try:
        from PIL import Image as _PILImage
    except Exception:
        _PILImage = None
    for p, cap in candidates:
        if not p.is_file() or p.stat().st_size < 80_000:
            continue
        try:
            if _PILImage is None:
                _ = plt.imread(p)
            else:
                with _PILImage.open(p) as im:
                    im.load()
        except Exception:
            continue
        found.append((p, cap))
    if not found:
        fig, ax = plt.subplots(figsize=(7, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.5, T("No local collection photo yet.", "本地还没有采集照片。"), ha="center")
        _save(fig, path)
        return fig
    fig, axes = plt.subplots(len(found), 1, figsize=(8.8, 3.6 * len(found)))
    if len(found) == 1:
        axes = [axes]
    for ax, (p, cap) in zip(axes, found):
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(cap, loc="left", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_tinymyo_mask_story(path: Optional[Path] = None) -> plt.Figure:
    """Schematic of TinyMyo masked reconstruction. Not a model forward pass."""
    if PLOT_LANG == "zh":
        configure_chinese_matplotlib()
    rng = np.random.RandomState(1)
    fs = 2000.0
    patch = 20
    n_patch = 16
    n = n_patch * patch
    t_ms = np.arange(n) / fs * 1000.0
    tt = np.arange(n) / fs
    env = (
        0.25
        + 0.95 * np.exp(-0.5 * ((tt - 0.055) / 0.018) ** 2)
        + 0.55 * np.exp(-0.5 * ((tt - 0.115) / 0.022) ** 2)
    )
    x = env * np.sin(2 * np.pi * 75.0 * tt)
    x = x + 0.06 * rng.randn(n)
    x = x / np.max(np.abs(x))
    masked = {0, 1, 4, 7, 8, 11, 14}

    def _blank_masked(sig: np.ndarray) -> np.ndarray:
        out = sig.astype(float).copy()
        for i in masked:
            out[i * patch : (i + 1) * patch] = np.nan
        return out

    kernel = np.ones(11) / 11.0
    smooth = np.convolve(x, kernel, mode="same")
    xhat = x.copy()
    for i in masked:
        sl = slice(i * patch, (i + 1) * patch)
        xhat[sl] = 0.85 * smooth[sl]

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.0), sharex=True)
    axes[0].plot(t_ms, x, color="0.12", lw=1.15)
    axes[0].set_title(T("1. Original window (one channel)", "1. 原始窗口（单通道）"), loc="left")

    axes[1].plot(t_ms, _blank_masked(x), color="0.12", lw=1.15)
    axes[1].set_title(
        T("2. Masked: grey holes are [MASK] tokens", "2. 遮住之后：灰色空洞就是 [MASK]"),
        loc="left",
    )

    axes[2].plot(t_ms, x, color="0.65", lw=1.0, ls="--", label=T("original", "原始"))
    axes[2].plot(t_ms, xhat, color="0.10", lw=1.25, label=T("schematic reconstruction", "示意重建"))
    axes[2].set_title(
        T("3. Job: fill the holes using the visible scraps", "3. 作业：用看得见的片段把洞填回去"),
        loc="left",
    )
    axes[2].legend(fontsize=8, loc="upper right")

    for ax in axes[1:]:
        for i in masked:
            left = t_ms[i * patch]
            right = t_ms[min((i + 1) * patch, n) - 1]
            ax.axvspan(left, right, color="#c8c8c8", zorder=0)
            ax.text(
                0.5 * (left + right),
                1.05,
                "M",
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.35",
            )
    for ax in axes:
        ax.set_ylabel(T("amplitude", "幅度"))
        ax.set_ylim(-1.35, 1.45)
        ax.set_xlim(t_ms[0], t_ms[-1])
    axes[-1].set_xlabel(
        T(
            "time (ms)    one patch = 20 samples = 10 ms at 2 kHz",
            "时间 (ms)    一个 patch = 20 个点 = 2 kHz 下 10 ms",
        )
    )
    fig.suptitle(
        T(
            "SCHEMATIC — TinyMyo pretraining (not a real model output)",
            "示意图 — TinyMyo 预训练在干什么（不是模型真实输出）",
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)
    return fig


def plot_three_pipelines(path: Optional[Path] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    pipelines = [
        (
            0.2,
            4.3,
            T("Traditional", "传统流程"),
            [
                "EMG",
                T("filtering", "滤波"),
                T("handcrafted features", "手工特征"),
                T("task-specific classifier", "任务分类器"),
            ],
        ),
        (
            0.2,
            2.4,
            T("Deep learning", "深度学习"),
            ["EMG", T("learned representation", "学到的表示"), T("task-specific prediction", "任务预测")],
        ),
        (
            0.2,
            0.5,
            T("Foundation model", "基础模型"),
            [
                T("large heterogeneous EMG", "大规模异构 EMG"),
                T("self-supervised pretraining", "自监督预训练"),
                T("reusable representation", "可复用表示"),
                T("small downstream model", "小型下游模型"),
            ],
        ),
    ]
    for x0, y, title, steps in pipelines:
        ax.text(x0, y + 1.05, title, fontsize=11, fontweight="bold")
        for i, step in enumerate(steps):
            x = x0 + i * 2.9
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    2.6,
                    0.85,
                    boxstyle="round,pad=0.03,rounding_size=0.1",
                    facecolor="white",
                    edgecolor="0.2",
                )
            )
            ax.text(x + 1.3, y + 0.42, step, ha="center", va="center", fontsize=8)
            if i < len(steps) - 1:
                ax.annotate(
                    "",
                    xy=(x + 2.7, y + 0.42),
                    xytext=(x + 2.6, y + 0.42),
                    arrowprops=dict(arrowstyle="->", color="0.2"),
                )
    _save(fig, path)
    return fig


def plot_summary_progression(path: Optional[Path] = None) -> plt.Figure:
    steps = [
        T("Muscle activity", "肌肉活动"),
        T("Surface EMG", "表面肌电"),
        T("Interference / crosstalk / artefact", "干扰 / 串扰 / 伪迹"),
        T("Signal processing", "信号处理"),
        T("Threshold control", "阈值控制"),
        T("LIMITATION: only simple decisions", "局限：只能做简单判决"),
        T("Multichannel patterns", "多通道模式"),
        T("Handcrafted features", "手工特征"),
        T("Machine learning", "机器学习"),
        T("LIMITATION: user / electrode variability", "局限：人/电极会变"),
        T("Pretrained representation", "预训练表示"),
        T("Linear probe", "线性探针"),
        T("Foundation-model idea", "基础模型的想法"),
        T("Movement intention", "运动意图"),
        T("Controller", "控制器"),
        T("Actuator / prosthetic or robot", "执行器 / 假肢或机器人"),
    ]
    path = path or (OUTPUT_DIR / "emg_to_foundation_model.png")
    fig, ax = plt.subplots(figsize=(7.2, 11.5))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) + 1)
    for i, text in enumerate(steps):
        y = len(steps) - i
        is_lim = text.startswith("LIMITATION") or text.startswith("局限")
        ax.add_patch(
            FancyBboxPatch(
                (1.2, y - 0.4),
                7.6,
                0.75,
                boxstyle="round,pad=0.03,rounding_size=0.1",
                facecolor="#f4f4f4" if is_lim else "white",
                edgecolor="0.15",
                lw=1.15,
            )
        )
        ax.text(5, y - 0.02, text, ha="center", va="center", fontsize=10)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(5, y - 0.48),
                xytext=(5, y - 0.62),
                arrowprops=dict(arrowstyle="->", color="0.2"),
            )
    ax.set_title(T("From muscle activity to a controller", "从肌肉活动到控制器"), pad=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_model_ab_table(path: Optional[Path] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ax.axis("off")
    cols = ["", T("Model A", "方案 A"), T("Model B", "方案 B")]
    rows = [
        [T("accuracy", "准确率"), "91%", "96%"],
        [T("memory", "内存"), "1 MB", "500 MB"],
        [T("latency", "延迟"), "2 ms", "100 ms"],
        [T("power", "功耗"), T("low", "低"), T("higher", "更高")],
        [T("cross-user", "跨用户"), T("moderate", "一般"), T("better", "更好")],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    ax.set_title(T(
        "Which model belongs inside a battery-powered prosthetic hand?",
        "电池供电的假肢手里，你会放哪一个？",
    ))
    _save(fig, path)
    return fig
