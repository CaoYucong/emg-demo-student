"""Frozen TinyMyo encoder and linear-probe embeddings.

Pooling matches official ``EMGClassificationHead`` *before* the Linear layer:
tokens (B, C*P, 192) -> concat channels -> mean over patches -> (B, C*192).

There is no official ``forward_features`` API. This wrapper reuses
``prepare_tokens``, transformer blocks, and ``norm`` from the vendored
``TinyMyo`` class, then applies that documented pooling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from einops import rearrange
from safetensors.torch import load_file

from . import CHECKPOINT_DIR
from .tinymyo_official import TinyMyo

HF_REPO = "PulpBio/TinyMyo"
# Generic pretrained reconstruction checkpoint. Do NOT use UCI_finetune_5sec
# as a "foundation" encoder: that file already saw UCI labels.
HF_PRETRAINED_FILE = "pretraining/TinyMyo/TinyMyo.safetensors"
UCI_FINETUNE_FILE = "UCI_EMG/UCI_finetune_5sec.safetensors"

PRETRAINED_CONFIG = {
    "img_size": 1000,
    "patch_size": 20,
    "in_chans": 16,
    "embed_dim": 192,
    "n_layer": 8,
    "n_head": 3,
    "mlp_ratio": 4,
    "qkv_bias": True,
    "attn_drop": 0.1,
    "proj_drop": 0.1,
    "drop_path": 0.0,
    "num_classes": 0,
    "task": "pretraining",
}


def default_checkpoint_path() -> Path:
    return CHECKPOINT_DIR / "TinyMyo" / HF_PRETRAINED_FILE


def select_device(prefer: str | None = None) -> str:
    """Pick a compute device.

    Intel laptops with integrated graphics have **no** PyTorch backend.
    They run on CPU. Apple Silicon may use MPS. NVIDIA may use CUDA.

    Override with ``prefer`` or env ``YIJUN_DEVICE`` = ``cpu`` / ``mps`` /
    ``cuda`` / ``auto``.
    """
    prefer = (prefer or os.environ.get("YIJUN_DEVICE") or "auto").lower()
    mps_ok = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if prefer == "cpu":
        return "cpu"
    if prefer in ("auto", "mps", "gpu", "metal") and mps_ok:
        return "mps"
    if prefer in ("auto", "cuda", "gpu") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def recommended_batch_size(device: str) -> int:
    if device == "mps":
        return 32
    if device == "cuda":
        return 64
    return 16


def download_pretrained_checkpoint(dest_dir: Optional[Path] = None) -> Path:
    from huggingface_hub import hf_hub_download

    dest_dir = dest_dir or (CHECKPOINT_DIR / "TinyMyo")
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_PRETRAINED_FILE,
        local_dir=str(dest_dir),
    )
    return Path(path)


def _strip_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state.keys())
    prefixes = ("model.", "module.", "net.")
    for prefix in prefixes:
        if keys and all(k.startswith(prefix) for k in keys):
            state = {k[len(prefix) :]: v for k, v in state.items()}
            keys = list(state.keys())
    return state


def inspect_checkpoint(path: Path) -> dict:
    state = _strip_prefixes(load_file(str(path)))
    info = {
        "path": str(path),
        "n_tensors": len(state),
        "keys_head": sorted(state.keys())[:20],
        "shapes": {k: list(v.shape) for k, v in list(state.items())[:30]},
    }
    if "channel_embed" in state:
        info["channel_embed"] = list(state["channel_embed"].shape)
    if "patch_embedding.patch_embed.proj.weight" in state:
        info["patch_embed"] = list(
            state["patch_embedding.patch_embed.proj.weight"].shape
        )
    return info


def build_encoder(
    img_size: int,
    in_chans: int = 8,
    device: str = "cpu",
) -> TinyMyo:
    """Build TinyMyo with the official pretrained hyperparameters.

    ``img_size`` must equal the window length in samples. Pretraining used
    1000 samples (0.5 s at 2 kHz). UCI windows may be 1000 (5 s at 200 Hz)
    or 200 (1 s at 200 Hz). Channel bank remains 16; TinyMyo uses the first
    ``in_chans`` learned channel embeddings.
    """
    model = TinyMyo(
        img_size=img_size,
        patch_size=20,
        in_chans=in_chans,
        embed_dim=192,
        n_layer=8,
        n_head=3,
        mlp_ratio=4,
        qkv_bias=True,
        attn_drop=0.1,
        proj_drop=0.1,
        drop_path=0.0,
        task="pretraining",
        num_classes=0,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_pretrained_weights(
    model: TinyMyo,
    checkpoint_path: Path,
) -> dict:
    raw = load_file(str(checkpoint_path))
    state = _strip_prefixes(raw)
    incompatible = model.load_state_dict(state, strict=False)
    model.to(next(model.parameters()).device)
    return {
        "n_loaded": len(state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": int(total), "trainable": int(trainable), "frozen": int(frozen)}


def pool_classification_embedding(tokens: torch.Tensor, in_chans: int) -> torch.Tensor:
    """Match EMGClassificationHead pooling, without the Linear classifier."""
    _b, n, _d = tokens.shape
    num_patches = n // in_chans
    x = rearrange(tokens, "b (c p) d -> b p (c d)", c=in_chans, p=num_patches)
    return x.mean(dim=1)


@torch.no_grad()
def encode_tokens(model: TinyMyo, x: torch.Tensor) -> torch.Tensor:
    """Encoder forward up to (and including) final LayerNorm."""
    b, c, t = x.shape
    tokens = model.prepare_tokens(x)
    p = t // model.patch_size
    pos_ids = torch.arange(p, device=x.device).repeat(c).unsqueeze(0).expand(b, -1)
    for blk in model.blocks:
        tokens = blk(tokens, pos_ids=pos_ids, attn_mask=None)
    return model.norm(tokens)


@torch.no_grad()
def encode_windows(
    model: TinyMyo,
    X: np.ndarray,
    batch_size: int = 8,
    device: str = "cpu",
) -> np.ndarray:
    """X (N, C, T) float -> embeddings (N, C*embed_dim)."""
    model = model.to(device)
    model.eval()
    n, c, t = X.shape
    assert t == model.img_size, f"T={t} != img_size={model.img_size}"
    out = []
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
        tokens = encode_tokens(model, xb)
        pooled = pool_classification_embedding(tokens, in_chans=c)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def embedding_dim(in_chans: int = 8, embed_dim: int = 192) -> int:
    return in_chans * embed_dim


def save_load_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
