"""Yijun EMG teaching demo package."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
CHECKPOINT_DIR = ROOT / "checkpoints"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "outputs"
VENDOR_DIR = ROOT / "vendor"

__all__ = [
    "ROOT",
    "DATA_DIR",
    "PROCESSED_DIR",
    "RAW_DIR",
    "CHECKPOINT_DIR",
    "CACHE_DIR",
    "OUTPUT_DIR",
    "VENDOR_DIR",
]
