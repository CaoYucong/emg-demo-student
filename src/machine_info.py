"""Print what this computer can actually run. Intel iGPU is not used."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any


def _ram_gb() -> float | None:
    try:
        if sys.platform == "darwin":
            import subprocess

            raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(raw) / (1024**3)
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024**3)
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        return (pages * page) / (1024**3)
    except Exception:
        return None


def machine_report() -> dict[str, Any]:
    try:
        import torch

        torch_ver = torch.__version__
        from src.pretrained_encoder import select_device

        device = select_device()
    except Exception as exc:
        torch_ver = f"not importable ({exc})"
        device = "unknown"

    ram = _ram_gb()
    arch = platform.machine().lower()
    intel = any(k in arch for k in ("x86_64", "amd64", "i386", "i686"))
    apple = arch in ("arm64", "aarch64") and sys.platform == "darwin"
    return {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "intel_cpu": intel,
        "apple_silicon": apple,
        "python": sys.version.split()[0],
        "torch": torch_ver,
        "device": device,
        "ram_gb": None if ram is None else round(ram, 1),
        "note": (
            "Intel integrated graphics (UHD/Iris) is not used. "
            "PyTorch runs this demo on CPU. 16–32 GB RAM is enough."
        ),
    }


def format_machine_report(rep: dict[str, Any] | None = None, lang: str = "zh") -> str:
    r = rep or machine_report()
    if lang.startswith("zh"):
        if r["device"] == "cpu":
            gpu = "英特尔核显不会被用到，计算在 CPU 上。这是学生机的预期情况"
        elif r["device"] == "mps":
            gpu = "这台是 Apple 芯片，计算走 MPS"
        else:
            gpu = f"计算走 {r['device']}"
        extra = "内存 16–32 GB 够用。保持 FAST_DEMO = True。"
        return (
            f"系统 {r['os']}  |  架构 {r['arch']}  |  Python {r['python']}\n"
            f"PyTorch {r['torch']}  |  计算设备 {r['device']}  |  内存约 {r['ram_gb']} GB\n"
            f"{gpu}。{extra}"
        )
    return (
        f"OS {r['os']}  |  arch {r['arch']}  |  Python {r['python']}\n"
        f"PyTorch {r['torch']}  |  device {r['device']}  |  RAM ~ {r['ram_gb']} GB\n"
        f"{r['note']}"
    )
