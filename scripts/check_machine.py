#!/usr/bin/env python3
"""Print whether this computer can run the classroom demo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.machine_info import format_machine_report, machine_report


def main() -> int:
    rep = machine_report()
    print(format_machine_report(rep, lang="zh"))
    print()
    if "not importable" in str(rep["torch"]):
        print("还没装好 PyTorch。请先按 运行说明.md 建虚拟环境并 pip install。")
        return 1
    if rep["device"] == "cpu":
        print("这是预期情况：英特尔 CPU + 核显会走 CPU，不要去装 CUDA。")
    print("下一步：python scripts/prepare.py   （只需联网做一次）")
    print("然后：jupyter notebook notebooks/01_emg_story_class_zh.ipynb")
    print("Kernel 选 yijun_emg_demo，保持 FAST_DEMO = True。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
