#!/usr/bin/env bash
# Intel Mac / Linux student setup. Does not use Intel integrated graphics.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null; then
  echo "找不到 python3。请先安装 Python 3.10–3.12（https://www.python.org/downloads/）"
  exit 1
fi

PY=python3
echo "Python: $($PY -c 'import sys; print(sys.version)')"
echo "Arch:   $($PY -c 'import platform; print(platform.machine())')"

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name yijun_emg_demo --display-name yijun_emg_demo
python scripts/check_machine.py
echo
echo "环境好了。下一步（需要网络，只需一次）："
echo "  source .venv/bin/activate"
echo "  python scripts/prepare.py"
echo "然后打开 notebooks/01_emg_story_class_zh.ipynb ，Kernel 选 yijun_emg_demo"
