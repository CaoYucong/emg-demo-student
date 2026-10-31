#!/usr/bin/env bash
# Build a classroom zip: source + scripts + one notebook. No venv, no data, no weights, no git.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d)"
DEST="${1:-$HOME/Desktop/yijun_emg_demo_student_${STAMP}.zip}"
STAGE="$(mktemp -d /tmp/yijun_emg_student_XXXXXX)"
PACK="$STAGE/yijun_emg_demo"

mkdir -p "$PACK/src" "$PACK/vendor" "$PACK/scripts" "$PACK/notebooks" \
  "$PACK/figures" "$PACK/data/raw" "$PACK/data/processed" "$PACK/cache" \
  "$PACK/checkpoints" "$PACK/outputs"

cp "$ROOT/README.md" "$PACK/"
cp "$ROOT/运行说明.md" "$PACK/"
cp "$ROOT/LICENSE" "$PACK/"
cp "$ROOT/SETUP_NOTES.md" "$PACK/"
cp "$ROOT/requirements.txt" "$PACK/"
cp "$ROOT/.gitignore" "$PACK/"
cp "$ROOT/vendor/NOTICE" "$PACK/vendor/"
cp "$ROOT/vendor/TinyMyo.py" "$PACK/vendor/"
cp "$ROOT/vendor/uci.py" "$PACK/vendor/"
cp "$ROOT/src/"*.py "$PACK/src/"
cp "$ROOT/scripts/prepare.py" "$PACK/scripts/"
cp "$ROOT/scripts/verify.py" "$PACK/scripts/"
cp "$ROOT/scripts/check_machine.py" "$PACK/scripts/"
cp "$ROOT/scripts/setup_student.sh" "$PACK/scripts/"
cp "$ROOT/scripts/setup_student.ps1" "$PACK/scripts/"
cp "$ROOT/scripts/make_student_zip.sh" "$PACK/scripts/"
cp "$ROOT/notebooks/01_emg_story_class_zh.ipynb" "$PACK/notebooks/"
cp "$ROOT/figures/"*.png "$PACK/figures/" 2>/dev/null || true
cp "$ROOT/figures/ATTRIBUTION.md" "$PACK/figures/" 2>/dev/null || true

touch "$PACK/data/raw/.gitkeep"
touch "$PACK/data/processed/.gitkeep"
touch "$PACK/cache/.gitkeep"
touch "$PACK/checkpoints/.gitkeep"
touch "$PACK/outputs/.gitkeep"

cat > "$PACK/这是什么.txt" <<'EOF'
发给学生的课堂压缩包（最小可行）。

不含：Python 环境、UCI 原始数据、处理后的窗口、模型权重、老师本机的运行结果。
第一次上课前请先读 README.md，按三步安装并下载数据。
需要联网一次。整套大约还要再下载 300MB 左右。
EOF

cd "$STAGE"
rm -f "$DEST"
zip -X -r "$DEST" yijun_emg_demo \
  -x "*.DS_Store" \
  -x "*__pycache__*" \
  -x "*.pyc"

rm -rf "$STAGE"
ls -lh "$DEST"
echo "$DEST"
