# Intel Windows student setup. Intel UHD/Iris is not used; PyTorch stays on CPU.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "找不到 python。请先安装 Python 3.10–3.12（勾选 Add python.exe to PATH）：https://www.python.org/downloads/"
    exit 1
}

Write-Host "Python:" (python -c "import sys; print(sys.version)")
Write-Host "Arch:  " (python -c "import platform; print(platform.machine())")

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name yijun_emg_demo --display-name yijun_emg_demo
python scripts/check_machine.py
Write-Host ""
Write-Host "环境好了。下一步（需要网络，只需一次）："
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python scripts/prepare.py"
Write-Host "然后打开 notebooks/01_emg_story_class_zh.ipynb ，Kernel 选 yijun_emg_demo"
