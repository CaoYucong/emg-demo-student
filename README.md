# 学生电脑：中文运行说明（请从这里开始）

打开这个文件夹，**先看这一页**。课堂笔记本是：

`notebooks/01_emg_story_class_zh.ipynb`

英特尔 CPU + 英特尔核显 + 32 GB 内存可以跑。**核显不会被用到**，计算走 **CPU**。不要装 CUDA，也不要找“英特尔 GPU 版 PyTorch”。

---

## 三步跑起来

| 步 | 做什么 | 命令 |
| --- | --- | --- |
| 1 | 建环境（只需一次） | Mac/Linux：`./scripts/setup_student.sh`　Windows：`.\scripts\setup_student.ps1` |
| 2 | 下数据（只需一次，要联网） | `python scripts/prepare.py` |
| 3 | 打开课 | `jupyter notebook notebooks/01_emg_story_class_zh.ipynb` |

打开后：**Kernel → Change Kernel → `yijun_emg_demo`**。第一格保持 `FAST_DEMO = True`。看到 `DEVICE cpu` 就对了。

---

## 0. 先装 Python

装 **Python 3.10 / 3.11 / 3.12**（64 位）：https://www.python.org/downloads/

- Windows 务必勾选 **Add python.exe to PATH**
- 不要用 3.8；不必追 3.14
- 外网要通（第一次下数据几百 MB）
- 磁盘空闲 2 GB 以上

检查：

```text
python3 --version     # Mac / Linux
python --version      # Windows
```

应看到 `Python 3.10` 或 `3.11` 或 `3.12`。

**不用 GitHub。** 老师会发给你一个压缩包（或 U 盘里的文件夹）。解压后进入项目根目录，能看到 `README.md`、`requirements.txt` 和 `scripts/` 就算对了。

文件夹不要放在“仅云端”的 iCloud / OneDrive 里，拷到本地磁盘再跑。

---

## 1. 建环境

### Mac / Linux（含英特尔 Mac）

在项目根目录打开终端：

```bash
chmod +x scripts/setup_student.sh
./scripts/setup_student.sh
```

### Windows

在项目根目录打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\scripts\setup_student.ps1
```

脚本失败就手动做：

**Mac / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name yijun_emg_demo --display-name yijun_emg_demo
python scripts/check_machine.py
```

**Windows**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name yijun_emg_demo --display-name yijun_emg_demo
python scripts/check_machine.py
```

`check_machine.py` 打印 `计算设备 cpu` 就是英特尔机的预期结果。

---

## 2. 下载数据（只做一次）

必须已经 `activate`，并且当前目录是项目根：

```bash
python scripts/prepare.py
```

要联网。英特尔 CPU 第一次大约 10–20 分钟。中断了再跑同一条即可。成功后可以断网上课。

---

## 3. 上课

先激活环境，再开笔记本：

```bash
source .venv/bin/activate          # Mac / Linux
# .\.venv\Scripts\Activate.ps1     # Windows
jupyter notebook notebooks/01_emg_story_class_zh.ipynb
```

1. Kernel 选 **`yijun_emg_demo`**
2. `FAST_DEMO = True`
3. 从上到下 Run

比老师的苹果电脑慢一点是正常的。

---

## 卡住了怎么办

| 现象 | 怎么办 |
| --- | --- |
| `No module named 'torch'` | Kernel 选错了。改成 `yijun_emg_demo` |
| `DEVICE cpu` | 正确。核显不参与 |
| pip / torch 装失败 | `python -m pip install -U pip` 再 `pip install -r requirements.txt`。不要混用 Anaconda base |
| Windows 不让跑 `.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 读数据卡住 / 找不到文件 | 项目拷到本地硬盘，不要放在仅云端的网盘 |
| `OMP: Error #15` | `export KMP_DUPLICATE_LIB_OK=TRUE`（Windows cmd：`set KMP_DUPLICATE_LIB_OK=TRUE`） |

老师若是 Apple 芯片，可能显示 `mps`。你是英特尔，应显示 `cpu`。数字接近即可，不必完全一样。

---

# 这节课在讲什么

从阈值控制的肌电机械臂，走到“基础模型”这个想法。不是普通 ML 教程。

1. 原来的控制器会在哪里失败？
2. 多通道模式能不能认？
3. 实验室分类器是不是假肢控制器？

编码器是冻结的 **TinyMyo**。不用 `UCI_finetune_5sec.safetensors`。探针没赢就照实留。

更多技术细节：`SETUP_NOTES.md`。英文课堂稿：`notebooks/01_emg_story_class.ipynb`。
