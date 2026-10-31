# SETUP_NOTES.md

This file records **verified** TinyMyo / BioFoundation / UCI-EMG facts used by
the demo. Nothing here is inferred from a blog post or from an imagined API.

Runtime observations (window counts, checkpoint key names after download,
whether rest vs fist is a valid amplitude pair) are written by
`python scripts/prepare.py` to `data/processed/meta.json`. Read that file
alongside this one after setup.

## Local runtime (do not put the venv in iCloud)

This vault lives in iCloud Drive. macOS can evict `.venv` (`libtorch_cpu.dylib`)
and `cache/` / `data/processed/` into **dataless** placeholders. Importing
`torch` or `load_meta()` then hangs on `read()`.

Use a venv **outside** iCloud:

```bash
/opt/homebrew/bin/python3.14 -m venv ~/.venvs/yijun_emg_demo
~/.venvs/yijun_emg_demo/bin/python -m pip install -r requirements.txt
~/.venvs/yijun_emg_demo/bin/python -m ipykernel install --user --name yijun_emg_demo --display-name yijun_emg_demo
```

If `ls -lO cache data/processed checkpoints` shows `dataless`, re-run
`python scripts/prepare.py` with that local interpreter (or copy a fresh local
prepare into those folders). Notebook kernel: **`yijun_emg_demo`**.

## Repositories

| Role | Location |
| --- | --- |
| Model code | https://github.com/pulp-bio/BioFoundation `models/TinyMyo.py` (Apache-2.0) |
| Weights + UCI conversion script | https://huggingface.co/PulpBio/TinyMyo (CC BY-ND 4.0) |
| Paper | arXiv:2512.15729 |
| Gesture dataset | UCI ML Repository id 481, https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures |
| Dataset DOI | https://doi.org/10.24432/C5ZP5C |

BioFoundation `scripts/` on GitHub currently contains only a README. The working
UCI converter lives in the Hugging Face repo:

`https://huggingface.co/PulpBio/TinyMyo/blob/main/scripts/uci.py`

A copy is vendored at `vendor/uci.py`. The TinyMyo class is vendored at
`vendor/TinyMyo.py` / `src/tinymyo_official.py`.

The classroom environment does **not** install full BioFoundation (Lightning,
tensorrt, CUDA). That stack is not appropriate for a Mac CPU demo.

## Checkpoints (exact names)

Hugging Face repo `PulpBio/TinyMyo`, inspected via the Hub tree API:

| File | Role in this demo |
| --- | --- |
| `pretraining/TinyMyo/TinyMyo.safetensors` (~14.3 MB) | **Used.** Generic pretrained reconstruction encoder. |
| `pretraining/TinyMyo/config.json` | Official architecture hyperparameters. |
| `UCI_EMG/UCI_finetune_5sec.safetensors` | **Not used as a foundation encoder.** This checkpoint was already fine-tuned on UCI labels. Using it for a “frozen linear probe on UCI” would leak task exposure. |
| `UCI_EMG/config.json` | Documented only; not loaded. |
| DB5 / EPN612 / DB8 fine-tunes | Not used. |

Official pretrained `config.json`:

```json
{
  "_target_": "models.TinyMyo.TinyMyo",
  "img_size": 1000,
  "patch_size": 20,
  "in_chans": 16,
  "embed_dim": 192,
  "n_layer": 8,
  "n_head": 3,
  "mlp_ratio": 4,
  "qkv_bias": true,
  "attn_drop": 0.1,
  "proj_drop": 0.1,
  "drop_path": 0.0,
  "num_classes": 0,
  "task": "pretraining"
}
```

Download command used by `scripts/prepare.py`:

```python
huggingface_hub.hf_hub_download(
    repo_id="PulpBio/TinyMyo",
    filename="pretraining/TinyMyo/TinyMyo.safetensors",
    local_dir="checkpoints/TinyMyo",
)
```

State-dict prefixes are inspected at load time (possible `model.` / `module.`
stripping). Missing/unexpected keys are stored in `meta.json`. Typical expected
mismatch: the reconstruction head vs a classification head, which we do not use.

## TinyMyo input / embedding (official)

- Default input: `(B, in_chans, img_size)` with `img_size = 1000`.
- Runtime length **must** equal `img_size` (`TinyMyo.forward` asserts this).
- Maximum channels: 16. Fewer channels use a prefix of learned `channel_embed`.
- Patch size: 20 samples. Tokens are channel-major.
- `embed_dim = 192`.
- There is **no** official `forward_features`.
- Classification head `EMGClassificationHead`:
  1. tokens `(B, C*P, 192)`
  2. rearrange to `(B, P, C*192)`
  3. mean over patches `P`
  4. `Linear(C*192, num_classes)`

This demo’s frozen embedding is step 3: **`(B, in_chans * 192)`**.
For 8-channel UCI that is **`(B, 1536)`**.

Pretraining physics (paper + model card): 16 channels, **2 kHz**, 1000 samples =
**0.5 s**, band-pass 20–450 Hz, 50 Hz notch, per-channel min-max.

UCI downstream protocol (`scripts/uci.py` + scripts README): 8 channels,
**200 Hz**, band-pass **20–90 Hz**, 50 Hz notch, per-channel **z-score**.
Official window recipes:

- `--seq_len 200 --stride 50` → 1 s
- `--seq_len 1000 --stride 250` → 5 s

`img_size=1000` matches UCI **sample count** for the 5 s recipe, not physical
duration vs pretraining. If 5 s windows are too rare because labelled runs are
shorter than 1000 samples, `prepare.py` falls back to the official 1 s recipe
and instantiates `TinyMyo(img_size=200)`. That is a documented domain/length
mismatch, not a silent pad-to-1000 hack.

## UCI EMG labels (verified from UCI documentation)

From the UCI dataset page, class column:

0. unmarked data
1. hand at rest
2. hand clenched in a fist
3. wrist flexion
4. wrist extension
5. radial deviations
6. ulnar deviations
7. extended palm (not performed by all subjects)

Sampling: MYO Thalmic bracelet, **eight channels**, Bluetooth to PC. UCI text
does not always print “200 Hz”; TinyMyo `uci.py` sets `fs = 200.0` and that is
the rate used here, attributed to the TinyMyo conversion script.

Gesture timing in the UCI description: each gesture about **3 s**, pause about
**3 s**. That matters for whether 5 s windows exist.

### TinyMyo script vs UCI codebook

`preprocess_emg(..., remove_class0=True)` keeps `class >= 1` and its docstring
says this removes “rest”. That docstring is **wrong relative to the UCI
codebook**. Class 0 is unmarked. Class 1 is rest and is kept. Labels are then
shifted with `labs - 1`, so classification label 0 is rest.

Subjects who performed gesture 7 are dropped (script: `users_with_gesture(..., 7)`).
Published split: train 1–24, val 25–30, test 31–36, after that filter.

Official HDF5 has datasets `data` `(N, 8, seq_len)` and `label` only.
**Subject IDs are a teaching extension** stored by this project so Experiment 9
is possible.

## Experiment 2 — do not assume rest vs fist

Requirement: verify semantics, then verify **amplitude**.

1. Docs: rest is class 1, fist is class 2. Unmarked is class 0. We never treat
   unmarked as rest.
2. `prepare.py` computes raw RMS by class on real recordings.
3. Rest vs fist is used only if rest RMS is clearly lower than fist RMS
   (`<= 0.85 × fist`).
4. Otherwise the notebook labels the demo **binary amplitude-based
   classification** and uses the lowest-RMS vs highest-RMS non-unmarked classes.
5. A synthetic rest segment, if ever used, is labelled SYNTHETIC and is **not**
   included in quantitative model comparisons.

Thresholding is done on **raw** EMG (rectified moving average / RMS envelope),
not on z-scored TinyMyo windows. Z-scoring would destroy the amplitude story.

## Linear probe honesty

The encoder is loaded from `pretraining/TinyMyo/TinyMyo.safetensors` with
`requires_grad=False`. We do not fine-tune TinyMyo on UCI.

Paper note, kept for class: on UCI, pretrained ≈ from-scratch because the task
is relatively simple. Handcrafted features may win. **Do not change dataset to
force a foundation-model win.**

Domain-mismatch reasons a probe might lose:

- 2 kHz / 16 ch pretraining vs 200 Hz / 8 ch UCI
- 20–450 Hz min-max vs 20–90 Hz z-score
- 0.5 s pretraining windows vs 1 s or 5 s UCI windows
- linear probe cannot adapt the encoder
- 6 static gestures are already easy for RMS/MAV/WL/ZC
- subject shift is hard

## 50 Hz

Official UCI preprocessing applies a 50 Hz notch. That does **not** prove the
released files contain a strong mains peak. Experiment 3 always plots the
**observed** PSD, then a **synthetic** `x + A sin(2π·50·t)` example labelled
SYNTHETIC.

## Runtime observations (this machine, after `prepare.py`)

These are **observed**, not claimed in advance.

- Gesture-7 users dropped by the official script: **11 and 30**.
- Labelled gesture runs in the raw files are longer than the UCI “3 s” blurb:
  median rest/fist/flexion holds were about **8.3–8.5 s**, so official
  `--seq_len 1000` (5 s) windows exist. Counts: **2803** windows, 6 classes,
  minimum class count 440. The 1 s recipe produced 25410 windows; we still
  used 5 s because it is the first official protocol that satisfied the
  count checks and matches TinyMyo `img_size=1000` in sample count.
- Raw RMS: rest \(2.8\times10^{-5}\), fist \(2.0\times10^{-4}\). Rest is
  clearly quieter, so Experiment 2 is honestly **hand at rest vs fist**.
- Pretrained checkpoint loaded with **0 missing and 0 unexpected keys** into
  `TinyMyo(in_chans=8, img_size=1000, task="pretraining")`.
- Encoder: 3,570,452 parameters, all frozen. Embedding `(N, 1536)`.
- FAST_DEMO cache: 625 windows.

On the FAST_DEMO subset, a headless run found (do not treat as a paper result):

- Random window split ≈ 97% for **both** systems, with the **same people**
  in train and test.
- Held-out subject split: handcrafted ≈ 90.7%, TinyMyo linear probe ≈ 90.2%.
- At 5% labelled train windows, handcrafted remained higher on average.
  The foundation-model probe did **not** win. That result is kept.

## What this demo is not

- Not a claim that TinyMyo “removes crosstalk”.
- Not a claim that a random window split is cross-user generalisation.
- Not a claim that PCA clusters prove a better representation.
- Not an API wrapper around an unsupported Hugging Face `AutoModel`.
