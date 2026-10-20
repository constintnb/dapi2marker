# dapi2marker

Conditional latent diffusion for **DAPI → CD45RO** image translation across
colon and liver tissue. The project provides a configurable UNet, a frozen
pretrained VAE, fixed ROI-level three-fold cross-validation, and full-data training.

This is a training implementation. No accuracy claim is made for the small
model: it must be retrained and compared with the baseline using identical folds.

## Model size

The budget counts **all trainable UNet and condition encoder parameters**.
The pretrained `sd-vae-ft-mse` VAE remains frozen and is excluded from this budget.

| Preset | UNet widths | Time embedding | SPADE hidden | Trainable parameters |
| --- | --- | --- | --- | --- |
| `configs/baseline.yaml` | 256 / 512 / 1024 | 1024 | 128 | 236,779,852 |
| `configs/small.yaml` | 80 / 160 / 320 | 256 | 40 | **23,560,572 (9.95%)** |

Both retain the three-stage condition encoder (32 / 64 / 128), residual block
layout, five attention blocks, and the final up stage without SPADE.
Attention heads, dropout, condition widths, optional global conditioning, and
inference branch scales are configurable. The frozen VAE adds 83,653,863
parameters; total small-model size including it is 107,214,435, or 33.46% of
the baseline total. The 1/10 goal applies to the trainable part, not the total.

## Install

Use Python 3.10+ and install PyTorch for your hardware, then install this project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
dapi2marker --help
```

CPU and CUDA are supported. `runtime.device: auto` selects CUDA when available,
otherwise CPU. CPU uses FP32; CUDA precision supports `auto`, `no`, `fp16`, and
`bf16`. Automatic precision selects native bf16 when supported, otherwise fp16.
Default training is single-device; the implementation uses Accelerate, but
multi-GPU and CUDA execution have not yet been validated for this release.
Compilation is off by default. Paths resolve relative to their declaring YAML,
so commands also work outside the repository when given an absolute config path.

## Conda environment

A portable Python 3.13 bootstrap is provided in
[`environment.project.yml`](environment.project.yml). It matches the Python
series in the supplied Linux base export. Create the environment, install a
hardware-appropriate PyTorch wheel, then install the project:

```bash
conda env create -f environment.project.yml
conda activate dapi2marker
# Install CPU/CUDA PyTorch for your hardware before the project.
python -m pip install -e '.[dev]'
python -m pip check
```

For Linux CPU/CUDA setup and runtime verification, see
[environment notes](docs/environment.md). Python 3.13 requires PyTorch >= 2.6
in the project dependency metadata. The original machine-specific
`environment.yml` is retained locally and excluded from releases. Local
validation used Python 3.12 on macOS CPU; Python 3.13/Linux is added to CI but
has not been validated on the user's training server.

For the supplied **RTX 4090 / CUDA 12.8** server, use the pinned PyTorch 2.11.0
CUDA wheel profile after activating the environment:

```bash
python -m pip install -r requirements-cu128.txt
python -m pip install -e '.[dev]'
python -m pip check
dapi2marker doctor --require-cuda
```

The [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v2110)
provide this CUDA 12.8 build. `doctor` reports package and driver versions, GPU
memory, selected precision, and a real CUDA forward/backward check. It needs no
data or VAE. `--require-cuda` exits unsuccessfully if CUDA is unavailable or a
visible GPU fails the check. The GPU presets require CUDA and keep their outputs
under `runs/small_cuda/` and `runs/smoke_cuda/`.

## Local data and VAE

Keep the two datasets in their existing locations; there is no need to copy or
rename images. Change `data.sources` in your configuration if needed:

```text
ds/                         # colon
  train/{DAPI,CD45RO}/ROIxxx_rr_cc.jpg
  val/{DAPI,CD45RO}/ROIxxx_rr_cc.jpg
  test/DAPI/ROIxxx_rr_cc.jpg
ds_round2/                  # liver; same layout
weights/sd-vae-ft-mse/       # config.json and diffusion_pytorch_model.safetensors
```

PNG/JPEG files are accepted. Each labeled DAPI tile needs a same-name CD45RO
target of the same size; the current datasets contain 256×256 tiles. Input and
target are read as grayscale, mapped to [-1, 1], and encoded with the frozen VAE
using its deterministic posterior mode and configured scaling factor.

Download the VAE if it is not already available locally:

```bash
dapi2marker download-vae --config configs/small.yaml
```

The downloader records the resolved upstream revision. Checkpoints also store a
fingerprint of the actual local VAE config and weights. See
[third-party notices](THIRD_PARTY_NOTICES.md). Dataset images, VAE weights,
generated images, training checkpoints, and the original local experiment
archive (`legacy/`) are excluded from Git and release archives.

## Fixed ROI cross-validation

The included [split manifest](splits/joint_cd45ro.json) pools each dataset's
original **train + val** and assigns every complete `(tissue, ROI)` group to
one of three fixed folds. Original test inputs stay outside training and CV.
Same-named colon/liver tiles remain distinct. The split contains file hashes,
portable relative paths, and a membership fingerprint; it contains no pixels.

```bash
dapi2marker prepare-splits --config configs/small.yaml
dapi2marker plan --config configs/small.yaml
dapi2marker params --config configs/small.yaml
```

`prepare-splits` checks paired files and hashes. It refuses to silently replace
an existing changed manifest; use `--force` only when intentionally changing
the dataset or split. When using different datasets, set a different manifest
path and regenerate it. The provided manifest uses seed 2026.

| Fold | Train tiles | Validation colon | Validation liver | Validation ROIs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 5,767 | 2,209 | 700 | 14 (9 colon + 5 liver) |
| 1 | 5,803 | 2,033 | 840 | 14 (8 colon + 6 liver) |
| 2 | 5,782 | 2,054 | 840 | 14 (8 colon + 6 liver) |
| Full training | **8,676** | — | — | All 42 labeled ROIs |

Original test inputs: 1,346 colon + 560 liver. They have no targets in this
pipeline and therefore yield predictions rather than validation metrics.

## Run

Check the four intended runs without starting training:

```bash
dapi2marker cross-validate --config configs/small.yaml --include-full --dry-run
```

Run three independently initialized models and a fourth model on all labels:

```bash
dapi2marker cross-validate --config configs/small.yaml --include-full
```

Or run/resume one fold and then train all labeled tiles:

```bash
dapi2marker train --config configs/small.yaml --fold 0
dapi2marker train --config configs/small.yaml --fold 0 \
  --resume runs/small/fold_0/checkpoint_last.pt
dapi2marker train --config configs/small.yaml --full --max-steps 100000
```

The starting preset uses 136,800 optimizer steps per run, batch size 8,
AdamW (LR 1e-4), warmup/cosine decay, EMA, v-prediction, pixel MSE and SSIM.
This is an explicit starting budget, not a tuned schedule. CV reports each
fold's best step and their median as a suggestion; full training uses the same
step budget unless `--full-max-steps` or `train --full --max-steps` overrides it.
Full training has no validation score or early stopping. Use CV to choose its
budget before a final run. Each full/CV run starts from fresh initialization.

Training samples colon/liver with equal expected probability. Validation
reports per-tile, per-ROI, and per-tissue PSNR/SSIM/S50; checkpoint selection
uses **equal-tissue, equal-ROI macro S50**. DDIM uses fixed per-tile noise seeds,
50 steps and an average of eight decoded draws by default. For adjacent crops,
only liver is enabled because its 256-pixel grid stride was verified; the
lookup includes tissue identity and only that fold's training tiles.

Evaluate a fold or predict the original test inputs:

```bash
dapi2marker evaluate --config configs/small.yaml --fold 0 \
  --checkpoint runs/small/fold_0/model_best.pt
dapi2marker predict --config configs/small.yaml --split test \
  --checkpoint runs/small/full/model_final.pt --output runs/test_predictions
```

Predictions are saved under separate `colon/` and `liver/` folders. Inference
weights contain their architecture; incompatible presets, data/VAE fingerprints,
or validation folds are rejected. Resume uses local trusted training-state
files and requires the original data, budget, and training settings. Old plain
state dictionaries and baseline-size weights cannot initialize the small model.

## Check and package

```bash
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
python -m build --no-isolation
python scripts/package_source.py
```

For a real-data two-step CPU check, use
`dapi2marker train --config configs/smoke.yaml`; it needs the local data/VAE.
For the NVIDIA server, first run
`dapi2marker train --config configs/smoke_cuda.yaml`, then preview the intended
GPU workflow with
`dapi2marker cross-validate --config configs/small_cuda.yaml --include-full --dry-run`.
Remove `--dry-run` to start the three folds and a fresh full-data run.
Automated tests use generated data and a tiny local VAE and need no download.
The tests also verify the exact real-small-model parameter count and a forward/
backward pass. [Usage notes](docs/usage.md) describe outputs and evaluation limits;
[local validation](docs/validation.md) records the tested environment and smoke run.

`scripts/package_source.py` creates `artifacts/dapi2marker-source.zip` from an
explicit source allowlist. Wheel/sdist files appear in `dist/`. To publish to
your own empty GitHub repository after reviewing the source:

```bash
git init -b main
git add .
git commit -m "Package configurable DAPI-to-CD45RO diffusion model"
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

No repository is published automatically by these setup or training commands.
