# Experiment protocol and outputs

## Grouping

The supervised pool combines the original train and val within each source,
then splits by `tissue:ROI`. There are 25 colon and 17 liver labeled groups.
Every ROI appears as validation exactly once; its tiles never enter that fold's
training or crop index. Seeded largest-group-first balancing keeps each fold
at 14 ROIs and approximately equal tile totals. These are three-fold CV runs,
not three small holdouts that leave many ROIs unvalidated.

If several ROIs belong to the same patient/specimen, patient-level generalization
requires that identifier and a new group-level manifest. The present source
filenames do not provide patient IDs; ROI-level results must be described as such.

The shared supervised target is CD45RO. Liver-only markers (CD68, HLA-DR,
Vimentin) are not pooled with colon CD45RO. The package checks complete pairs
for its configured target rather than silently dropping missing labels.

## Training

`small.yaml` and `small_cuda.yaml` share the same 23,560,572-parameter trainable
model. Keep the manifest, sampling, loss, VAE, DDIM seeds/draw count, and optimizer
step budget fixed when comparing experiments. Equal expected tissue sampling
prevents colon's 6,296 tiles from overwhelming liver's 2,380 tiles; it uses
sampling with replacement and does not guarantee a 50/50 ratio in each batch.

All augmentation applies identical transforms to input and target. Native-grid
crop neighbors are indexed only from selected training records. Crops fall back
to the anchor when required neighbors are missing. Colon remains on ordinary
paired flips/rotations until its physical grid stride is confirmed. Dataset
workers are recreated each epoch so they receive the updated deterministic epoch.

The VAE is frozen/eval/FP32, but decoding retains the gradient to the predicted
latent for image losses. Disable pixel and SSIM weights together to skip decoding
in the training objective. Validation always uses DDIM and decoded images.

`max_steps` counts completed optimizer updates, not epochs or samples. A fold
and a full run with equal step budgets therefore have different numbers of
data passes. The default 136,800 is an explicit update budget. The completed
reference runs used this budget; each fold selected its best checkpoint at
130,000 or 136,800 steps. See the [reference results](../README.md#reference-results)
before choosing the budget for a new dataset or experiment.

## Selection and reporting

S50 per image is `100 * (0.7 * SSIM + 0.3 * min(PSNR, 50) / 50)` with images
rounded once to grayscale uint8. Average tiles within each ROI, then average
ROIs within tissue, then colon and liver equally. `macro.s50` selects EMA weights.
`micro` is also reported for comparison but does not select checkpoints.
Infinite PSNR for a perfect reconstruction is stored as 100 dB in reports;
the S50 contribution remains capped at 50 dB.

CV writes both the mean and sample standard deviation of the three fold macro
scores, plus pooled out-of-fold metrics across 8,676 tiles. The pooled result
and mean fold score can differ slightly because folds have different group
counts by tissue. `data.max_val_samples` makes a run a smoke check, so it does
not claim complete OOF coverage. No score from a smoke run establishes quality.

The best score across repeated checkpoint evaluations is a model-selection
estimate. Original unlabeled test data cannot supply an independent accuracy
estimate; labeled external test data would be needed for that purpose.

## Files

For single images outside the dataset, use `predict-image` as shown in the
[README](../README.md#predict-a-single-image). It reads checkpoint architecture,
diffusion metadata, and a matching local VAE without reading a training YAML or
manifest. It produces one PNG and prints a JSON settings report. `predict` and
`evaluate` retain their fixed-split fingerprint and validation-fold checks.
Single-image sampling IDs depend on grayscale pixel contents and dimensions;
fixed-split sampling IDs depend on tissue/ROI/tile identity. The same numeric
seed therefore need not reproduce a reference tile's exact fixed-split output.

Each fold uses `runs/<preset>/fold_0/`, `fold_1/`, or `fold_2/`. The full model
uses `runs/<preset>/full/` and starts with a fresh model, not a CV checkpoint.

| File | Purpose |
| --- | --- |
| `resolved_config.json` | Effective absolute paths and settings |
| `train.jsonl` | Step, losses, and LR at the configured logging interval |
| `checkpoint_last.pt` | Online/EMA model, AdamW, LR scheduler, AMP scaler and per-process RNG for resume |
| `model_last.pt` | Latest self-describing EMA inference weights |
| `model_best.pt` | Best validation macro S50 EMA weights (CV only) |
| `validation_best.json`, `.csv` | Best validation aggregation and per-tile scores |
| `validation_last.json` | Most recent validation aggregation |
| `model_final.pt` | Final EMA inference weights (full training only) |
| `result.json` | Final step, sizes, fingerprints, and best validation step |
| `cross_validation.json`, `out_of_fold.csv` | Combined report at the preset's output root |

Runs refuse to overwrite a populated run directory. Resume individual folds
with the same configuration and budget; use a new output directory for a new
experiment. The cross-validation wrapper launches each fold in its own Python
process, so model/optimizer/RNG state does not carry between folds. If interrupted,
resume the individual run; the wrapper deliberately refuses to restart occupied
runs. The JSON/CSV outputs remain usable for manual aggregation.

For NVIDIA setup, use the CUDA presets and runtime check described in
[environment notes](environment.md). CUDA presets select native bf16 or fp16
automatically. The resolved precision is recorded in `result.json` and inference
weight metadata; changing it invalidates an automatic-precision resume.

The default evaluation averages eight decoded draws per tile over 50 DDIM
steps. On thousands of validation tiles this can be expensive; a short smoke
configuration uses two steps and one draw. Keep final evaluation settings fixed
when comparing experiments. CPU smoke and tests have been run locally, and
single-device CUDA/bf16 training results are included in the reference record.
Distributed training and throughput/memory benchmarks remain unvalidated.
