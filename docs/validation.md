# Validation record

Validated on 2026-10-10 with Python 3.12.14, macOS arm64, CPU FP32.
The CPU test suite was rechecked on 2026-10-20: **41 tests passed**;
Ruff lint/format checks and `pip check` also passed.
Completed single-device CUDA/bf16 results were reviewed on 2026-10-20 and are
summarized below. Distributed training and throughput/memory benchmarks remain
unvalidated.

| Dependency | Tested version |
| --- | --- |
| PyTorch | 2.14.1 |
| Diffusers | 0.41.0 |
| Accelerate | 1.15.0 |
| NumPy | 2.5.3 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |
| scikit-image | 0.26.0 |
| pytest | 9.1.1 |
| Ruff | 0.17.0 |
| build / setuptools / wheel | 1.6.1 / 84.0.0 / 0.48.0 |

Automated checks cover exact parameter counts, a real small-model backward pass,
odd spatial sizes, optional global conditioning, condition caching, VAE decode
gradients, deterministic sampling, the SSIM reference, ROI/tissue metric
weighting, complete fold coverage and isolation, missing pairs, source name
collisions, crop neighbor isolation, checkpoint architecture guards, interrupted
resume equivalence, CLI use outside the repository, and the full three-fold
subprocess workflow followed by a fresh full-data model.

CUDA setup adds `requirements-cu128.txt` (PyTorch 2.11.0/cu128),
`configs/small_cuda.yaml`, `configs/smoke_cuda.yaml`, and `doctor --require-cuda`.
The new tests cover CPU precision fallback, native-bf16 selection, failed-kernel
reporting, strict automatic-precision resume signatures, unchanged model/fold
configuration, and failure before creating a training directory when CUDA is
unavailable. CUDA-dependent branches use mocks in this CPU suite; they do not
establish CUDA execution or performance. A local real `doctor` invocation
reported CPU-only PyTorch, and `--require-cuda` correctly returned exit code 1.
The official cu128 index was checked for a Python 3.13/Linux x86_64 wheel.
The installation profile is separate from the returned training artifacts;
those artifacts do not contain a complete package-version or GPU model report.

A separate real-data smoke run used `configs/smoke.yaml`: the actual 23,560,572
parameter small model, local pretrained VAE, four train tiles, two validation
tiles, two optimizer steps, two DDIM steps and one draw. Training and validation
completed, self-describing weights were saved, and standalone CLI evaluation
reproduced the final validation scores. This verifies the code path; it does
not measure final model quality.

The included data manifest has 8,676 labeled tiles from 42 source-specific ROIs
and 1,906 original test inputs. Its membership fingerprint is:

```text
55ad62316a6fd7b83bde13ac14a2563ffad495f8312854b381d7bcb82dd637b1
```

The local VAE config/weights fingerprint used for the smoke run is:

```text
9037a4e38afc665628bce1736c5ffa7827af09650f2be49615b1ba1677e4415d
```

Project inference weights record architecture and provenance metadata in a
self-describing checkpoint format.

## Returned CUDA training results

The returned `runs/small_cuda/` artifacts contain completed three-fold runs and
a separate full-data run. All four used 136,800 optimizer steps, batch size 8,
one device, and bf16. The full model used all 8,676 labeled tiles without a
validation set. Each fold selected EMA weights using macro S50, evaluated with
50 DDIM steps and eight decoded draws per input. The best steps were 130,000,
136,800, and 130,000; mean fold S50 was 66.8384 with sample standard deviation
0.8263. See the [README](../README.md#reference-results) and
[reference metrics](reference_results.json) for all reported values.

The returned OOF CSV was checked to cover all 8,676 labeled samples exactly once.
Its reaggregated metrics match the JSON report; each fold's validation IDs match
the fixed manifest, and its training and validation ROI groups are disjoint.
The four train logs have complete, finite records at every 100 steps. Three
best-fold weights and the final full-data weights were loaded with
`weights_only=True`; all tensors are finite and the recorded step, fold/full,
manifest fingerprint, and VAE fingerprint agree with the result files.

These checks inspect returned files and recompute metrics from saved per-tile
scores. CUDA inference was not repeated locally. The package-version table
above describes local CPU validation, and the cu128 installation profile does
not establish the exact packages used by the returned training runs. The
reference scores are validation-checkpoint selection results; independent
external test performance and patient-level generalization remain unevaluated.
