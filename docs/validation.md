# Local validation record

Validated on 2026-10-10 with Python 3.12.14, macOS arm64, CPU FP32.
The CPU test suite was rechecked on 2026-10-11: **41 tests passed**;
Ruff lint/format checks and `pip check` also passed.
The release is not benchmarked for CUDA, mixed precision, distributed training,
or predictive accuracy.

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
The intended RTX 4090 server has not been accessed or tested.

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

The baseline factory's 338 state entries have the same names and shapes as
the archived original model. Project inference weights add architecture and
provenance metadata; an old plain state dictionary still needs conversion into
this format before use. Full small-model training has not been run.
