from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dapi2marker.cli import main
from dapi2marker.config import autocast_context, load_config, resolve_precision
from dapi2marker import diagnostics
from dapi2marker.training import _signature, train_run


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("requested", ["auto", "no", "fp16", "bf16"])
def test_cpu_precision_never_probes_cuda(requested, monkeypatch):
    def unexpected_probe(*args, **kwargs):
        raise AssertionError("CPU precision must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", unexpected_probe)
    device = torch.device("cpu")
    assert resolve_precision(device, requested) == "no"
    with autocast_context(device, requested):
        assert not torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize("native_support, expected", [(True, "bf16"), (False, "fp16")])
def test_auto_precision_requires_native_bf16(native_support, expected, monkeypatch):
    probes = []

    def supported(including_emulation=True):
        probes.append(including_emulation)
        return native_support if not including_emulation else True

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    assert resolve_precision(torch.device("cuda"), "auto") == expected
    assert probes == [False]


def test_explicit_unsupported_bf16_fails_early(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimeError, match="does not support bf16"):
        resolve_precision(torch.device("cuda"), "bf16")


@pytest.mark.parametrize(
    "available, usable, require_cuda, expected",
    [(False, False, False, 0), (True, False, True, 1), (True, True, True, 0)],
)
def test_doctor_cli_requires_usable_kernels_without_config(
    available, usable, require_cuda, expected, monkeypatch, capsys
):
    report = {"cuda_available": available, "cuda_usable": usable}
    monkeypatch.setattr(diagnostics, "runtime_report", lambda: report)
    arguments = ["doctor", "--require-cuda"] if require_cuda else ["doctor"]
    assert main(arguments) == expected
    assert json.loads(capsys.readouterr().out) == report


def test_doctor_reports_kernel_failure_when_cuda_is_available(monkeypatch):
    monkeypatch.setattr(diagnostics, "_driver_report", lambda: {"available": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(name="test GPU", total_memory=24 * 2**30),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: True)

    def incompatible_kernel(*args, **kwargs):
        raise RuntimeError("no kernel image is available for execution on the device")

    monkeypatch.setattr(torch, "ones", incompatible_kernel)
    report = diagnostics.runtime_report()
    assert report["cuda_available"] and not report["cuda_usable"]
    assert report["devices"][0]["selected_precision"] == "bf16"
    assert not report["devices"][0]["kernel_check_passed"]
    assert "no kernel image" in report["devices"][0]["error"]


def test_gpu_presets_use_same_model_and_folds_with_independent_outputs():
    small = load_config(ROOT / "configs" / "small.yaml")
    gpu = load_config(ROOT / "configs" / "small_cuda.yaml")
    smoke = load_config(ROOT / "configs" / "smoke_cuda.yaml")
    assert gpu["model"] == small["model"] == smoke["model"]
    assert gpu["data"] == small["data"]
    assert gpu["paths"]["manifest"] == smoke["paths"]["manifest"]
    assert gpu["runtime"]["device"] == smoke["runtime"]["device"] == "cuda"
    assert gpu["train"]["mixed_precision"] == smoke["train"]["mixed_precision"] == "auto"
    assert gpu["inference"]["mixed_precision"] == smoke["inference"]["mixed_precision"] == "auto"
    assert (
        len(
            {gpu["paths"]["output_dir"], smoke["paths"]["output_dir"], small["paths"]["output_dir"]}
        )
        == 3
    )
    assert smoke["train"]["max_steps"] == 2


def test_auto_precision_is_part_of_resume_signature():
    config = load_config(ROOT / "configs" / "small_cuda.yaml")
    accelerator = SimpleNamespace(num_processes=1, device=torch.device("cuda"))
    arguments = (config, {"fingerprint": "data"}, 0, False, accelerator, "vae")
    assert _signature(*arguments, "bf16") != _signature(*arguments, "fp16")
    config["train"]["mixed_precision"] = "bf16"
    assert _signature(*arguments, "bf16") == _signature(*arguments)


def test_gpu_training_fails_before_writing_outputs_when_cuda_missing(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs" / "smoke_cuda.yaml")
    output = tmp_path / "gpu_runs"
    config["paths"]["output_dir"] = str(output)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        train_run(config)
    assert not output.exists()
