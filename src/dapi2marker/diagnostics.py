"""Report the current Python/CUDA runtime without loading data or model weights."""

from importlib.metadata import PackageNotFoundError, version
import platform
import subprocess

import torch

from .config import autocast_context, resolve_precision


def _driver_report():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    return {
        "available": result.returncode == 0,
        "output": result.stdout.strip(),
        "error": result.stderr.strip() or None,
    }


def runtime_report():
    packages = {}
    for name in (
        "torch",
        "diffusers",
        "accelerate",
        "numpy",
        "Pillow",
        "PyYAML",
        "scikit-image",
        "safetensors",
        "huggingface-hub",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    available = torch.cuda.is_available()
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": available,
        "cuda_usable": False,
        "nvidia_smi": _driver_report(),
        "devices": [],
    }
    if not available:
        return report
    for index in range(torch.cuda.device_count()):
        entry = {"index": index, "kernel_check_passed": False}
        try:
            device = torch.device("cuda", index)
            with torch.cuda.device(device):
                properties = torch.cuda.get_device_properties(device)
                entry.update(
                    {
                        "name": properties.name,
                        "vram_gib": round(properties.total_memory / 2**30, 2),
                        "compute_capability": list(torch.cuda.get_device_capability(device)),
                        "native_bf16_supported": torch.cuda.is_bf16_supported(
                            including_emulation=False
                        ),
                        "selected_precision": resolve_precision(device, "auto"),
                    }
                )
                # A tiny real forward/backward detects incompatible wheel/GPU
                # architectures that is_available() alone cannot establish.
                inputs = torch.ones((32, 32), device=device, requires_grad=True)
                with autocast_context(device, entry["selected_precision"]):
                    loss = ((inputs @ inputs).float() / 32).square().mean()
                loss.backward()
                torch.cuda.synchronize(device)
                entry["kernel_check_passed"] = bool(
                    torch.isfinite(loss).item() and torch.isfinite(inputs.grad).all().item()
                )
        except (RuntimeError, AssertionError) as error:
            entry["error"] = str(error)
        report["devices"].append(entry)
    report["cuda_usable"] = bool(report["devices"]) and all(
        item["kernel_check_passed"] for item in report["devices"]
    )
    return report
