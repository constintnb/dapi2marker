"""YAML presets, portable paths, and experiment validation."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

from .models.factory import ModelSpec


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def _read(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    if path in stack:
        raise ValueError(f"Circular configuration inheritance: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    parent = config.pop("extends", None)
    base = _read((path.parent / parent).resolve(), (*stack, path)) if parent else {}

    def resolve(value):
        candidate = Path(value).expanduser()
        return str(candidate if candidate.is_absolute() else (path.parent / candidate).resolve())

    for key, value in config.get("paths", {}).items():
        if value is not None:
            config["paths"][key] = resolve(value)
    if "sources" in config.get("data", {}):
        config["data"]["sources"] = {
            key: resolve(value) for key, value in config["data"]["sources"].items()
        }
    if config.get("vae", {}).get("path") is not None:
        config["vae"]["path"] = resolve(config["vae"]["path"])
    return _merge(base, config)


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    config = _read(path)
    config["_config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config):
    for section in (
        "runtime",
        "paths",
        "model",
        "vae",
        "data",
        "diffusion",
        "train",
        "evaluation",
        "inference",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    ModelSpec.from_dict(config["model"])
    for key in ("manifest", "output_dir"):
        if not isinstance(config["paths"].get(key), str) or not config["paths"][key]:
            raise ValueError(f"paths.{key} must be a nonempty path.")
    threads = config["runtime"].get("cpu_threads", 2)
    if type(threads) is not int or threads < 1:
        raise ValueError("runtime.cpu_threads must be a positive integer.")
    if config["runtime"].get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("runtime.device must be auto, cpu, or cuda.")
    if config["runtime"].get("matmul_precision", "high") not in {"highest", "high", "medium"}:
        raise ValueError("Invalid matmul precision.")
    for section, keys in {
        "train": ("batch_size", "max_steps", "checkpoint_every_steps", "log_every_steps"),
        "evaluation": ("every_steps", "num_samples_per_input"),
        "inference": ("batch_size", "num_samples_per_input"),
        "diffusion": ("timesteps", "ddim_steps"),
    }.items():
        for key in keys:
            value = config[section].get(key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer.")
    for section in ("train", "inference"):
        if config[section].get("mixed_precision") not in {"auto", "no", "fp16", "bf16"}:
            raise ValueError(f"{section}.mixed_precision must be auto, no, fp16, or bf16.")
        workers = config[section].get("num_workers", 0)
        if type(workers) is not int or workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")
    for key in ("learning_rate", "gradient_clip_norm"):
        value = config["train"].get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"train.{key} must be finite and positive.")
    for key in ("pixel_weight", "ssim_weight"):
        value = config["train"]["loss"].get(key, 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"train.loss.{key} must be finite and nonnegative.")
    decay = config["train"].get("weight_decay")
    if type(decay) not in (int, float) or not math.isfinite(decay) or decay < 0:
        raise ValueError("train.weight_decay must be finite and nonnegative.")
    if not 0 <= config["train"]["ema_decay"] < 1 or not 0 <= config["train"]["warmup_fraction"] < 1:
        raise ValueError("EMA decay and warmup fraction must be in [0,1).")
    if (
        config["diffusion"]["timesteps"] < 2
        or config["diffusion"]["ddim_steps"] > config["diffusion"]["timesteps"]
    ):
        raise ValueError("DDIM steps must be within the diffusion time range.")
    if config["data"].get("sampling") not in {"natural", "tissue_balanced"}:
        raise ValueError("Invalid data.sampling mode.")
    for key in ("max_train_samples", "max_val_samples"):
        value = config["data"].get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"data.{key} must be null or a positive integer.")
    if not config["data"].get("sources") or type(config["data"].get("seed")) is not int:
        raise ValueError("data.sources and an integer data.seed are required.")
    if (
        config["data"]["seed"] < 0
        or type(config["data"].get("folds")) is not int
        or config["data"]["folds"] < 2
    ):
        raise ValueError("data.seed must be nonnegative and data.folds must be >= 2.")
    crop = config["data"].get("adjacent_crop", {})
    if crop.get("enabled") and not set(crop.get("tissues", ("liver",))).issubset(
        config["data"]["sources"]
    ):
        raise ValueError("Adjacent crop tissue IDs must be declared data.sources.")
    for section in ("train", "evaluation", "inference"):
        if type(config[section].get("seed")) is not int or config[section]["seed"] < 0:
            raise ValueError(f"{section}.seed must be a nonnegative integer.")
    if not config["vae"].get("path"):
        raise ValueError("A local vae.path is required; use download-vae to populate it.")


def resolve_device(requested):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def resolve_precision(device, requested):
    """Choose native bf16 or fp16 on CUDA, and FP32 on CPU."""
    import torch

    if requested not in {"auto", "no", "fp16", "bf16"}:
        raise ValueError("Invalid mixed precision mode.")
    if device.type != "cuda":
        return "no"
    with torch.cuda.device(device):
        if requested == "auto":
            return "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"
        if requested == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support bf16; select auto, fp16 or no.")
    return requested


def autocast_context(device, precision):
    from contextlib import nullcontext
    import torch

    precision = resolve_precision(device, precision)
    if precision == "no":
        return nullcontext()
    return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def run_directory(config, fold=0, full=False):
    return Path(config["paths"]["output_dir"]) / ("full" if full else f"fold_{fold}")
