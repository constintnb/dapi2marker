"""Fixed-step training, complete-ROI validation, EMA, and strict resume."""

import copy
import json
import math
from pathlib import Path
import random

from accelerate import Accelerator
from accelerate.utils import gather_object, set_seed
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .checkpoint import EMA, atomic_save, save_weights, to_cpu
from .config import resolve_device, resolve_precision, run_directory
from .data.dataset import PairedDataset, make_sampler
from .data.manifest import read_manifest, select_samples, stable_digest
from .diffusion.schedule import DiffusionScheduler
from .evaluation import evaluate_loader
from .losses import reconstruction_losses
from .metrics import write_rows
from .models.factory import ModelSpec, create_model
from .models.vae import VAE, vae_fingerprint


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _rng_state(device):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _signature(config, manifest, fold, full, accelerator, vae_digest, precision=None):
    # Absolute paths can change on another machine. Training choices/data may not.
    choices = copy.deepcopy(
        {key: config[key] for key in ("model", "data", "diffusion", "train", "evaluation")}
    )
    choices["data"].pop("sources", None)
    if config["train"]["mixed_precision"] == "auto":
        choices["train"]["resolved_mixed_precision"] = precision
    return stable_digest(
        {
            "choices": choices,
            "manifest": manifest["fingerprint"],
            "vae_fingerprint": vae_digest,
            "fold": None if full else fold,
            "full": full,
            "world_size": accelerator.num_processes,
            "device_type": accelerator.device.type,
        }
    )


def train_run(config, *, fold=0, full=False, resume=None):
    requested_device = resolve_device(config["runtime"]["device"])
    if requested_device.type == "cpu":
        torch.set_num_threads(config["runtime"].get("cpu_threads", 2))
    torch.set_float32_matmul_precision(config["runtime"].get("matmul_precision", "high"))
    precision = resolve_precision(requested_device, config["train"]["mixed_precision"])
    accelerator = Accelerator(cpu=requested_device.type == "cpu", mixed_precision=precision)
    device = accelerator.device
    if (
        config["train"]["mixed_precision"] == "auto"
        and resolve_precision(device, "auto") != precision
    ):
        raise RuntimeError(
            "Selected distributed CUDA devices require different precision modes; set one supported mode explicitly."
        )
    if accelerator.is_main_process:
        print(f"device={device} precision={precision}", flush=True)
    set_seed(config["train"]["seed"], device_specific=True)
    manifest = read_manifest(config["paths"]["manifest"], sources=config["data"]["sources"])
    if (
        manifest["target"] != config["data"]["target"]
        or manifest["n_folds"] != config["data"]["folds"]
    ):
        raise ValueError("Configuration target/fold count does not match the fixed manifest.")
    directory = run_directory(config, fold, full)
    if resume is None and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Run already exists: {directory}; use --resume or a new output_dir.")
    if accelerator.is_main_process:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(config, directory / "resolved_config.json")
    accelerator.wait_for_everyone()
    train_data = PairedDataset(
        select_samples(manifest, fold=fold, full=full),
        augment=True,
        seed=config["data"]["seed"],
        adjacent_crop=config["data"].get("adjacent_crop"),
        limit=config["data"].get("max_train_samples"),
    )
    val_data = (
        None
        if full
        else PairedDataset(
            select_samples(manifest, split="val", fold=fold),
            limit=config["data"].get("max_val_samples"),
        )
    )
    generator = torch.Generator()
    sampler = make_sampler(train_data, config["data"]["sampling"], generator)
    loader_generator = torch.Generator().manual_seed(config["data"]["seed"])
    loader_options = dict(
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    train_loader = DataLoader(
        train_data, sampler=sampler, shuffle=sampler is None, **loader_options
    )
    val_loader = (
        DataLoader(val_data, shuffle=False, **loader_options) if val_data is not None else None
    )
    spec = ModelSpec.from_dict(config["model"])
    raw_model = create_model(spec, device=device)
    vae = VAE(config["vae"]["path"], device=device)
    for dataset in (train_data, val_data):
        if dataset and any(
            min(record["image_size"]) // vae.downsample_factor < 4 for record in dataset.records
        ):
            raise ValueError(
                "Input images produce a VAE latent smaller than the UNet's 4x4 minimum."
            )
    optimizer = AdamW(
        raw_model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    max_steps = config["train"]["max_steps"]
    warmup = int(max_steps * config["train"]["warmup_fraction"])

    def multiplier(step):
        if step < warmup:
            return step / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, max_steps - warmup))))

    lr_scheduler = LambdaLR(optimizer, multiplier)
    ema = EMA(raw_model, config["train"]["ema_decay"])
    vae_digest = vae_fingerprint(config["vae"]["path"])
    signature = _signature(config, manifest, fold, full, accelerator, vae_digest, precision)
    step, epoch, batch_offset, best_step, best_score = 0, 0, 0, None, None
    resume_rng = None
    resume_scaler = None
    if resume is not None:
        # Training checkpoints contain optimizer and RNG Python state. Only
        # resume local, trusted files; inference uses weights_only=True instead.
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state.get("format_version") != 1 or state.get("signature") != signature:
            raise ValueError(
                "Resume requires identical data, model, budget, settings, VAE config, and device/world size."
            )
        raw_model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        ema.load_state_dict(state["ema"])
        step, epoch, batch_offset = state["step"], state["epoch"], state["batch_offset"]
        best_step, best_score = state["best_step"], state["best_score"]
        resume_rng = state["rng"][accelerator.process_index]
        resume_scaler = state.get("scaler")
        del state
    model, optimizer, train_loader = accelerator.prepare(raw_model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if resume_scaler is not None and accelerator.scaler is not None:
        accelerator.scaler.load_state_dict(resume_scaler)
    if config["train"].get("compile"):
        model = torch.compile(model)
    diffusion = DiffusionScheduler(config["diffusion"]["timesteps"], device)
    metadata = {
        "target": manifest["target"],
        "manifest_fingerprint": manifest["fingerprint"],
        "fold": None if full else fold,
        "full": full,
        "weights": "ema",
        "vae": config["vae"],
        "vae_fingerprint": vae_digest,
        "diffusion": config["diffusion"],
        "training_signature": signature,
        "precision": precision,
    }

    def checkpoint():
        rng = gather_object([_rng_state(device)])
        if accelerator.is_main_process:
            atomic_save(
                {
                    "format_version": 1,
                    "signature": signature,
                    "config": config,
                    "step": step,
                    "epoch": epoch,
                    "batch_offset": batch_offset,
                    "best_score": best_score,
                    "best_step": best_step,
                    "model": to_cpu(raw_model.state_dict()),
                    "optimizer": to_cpu(optimizer.state_dict()),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "ema": to_cpu(ema.shadow),
                    "rng": rng,
                    "scaler": accelerator.scaler.state_dict()
                    if accelerator.scaler is not None
                    else None,
                },
                directory / "checkpoint_last.pt",
            )
            with ema.applied():
                save_weights(
                    raw_model, spec, directory / "model_last.pt", {**metadata, "step": step}
                )
        accelerator.wait_for_everyone()

    while step < max_steps:
        train_data.set_epoch(epoch)
        generator.manual_seed(config["data"]["seed"] + epoch)
        loader_generator.manual_seed(config["data"]["seed"] + epoch)
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        # Rebuild this epoch's deterministic sampler order, then skip completed
        # batches. Its loader generator is independent of diffusion/dropout RNG.
        active_loader = (
            accelerator.skip_first_batches(train_loader, batch_offset)
            if batch_offset
            else train_loader
        )
        if resume_rng is not None:
            _restore_rng(resume_rng, device)
            resume_rng = None
        model.train()
        for batch in active_loader:
            batch_offset += 1
            inputs, targets = batch["dapi"], batch["targets"]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                condition, latent = vae.encode(inputs), vae.encode(targets)
            time = torch.randint(diffusion.timesteps, (len(inputs),), device=device)
            noise = torch.randn_like(latent)
            noisy = diffusion.add_noise(latent, noise, time)
            with accelerator.autocast():
                prediction = model(noisy, condition, time)
            v_loss = F.mse_loss(prediction.float(), diffusion.get_v(latent, noise, time))
            loss_config = config["train"]["loss"]
            if loss_config["pixel_weight"] or loss_config["ssim_weight"]:
                alpha = diffusion.extract(diffusion.alphas_cumprod.sqrt(), time, latent.shape)
                sigma = diffusion.extract((1 - diffusion.alphas_cumprod).sqrt(), time, latent.shape)
                image_loss, _, _ = reconstruction_losses(
                    vae.decode(alpha * noisy - sigma * prediction.float()), targets, loss_config
                )
            else:
                image_loss = v_loss.new_zeros(())
            loss = v_loss + image_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}.")
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), config["train"]["gradient_clip_norm"])
            optimizer.step()
            if accelerator.optimizer_step_was_skipped:
                continue
            lr_scheduler.step()
            ema.update()
            step += 1
            if step % config["train"].get("log_every_steps", 100) == 0 or step == max_steps:
                reduced = (
                    accelerator.reduce(
                        torch.stack([loss.detach(), v_loss.detach(), image_loss.detach()]),
                        reduction="mean",
                    )
                    .cpu()
                    .tolist()
                )
                if accelerator.is_main_process:
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "loss": reduced[0],
                        "v_loss": reduced[1],
                        "image_loss": reduced[2],
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    }
                    with (directory / "train.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    print(f"step={step}/{max_steps} loss={reduced[0]:.5f}", flush=True)
            if val_loader is not None and (
                step % config["evaluation"]["every_steps"] == 0 or step == max_steps
            ):
                with ema.applied():
                    report, rows = evaluate_loader(
                        raw_model,
                        vae,
                        diffusion,
                        val_loader,
                        val_data,
                        device=device,
                        seed=config["evaluation"]["seed"],
                        steps=config["diffusion"]["ddim_steps"],
                        draws=config["evaluation"]["num_samples_per_input"],
                        precision=precision,
                        accelerator=accelerator,
                    )
                    if accelerator.is_main_process:
                        report.update(
                            {"step": step, "manifest_fingerprint": manifest["fingerprint"]}
                        )
                        write_json(report, directory / "validation_last.json")
                        if best_score is None or report["macro"]["s50"] > best_score:
                            best_score, best_step = report["macro"]["s50"], step
                            save_weights(
                                raw_model,
                                spec,
                                directory / "model_best.pt",
                                {**metadata, "step": step},
                            )
                            write_json(report, directory / "validation_best.json")
                            write_rows(rows, directory / "validation_best.csv")
                        print(
                            f"validation step={step} macro S50={report['macro']['s50']:.4f}",
                            flush=True,
                        )
                model.train()
            if step % config["train"]["checkpoint_every_steps"] == 0 or step == max_steps:
                checkpoint()
            if step == max_steps:
                break
        epoch += 1
        batch_offset = 0
    result = {
        "mode": "full" if full else "cv",
        "fold": None if full else fold,
        "step": step,
        "best_step": best_step,
        "best_macro_s50": best_score,
        "train_tiles": len(train_data),
        "val_tiles": len(val_data) if val_data else 0,
        "trainable_parameters": sum(p.numel() for p in raw_model.parameters()),
        "manifest_fingerprint": manifest["fingerprint"],
        "signature": signature,
        "world_size": accelerator.num_processes,
        "precision": precision,
    }
    if accelerator.is_main_process:
        if full:
            with ema.applied():
                save_weights(
                    raw_model, spec, directory / "model_final.pt", {**metadata, "step": step}
                )
        write_json(result, directory / "result.json")
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return result
