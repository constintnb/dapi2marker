"""One prediction path shared by validation and command-line inference."""

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint import load_weights
from .config import resolve_device, resolve_precision
from .data.dataset import PairedDataset
from .data.manifest import read_manifest, select_samples
from .diffusion.sampling import sample_images
from .diffusion.schedule import DiffusionScheduler
from .metrics import aggregate_metrics, image_metrics, write_rows
from .models.vae import VAE, vae_fingerprint


@torch.no_grad()
def evaluate_loader(
    model,
    vae,
    scheduler,
    loader,
    dataset,
    *,
    device,
    seed,
    steps,
    draws,
    precision="no",
    accelerator=None,
    output_dir=None,
):
    model.eval()
    rows, predicted = [], 0
    main = accelerator is None or accelerator.is_main_process
    for batch in tqdm(loader, desc="Predict", disable=not main):
        inputs = batch["dapi"].to(device)
        prediction = (
            sample_images(
                model,
                vae,
                scheduler,
                vae.encode(inputs),
                sample_ids=batch["sample_id"],
                seed=seed,
                steps=steps,
                draws=draws,
                precision=precision,
            )
            .mul(255)
            .round()
            .to(torch.uint8)
        )
        indices = batch["sample_index"].to(device)
        targets = (
            ((batch["targets"].to(device) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
            if "targets" in batch
            else None
        )
        if accelerator is not None:
            gathered = (
                accelerator.gather_for_metrics((indices, prediction, targets))
                if targets is not None
                else accelerator.gather_for_metrics((indices, prediction))
            )
            indices, prediction = gathered[:2]
            targets = gathered[2] if targets is not None else None
        if not main:
            continue
        for offset, index in enumerate(indices.cpu().tolist()):
            record = dataset.records[index]
            image = prediction[offset, 0].cpu().numpy()
            predicted += 1
            if output_dir is not None:
                path = (
                    Path(output_dir)
                    / record["tissue"]
                    / Path(record["filename"]).with_suffix(".png")
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(path)
            if targets is not None:
                target = targets[offset, 0].cpu().numpy()
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "tissue": record["tissue"],
                        "group_id": record["group_id"],
                        **image_metrics(target, image),
                    }
                )
    return (aggregate_metrics(rows) if rows else {"predicted_tiles": predicted}), rows


def predict_run(
    config, checkpoint, *, fold=0, split="val", full=False, output=None, save_images=True
):
    device = resolve_device(config["runtime"]["device"])
    precision = resolve_precision(device, config["inference"]["mixed_precision"])
    if device.type == "cpu":
        torch.set_num_threads(config["runtime"].get("cpu_threads", 2))
    manifest = read_manifest(config["paths"]["manifest"], sources=config["data"]["sources"])
    model, metadata = load_weights(checkpoint, device=device, expected_spec=config["model"])
    expected = {"target": manifest["target"], "manifest_fingerprint": manifest["fingerprint"]}
    expected["vae_fingerprint"] = vae_fingerprint(config["vae"]["path"])
    expected["diffusion"] = config["diffusion"]
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match the selected data.")
    if split == "val" and (metadata.get("full") or metadata.get("fold") != fold):
        raise ValueError("Validation requires a checkpoint trained on this fold's training ROIs.")
    if config["data"]["target"] != manifest["target"]:
        raise ValueError("Configured target does not match the manifest.")
    records = select_samples(manifest, split=split, fold=fold, full=full)
    dataset = PairedDataset(
        records, limit=config["data"].get("max_val_samples") if split == "val" else None
    )
    loader = DataLoader(
        dataset,
        batch_size=config["inference"]["batch_size"],
        num_workers=config["inference"]["num_workers"],
        shuffle=False,
    )
    vae = VAE(config["vae"]["path"], device=device)
    scheduler = DiffusionScheduler(config["diffusion"]["timesteps"], device)
    output = (
        Path(output) if output is not None else Path(checkpoint).parent / f"predictions_{split}"
    )
    if save_images and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prediction output is not empty: {output}; select a new --output.")
    report, rows = evaluate_loader(
        model,
        vae,
        scheduler,
        loader,
        dataset,
        device=device,
        seed=config["inference"]["seed"],
        steps=config["diffusion"]["ddim_steps"],
        draws=config["inference"]["num_samples_per_input"],
        precision=precision,
        output_dir=output if save_images else None,
    )
    report.update(
        {
            "checkpoint": str(Path(checkpoint).resolve()),
            "split": split,
            "precision": precision,
            "manifest_fingerprint": manifest["fingerprint"],
            "sampling": {
                "seed": config["inference"]["seed"],
                "steps": config["diffusion"]["ddim_steps"],
                "draws": config["inference"]["num_samples_per_input"],
            },
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    from .training import write_json

    write_json(report, output / "metrics.json")
    if rows:
        write_rows(rows, output / "per_tile.csv")
    return report
