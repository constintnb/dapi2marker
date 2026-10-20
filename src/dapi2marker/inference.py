"""Standalone image prediction from inference weights and a matching local VAE."""

import hashlib
from pathlib import Path
import pickle
import tempfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .checkpoint import load_weights
from .config import resolve_device, resolve_precision
from .diffusion.sampling import sample_images, sampling_times
from .diffusion.schedule import DiffusionScheduler
from .models.vae import VAE, vae_fingerprint


def _find_vae(checkpoint, metadata, requested):
    if requested is not None:
        return Path(requested).expanduser().resolve()
    candidates = [checkpoint.parent / "sd-vae-ft-mse", Path.cwd() / "weights/sd-vae-ft-mse"]
    recorded = metadata.get("vae", {}).get("path")
    if recorded:
        candidates.append(Path(recorded).expanduser())
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(
        "Matching local VAE not found. Place it in weights/sd-vae-ft-mse "
        "or provide --vae /path/to/vae. "
        "The project downloader is: dapi2marker download-vae --config configs/small.yaml"
    )


def _read_image(path):
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Input must be a single image, not an animation or image stack.")
        if image.mode in {"I", "F"} or image.mode.startswith("I;16"):
            raise ValueError(
                "Input must be an 8-bit image. Convert high-bit-depth/float data "
                "to the training intensity range before prediction."
            )
        pixels = np.array(image.convert("L"), copy=True)
    height, width = pixels.shape
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    sample_id = f"image:{width}x{height}:{digest}"
    tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0).float() / 127.5 - 1
    return tensor, sample_id


def _save_png(image, output, overwrite):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            image.save(handle, format="PNG")
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output}; use --overwrite to replace it."
            )
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@torch.no_grad()
def predict_image(
    checkpoint,
    input_path,
    output_path,
    *,
    vae_path=None,
    steps=None,
    draws=8,
    seed=42026,
    device="auto",
    precision="auto",
    overwrite=False,
):
    """Predict one 8-bit DAPI image without a config, labels, or a split manifest."""
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps must be a positive integer.")
    checkpoint = Path(checkpoint).expanduser().resolve()
    input_path = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output == input_path:
        raise ValueError("Input and output paths must be different.")
    if output.suffix.lower() != ".png":
        raise ValueError("Output must use the .png extension for a lossless grayscale image.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it.")
    inputs, sample_id = _read_image(input_path)
    device = resolve_device(device)
    precision = resolve_precision(device, precision)
    if device.type == "cpu":
        torch.set_num_threads(2)
    try:
        model, metadata = load_weights(checkpoint, device=device)
    except pickle.UnpicklingError as error:
        raise ValueError(
            "Use a self-describing inference checkpoint: model_final.pt, model_best.pt, "
            "or model_last.pt. Training-state checkpoint_last.pt is not supported."
        ) from error
    diffusion = metadata.get("diffusion", {})
    timesteps = diffusion.get("timesteps")
    if type(timesteps) is not int or timesteps < 2:
        raise ValueError("Checkpoint must contain diffusion.timesteps >= 2.")
    steps = diffusion.get("ddim_steps", min(50, timesteps)) if steps is None else steps
    scheduler = DiffusionScheduler(timesteps, device)
    sampling_times(scheduler, steps)
    expected_vae = metadata.get("vae_fingerprint")
    if not isinstance(expected_vae, str) or not expected_vae:
        raise ValueError("Checkpoint must contain a VAE fingerprint for compatible image decoding.")
    vae_path = _find_vae(checkpoint, metadata, vae_path)
    actual_vae = vae_fingerprint(vae_path)
    if actual_vae != expected_vae:
        raise ValueError(
            "VAE fingerprint does not match the checkpoint; select the matching --vae."
        )
    vae = VAE(vae_path, device=device)
    height, width = inputs.shape[-2:]
    factor = vae.downsample_factor
    padded_height = max(4 * factor, ((height + factor - 1) // factor) * factor)
    padded_width = max(4 * factor, ((width + factor - 1) // factor) * factor)
    inputs = F.pad(inputs, (0, padded_width - width, 0, padded_height - height), mode="replicate")
    prediction = sample_images(
        model,
        vae,
        scheduler,
        vae.encode(inputs.to(device)),
        sample_ids=[sample_id],
        seed=seed,
        steps=steps,
        draws=draws,
        precision=precision,
    )
    prediction = prediction[0, 0, :height, :width]
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Image prediction contains non-finite values.")
    pixels = prediction.mul(255).round().to(torch.uint8).cpu().numpy()
    _save_png(Image.fromarray(pixels), output, overwrite)
    return {
        "checkpoint": str(checkpoint),
        "input": str(input_path),
        "output": str(output),
        "target": metadata.get("target"),
        "image_size": [width, height],
        "padded_size": [padded_width, padded_height],
        "vae_path": str(vae_path),
        "vae_fingerprint": actual_vae,
        "device": str(device),
        "precision": precision,
        "steps": steps,
        "draws": draws,
        "seed": seed,
        "sample_id": sample_id,
    }
