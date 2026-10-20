"""Frozen FP32 VAE with gradients retained through decoding during training."""

from pathlib import Path
import hashlib

import torch
from torch import nn
from diffusers import AutoencoderKL


def vae_fingerprint(source):
    """Fingerprint local config/weights, independent of their absolute path."""
    source = Path(source)
    files = [source / "config.json", *sorted(source.glob("*.safetensors"))]
    if len(files) < 2:
        raise FileNotFoundError(f"VAE config/safetensors are missing: {source}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class VAE(nn.Module):
    def __init__(self, source, *, device="cpu", revision=None, local_files_only=True):
        super().__init__()
        if local_files_only and not Path(source).is_dir():
            raise FileNotFoundError(
                f"VAE directory not found: {source}. Run dapi2marker download-vae."
            )
        self.vae = AutoencoderKL.from_pretrained(
            str(source),
            revision=revision,
            local_files_only=local_files_only,
            use_safetensors=True,
            torch_dtype=torch.float32,
        ).to(device)
        if self.vae.config.latent_channels != 4:
            raise ValueError("This UNet requires a four-channel VAE latent.")
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.scaling_factor = self.vae.config.scaling_factor
        self.downsample_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def train(self, mode=True):
        super().train(False)
        return self

    def encode(self, image):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        with torch.autocast(device_type=image.device.type, enabled=False):
            return self.vae.encode(image.float()).latent_dist.mode() * self.scaling_factor

    def decode(self, latent):
        with torch.autocast(device_type=latent.device.type, enabled=False):
            result = self.vae.decode(latent.float() / self.scaling_factor).sample
        return result.mean(dim=1, keepdim=True)
