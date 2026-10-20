from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from dapi2marker.config import load_config
from dapi2marker.data.manifest import build_manifest
from dapi2marker.models.factory import ModelSpec


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def paired_sources(tmp_path):
    sources = {}
    for tissue_index, tissue in enumerate(("colon", "liver")):
        root = tmp_path / tissue
        sources[tissue] = str(root)
        for split in ("train", "val", "test"):
            (root / split / "DAPI").mkdir(parents=True)
            if split != "test":
                (root / split / "CD45RO").mkdir()
        for roi in range(3):
            for col in range(2):
                split = "val" if roi == 0 and col == 1 else "train"
                name = f"ROI{roi:03}_00_{col:02}.png"
                rng = np.random.default_rng(roi * 10 + col + tissue_index * 100)
                image = rng.integers(0, 255, (32, 32), dtype=np.uint8)
                Image.fromarray(image).save(root / split / "DAPI" / name)
                Image.fromarray(255 - image).save(root / split / "CD45RO" / name)
        Image.fromarray(np.arange(1024, dtype=np.uint8).reshape(32, 32)).save(
            root / "test" / "DAPI" / "ROI099_00_00.png"
        )
    return sources


@pytest.fixture
def tiny_spec():
    return ModelSpec(
        base_channels=8,
        time_dim=16,
        spade_hidden_channels=8,
        condition_channels=(8, 16, 32),
        dropout=0.15,
    )


@pytest.fixture
def tiny_vae(tmp_path):
    from diffusers import AutoencoderKL

    torch.manual_seed(10)
    model = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=(8, 8, 8),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=32,
        scaling_factor=0.18215,
    )
    path = tmp_path / "vae"
    model.save_pretrained(path, safe_serialization=True)
    return path


@pytest.fixture
def experiment(tmp_path, paired_sources, tiny_vae, tiny_spec):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "smoke.yaml")
    config["model"] = tiny_spec.to_dict()
    config["vae"]["path"] = str(tiny_vae)
    config["paths"]["output_dir"] = str(tmp_path / "runs")
    config["paths"]["manifest"] = str(tmp_path / "splits.json")
    config["data"]["sources"] = paired_sources
    build_manifest(paired_sources, config["paths"]["manifest"])
    return config
