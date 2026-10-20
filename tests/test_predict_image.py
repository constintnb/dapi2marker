import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from dapi2marker.checkpoint import load_weights, save_weights
from dapi2marker.cli import main
from dapi2marker.data.dataset import PairedDataset
from dapi2marker.diffusion.sampling import sample_images
from dapi2marker.diffusion.schedule import DiffusionScheduler
from dapi2marker.inference import predict_image
from dapi2marker.models.factory import create_model
from dapi2marker.models.vae import VAE, vae_fingerprint


@pytest.fixture
def image_setup(tmp_path, tiny_spec, tiny_vae):
    weights = tmp_path / "weights"
    weights.mkdir()
    vae = weights / "sd-vae-ft-mse"
    shutil.move(str(tiny_vae), vae)
    checkpoint = weights / "model_final.pt"
    save_weights(
        create_model(tiny_spec),
        tiny_spec,
        checkpoint,
        {
            "target": "CD45RO",
            "full": True,
            "manifest_fingerprint": "unavailable-training-manifest",
            "vae": {"path": "/unavailable-training-server/vae"},
            "vae_fingerprint": vae_fingerprint(vae),
            "diffusion": {"timesteps": 8, "ddim_steps": 2},
        },
    )
    image = tmp_path / "arbitrary-dapi-name.png"
    pixels = np.random.default_rng(23).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image)
    return {"checkpoint": checkpoint, "image": image, "vae": vae, "root": tmp_path}


def test_image_cli_uses_checkpoint_without_config_or_manifest(image_setup, monkeypatch, capsys):
    import dapi2marker.cli as cli
    import dapi2marker.data.manifest as manifest

    def unexpected_data_access(*args, **kwargs):
        raise AssertionError("Standalone prediction must not read training config or manifest")

    monkeypatch.setattr(cli, "load_config", unexpected_data_access)
    monkeypatch.setattr(cli, "read_manifest", unexpected_data_access)
    monkeypatch.setattr(manifest, "read_manifest", unexpected_data_access)
    monkeypatch.chdir(image_setup["root"])
    output = Path("outputs/cd45ro.png")
    assert (
        main(
            [
                "predict-image",
                "--checkpoint",
                "weights/model_final.pt",
                "--input",
                str(image_setup["image"]),
                "--output",
                str(output),
                "--steps",
                "2",
                "--draws",
                "8",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["target"] == "CD45RO"
    assert report["steps"] == 2 and report["draws"] == 8
    assert report["precision"] == "no"
    assert report["vae_path"] == str(image_setup["vae"])
    with Image.open(output) as prediction:
        assert prediction.mode == "L" and prediction.size == (32, 32)
    assert not any((image_setup["root"] / name).exists() for name in ("configs", "ds", "splits"))


def test_image_cli_works_outside_repository_and_without_original_paths(image_setup):
    elsewhere = image_setup["root"] / "elsewhere"
    elsewhere.mkdir()
    output = elsewhere / "prediction.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dapi2marker",
            "predict-image",
            "--checkpoint",
            str(image_setup["checkpoint"]),
            "--input",
            str(image_setup["image"]),
            "--output",
            str(output),
            "--draws",
            "1",
            "--device",
            "cpu",
        ],
        cwd=elsewhere,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["steps"] == 2 and report["vae_path"] == str(image_setup["vae"])
    assert output.is_file()


def test_image_prediction_matches_shared_sampler_and_training_preprocessing(image_setup):
    output = image_setup["root"] / "prediction.png"
    report = predict_image(
        image_setup["checkpoint"], image_setup["image"], output, draws=2, device="cpu"
    )
    dataset = PairedDataset(
        [
            {
                "dapi_path": image_setup["image"].name,
                "target_path": None,
                "source_root": str(image_setup["root"]),
                "sample_id": report["sample_id"],
                "tissue": "custom",
                "group_id": "custom:image",
            }
        ]
    )
    model, _ = load_weights(image_setup["checkpoint"])
    vae = VAE(image_setup["vae"])
    with torch.no_grad():
        expected = (
            sample_images(
                model,
                vae,
                DiffusionScheduler(8),
                vae.encode(dataset[0]["dapi"][None]),
                sample_ids=[report["sample_id"]],
                seed=42026,
                steps=2,
                draws=2,
            )[0, 0]
            .mul(255)
            .round()
            .to(torch.uint8)
            .numpy()
        )
    with Image.open(output) as prediction:
        np.testing.assert_array_equal(np.array(prediction), expected)
    copied = image_setup["root"] / "renamed.png"
    shutil.copyfile(image_setup["image"], copied)
    second = output.with_name("second.png")
    repeated = predict_image(image_setup["checkpoint"], copied, second, draws=2, device="cpu")
    assert repeated["sample_id"] == report["sample_id"]
    assert second.read_bytes() == output.read_bytes()


@pytest.mark.parametrize("size", [(35, 27), (7, 9)])
def test_image_padding_preserves_original_dimensions(image_setup, size):
    Image.fromarray(np.full((size[1], size[0]), 42, dtype=np.uint8)).save(image_setup["image"])
    output = image_setup["root"] / "odd.png"
    report = predict_image(
        image_setup["checkpoint"], image_setup["image"], output, draws=1, device="cpu"
    )
    assert report["image_size"] == list(size)
    assert all(value % 4 == 0 and value >= 16 for value in report["padded_size"])
    with Image.open(output) as prediction:
        assert prediction.size == size


@pytest.mark.parametrize(
    "options, message",
    [
        ({"steps": 0}, "steps"),
        ({"steps": 9}, "DDIM steps"),
        ({"draws": 0}, "draws"),
        ({"seed": -1}, "seed"),
    ],
)
def test_image_prediction_rejects_invalid_sampling_without_output(image_setup, options, message):
    output = image_setup["root"] / "invalid.png"
    with pytest.raises(ValueError, match=message):
        predict_image(
            image_setup["checkpoint"], image_setup["image"], output, device="cpu", **options
        )
    assert not output.exists()


def test_image_prediction_requires_matching_vae(image_setup):
    config_path = image_setup["vae"] / "config.json"
    config = json.loads(config_path.read_text())
    config["scaling_factor"] = 0.25
    config_path.write_text(json.dumps(config))
    output = image_setup["root"] / "mismatch.png"
    with pytest.raises(ValueError, match="VAE fingerprint"):
        predict_image(image_setup["checkpoint"], image_setup["image"], output, device="cpu")
    assert not output.exists()


def test_explicit_vae_path_overrides_auto_discovery(image_setup, monkeypatch):
    monkeypatch.chdir(image_setup["root"])
    moved = image_setup["root"] / "custom_vae"
    shutil.move(str(image_setup["vae"]), moved)
    output = image_setup["root"] / "explicit.png"
    with pytest.raises(FileNotFoundError, match="--vae"):
        predict_image(image_setup["checkpoint"], image_setup["image"], output, device="cpu")
    report = predict_image(
        image_setup["checkpoint"],
        image_setup["image"],
        output,
        vae_path=moved,
        draws=1,
        device="cpu",
    )
    assert report["vae_path"] == str(moved) and output.exists()


@pytest.mark.parametrize("missing", ["diffusion", "vae_fingerprint"])
def test_image_prediction_requires_sampling_and_vae_metadata(image_setup, missing):
    payload = torch.load(image_setup["checkpoint"], map_location="cpu", weights_only=True)
    payload["metadata"].pop(missing)
    torch.save(payload, image_setup["checkpoint"])
    output = image_setup["root"] / "missing.png"
    with pytest.raises(ValueError, match="diffusion|VAE fingerprint"):
        predict_image(image_setup["checkpoint"], image_setup["image"], output, device="cpu")
    assert not output.exists()


def test_output_protection_and_explicit_overwrite(image_setup):
    output = image_setup["root"] / "existing.png"
    output.write_bytes(b"existing result")
    with pytest.raises(FileExistsError):
        predict_image(image_setup["checkpoint"], image_setup["image"], output, device="cpu")
    assert output.read_bytes() == b"existing result"
    original = image_setup["image"].read_bytes()
    with pytest.raises(ValueError, match="different"):
        predict_image(
            image_setup["checkpoint"],
            image_setup["image"],
            image_setup["image"],
            device="cpu",
            overwrite=True,
        )
    assert image_setup["image"].read_bytes() == original
    predict_image(
        image_setup["checkpoint"],
        image_setup["image"],
        output,
        device="cpu",
        draws=1,
        overwrite=True,
    )
    with Image.open(output) as prediction:
        assert prediction.mode == "L"
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_high_bit_depth_and_invalid_image_fail_cleanly(image_setup, capsys):
    output = image_setup["root"] / "invalid.png"
    Image.fromarray(np.full((32, 32), 50000, dtype=np.uint16)).save(image_setup["image"])
    with pytest.raises(ValueError, match="8-bit"):
        predict_image(image_setup["checkpoint"], image_setup["image"], output, device="cpu")
    image_setup["image"].write_bytes(b"not an image")
    assert (
        main(
            [
                "predict-image",
                "--checkpoint",
                str(image_setup["checkpoint"]),
                "--input",
                str(image_setup["image"]),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err
    assert not output.exists()


def test_training_state_is_rejected_without_unsafe_pickle_loading(image_setup):
    state = image_setup["root"] / "checkpoint_last.pt"
    torch.save({"rng": np.random.get_state()}, state)
    output = image_setup["root"] / "unsafe.png"
    with pytest.raises(ValueError, match="inference checkpoint"):
        predict_image(state, image_setup["image"], output, device="cpu")
    assert not output.exists()
