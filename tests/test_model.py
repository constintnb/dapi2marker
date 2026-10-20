import pytest
import torch

from dapi2marker.checkpoint import load_weights, save_weights
from dapi2marker.models.factory import ModelSpec, create_model, parameter_count
from dapi2marker.models.vae import VAE


def test_exact_parameter_budget():
    small = parameter_count(ModelSpec())
    baseline = parameter_count(
        ModelSpec(base_channels=256, time_dim=1024, spade_hidden_channels=128)
    )
    assert (small, baseline) == (23560572, 236779852)
    assert 0.099 < small / baseline < 0.1


def test_small_forward_backward():
    model = create_model(ModelSpec())
    noise, condition = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    output = model(noise, condition, torch.tensor([5]))
    assert output.shape == noise.shape
    output.square().mean().backward()
    assert torch.isfinite(model.inp.conv1[1].weight.grad).all()


@pytest.mark.parametrize("size", [(8, 8), (9, 11)])
def test_cached_condition_and_optional_global_film(tiny_spec, size):
    spec = ModelSpec(**{**tiny_spec.__dict__, "global_condition_hidden": 8})
    model = create_model(spec).eval()
    noise, condition = torch.randn(1, 4, *size), torch.randn(1, 4, *size)
    time = torch.tensor([1])
    with torch.no_grad():
        direct = model(noise, condition, time)
        cached = model(noise, model.encode_condition(condition), time)
    assert direct.shape == noise.shape
    torch.testing.assert_close(direct, cached, rtol=0, atol=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"base_channels": 79},
        {"time_dim": 3},
        {"attention_heads": 7},
        {"global_condition_hidden": 8, "condition_channels": None},
    ],
)
def test_invalid_architecture_rejected(changes):
    with pytest.raises(ValueError):
        ModelSpec(**changes)


def test_checkpoint_roundtrip_and_architecture_guard(tmp_path, tiny_spec):
    model = create_model(tiny_spec)
    path = tmp_path / "model.pt"
    save_weights(model, tiny_spec, path, {"target": "CD45RO"})
    restored, metadata = load_weights(path, expected_spec=tiny_spec)
    assert metadata["target"] == "CD45RO"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="architecture"):
        load_weights(path, expected_spec=ModelSpec())


def test_frozen_vae_decode_keeps_input_gradient(tiny_vae):
    vae = VAE(tiny_vae)
    vae.train()
    assert not vae.vae.training
    assert not any(param.requires_grad for param in vae.parameters())
    image = torch.randn(1, 1, 32, 32)
    latent = vae.encode(image).detach().requires_grad_()
    decoded = vae.decode(latent)
    assert decoded.shape == image.shape and decoded.dtype == torch.float32
    decoded.square().mean().backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0
    assert all(param.grad is None for param in vae.parameters())
