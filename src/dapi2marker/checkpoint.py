"""Self-describing inference weights and resumable local training state."""

from contextlib import contextmanager
from pathlib import Path

import torch

from .models.factory import ModelSpec, create_model


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(item) for item in value)
    return value


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_weights(model, spec, path, metadata=None):
    spec = spec if isinstance(spec, ModelSpec) else ModelSpec.from_dict(spec)
    atomic_save(
        {
            "format_version": 1,
            "model_config": spec.to_dict(),
            "state_dict": to_cpu(model.state_dict()),
            "metadata": metadata or {},
        },
        path,
    )


def load_weights(path, *, device="cpu", expected_spec=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or "model_config" not in payload:
        raise ValueError("Expected a dapi2marker inference checkpoint with model_config.")
    spec = ModelSpec.from_dict(payload["model_config"])
    if expected_spec is not None:
        expected_spec = (
            expected_spec
            if isinstance(expected_spec, ModelSpec)
            else ModelSpec.from_dict(expected_spec)
        )
        if expected_spec != spec:
            raise ValueError("Checkpoint architecture does not match the configuration.")
    model = create_model(spec, device=device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload["metadata"]


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name].lerp_(param.detach(), 1 - self.decay)

    def load_state_dict(self, state):
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA state has a different architecture.")
        for name, value in state.items():
            self.shadow[name].copy_(value)

    @contextmanager
    def applied(self):
        backup = to_cpu(dict(self.model.named_parameters()))
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(backup[name])
