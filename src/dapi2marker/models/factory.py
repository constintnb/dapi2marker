"""One architecture schema and factory shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ModelSpec:
    base_channels: int = 80
    time_dim: int = 256
    spade_hidden_channels: int = 40
    attention_heads: int = 8
    dropout: float = 0.15
    condition_channels: tuple[int, int, int] | None = (32, 64, 128)
    global_condition_hidden: int | None = None
    skip_scale: float = 1.0
    backbone_scale: float = 1.0

    def __post_init__(self):
        for key in ("base_channels", "time_dim", "spade_hidden_channels", "attention_heads"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"model.{key} must be a positive integer.")
        if self.base_channels % 8:
            raise ValueError("model.base_channels must be divisible by 8.")
        if self.time_dim <= 2 or self.time_dim % 2:
            raise ValueError("model.time_dim must be an even integer greater than 2.")
        if any(self.base_channels * scale % self.attention_heads for scale in (2, 4)):
            raise ValueError("Attention widths must be divisible by model.attention_heads.")
        if type(self.dropout) not in (int, float) or not 0 <= self.dropout < 1:
            raise ValueError("model.dropout must be in [0,1).")
        if self.condition_channels is not None and (
            len(self.condition_channels) != 3
            or any(type(c) is not int or c <= 0 for c in self.condition_channels)
        ):
            raise ValueError("Condition encoder requires three positive widths.")
        if self.global_condition_hidden is not None:
            if self.condition_channels is None:
                raise ValueError("Global conditioning requires the condition encoder.")
            if type(self.global_condition_hidden) is not int or self.global_condition_hidden <= 0:
                raise ValueError("Global conditioning hidden width must be positive.")
        for value in (self.skip_scale, self.backbone_scale):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Inference branch scales must be finite and nonnegative.")

    @classmethod
    def from_dict(cls, config: dict) -> ModelSpec:
        values = dict(config)
        if "condition_encoder" in values:
            encoder = values.pop("condition_encoder")
            values["condition_channels"] = (
                tuple(encoder.get("channels", (32, 64, 128)))
                if encoder.get("enabled", True)
                else None
            )
        elif values.get("condition_channels") is not None:
            values["condition_channels"] = tuple(values["condition_channels"])
        if "global_condition" in values:
            global_options = values.pop("global_condition")
            values["global_condition_hidden"] = (
                global_options.get("hidden_channels", 64)
                if global_options.get("enabled", False)
                else None
            )
        if "freeu" in values:
            values.update(values.pop("freeu"))
        return cls(**values)

    def to_dict(self) -> dict:
        values = asdict(self)
        if self.condition_channels is not None:
            values["condition_channels"] = list(self.condition_channels)
        return values


def create_model(config: dict | ModelSpec, *, device=None):
    import torch
    from .unet import UNet

    spec = config if isinstance(config, ModelSpec) else ModelSpec.from_dict(config)
    with torch.device(device or "cpu"):
        return UNet(**asdict(spec))


def parameter_count(config: dict | ModelSpec) -> int:
    model = create_model(config, device="meta")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
