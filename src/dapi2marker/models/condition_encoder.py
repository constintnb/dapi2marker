"""Small learned pyramid for the frozen VAE's DAPI latent."""

import torch
from torch import nn


class PatchGlobalFiLM(nn.Module):
    """Single-patch channel mean/std, followed by one residual FiLM operation."""

    def __init__(self, condition_channels, feature_channels, hidden_channels=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * condition_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 2 * feature_channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features, condition):
        # Only spatial axes: no batch statistics or information from other patches.
        with torch.autocast(device_type=condition.device.type, enabled=False):
            variance, mean = torch.var_mean(condition.float(), dim=(-2, -1), correction=0)
            descriptor = torch.cat((mean, (variance + 1e-6).sqrt()), dim=1)
        scale, shift = self.mlp(descriptor).to(features.dtype).chunk(2, dim=1)
        return features * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ConditionEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 128)):
        super().__init__()
        c0, c1, c2 = channels
        # No spatial normalization: preserve useful absolute intensity cues.
        self.level0 = nn.Sequential(
            nn.Conv2d(4, c0, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c0, c0, 3, padding=1),
            nn.SiLU(),
        )
        self.level1 = self.down(c0, c1)
        self.level2 = self.down(c1, c2)

    @staticmethod
    def down(inc, outc):
        # floor(H/2), matching the main UNet's MaxPool even for odd sizes.
        return nn.Sequential(
            nn.Conv2d(inc, outc, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(outc, outc, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, condition):
        c0 = self.level0(condition)
        c1 = self.level1(c0)
        c2 = self.level2(c1)
        return c0, c1, c2
