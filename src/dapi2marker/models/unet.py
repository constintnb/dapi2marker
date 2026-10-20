"""Configurable UNet retaining the original residual and attention topology."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .condition_encoder import ConditionEncoder, PatchGlobalFiLM

LATENT_CH = 4  # channels of the target/noise latent (VAE latent dim)
COND_CH = 4  # channels of the DAPI conditioning latent


class time_embedding(nn.Module):
    def __init__(self, dim):
        super(time_embedding, self).__init__()
        if type(dim) is not int or dim <= 2 or dim % 2:
            raise ValueError("Time embedding dimension must be even and greater than 2.")
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def zero_module(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class SPADE(nn.Module):
    def __init__(self, out_ch, cond_ch=COND_CH, hidden_ch=128, groups=8):
        super().__init__()

        g = groups
        while out_ch % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(g, out_ch, affine=False)

        self.shared = nn.Sequential(
            nn.Conv2d(cond_ch, hidden_ch, kernel_size=3, padding=1), nn.SiLU(inplace=True)
        )
        self.gamma = zero_module(nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1))
        self.beta = zero_module(nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, h, cond):
        normalized = self.norm(h)
        if cond.shape[-2:] != h.shape[-2:]:
            cond = F.interpolate(cond, size=h.shape[-2:], mode="nearest")
        act = self.shared(cond)
        gamma = self.gamma(act)
        beta = self.beta(act)
        return normalized * (1 + gamma) + beta


class transformer_block(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.15):
        super().__init__()
        if heads <= 0 or dim % heads:
            raise ValueError("Attention channels must be divisible by a positive head count.")
        self.heads = heads
        self.dim_head = dim // heads

        self.norm1 = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm2 = nn.GroupNorm(8, dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(dim * 4, dim, kernel_size=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm1(x)).view(B, 3, self.heads, self.dim_head, H * W)
        q, k, v = [tensor.transpose(2, 3) for tensor in (qkv[:, 0], qkv[:, 1], qkv[:, 2])]

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(2, 3).reshape(B, C, H, W)

        x = x + self.proj(out)
        x = x + self.ffn(self.norm2(x))

        return x


class Conv(nn.Module):
    def __init__(
        self, inc, outc, time_dim, cond_ch=COND_CH, spade_hidden_channels=128, dropout=0.15
    ):
        super(Conv, self).__init__()

        self.spade1 = SPADE(inc, cond_ch, hidden_ch=spade_hidden_channels)
        self.conv1 = nn.Sequential(
            nn.SiLU(inplace=True), nn.Conv2d(inc, outc, kernel_size=3, padding=1)
        )

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, outc * 2))

        self.spade2 = SPADE(outc, cond_ch, hidden_ch=spade_hidden_channels)
        self.conv2 = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1),
        )

        self.shortcut = nn.Conv2d(inc, outc, kernel_size=1) if inc != outc else nn.Identity()

    def forward(self, x, t_emb, cond=None):

        h = self.conv1(self.spade1(x, cond))
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = t_emb.chunk(2, dim=1)

        h = self.spade2(h, cond)
        h = h * (1 + scale) + shift

        h = self.conv2(h)

        return h + self.shortcut(x)


class downs(nn.Module):
    def __init__(
        self, inc, outc, time_dim, cond_ch=COND_CH, spade_hidden_channels=128, dropout=0.15
    ):
        super(downs, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = Conv(inc, outc, time_dim, cond_ch, spade_hidden_channels, dropout)
        self.conv2 = Conv(outc, outc, time_dim, cond_ch, spade_hidden_channels, dropout)
        self.conv3 = Conv(outc, outc, time_dim, cond_ch, spade_hidden_channels, dropout)

    def forward(self, x, t_emb, cond=None):
        x = self.pool(x)
        x = self.conv1(x, t_emb, cond)
        x = self.conv2(x, t_emb, cond)
        x = self.conv3(x, t_emb, cond)
        return x


class ups(nn.Module):
    def __init__(
        self,
        inc,
        outc,
        time_dim,
        skip_scale,
        backbone_scale,
        cond_ch=COND_CH,
        spade_hidden_channels=128,
        dropout=0.15,
    ):
        super(ups, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(inc, inc // 2, kernel_size=3, padding=1),
        )

        self.conv1 = Conv(inc, outc, time_dim, cond_ch, spade_hidden_channels, dropout)
        self.conv2 = Conv(outc, outc, time_dim, cond_ch, spade_hidden_channels, dropout)
        self.conv3 = Conv(outc, outc, time_dim, cond_ch, spade_hidden_channels, dropout)

        self.skip_scale = skip_scale
        self.backbone_scale = backbone_scale

    def forward(self, x1, x2, t_emb, cond=None):
        x1 = self.up(x1)

        dh = x2.size()[2] - x1.size()[2]
        dw = x2.size()[3] - x1.size()[3]

        if dw > 0 or dh > 0:
            x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])

        # Empirically effective legacy FreeU-like reweighting.  Here x1 is
        # the decoder backbone and x2 is the encoder skip.  This is kept as
        # an explicitly configurable inference heuristic rather than being
        # presented as the canonical Fourier-domain FreeU implementation.
        if not self.training:
            x2 = x2 * self.skip_scale
            x1 = x1 * self.backbone_scale

        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x, t_emb, cond)
        x = self.conv2(x, t_emb, cond)
        x = self.conv3(x, t_emb, cond)
        return x


class outconv(nn.Module):
    def __init__(self, inc, outc):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def group_norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    """Construct the same group layout used by SPADE, without conditioning."""
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels, affine=True)


class ConvNoSPADE(nn.Module):
    """Residual convolution block with GroupNorm and time conditioning."""

    def __init__(self, inc: int, outc: int, time_dim: int, dropout: float = 0.15):
        super().__init__()

        self.norm1 = group_norm(inc)
        self.conv1 = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(inc, outc, kernel_size=3, padding=1),
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, outc * 2),
        )

        self.norm2 = group_norm(outc)
        self.conv2 = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1),
        )

        self.shortcut = nn.Conv2d(inc, outc, kernel_size=1) if inc != outc else nn.Identity()

    def forward(self, x, t_emb, cond=None):
        # Accept the shared block interface; this block uses time conditioning.
        h = self.conv1(self.norm1(x))

        scale_shift = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(h)

        return h + self.shortcut(x)


class UpNoSPADE(nn.Module):
    """Highest-resolution up block whose residual Convs do not use SPADE."""

    def __init__(
        self,
        inc: int,
        outc: int,
        time_dim: int,
        skip_scale: float,
        backbone_scale: float,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(inc, inc // 2, kernel_size=3, padding=1),
        )

        self.conv1 = ConvNoSPADE(inc, outc, time_dim, dropout)
        self.conv2 = ConvNoSPADE(outc, outc, time_dim, dropout)
        self.conv3 = ConvNoSPADE(outc, outc, time_dim, dropout)

        self.skip_scale = skip_scale
        self.backbone_scale = backbone_scale

    def forward(self, x1, x2, t_emb, cond=None):
        x1 = self.up(x1)

        dh = x2.size(2) - x1.size(2)
        dw = x2.size(3) - x1.size(3)
        if dw > 0 or dh > 0:
            x1 = F.pad(
                x1,
                [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2],
            )

        # Apply the configured skip/backbone scales during inference.
        if not self.training:
            x2 = x2 * self.skip_scale
            x1 = x1 * self.backbone_scale

        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x, t_emb, cond)
        x = self.conv2(x, t_emb, cond)
        x = self.conv3(x, t_emb, cond)
        return x


class UNet(nn.Module):
    """Three-resolution UNet with SPADE omitted from the final up stage."""

    def __init__(
        self,
        time_dim=256,
        skip_scale=1.0,
        backbone_scale=1.0,
        condition_channels=(32, 64, 128),
        global_condition_hidden=None,
        base_channels=80,
        spade_hidden_channels=40,
        attention_heads=8,
        dropout=0.15,
    ):
        super().__init__()
        from .factory import ModelSpec

        self.spec = ModelSpec(
            base_channels=base_channels,
            time_dim=time_dim,
            spade_hidden_channels=spade_hidden_channels,
            attention_heads=attention_heads,
            dropout=dropout,
            condition_channels=condition_channels,
            global_condition_hidden=global_condition_hidden,
            skip_scale=skip_scale,
            backbone_scale=backbone_scale,
        )
        self.time_dim = time_dim
        self.skip_scale = skip_scale
        self.backbone_scale = backbone_scale
        self.condition_encoder = (
            ConditionEncoder(condition_channels) if condition_channels is not None else None
        )
        c0, c1, c2 = condition_channels if condition_channels is not None else (4, 4, 4)
        b0, b1, b2 = (base_channels * multiplier for multiplier in (1, 2, 4))
        self.time_mlp = nn.Sequential(
            time_embedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        block_args = {"spade_hidden_channels": spade_hidden_channels, "dropout": dropout}
        attention_args = {"heads": attention_heads, "dropout": dropout}
        self.inp = Conv(LATENT_CH, b0, time_dim, c0, **block_args)
        self.down1 = downs(b0, b1, time_dim, c1, **block_args)
        self.attn_down1 = transformer_block(b1, **attention_args)
        self.down2 = downs(b1, b2, time_dim, c2, **block_args)
        self.attn_down2 = transformer_block(b2, **attention_args)
        self.mid_conv1 = Conv(b2, b2, time_dim, c2, **block_args)
        self.mid_attn = transformer_block(b2, **attention_args)
        self.mid_conv2 = Conv(b2, b2, time_dim, c2, **block_args)
        self.attn_up1 = transformer_block(b2, **attention_args)
        self.up1 = ups(b2, b1, time_dim, skip_scale, backbone_scale, c1, **block_args)
        self.attn_up2 = transformer_block(b1, **attention_args)
        self.up2 = UpNoSPADE(b1, b0, time_dim, skip_scale, backbone_scale, dropout)
        self.out = outconv(b0, LATENT_CH)
        self.global_film = None
        if global_condition_hidden is not None:
            with torch.random.fork_rng(devices=[]):
                self.global_film = PatchGlobalFiLM(c2, b2, global_condition_hidden)

    def encode_condition(self, condition):
        """Build a condition pyramid once for all inference steps and draws."""
        if condition.ndim != 4 or condition.shape[1] != COND_CH:
            raise ValueError("Condition must have shape [batch,4,height,width].")
        return (
            self.condition_encoder(condition)
            if self.condition_encoder is not None
            else (condition, condition, condition)
        )

    def forward(self, x_t, cond, t):
        if x_t.ndim != 4 or x_t.shape[1] != LATENT_CH or min(x_t.shape[-2:]) < 4:
            raise ValueError("Latent input needs four channels and spatial dimensions >= 4.")
        c0, c1, c2 = cond if isinstance(cond, (tuple, list)) else self.encode_condition(cond)
        t_emb = self.time_mlp(t)
        x1 = self.inp(x_t, t_emb, c0)
        x2 = self.attn_down1(self.down1(x1, t_emb, c1))
        x3 = self.attn_down2(self.down2(x2, t_emb, c2))
        x_mid = self.mid_conv2(self.mid_attn(self.mid_conv1(x3, t_emb, c2)), t_emb, c2)
        if self.global_film is not None:
            x_mid = self.global_film(x_mid, c2)
        x = self.up1(self.attn_up1(x_mid), x2, t_emb, c1)
        x = self.up2(self.attn_up2(x), x1, t_emb, c0)
        return self.out(x)
