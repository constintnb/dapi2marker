"""Differentiable SSIM matching skimage's default 7x7 uniform-window metric."""

import torch
import torch.nn.functional as F


def ssim_loss(prediction, target):
    """Mean 1-SSIM for NCHW images in nominal [0, 1], computed in FP32.

    No clamp: decoded out-of-range values must still receive gradients.
    Use valid windows, sample covariance, K1=.01, K2=.03, data_range=1.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM needs matching NCHW tensors.")
    if min(prediction.shape[-2:]) < 7:
        raise ValueError("SSIM needs images at least 7x7.")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        x, y = prediction.float(), target.float()

        def mean(a):
            return F.avg_pool2d(a, 7, stride=1)

        ux, uy = mean(x), mean(y)
        vx = (mean(x * x) - ux * ux) * (49 / 48)
        vy = (mean(y * y) - uy * uy) * (49 / 48)
        cov = (mean(x * y) - ux * uy) * (49 / 48)
        score = (
            (2 * ux * uy + 0.01**2)
            * (2 * cov + 0.03**2)
            / ((ux * ux + uy * uy + 0.01**2) * (vx + vy + 0.03**2))
        )
        return 1 - score.mean()


def s50_score(psnr, ssim):
    """One image's score; average these values, never cap the mean PSNR."""
    return 100 * (0.7 * ssim + 0.3 * min(float(psnr), 50.0) / 50.0)


def reconstruction_losses(prediction, target, config):
    """Weighted image objective plus raw diagnostics, on nominal [-1,1] images."""
    pixel_weight = config.get("pixel_weight", 0.0)
    ssim_weight = config.get("ssim_weight", 0.0)
    # Disabled pixel MSE is diagnostic only; no gradient contribution.
    pixel = F.mse_loss(
        prediction.float() if pixel_weight else prediction.detach().float(), target.float()
    )
    structural = (
        ssim_loss((prediction.float() + 1) / 2, (target.float() + 1) / 2)
        if ssim_weight
        else prediction.new_zeros((), dtype=torch.float32)
    )
    return pixel_weight * pixel + ssim_weight * structural, pixel, structural
