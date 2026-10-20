import numpy as np
from skimage.metrics import structural_similarity
import torch

from dapi2marker.diffusion.sampling import sample_images
from dapi2marker.diffusion.schedule import DiffusionScheduler
from dapi2marker.losses import ssim_loss
from dapi2marker.metrics import aggregate_metrics


class ZeroModel:
    def __init__(self):
        self.calls = 0

    def encode_condition(self, condition):
        self.calls += 1
        return condition

    def __call__(self, noise, condition, time):
        return torch.zeros_like(noise)


class IdentityVAE:
    def decode(self, latent):
        return latent[:, :1]


def test_sampling_invariant_to_batch_partition_and_caches_condition():
    scheduler = DiffusionScheduler(8)
    condition = torch.zeros(2, 4, 8, 8)
    model = ZeroModel()
    options = dict(seed=2026, steps=3, draws=2)
    together = sample_images(
        model,
        IdentityVAE(),
        scheduler,
        condition,
        sample_ids=["colon:ROI001", "liver:ROI001"],
        **options,
    )
    assert model.calls == 1
    separate = torch.cat(
        [
            sample_images(
                model, IdentityVAE(), scheduler, condition[i : i + 1], sample_ids=[sid], **options
            )
            for i, sid in enumerate(["colon:ROI001", "liver:ROI001"])
        ]
    )
    torch.testing.assert_close(together, separate, rtol=0, atol=0)


def test_v_parameterization_recovers_clean_latent():
    scheduler = DiffusionScheduler(8)
    clean, noise = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    time = torch.tensor([0, 7])
    noisy = scheduler.add_noise(clean, noise, time)
    velocity = scheduler.get_v(clean, noise, time)
    alpha = scheduler.extract(scheduler.alphas_cumprod.sqrt(), time, clean.shape)
    sigma = scheduler.extract((1 - scheduler.alphas_cumprod).sqrt(), time, clean.shape)
    torch.testing.assert_close(alpha * noisy - sigma * velocity, clean)


def test_differentiable_ssim_matches_reference():
    rng = np.random.default_rng(42)
    x, y = rng.random((32, 32)).astype(np.float32), rng.random((32, 32)).astype(np.float32)
    prediction = torch.from_numpy(x)[None, None].requires_grad_()
    loss = ssim_loss(prediction, torch.from_numpy(y)[None, None])
    expected = 1 - structural_similarity(x, y, data_range=1)
    assert abs(float(loss.detach()) - expected) < 1e-5
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_selection_weights_equal_rois_then_equal_tissues():
    rows = [
        {
            "sample_id": str(i),
            "tissue": tissue,
            "group_id": group,
            "psnr": score,
            "ssim": score,
            "s50": score,
        }
        for i, (tissue, group, score) in enumerate(
            [
                ("colon", "colon:ROI0", 10),
                ("colon", "colon:ROI0", 10),
                ("colon", "colon:ROI1", 30),
                ("liver", "liver:ROI0", 60),
            ]
        )
    ]
    report = aggregate_metrics(rows)
    assert report["macro"]["s50"] == 40
    assert report["micro"]["s50"] == 27.5
