"""DDIM sampling with condition caching and stable per-tile random streams."""

import hashlib

import torch

from ..config import autocast_context


def sampling_times(scheduler, steps):
    if type(steps) is not int or not 1 <= steps <= scheduler.timesteps:
        raise ValueError("DDIM steps must be between 1 and diffusion.timesteps.")
    return torch.linspace(scheduler.timesteps - 1, 0, steps).long().tolist()


@torch.no_grad()
def sample_images(
    model, vae, scheduler, condition, *, sample_ids, seed, steps, draws=1, precision="no"
):
    if type(draws) is not int or draws < 1 or len(sample_ids) != len(condition):
        raise ValueError("Sampling needs a positive draw count and one stable ID per input.")
    device = condition.device
    times = sampling_times(scheduler, steps)
    # The caller supplies an unwrapped EMA model for evaluation, so this method
    # runs the condition encoder once per batch, independent of DDIM steps/draws.
    with autocast_context(device, precision):
        features = model.encode_condition(condition)
    accumulated = None
    for draw in range(draws):
        noise = []
        for sample_id in sample_ids:
            digest = hashlib.blake2b(f"{seed}:{sample_id}:{draw}".encode(), digest_size=8).digest()
            generator = torch.Generator(device=device).manual_seed(int.from_bytes(digest, "little"))
            noise.append(
                torch.randn(
                    (1, *condition.shape[1:]),
                    device=device,
                    dtype=condition.dtype,
                    generator=generator,
                )
            )
        x = torch.cat(noise)
        with autocast_context(device, precision):
            for index, t1 in enumerate(times):
                t0 = times[index + 1] if index + 1 < len(times) else -1
                time = torch.full((len(x),), t1, device=device, dtype=torch.long)
                x = scheduler.step(model(x, features, time), t1, t0, x)
        decoded = ((vae.decode(x).float() + 1) / 2).clamp(0, 1)
        accumulated = decoded if accumulated is None else accumulated + decoded
    return accumulated / draws
