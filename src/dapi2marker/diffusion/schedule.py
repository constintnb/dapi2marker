import torch
import math


class DiffusionScheduler:
    def __init__(self, timesteps, device="cpu"):
        if type(timesteps) is not int or timesteps < 2:
            raise ValueError("diffusion.timesteps must be an integer >= 2.")
        self.timesteps = timesteps
        self.device = device

        self.betas = self.cos_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def cos_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alpha_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]

        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def extract(self, arr, timesteps, broadcast_shape):
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res

    def get_v(self, origin, noise, timestep):
        alpha1 = torch.sqrt(self.alphas_cumprod)
        alpha1 = self.extract(alpha1, timestep, origin.shape)

        alpha2 = torch.sqrt(1.0 - self.alphas_cumprod)
        alpha2 = self.extract(alpha2, timestep, origin.shape)

        v = alpha1 * noise - alpha2 * origin
        return v

    def add_noise(self, origin, noise, timesteps):
        alpha1 = torch.sqrt(self.alphas_cumprod)
        alpha1 = self.extract(alpha1, timesteps, origin.shape)

        alpha2 = torch.sqrt(1.0 - self.alphas_cumprod)
        alpha2 = self.extract(alpha2, timesteps, origin.shape)

        nosiy_samples = alpha1 * origin + alpha2 * noise
        return nosiy_samples

    def step(self, model_output, timestep1, timestep0, sample):
        alpha1_val = self.alphas_cumprod[timestep1]
        alpha0_val = (
            self.alphas_cumprod[timestep0]
            if timestep0 >= 0
            else torch.tensor(1.0, device=self.device)
        )

        sigma1 = torch.sqrt(1 - alpha1_val)
        alpha1 = torch.sqrt(alpha1_val)

        pred_x0 = alpha1 * sample - sigma1 * model_output
        pred_noise = sigma1 * sample + alpha1 * model_output

        pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)

        direction = torch.sqrt(1 - alpha0_val) * pred_noise
        prev_sample = torch.sqrt(alpha0_val) * pred_x0 + direction
        return prev_sample
