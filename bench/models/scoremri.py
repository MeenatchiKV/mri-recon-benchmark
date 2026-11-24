"""score-MRI (Chung & Ye 2022) with the official code: NCSN++ score network, VE-SDE, continuous denoising
score matching, reverse-diffusion predictor + Langevin corrector (all imported from ../score-MRI-main).

As in score-MRI, the network is trained on magnitude images scaled to [0, 1] and, at inference, is applied
to the real and imaginary parts of the complex image separately. Multi-coil data consistency follows the
paper's SENSE variant: after every predictor and corrector step, a Kaczmarz step
    x <- x + lambda_i * A^H (y - A x),    A = M F S (ESPIRiT maps),  lambda_i linear 1.0 -> 0.2
is applied to the coil-combined image. (The official hybrid sampler additionally runs the prior on every
coil image separately, which costs C times more network calls; it is not used here.)

Deviations forced by the 6 GB GPU, reported with the results: NCSN++ width nf = 64 instead of 128,
and N reverse steps (default 500, the official default) is configurable.
"""
import numpy as np
import torch
from torch import nn

from ..mri import adjoint_op, forward_op
from ..score_import import load_score_mri


def make_config(nf=64, image_size=320):
    s = load_score_mri()
    cfg = s.config_module.get_config()
    cfg.model.nf = nf
    cfg.data.image_size = image_size
    return cfg


class ScoreMRI(nn.Module):
    default_train = {"optimizer": "adam", "lr": 2e-4, "warmup": 5000, "grad_clip": 1.0, "steps": 100000,
                     "ema_rate": 0.999, "batch_size": 1}

    def __init__(self, nf=64, image_size=320, n_steps=500, snr=0.16, lamb_start=1.0, lamb_end=0.2, seed=0):
        super().__init__()
        s = load_score_mri()
        self.cfg = make_config(nf, image_size)
        self.score_model = s.mutils.create_model(self.cfg)
        if isinstance(self.score_model, nn.DataParallel):    # create_model wraps in DataParallel
            self.score_model = self.score_model.module
        self.n_steps, self.snr, self.seed = n_steps, snr, seed
        self.lamb_start, self.lamb_end = lamb_start, lamb_end

    def sde(self, n=None):
        m = self.cfg.model
        return load_score_mri().sde_lib.VESDE(m.sigma_min, m.sigma_max, n or m.num_scales)

    @torch.no_grad()
    def forward(self, batch):
        s = load_score_mri()
        y, maps, mask = batch["masked_kspace"], batch["maps"], batch["mask"][:, None, None, :]
        zf = adjoint_op(y, maps, mask)
        scale = zf.abs().amax(dim=(-2, -1), keepdim=True) + 1e-12      # complex image -> |x| <= ~1
        y = y / scale[..., None]

        sde = self.sde(self.n_steps)
        sde.discrete_sigmas = sde.discrete_sigmas.to(zf.device)   # official code predates strict device indexing
        score_fn = s.mutils.get_score_fn(sde, self.score_model, train=False, continuous=True)
        predictor = s.sampling.ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        corrector = s.sampling.LangevinCorrector(sde, score_fn, self.snr, 1)

        b, h, w = zf.shape
        gen = torch.Generator(device=zf.device).manual_seed(self.seed)
        x = torch.complex(*(torch.randn(2, b, h, w, generator=gen, device=zf.device) * sde.sigma_max))
        x_mean = x
        timesteps = torch.linspace(sde.T, 1e-5, sde.N, device=zf.device)
        for i, t in enumerate(timesteps):
            lamb = self.lamb_start + (self.lamb_end - self.lamb_start) * i / max(sde.N - 1, 1)
            for step in (predictor, corrector):
                ri = torch.stack([x.real, x.imag], 1).reshape(2 * b, 1, h, w)   # real & imag in one batch
                vec_t = torch.full((2 * b,), float(t), device=zf.device)
                ri, ri_mean = step.update_fn(ri, vec_t)
                ri, ri_mean = ri.reshape(b, 2, h, w), ri_mean.reshape(b, 2, h, w)
                x = torch.complex(ri[:, 0], ri[:, 1])
                x_mean = torch.complex(ri_mean[:, 0], ri_mean[:, 1])
                x = x + lamb * adjoint_op(y - forward_op(x, maps, mask), maps, mask)
                x_mean = x_mean + lamb * adjoint_op(y - forward_op(x_mean, maps, mask), maps, mask)
        return x_mean.abs() * scale

    def loss(self, batch, pred):
        raise NotImplementedError("score-MRI is trained with bench.train_score (denoising score matching)")
