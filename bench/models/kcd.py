"""K-space cold diffusion (KCD; Shen et al. 2023, "K-space Cold Diffusion: learning to reconstruct
accelerated MRI without noise"), extended to multi-coil data with the benchmark's ESPIRiT maps.

Degradation (deterministic, no noise): D_t removes floor(t * n_u / T) of the n_u unsampled PE columns in a
random but fixed order per slice, applied to the coil k-space and SENSE-combined:
    x_t = S^H F^H (M_t k),   M_0 = all columns (x_0 = fully sampled SENSE image),  M_T = M (x_T = A^H y)
Training: a time-conditioned ConvNeXt U-Net f(x_t, t) predicts x_0 (L1 loss), t ~ U{0..T}, T = 1000.
Sampling (cold-diffusion "x0 step-down", Bansal et al. 2022, Alg. 2) from x_T = A^H y. The sampler state
is the coil k-space k_t (zero outside M_t), so every network input has exactly the training form
x_t = S^H F^H k_t:
    x0 = f(x_t, t);  k0 = DC(F S x0) = M y + (1 - M) F S x0
    k_{t'} = k_t - M_t k0 + M_{t'} k0      (= D_{t'}(k0) + D_t(k_t - k0), Alg. 2 in k-space; M y is kept exact)
The first implementation (sampler="image", kept for the record) held a SENSE-combined image as state and
re-expanded it with the maps at every level; S S^H is not the identity per coil, so energy leaked into
still-removed columns and the network inputs drifted away from the training distribution with each level
(validation SSIM fell monotonically from 0.755 at 1 level to 0.556 at 100).
Sampling runs on a grid of `sample_steps` levels (1000 = one step per t, the KCD default). Multi-coil sampling costs
~0.25 s per level on the 6 GB GPU, so the default is 100 levels (the authors' single-coil study found uniform
shorter schedules as accurate); the final number is chosen on the validation volumes and reported.
Per-epoch validation (checkpoint selection) uses `val_steps` = 10: on the multi-coil data the x0 step-down
sampler accumulates error with many levels (an undertrained checkpoint scored SSIM 0.769 / 0.762 / 0.671 at
1 / 10 / 50 levels), so validating at 50 levels would select checkpoints on the sampler's worst regime.

Network, T, loss and bf16 precision follow the single-coil KSC-Diff code in ../MRI-Work1/ksc. The
multi-coil degradation operator and DC are the extension made for this benchmark.
"""
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..mri import chan_to_complex, complex_to_chan, fft2c, ifft2c
from .kcd_unet import Unet


class KCD(nn.Module):
    default_train = {"optimizer": "adam", "lr": 2e-5, "weight_decay": 0.0, "epochs": 50, "lr_step": 40,
                     "lr_gamma": 0.1}

    def __init__(self, dim=64, timesteps=1000, sample_steps=100, val_steps=10, sampler="kspace"):
        super().__init__()
        self.sampler = sampler
        self.net = Unet(dim=dim, dim_mults=(1, 2, 4, 8), channels=2)
        self.T, self.sample_steps, self.val_steps = timesteps, sample_steps, val_steps

    # ------------------------------------------------------------------ operators
    @staticmethod
    def _combine(k, maps):
        return torch.sum(torch.conj(maps) * ifft2c(k), dim=1)

    @staticmethod
    def _expand(x, maps):
        return fft2c(maps * x[:, None])

    def _orders(self, batch):
        """Removal rank per PE column (inf for acquired columns); fixed per slice at evaluation."""
        mask = batch["mask"].cpu().numpy()
        orders = np.full(mask.shape, np.inf, dtype=np.float32)
        for b in range(mask.shape[0]):
            unsampled = np.flatnonzero(mask[b] == 0)
            if self.training:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(zlib.crc32(
                    f"kcd/{batch['volume'][b]}/{int(batch['slice'][b])}/{int(batch['accel'][b])}".encode()))
            orders[b, unsampled[rng.permutation(len(unsampled))]] = np.arange(len(unsampled))
        return torch.from_numpy(orders).to(batch["mask"].device)

    def _keep(self, order, mask, t):
        """(B, 1, 1, W) column mask of D_t."""
        n_removed = torch.floor(t.float() * (mask == 0).sum(1).float() / self.T)
        return (order >= n_removed[:, None]).float()[:, None, None, :]

    def _predict(self, x, t, scale):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            out = self.net(complex_to_chan(x / scale), t)
        return chan_to_complex(out.float()) * scale

    # ------------------------------------------------------------------ interface
    def forward(self, batch):
        maps, mask = batch["maps"], batch["mask"]
        y = batch["masked_kspace"]
        scale = self._combine(y, maps).abs().amax(dim=(-2, -1)).reshape(-1, 1, 1) + 1e-12
        order = self._orders(batch)
        if self.training:
            k = batch["kspace"]
            t = torch.randint(0, self.T + 1, (k.shape[0],), device=k.device)
            x_t = self._combine(k * self._keep(order, mask, t), maps)
            self._target = self._combine(k, maps)
            self._scale = scale
            return self._predict(x_t, t, scale)
        return self.sample(y, maps, mask, order, scale, self.sample_steps)

    @torch.no_grad()
    def sample(self, y, maps, mask, order, scale, steps):
        m = mask[:, None, None, :]
        grid = torch.round(torch.linspace(self.T, 0, steps + 1, device=y.device)).long()

        def dc(k):
            return m * y + (1 - m) * k

        if self.sampler == "kspace":
            k = y                                                     # k_T = M y  (x_T = A^H y)
            for i in range(steps):
                t, t_next = grid[i].expand(y.shape[0]), grid[i + 1].expand(y.shape[0])
                k0 = dc(self._expand(self._predict(self._combine(k, maps), t, scale), maps))
                k = k - k0 * self._keep(order, mask, t) + k0 * self._keep(order, mask, t_next)
                k = dc(k)                                             # acquired columns stay exactly y
            return self._combine(k, maps).abs()

        x = self._combine(y, maps)                                    # x_T = A^H y
        for i in range(steps):
            t, t_next = grid[i].expand(y.shape[0]), grid[i + 1].expand(y.shape[0])
            k0 = dc(self._expand(self._predict(x, t, scale), maps))
            x = (x - self._combine(k0 * self._keep(order, mask, t), maps)
                 + self._combine(k0 * self._keep(order, mask, t_next), maps))
            x = self._combine(dc(self._expand(x, maps)), maps)
        return x.abs()

    def loss(self, batch, pred):
        s = self._scale[:, None]
        return F.l1_loss(complex_to_chan(pred) / s, complex_to_chan(self._target) / s)
