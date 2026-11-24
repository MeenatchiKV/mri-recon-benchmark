"""MoDL (Aggarwal, Mani & Jacob 2019), following the authors' reference implementation:
  x_{k+1} = argmin_x ||A x - y||^2 + lambda ||x - D_w(x_k)||^2,  solved with 10 CG iterations
  D_w: 5 conv layers, 64 filters, 3x3, BatchNorm + ReLU (none on the last), residual; shared over K = 10
  lambda learnable, initialised to 0.05; A = M F S with the ESPIRiT maps; MSE loss; Adam lr 1e-3.

Loss (target="sense", default): complex MSE against the fully sampled SENSE combination S^H F^H y_full with the
same maps, as in the original paper. MoDL's data consistency constrains its output to a SENSE-combined image, so
the RSS reference (with its noise floor) is not reachable; target="rss" (|x| vs RSS, MSE) is kept only to
reproduce the first knee run, which did not exceed zero-filling (runs/_superseded/knee_modl_rss_target).
Details matched to the reference code (github.com/hkaggarwal/modl): CG starts from x = 0; BatchNorm uses the
TensorFlow defaults (moving-average decay 0.99, eps 1e-3; PyTorch's default momentum 0.1 averages over only ~1
slice because the shared denoiser is called K times per slice); gradients clipped to [-1, 1] by value; the K = 10
model is initialised from a trained K = 1 model (bench.train --init-from), batch size 1, Adam lr 1e-3.
Gradient checkpointing per unrolled iteration re-runs each forward step in the backward pass, which applies the
BatchNorm running-stat update twice; the momentum is halved in that case so the effective decay stays 0.99.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..mri import adjoint_op, chan_to_complex, complex_to_chan, conjugate_gradient, normal_op


class Denoiser(nn.Module):
    def __init__(self, layers=5, features=64, bn_momentum=0.01):
        super().__init__()
        mods, cin = [], 2
        for i in range(layers):
            last = i == layers - 1
            mods.append(nn.Conv2d(cin, 2 if last else features, 3, padding=1))
            mods.append(nn.BatchNorm2d(2 if last else features, eps=1e-3, momentum=bn_momentum))
            if not last:
                mods.append(nn.ReLU(inplace=True))
            cin = features
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return x + chan_to_complex(self.net(complex_to_chan(x)))


class MoDL(nn.Module):
    default_train = {"optimizer": "adam", "lr": 1e-3, "weight_decay": 0.0, "epochs": 50, "lr_step": 40,
                     "lr_gamma": 0.1}
    clip_grad_value = 1.0

    def __init__(self, iterations=10, cg_iterations=10, layers=5, features=64, lam_init=0.05,
                 grad_checkpoint=True, target="sense"):
        super().__init__()
        self.target = target
        self.dw = Denoiser(layers, features, bn_momentum=0.005 if grad_checkpoint else 0.01)
        self.lam = nn.Parameter(torch.tensor(float(lam_init)))
        self.iterations, self.cg_iterations = iterations, cg_iterations
        self.grad_checkpoint = grad_checkpoint

    def _step(self, x, atb, maps, mask):
        z = self.dw(x)
        lam = self.lam.abs()
        return conjugate_gradient(lambda v: normal_op(v, maps, mask) + lam * v, atb + lam * z,
                                  torch.zeros_like(x), self.cg_iterations)

    def forward(self, batch):
        maps, mask = batch["maps"], batch["mask"][:, None, None, :]
        atb = adjoint_op(batch["masked_kspace"], maps, mask)
        x = atb
        for _ in range(self.iterations):
            if self.grad_checkpoint and self.training:
                x = checkpoint(self._step, x, atb, maps, mask, use_reentrant=False)
            else:
                x = self._step(x, atb, maps, mask)
        self._complex = x
        return x.abs()

    def loss(self, batch, pred):
        if self.target == "rss":
            return F.mse_loss(pred, batch["target"])
        ones = torch.ones_like(batch["mask"])[:, None, None, :]
        ref = adjoint_op(batch["kspace"], batch["maps"], ones)
        return F.mse_loss(torch.view_as_real(self._complex), torch.view_as_real(ref))
