"""Recurrent Inference Machine (Putzky & Welling 2017; Lonning et al. 2019 for multi-coil MRI).

Each of T = 8 time steps takes the current estimate x_t and the data-fidelity gradient A^H(A x_t - y),
passes them through  conv5x5 -> ReLU -> ConvGRU -> conv3x3(dil 2) -> ReLU -> ConvGRU -> conv3x3,
and adds the output to x_t. Weights are shared over time; the loss is averaged over all time steps.
A = M F S uses the ESPIRiT maps; x_0 = A^H y. L1 loss on |x_t| averaged over t; Adam.

The hidden size, learning rate and loss are configuration choices; they are reported with the results
and should be checked against Lonning et al. (2019) / the DIRECT RIM config before the final runs.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..mri import adjoint_op, chan_to_complex, complex_to_chan, forward_op


class ConvGRUCell(nn.Module):
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 2 * channels, kernel, padding=kernel // 2)
        self.cand = nn.Conv2d(2 * channels, channels, kernel, padding=kernel // 2)

    def forward(self, x, h):
        r, z = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class RIMCell(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv2d(4, hidden, 5, padding=2)
        self.gru1 = ConvGRUCell(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2)
        self.gru2 = ConvGRUCell(hidden)
        self.conv3 = nn.Conv2d(hidden, 2, 3, padding=1)

    def forward(self, x, grad, h1, h2):
        a = F.relu(self.conv1(torch.cat([complex_to_chan(x), complex_to_chan(grad)], 1)))
        h1 = self.gru1(a, h1)
        h2 = self.gru2(F.relu(self.conv2(h1)), h2)
        return x + chan_to_complex(self.conv3(h2)), h1, h2


class RIM(nn.Module):
    default_train = {"optimizer": "adam", "lr": 1e-3, "weight_decay": 0.0, "epochs": 50, "lr_step": 40,
                     "lr_gamma": 0.1}

    def __init__(self, steps=8, hidden=64, grad_checkpoint=True):
        super().__init__()
        self.cell = RIMCell(hidden)
        self.steps, self.hidden, self.grad_checkpoint = steps, hidden, grad_checkpoint

    def forward(self, batch):
        y, maps, mask = batch["masked_kspace"], batch["maps"], batch["mask"][:, None, None, :]
        x = adjoint_op(y, maps, mask)
        b, h, w = x.shape
        h1 = x.real.new_zeros(b, self.hidden, h, w)
        h2 = x.real.new_zeros(b, self.hidden, h, w)
        self._trajectory = []
        for _ in range(self.steps):
            grad = adjoint_op(forward_op(x, maps, mask) - y, maps, mask)
            if self.grad_checkpoint and self.training:
                x, h1, h2 = checkpoint(self.cell, x, grad, h1, h2, use_reentrant=False)
            else:
                x, h1, h2 = self.cell(x, grad, h1, h2)
            self._trajectory.append(x.abs())
        return x.abs()

    def loss(self, batch, pred):
        return torch.stack([F.l1_loss(p, batch["target"]) for p in self._trajectory]).mean()
