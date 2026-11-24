"""fastMRI U-Net baseline (Ronneberger et al. 2015; Zbontar et al. 2018), official fastmri.models.Unet.

Input: RSS of the zero-filled coil images, instance-normalised and clamped to [-6, 6] as in the fastMRI
UnetDataTransform; output de-normalised. Training recipe of the fastMRI U-Net baseline: L1 loss,
RMSprop, lr 1e-3, 50 epochs, lr x0.1 at epoch 40.
"""
import torch
import torch.nn.functional as F
from fastmri.models import Unet
from torch import nn

from ..mri import ifft2c, rss


class UNetRecon(nn.Module):
    default_train = {"optimizer": "rmsprop", "lr": 1e-3, "weight_decay": 0.0, "epochs": 50, "lr_step": 40,
                     "lr_gamma": 0.1}

    def __init__(self, chans=32, num_pools=4, drop_prob=0.0):
        super().__init__()
        self.unet = Unet(1, 1, chans=chans, num_pool_layers=num_pools, drop_prob=drop_prob)

    def _normalise(self, batch):
        zf = rss(ifft2c(batch["masked_kspace"]))
        mean = zf.mean(dim=(-2, -1), keepdim=True)
        std = zf.std(dim=(-2, -1), keepdim=True) + 1e-11
        return ((zf - mean) / std).clamp(-6, 6), mean, std

    def forward(self, batch):
        x, mean, std = self._normalise(batch)
        self._stats = (mean, std)
        return self.unet(x.unsqueeze(1)).squeeze(1) * std + mean

    def loss(self, batch, pred):
        mean, std = self._stats
        target = ((batch["target"] - mean) / std).clamp(-6, 6)
        return F.l1_loss((pred - mean) / std, target)
