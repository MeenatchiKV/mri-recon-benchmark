"""End-to-end VarNet (Sriram et al. 2020), official fastmri.models.VarNet with its default (paper) size:
12 cascades, 18-channel U-Net regularisers, learned sensitivity-map network from the ACS lines.

The only change is gradient checkpointing over the cascades, which leaves the computation unchanged and
lets the full-size network train on a 6 GB GPU. Training recipe of the fastMRI VarNet: SSIM loss, Adam,
lr 3e-4, 50 epochs, lr x0.1 at epoch 40.
"""
import fastmri
import torch
from fastmri.losses import SSIMLoss
from fastmri.models import VarNet
from torch import nn
from torch.utils.checkpoint import checkpoint


def to_real(x):
    return torch.view_as_real(x)


class E2EVarNet(nn.Module):
    default_train = {"optimizer": "adam", "lr": 3e-4, "weight_decay": 0.0, "epochs": 50, "lr_step": 40,
                     "lr_gamma": 0.1}

    def __init__(self, num_cascades=12, sens_chans=8, sens_pools=4, chans=18, pools=4, grad_checkpoint=True):
        super().__init__()
        self.net = VarNet(num_cascades, sens_chans, sens_pools, chans, pools)
        self.grad_checkpoint = grad_checkpoint
        self.ssim = SSIMLoss()

    def forward(self, batch):
        masked = to_real(batch["masked_kspace"])                       # (B, C, H, W, 2)
        mask = batch["mask"].bool().reshape(1, 1, 1, -1, 1)
        n_low = int(batch["num_low"][0]) if torch.is_tensor(batch["num_low"]) else int(batch["num_low"])
        net = self.net
        sens = net.sens_net(masked, mask, n_low)
        k = masked.clone()
        for cascade in net.cascades:
            if self.grad_checkpoint and self.training:
                k = checkpoint(cascade, k, masked, mask, sens, use_reentrant=False)
            else:
                k = cascade(k, masked, mask, sens)
        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(k)), dim=1)

    def loss(self, batch, pred):
        return self.ssim(pred.unsqueeze(1), batch["target"].unsqueeze(1), data_range=batch["max_value"])
