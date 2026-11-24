"""Learned reconstruction models. Each model exposes

    forward(batch) -> magnitude image (B, H, W)
    loss(batch, pred) -> scalar
    default_train  -> dict(optimizer, lr, epochs, lr_step, lr_gamma, ...) from the original paper / code

`batch` is the collated output of bench.data.SliceDataset (batch size 1, tensors on the target device).
"""
from .kcd import KCD
from .modl import MoDL
from .rim import RIM
from .scoremri import ScoreMRI
from .swinmr import SwinMR
from .unet import UNetRecon
from .varnet import E2EVarNet

MODELS = {"unet": UNetRecon, "varnet": E2EVarNet, "modl": MoDL, "rim": RIM, "swinmr": SwinMR,
          "kcd": KCD, "scoremri": ScoreMRI}   # scoremri is trained with bench.train_score


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)
