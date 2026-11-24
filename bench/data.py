"""Slice dataset over the preprocessed volumes.

Every sample carries what any method needs, so all methods get identical inputs:
  masked_kspace (C, H, W) complex64    mask (W,) float32    num_low (int)
  maps (C, H, W) complex64  ESPIRiT maps from the ACS lines of this R
  target (H, W) float32     RSS of the fully sampled compressed coils
  max_value                 volume maximum of the target (data range for PSNR / SSIM, as in fastMRI)
"""
import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .espirit import load_maps, maps_path
from .masks import eval_mask, make_mask


class SliceDataset(Dataset):
    def __init__(self, cfg, anatomy, split, accels=None, train=False, mask_type=None, with_maps=True,
                 volumes=None):
        self.cfg, self.anatomy, self.train, self.with_maps = cfg, anatomy, train, with_maps
        self.accels = accels or cfg["masks"]["accelerations"]
        self.mask_type = mask_type or cfg["masks"]["type"]
        self.cf = cfg["masks"]["center_fractions"]
        names = volumes or json.load(open(cfg["data_dir"] / anatomy / "split.json"))[split]
        info = {v["name"]: v for v in json.load(open(cfg["data_dir"] / anatomy / "volumes.json"))}
        self.vol_dir = cfg["data_dir"] / anatomy / "volumes"
        self.items = [(n, s) for n in names for s in range(info[n]["slices"])]
        if not train:  # evaluation enumerates every (slice, R) pair
            self.items = [(n, s, r) for n, s in self.items for r in self.accels]
        self.max_value = {}
        for n in names:
            with h5py.File(self.vol_dir / f"{n}.h5", "r") as f:
                self.max_value[n] = float(f.attrs["max"])
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.train:
            name, s = self.items[i]
            accel = int(self.rng.choice(self.accels))
        else:
            name, s, accel = self.items[i]
        with h5py.File(self.vol_dir / f"{name}.h5", "r") as f:
            kspace = f["kspace"][s]
            target = f["target"][s]
        w = kspace.shape[-1]
        if self.train:
            mask, n_low = make_mask(w, accel, self.cf[accel], self.mask_type,
                                    seed=int(self.rng.integers(2 ** 31)))
        else:
            mask, n_low = eval_mask(name, s, w, accel, self.cf[accel], self.mask_type)
        sample = {
            "kspace": torch.from_numpy(kspace),
            "masked_kspace": torch.from_numpy(kspace * mask),
            "mask": torch.from_numpy(mask),
            "num_low": n_low,
            "target": torch.from_numpy(target),
            "max_value": torch.tensor(self.max_value[name], dtype=torch.float32),
            "volume": name, "slice": s, "accel": accel,
        }
        if self.with_maps:
            sample["maps"] = torch.from_numpy(
                load_maps(maps_path(self.cfg, self.anatomy, accel, name), s).astype(np.complex64))
        return sample

    def worker_init(self, worker_id):
        self.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32 + worker_id)
