"""Undersampling masks from the official fastMRI mask functions.

Evaluation masks are deterministic per (volume, slice, R), so every method sees exactly the same samples.
Training masks are drawn fresh every iteration (as in the fastMRI baselines).
"""
import zlib

import numpy as np
from fastmri.data.subsample import EquispacedMaskFractionFunc, RandomMaskFunc

MASK_FUNCS = {"random": RandomMaskFunc, "equispaced": EquispacedMaskFractionFunc}


def num_low_frequencies(width, center_fraction):
    return int(round(width * center_fraction))


def make_mask(width, accel, center_fraction, mask_type="random", seed=None):
    """Returns (mask (W,) float32, num_low_frequencies)."""
    func = MASK_FUNCS[mask_type]([center_fraction], [accel])
    mask, n_low = func((1, width, 1), seed=seed)
    return mask.reshape(-1).numpy().astype(np.float32), int(n_low)


def eval_seed(volume, slice_idx, accel, mask_type="random"):
    return zlib.crc32(f"{volume}/{slice_idx}/{accel}/{mask_type}".encode())


def eval_mask(volume, slice_idx, width, accel, center_fraction, mask_type="random"):
    return make_mask(width, accel, center_fraction, mask_type, eval_seed(volume, slice_idx, accel, mask_type))


def acs_slice(width, n_low):
    start = (width - n_low + 1) // 2
    return slice(start, start + n_low)
