"""Classical (training-free) reconstructions, each from a reference implementation:

  zf       zero-filled IFFT + RSS
  grappa   GRAPPA (Griswold et al. 2002), pygrappa.cgrappa, 5x5 kernel, calibrated on the ACS lines; RSS
  sense    CG-SENSE (Pruessmann et al. 2001) with Tikhonov regularisation, sigpy.mri.app.SenseRecon
  cs       L1-wavelet compressed sensing (Lustig et al. 2007), sigpy.mri.app.L1WaveletRecon (db4)

SENSE and CS use the same ESPIRiT maps as MoDL / RIM. Their regularisation weights are tuned on the
validation volumes (bench.tune) and read from results/<anatomy>/classical_params.json.
All functions take numpy arrays: masked k-space (C, H, W), mask (W,), maps (C, H, W), n_low.
"""
import numpy as np
import sigpy as sp
import sigpy.mri.app as app
from pygrappa import cgrappa

from .masks import acs_slice
from .mri import ifft2c_np, rss_np

CLASSICAL = ("zf", "grappa", "sense", "cs", "full_rss", "full_sense")
DEFAULTS = {"sense": {"lamda": 1e-3, "max_iter": 30}, "cs": {"lamda": 1e-3, "max_iter": 100}}


def zf(kspace, mask, maps=None, n_low=None, **_):
    return rss_np(ifft2c_np(kspace), axis=0)


def grappa(kspace, mask, maps=None, n_low=None, kernel=(5, 5), lamda=0.01, **_):
    calib = kspace[:, :, acs_slice(kspace.shape[-1], n_low)]
    filled = cgrappa(np.ascontiguousarray(kspace.transpose(1, 2, 0)).astype(np.complex128),
                     np.ascontiguousarray(calib.transpose(1, 2, 0)).astype(np.complex128),
                     kernel_size=tuple(kernel), coil_axis=-1, lamda=lamda)
    return rss_np(ifft2c_np(filled.transpose(2, 0, 1)), axis=0)


def _weights(mask, shape):
    return np.broadcast_to(mask[None, None, :], shape).astype(np.float32)


def sense(kspace, mask, maps, n_low=None, lamda=1e-3, max_iter=30, device=-1, **_):
    dev = sp.Device(device)
    x = app.SenseRecon(sp.to_device(kspace, dev), sp.to_device(maps, dev), lamda=lamda,
                       weights=sp.to_device(_weights(mask, kspace.shape), dev), max_iter=max_iter,
                       device=dev, show_pbar=False).run()
    return np.abs(sp.to_device(x, sp.cpu_device))


def cs(kspace, mask, maps, n_low=None, lamda=1e-3, max_iter=100, device=-1, **_):
    dev = sp.Device(device)
    x = app.L1WaveletRecon(sp.to_device(kspace, dev), sp.to_device(maps, dev), lamda=lamda,
                           weights=sp.to_device(_weights(mask, kspace.shape), dev), max_iter=max_iter,
                           device=dev, show_pbar=False).run()
    return np.abs(sp.to_device(x, sp.cpu_device))


def full_rss(kspace, mask=None, maps=None, n_low=None, **_):
    """Ceiling row (not a method): fully sampled k-space, RSS-combined."""
    return rss_np(ifft2c_np(kspace), axis=0)


def full_sense(kspace, mask=None, maps=None, n_low=None, **_):
    """Ceiling row (not a method): fully sampled k-space combined with the ACS ESPIRiT maps that SENSE / CS /
    MoDL / RIM / the diffusion models use, i.e. the best any sensitivity-combining method can reach."""
    return np.abs(np.sum(np.conj(maps) * ifft2c_np(kspace), axis=0))


FUNCS = {"zf": zf, "grappa": grappa, "sense": sense, "cs": cs, "full_rss": full_rss, "full_sense": full_sense}
CEILINGS = ("full_rss", "full_sense")   # evaluated on the fully sampled k-space


def reconstruct(method, kspace, mask, maps, n_low, **params):
    return FUNCS[method](kspace, mask, maps, n_low, **params).astype(np.float32)
