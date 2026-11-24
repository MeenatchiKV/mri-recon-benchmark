"""ESPIRiT sensitivity maps (Uecker et al. 2014) from the ACS lines only, on the GPU.

This follows sigpy.mri.app.EspiritCalib step for step, except that the calibration block is rectangular
(`calib_readout` x number of ACS lines) instead of square; with only 13 ACS lines at R = 8 a square block
would throw away most of the fully sampled readout.

Maps are stored as float16 (real, imag) to halve disk use; |S| <= 1, so the rounding error is ~1e-3.

Usage:  python -m bench.espirit [--anatomy knee brain] [--accel 4 8]
"""
import argparse
import json

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .config import load_config
from .masks import acs_slice, num_low_frequencies
from .mri import ifft2c


@torch.no_grad()
def espirit_maps(kspace, n_low, calib_readout=64, kernel=6, thresh=0.02, crop=0.95, power_iters=100,
                 kernel_batch=32):
    """kspace: (C, H, W) complex torch tensor (only its ACS lines are used). Returns maps (C, H, W)."""
    c, h, w = kspace.shape
    r0 = h // 2 - calib_readout // 2
    calib = kspace[:, r0:r0 + calib_readout, acs_slice(w, n_low)]
    blocks = calib.unfold(1, kernel, 1).unfold(2, kernel, 1)          # (C, nbh, nbw, k, k)
    mat = blocks.permute(1, 2, 0, 3, 4).reshape(-1, c * kernel * kernel)
    _, s, vh = torch.linalg.svd(mat, full_matrices=False)
    kernels = vh[s > thresh * s.max()].reshape(-1, c, kernel, kernel)

    aha = torch.zeros(h, w, c, c, dtype=kspace.dtype, device=kspace.device)
    top, left = (h - kernel) // 2, (w - kernel) // 2
    for i in range(0, len(kernels), kernel_batch):
        kb = kernels[i:i + kernel_batch]
        padded = torch.zeros(len(kb), c, h, w, dtype=kspace.dtype, device=kspace.device)
        padded[..., top:top + kernel, left:left + kernel] = kb
        v = ifft2c(padded)                                               # (nk, C, H, W)
        aha += torch.einsum("kchw,kdhw->hwcd", v, v.conj())
    aha *= h * w / kernel ** 2

    x = torch.ones(h, w, c, 1, dtype=kspace.dtype, device=kspace.device)
    for _ in range(power_iters):
        y = aha @ x
        eig = torch.linalg.vector_norm(y, dim=-2, keepdim=True)
        x = y / (eig + 1e-12)
    maps = x[..., 0].permute(2, 0, 1)                                    # (C, H, W)
    maps = maps * torch.conj(maps[0] / (maps[0].abs() + 1e-12))
    return maps * (eig[..., 0, 0] > crop)


def maps_path(cfg, anatomy, accel, volume):
    return cfg["data_dir"] / anatomy / "maps" / f"R{accel}" / f"{volume}.h5"


def load_maps(path, slice_idx=None):
    with h5py.File(path, "r") as f:
        m = f["maps"][()] if slice_idx is None else f["maps"][slice_idx]
    m = m.astype(np.float32)
    return m[..., 0] + 1j * m[..., 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--anatomy", nargs="+", default=["knee", "brain"])
    ap.add_argument("--accel", nargs="+", type=int)
    ap.add_argument("--split", choices=["train", "val", "test"], help="only volumes of this split")
    args = ap.parse_args()
    cfg = load_config(args.config)
    es = cfg["espirit"]
    accels = args.accel or cfg["masks"]["accelerations"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for anatomy in args.anatomy:
        volumes = json.load(open(cfg["data_dir"] / anatomy / "volumes.json"))
        if args.split:
            keep = set(json.load(open(cfg["data_dir"] / anatomy / "split.json"))[args.split])
            volumes = [v for v in volumes if v["name"] in keep]
        for accel in accels:
            n_low = num_low_frequencies(cfg["preprocess"]["crop"][anatomy][1],
                                        cfg["masks"]["center_fractions"][accel])
            for v in tqdm(volumes, desc=f"ESPIRiT {anatomy} R{accel} ({n_low} ACS lines)"):
                out = maps_path(cfg, anatomy, accel, v["name"])
                if out.exists():
                    continue
                with h5py.File(cfg["data_dir"] / anatomy / "volumes" / f"{v['name']}.h5", "r") as f:
                    kspace = f["kspace"][()]
                maps = np.stack([
                    espirit_maps(torch.from_numpy(k).to(dev), n_low, es["calib_readout"], es["kernel"],
                                 es["thresh"], es["crop"]).cpu().numpy() for k in kspace])
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp")   # atomic: an interrupted run never leaves a half-written .h5
                with h5py.File(tmp, "w") as f:
                    f.create_dataset("maps", data=np.stack([maps.real, maps.imag], -1).astype(np.float16))
                    f.attrs["num_low_frequencies"] = n_low
                tmp.replace(out)


if __name__ == "__main__":
    main()
