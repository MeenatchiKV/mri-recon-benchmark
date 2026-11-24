"""Second reference image: the SENSE-combined fully sampled image  |sum_c conj(S_c) F^H k_c|.

Why: the RSS reference carries a positive noise bias in low-signal tissue that RSS-combining methods share and
sensitivity-combining methods do not; on noisy PD-FS knee scans even a fully sampled SENSE-combined image
reaches only SSIM ~0.87 against RSS. Every method is therefore scored against both references.

The reference maps are calibrated on the FULLY SAMPLED k-space (central 32 PE lines x 64 readout), so they are
independent of R and of the maps any method uses. Stored in data/<anatomy>/ref_sense/<volume>.h5 ('target').

Usage:  python -m bench.reference --anatomy knee
"""
import argparse
import json

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .config import load_config
from .espirit import espirit_maps
from .mri import ifft2c

CALIB_LINES = 32


def ref_path(cfg, anatomy, volume):
    return cfg["data_dir"] / anatomy / "ref_sense" / f"{volume}.h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--anatomy", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    es = cfg["espirit"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for v in tqdm(json.load(open(cfg["data_dir"] / args.anatomy / "volumes.json")), desc="SENSE reference"):
        out = ref_path(cfg, args.anatomy, v["name"])
        if out.exists():
            continue
        with h5py.File(cfg["data_dir"] / args.anatomy / "volumes" / f"{v['name']}.h5", "r") as f:
            kspace = f["kspace"][()]
        imgs = []
        for k in kspace:
            k = torch.from_numpy(k).to(dev)
            maps = espirit_maps(k, CALIB_LINES, es["calib_readout"], es["kernel"], es["thresh"], es["crop"])
            imgs.append(torch.sum(torch.conj(maps) * ifft2c(k), 0).abs().cpu().numpy())
        imgs = np.stack(imgs).astype(np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with h5py.File(tmp, "w") as f:
            f.create_dataset("target", data=imgs)
            f.attrs["max"] = float(imgs.max())
            f.attrs["calib_lines"] = CALIB_LINES
        tmp.replace(out)


if __name__ == "__main__":
    main()
