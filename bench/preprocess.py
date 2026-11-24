"""Raw fastMRI multi-coil volumes -> cropped, coil-compressed, intensity-scaled volumes + volume-level split.

For every raw volume:
  1. coil images = IFFT2(k-space); centre crop to 320 x 320 (the fastMRI reference FOV)
  2. SVD coil compression to `virtual_coils` (one compression matrix per volume)
  3. k-space = FFT2(compressed coil images)
  4. one scale factor per volume from the central `scale_lines` PE lines only (acquired at every R),
     so the scaling uses no information a real undersampled scan would not have
  5. reference = RSS of the fully sampled compressed coil images

Usage:  python -m bench.preprocess [--anatomy knee brain] [--workers 3] [--limit N]
"""
import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from .config import load_config
from .mri import center_crop_np, fft2c_np, ifft2c_np, rss_np


def process_volume(raw_path, out_path, crop, virtual_coils, scale_lines):
    with h5py.File(raw_path, "r") as f:
        kspace = f["kspace"][()]                          # (S, C, H, W)
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    coil_imgs = center_crop_np(ifft2c_np(kspace), crop)  # (S, C, 320, 320)
    del kspace
    n_slices, n_coils = coil_imgs.shape[:2]

    nv = min(virtual_coils, n_coils)
    flat = coil_imgs.transpose(1, 0, 2, 3).reshape(n_coils, -1)
    eigval, eigvec = np.linalg.eigh(flat @ flat.conj().T)
    order = np.argsort(eigval)[::-1]
    u = eigvec[:, order[:nv]]                             # (C, nv)
    energy = float(eigval[order[:nv]].sum() / eigval.sum())
    coil_imgs = np.einsum("cv,schw->svhw", u.conj(), coil_imgs).astype(np.complex64)

    kspace = fft2c_np(coil_imgs)
    w = kspace.shape[-1]
    low = np.zeros_like(kspace)
    c0 = w // 2 - scale_lines // 2
    low[..., c0:c0 + scale_lines] = kspace[..., c0:c0 + scale_lines]
    scale = 1.0 / float(rss_np(ifft2c_np(low), axis=1).max())
    kspace *= scale
    target = (rss_np(coil_imgs, axis=1) * scale).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("kspace", data=kspace.astype(np.complex64))
        f.create_dataset("target", data=target)
        f.attrs["acquisition"] = str(attrs.get("acquisition", "unknown"))
        f.attrs["patient_id"] = str(attrs.get("patient_id", ""))
        f.attrs["raw_coils"] = n_coils
        f.attrs["coil_energy_kept"] = energy
        f.attrs["scale"] = scale
        f.attrs["max"] = float(target.max())
    return {"name": out_path.stem, "slices": int(n_slices), "raw_coils": int(n_coils), "coils": int(nv),
            "acquisition": str(attrs.get("acquisition", "unknown")),
            "patient_id": str(attrs.get("patient_id", "")), "coil_energy_kept": energy}


def make_split(volumes, n_train, n_val, n_test, seed):
    """Volume-level split stratified by acquisition, with all volumes of a patient kept together."""
    rng = np.random.default_rng(seed)
    by_patient = defaultdict(list)
    for v in volumes:
        by_patient[v["patient_id"] or v["name"]].append(v)
    groups = list(by_patient.values())
    by_acq = defaultdict(list)
    for g in groups:
        by_acq[g[0]["acquisition"]].append(g)
    total = n_train + n_val + n_test
    quota = {"train": n_train / total, "val": n_val / total, "test": n_test / total}
    split = {"train": [], "val": [], "test": []}
    # round-robin over shuffled strata keeps each subset's contrast mix close to the whole
    ordered = []
    for acq in sorted(by_acq):
        gs = by_acq[acq]
        rng.shuffle(gs)
        ordered.extend(gs)
    for g in ordered:
        n_done = {k: sum(len(x) for x in split[k]) for k in split}
        n_all = sum(n_done.values()) + len(g)
        deficit = {k: quota[k] * n_all - n_done[k] for k in split}
        split[max(deficit, key=deficit.get)].append(g)
    return {k: sorted(v["name"] for g in gs for v in g) for k, gs in split.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--anatomy", nargs="+", default=["knee", "brain"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, help="only the first N volumes (smoke test)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pp, sp = cfg["preprocess"], cfg["split"]

    for anatomy in args.anatomy:
        raws = sorted(Path(cfg["raw_dirs"][anatomy]).glob("*.h5"))[: args.limit]
        out_dir = cfg["data_dir"] / anatomy / "volumes"
        jobs = [(r, out_dir / r.name, tuple(pp["crop"][anatomy]), pp["virtual_coils"], pp["scale_lines"])
                for r in raws]
        with ProcessPoolExecutor(args.workers) as ex:
            volumes = list(ex.map(process_volume, *zip(*jobs)))
        for v in volumes:
            print(f"{anatomy} {v['name']}: {v['slices']} slices, {v['raw_coils']}->{v['coils']} coils, "
                  f"energy kept {v['coil_energy_kept']:.4f}, {v['acquisition']}")
        n = len(volumes)
        test_dir = cfg["test_raw_dirs"].get(anatomy)
        test_raws = sorted(Path(test_dir).glob("*.h5"))[: args.limit] if test_dir and Path(test_dir).exists() else []
        if test_raws:   # external test set: fastMRI validation volumes, disjoint from the training batch
            jobs = [(r, out_dir / r.name, tuple(pp["crop"][anatomy]), pp["virtual_coils"], pp["scale_lines"])
                    for r in test_raws]
            with ProcessPoolExecutor(args.workers) as ex:
                test_volumes = list(ex.map(process_volume, *zip(*jobs)))
            n_val = max(1, n // 6) if args.limit else sp["val_with_external_test"]
            split = make_split(volumes, n - n_val, n_val, 0, seed=sp["seed"])
            split["test"] = sorted(v["name"] for v in test_volumes)
            volumes += test_volumes
        else:
            if args.limit:
                n_test = max(1, n // 4)
                n_val = max(1, n // 6)
                counts = (n - n_val - n_test, n_val, n_test)
            else:
                counts = (sp["train"], sp["val"], sp["test"])
            split = make_split(volumes, *counts, seed=sp["seed"])
        with open(cfg["data_dir"] / anatomy / "split.json", "w") as f:
            json.dump(split, f, indent=1)
        with open(cfg["data_dir"] / anatomy / "volumes.json", "w") as f:
            json.dump(volumes, f, indent=1)
        print(anatomy, {k: len(v) for k, v in split.items()})


if __name__ == "__main__":
    main()
