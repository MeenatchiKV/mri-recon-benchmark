"""Choose the number of sampler steps of an iterative diffusion model on the VALIDATION volumes.

Rule (fixed before looking at test data): the smallest step count whose mean foreground SSIM is within
`--tol` (default 0.001) of the best candidate. Writes runs/<anatomy>/<model>/inference.json, which
bench.evaluate applies, and the full comparison (SSIM, PSNR, seconds per slice) to select_steps.csv.

Usage:  python -m bench.select_steps --model kcd --anatomy knee --steps 25 50 100 200 1000
        python -m bench.select_steps --model scoremri --anatomy knee --steps 100 200 500
"""
import argparse
import csv
import json
import time

import numpy as np
import torch

from .config import load_config
from .data import SliceDataset
from .evaluate import load_model
from .metrics import foreground_mask, psnr, ssim

ATTR = {"kcd": "sample_steps", "scoremri": "n_steps"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--model", required=True, choices=list(ATTR))
    ap.add_argument("--anatomy", required=True)
    ap.add_argument("--steps", nargs="+", type=int, required=True)
    ap.add_argument("--items", type=int, default=24, help="validation (slice, R) pairs, spread over the val set")
    ap.add_argument("--tol", type=float, default=0.001)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda")
    model = load_model(cfg, args.model, args.anatomy, device)
    ds = SliceDataset(cfg, args.anatomy, "val")
    idx = np.linspace(0, len(ds) - 1, args.items).astype(int)
    fg_thr = cfg["metrics"]["foreground_threshold"]
    run_dir = cfg["runs_dir"] / args.anatomy / args.model

    rows = []
    for n in sorted(args.steps):
        setattr(model, ATTR[args.model], n)
        scores, secs = [], []
        for i in idx:
            s = ds[i]
            batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else [v]) for k, v in s.items()}
            batch["accel"] = torch.tensor([s["accel"]])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = model(batch)[0].float().cpu().numpy()
            torch.cuda.synchronize()
            secs.append(time.perf_counter() - t0)
            ref, dr = s["target"].numpy(), float(s["max_value"])
            fg = foreground_mask(ref, dr, fg_thr)
            scores.append((ssim(ref, x, dr, fg), psnr(ref, x, dr, fg)))
        sc = np.mean(scores, axis=0)
        rows.append({"steps": n, "SSIM": float(sc[0]), "PSNR": float(sc[1]), "sec_per_slice": float(np.median(secs)),
                     "n_items": len(idx)})
        print(rows[-1], flush=True)

    best = max(r["SSIM"] for r in rows)
    chosen = min(r["steps"] for r in rows if r["SSIM"] >= best - args.tol)
    with open(run_dir / "select_steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({ATTR[args.model]: chosen}, open(run_dir / "inference.json", "w"), indent=1)
    print(f"chosen: {ATTR[args.model]} = {chosen}")


if __name__ == "__main__":
    main()
