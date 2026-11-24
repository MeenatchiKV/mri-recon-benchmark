"""Grid-search the regularisation of the classical methods on the VALIDATION volumes (never the test set),
per anatomy and R, maximising mean SSIM. Writes results/<anatomy>/classical_params.json and the full grid
to results/<anatomy>/classical_tuning.csv (reported in the supplement).

Usage:  python -m bench.tune --anatomy knee [--every 3] [--workers 8]
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

from .classical import DEFAULTS, reconstruct
from .config import load_config
from .data import SliceDataset
from .evaluate import numpy_sample
from .metrics import foreground_mask, psnr, ssim

GRIDS = {
    # extended after the first pass picked the largest lambda / smallest kernel (grid edge)
    "grappa": {"lamda": [1e-3, 1e-2, 5e-2, 1e-1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], "kernel": [(5, 5), (3, 5), (3, 3)]},
    "sense": {"lamda": [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0]},
    "cs": {"lamda": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]},
}
_DS = None


_FG = 0.1


def _init(cfg, anatomy):
    global _DS, _FG
    _DS = SliceDataset(cfg, anatomy, "val")
    _FG = cfg["metrics"]["foreground_threshold"]


def _job(args):
    """Foreground SSIM / PSNR, the primary metrics of the benchmark."""
    i, method, params = args
    s = numpy_sample(_DS[i])
    x = reconstruct(method, s["masked_kspace"], s["mask"], s["maps"], s["num_low"], **params)
    fg = foreground_mask(s["target"], s["max_value"], _FG)
    return ssim(s["target"], x, s["max_value"], fg), psnr(s["target"], x, s["max_value"], fg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--anatomy", required=True)
    ap.add_argument("--methods", nargs="+", default=list(GRIDS))
    ap.add_argument("--every", type=int, default=3, help="use every n-th validation slice")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = SliceDataset(cfg, args.anatomy, "val")
    out = cfg["results_dir"] / args.anatomy
    out.mkdir(parents=True, exist_ok=True)
    grid_csv = out / "classical_tuning.csv"
    fields = ["method", "accel", "params", "SSIM", "PSNR", "n_slices"]
    done = {}   # every finished grid point is appended at once, so an interrupted run resumes
    if grid_csv.exists():
        with open(grid_csv) as f:
            for r in csv.DictReader(f):
                done[(r["method"], int(r["accel"]), r["params"])] = r
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(cfg, args.anatomy)) as ex:
        for method in args.methods:
            keys = list(GRIDS[method])
            for accel in cfg["masks"]["accelerations"]:
                idx = [i for i, (_, s, r) in enumerate(ds.items) if r == accel and s % args.every == 0]
                for values in product(*GRIDS[method].values()):
                    params = json.dumps(dict(DEFAULTS.get(method, {})) | dict(zip(keys, values)))
                    if (method, accel, params) in done:
                        continue
                    scores = np.array(list(ex.map(_job, [(i, method, json.loads(params)) for i in idx])))
                    row = {"method": method, "accel": accel, "params": params,
                           "SSIM": float(scores[:, 0].mean()), "PSNR": float(scores[:, 1].mean()), "n_slices": len(idx)}
                    done[(method, accel, params)] = row
                    new = not grid_csv.exists()
                    with open(grid_csv, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        if new:
                            w.writeheader()
                        w.writerow(row)
                    print(row, flush=True)
    best = {}
    for (method, accel, params), r in done.items():
        cur = best.setdefault(method, {}).get(str(accel))
        if cur is None or float(r["SSIM"]) > cur[0]:
            best[method][str(accel)] = (float(r["SSIM"]), json.loads(params))
    params = {m: {r: p for r, (_, p) in d.items()} for m, d in best.items()}
    json.dump(params, open(out / "classical_params.json", "w"), indent=1)
    print(json.dumps(params, indent=1))


if __name__ == "__main__":
    main()
