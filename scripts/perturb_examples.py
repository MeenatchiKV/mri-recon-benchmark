"""Reconstructions of ONE test slice under selected k-space perturbations, for the visual robustness figure.

Uses exactly the code path of bench.evaluate (same deterministic perturbation per volume/slice/R/condition, ESPIRiT
maps re-estimated from the perturbed data for the map-based methods, tuned classical parameters, best checkpoints),
so the metrics written here must equal the rows of results/knee/per_slice/<method>__random_<condition>.csv for this
slice; the script checks that and prints the largest difference.

    python scripts/perturb_examples.py            -> results/knee/recon_perturb/<method>__<condition>.npy + metrics.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import perturb  # noqa: E402
from bench.classical import CLASSICAL, reconstruct  # noqa: E402
from bench.config import load_config  # noqa: E402
from bench.data import SliceDataset  # noqa: E402
from bench.evaluate import (MAP_METHODS, _sense_target, _target, classical_params, load_model,  # noqa: E402
                            numpy_sample, perturbed_maps)
from bench.metrics import Evaluator  # noqa: E402

VOLUME, SLICE, ACCEL = "file1000495", 24, 4
METHODS = ["zf", "grappa", "sense", "cs", "unet", "varnet", "modl", "rim", "swinmr", "kcd"]
# spike:2 rather than spike:10: on this slice the ten-spike results are far from the test means (e.g. KCD 0.06 vs
# 0.39 SSIM), whereas the two-spike results are close to them, so the figure stays representative.
CONDITIONS = ["nominal", "noise:0.04", "motion:4", "dropout:0.2", "spike:2"]


def main():
    cfg = load_config(None)
    dev = torch.device("cuda")
    out = cfg["results_dir"] / "knee" / "recon_perturb"
    out.mkdir(parents=True, exist_ok=True)
    ds = SliceDataset(cfg, "knee", "test", accels=[ACCEL], mask_type="random")
    i = ds.items.index((VOLUME, SLICE, ACCEL))
    mc = cfg["metrics"]
    ev = Evaluator(dev, mc["hfen_sigma"], mc["lpips_net"], mc["foreground_threshold"])
    ref = _target(ds, VOLUME, SLICE)
    sref, sref_max = _sense_target(cfg, "knee", VOLUME, SLICE)
    rows = []
    for method in METHODS:
        model = None if method in CLASSICAL else load_model(cfg, method, "knee", dev)
        for cond in CONDITIONS:
            s = ds[i]
            ns = numpy_sample(s)
            k = perturb.apply(ns, cond)
            maps = ns["maps"]
            if cond != "nominal" and method in MAP_METHODS:
                maps = perturbed_maps(cfg, k, ns["num_low"], dev)
            if model is None:
                x = reconstruct(method, k, ns["mask"], maps, ns["num_low"],
                                **classical_params(cfg, "knee", method, ACCEL))
            else:
                s["masked_kspace"] = torch.from_numpy(k)
                s["maps"] = torch.from_numpy(maps)
                batch = {kk: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else [v]) for kk, v in s.items()}
                batch["num_low"] = torch.tensor([s["num_low"]])
                batch["accel"] = torch.tensor([s["accel"]])
                with torch.no_grad():
                    x = model(batch)[0].float().cpu().numpy()
            x = np.asarray(x, np.float32)
            np.save(out / f"{method}__{cond.replace(':', '-')}.npy", x.astype(np.float16))
            m = ev(ref, x, ds.max_value[VOLUME], sref, sref_max)
            stored = pd.read_csv(cfg["results_dir"] / "knee" / "per_slice" /
                                 f"{method}__random_{cond.replace(':', '-')}.csv")
            st = stored[(stored.volume == VOLUME) & (stored.slice == SLICE) & (stored.accel == ACCEL)]
            rows.append({"method": method, "condition": cond, "SSIM": m["SSIM"], "PSNR": m["PSNR"],
                         "SSIM_stored": float(st.SSIM.iloc[0]) if len(st) else np.nan})
            print(rows[-1], flush=True)
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    print("max |SSIM - stored SSIM| =", float((df.SSIM - df.SSIM_stored).abs().max()))


if __name__ == "__main__":
    main()
