"""Run one method on the test split under one or more conditions and write per-slice metrics.

Output: results/<anatomy>/per_slice/<method>__<setting>.csv, one row per (volume, slice, R) with
NMSE PSNR SSIM HFEN VIF LPIPS and the reconstruction time (seconds per slice, excluding data loading and
metric computation; GPU timings are synchronised and exclude 3 warm-up slices).

Usage:
  python -m bench.evaluate --method grappa --anatomy knee --workers 8
  python -m bench.evaluate --method varnet --anatomy knee --conditions nominal noise:0.01 motion:2
  python -m bench.evaluate --method varnet --anatomy brain --train-anatomy knee          # cross-anatomy
  python -m bench.evaluate --method varnet --anatomy knee --mask-type equispaced         # mask shift
  python -m bench.evaluate --method varnet --anatomy knee --accels 12                    # unseen R
"""
import argparse
import csv
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from . import perturb
from .classical import CEILINGS, CLASSICAL, DEFAULTS, reconstruct
from .config import load_config
from .data import SliceDataset
from .metrics import METRICS, Evaluator

_DS = None


MAP_METHODS = {"sense", "cs", "modl", "rim", "kcd", "scoremri"}   # methods that take the ESPIRiT maps as input


def perturbed_maps(cfg, kspace, n_low, device):
    """ESPIRiT maps re-estimated from the perturbed acquisition's ACS lines, as on a scanner. The stored maps
    come from the clean ACS and are used only when the acquisition is unperturbed; using them under noise /
    motion / spikes would give the map-based methods oracle calibration that VarNet (which estimates its maps
    from the measured ACS), GRAPPA and the image-domain networks do not get. Rounded to float16 like the
    stored maps."""
    from .espirit import espirit_maps
    es = cfg["espirit"]
    m = espirit_maps(torch.from_numpy(kspace).to(device), n_low, es["calib_readout"], es["kernel"], es["thresh"],
                     es["crop"]).cpu().numpy()
    return (m.real.astype(np.float16).astype(np.float32)
            + 1j * m.imag.astype(np.float16).astype(np.float32)).astype(np.complex64)


def numpy_sample(s):
    return {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in s.items()} | {
        "max_value": float(s["max_value"])}


def classical_params(cfg, anatomy, method, accel):
    path = cfg["results_dir"] / anatomy / "classical_params.json"
    params = dict(DEFAULTS.get(method, {}))
    if path.exists():
        tuned = json.load(open(path)).get(method, {})
        if tuned:   # an untuned R (12, the unseen-acceleration test) uses the nearest tuned R, like the networks
            near = min(tuned, key=lambda r: abs(int(r) - int(accel)))
            params.update(tuned[str(accel)] if str(accel) in tuned else tuned[near])
    return params


def _init_worker(cfg, anatomy, accels, mask_type):
    global _DS
    _DS = SliceDataset(cfg, anatomy, "test", accels=accels, mask_type=mask_type)


def _classical_job(args):
    i, method, condition, params, maps = args
    s = numpy_sample(_DS[i])
    k = s["kspace"] if method in CEILINGS else perturb.apply(s, condition)
    maps = s["maps"] if maps is None else maps
    t0 = time.perf_counter()
    x = reconstruct(method, k, s["mask"], maps, s["num_low"], **params)
    return i, x, time.perf_counter() - t0


def load_model(cfg, method, train_anatomy, device, tag=""):
    from .models import build_model
    run_dir = cfg["runs_dir"] / train_anatomy / (method + (f"_{tag}" if tag else ""))
    conf = json.load(open(run_dir / "config.json"))
    model = build_model(method, **conf["model_args"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
    inference = run_dir / "inference.json"   # sampler settings chosen on validation (bench.select_steps)
    if inference.exists():
        for k, v in json.load(open(inference)).items():
            setattr(model, k, v)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--method", required=True)
    ap.add_argument("--anatomy", required=True)
    ap.add_argument("--train-anatomy", help="anatomy the model was trained on (default: same)")
    ap.add_argument("--conditions", nargs="+", default=["nominal"])
    ap.add_argument("--accels", nargs="+", type=int)
    ap.add_argument("--mask-type")
    ap.add_argument("--workers", type=int, default=8, help="processes for classical methods")
    ap.add_argument("--limit", type=int, help="first N test items only (smoke test)")
    ap.add_argument("--save-recon", action="store_true", help="store reconstructions (float16) for figures")
    ap.add_argument("--run-tag", default="", help="checkpoint folder suffix, e.g. smoke -> runs/<anatomy>/<method>_smoke")
    ap.add_argument("--slice-stride", type=int, default=1,
                    help="only slices with index %% stride == 0 (fixed subset for slow samplers; bench.stats "
                         "then also compares every method on the common subset)")
    ap.add_argument("--cache", action="store_true",
                    help="store each learned-model reconstruction as it is made and reuse stored ones on restart "
                         "(for slow samplers; results/<anatomy>/recon_cache/<method>/<setting>/)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    accels = args.accels or cfg["masks"]["accelerations"]
    mask_type = args.mask_type or cfg["masks"]["type"]
    train_anatomy = args.train_anatomy or args.anatomy
    is_classical = args.method in CLASSICAL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SliceDataset(cfg, args.anatomy, "test", accels=accels, mask_type=mask_type)
    items = [i for i in range(len(ds)) if ds.items[i][1] % args.slice_stride == 0][: args.limit]
    mc = cfg["metrics"]
    evaluator = Evaluator(device, mc["hfen_sigma"], mc["lpips_net"], mc["foreground_threshold"])
    model = None if is_classical else load_model(cfg, args.method, train_anatomy, device, args.run_tag)
    hw = platform.processor() if is_classical else torch.cuda.get_device_name()
    acquisitions = {v["name"]: v["acquisition"] for v in json.load(open(cfg["data_dir"] / args.anatomy / "volumes.json"))}

    out_dir = cfg["results_dir"] / args.anatomy / "per_slice"
    out_dir.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        setting = f"{mask_type}_{condition.replace(':', '-')}" + ("" if accels == cfg["masks"]["accelerations"]
                                                                  else "_R" + "-".join(map(str, accels)))
        if train_anatomy != args.anatomy:
            setting += f"_train-{train_anatomy}"
        recons, times = {}, {}
        remap = condition != "nominal" and args.method in MAP_METHODS

        def job(i):
            maps = None
            if remap:
                s = numpy_sample(ds[i])
                maps = perturbed_maps(cfg, perturb.apply(s, condition), s["num_low"], device)
            return i, args.method, condition, classical_params(cfg, args.anatomy, args.method, ds.items[i][2]), maps

        if is_classical:
            chunk = 4 * args.workers    # maps are made on the GPU here and shipped in chunks to bound memory
            with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                                     initargs=(cfg, args.anatomy, accels, mask_type)) as ex:
                n = 0
                for c0 in range(0, len(items), chunk):
                    for i, x, dt in ex.map(_classical_job, [job(i) for i in items[c0:c0 + chunk]]):
                        recons[i], times[i] = x, dt
                        if n % 50 == 0:
                            print(f"{args.method} {condition}: {n}/{len(items)}", flush=True)
                        n += 1
        else:
            cache_dir = cfg["results_dir"] / args.anatomy / "recon_cache" / args.method / setting
            if args.cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                for n, i in enumerate(items):
                    cached = cache_dir / f"{i}.npz"
                    if args.cache and cached.exists():
                        c = np.load(cached)
                        recons[i], times[i] = c["x"], float(c["t"])
                        continue
                    s = ds[i]
                    s["masked_kspace"] = torch.from_numpy(perturb.apply(numpy_sample(s), condition))
                    if remap:
                        s["maps"] = torch.from_numpy(perturbed_maps(cfg, s["masked_kspace"].numpy(),
                                                                    s["num_low"], device))
                    batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else [v]) for k, v in s.items()}
                    batch["num_low"] = torch.tensor([s["num_low"]])
                    batch["accel"] = torch.tensor([s["accel"]])
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    x = model(batch)
                    torch.cuda.synchronize()
                    recons[i], times[i] = x[0].float().cpu().numpy(), time.perf_counter() - t0
                    if args.cache:
                        np.savez(cache_dir / f"{i}.tmp.npz", x=recons[i], t=times[i])
                        (cache_dir / f"{i}.tmp.npz").replace(cached)      # atomic: no half-written files
                    if n % 50 == 0:
                        print(f"{args.method} {condition}: {n}/{len(items)}", flush=True)
            for i in list(items)[:3]:   # warm-up slices: CUDA kernels / cuFFT plans not yet cached
                times[i] = float("nan")

        rows = []
        for i in items:
            name, sl, r = ds.items[i]
            sref, sref_max = _sense_target(cfg, args.anatomy, name, sl)
            m = evaluator(_target(ds, name, sl), recons[i], ds.max_value[name], sref, sref_max)
            rows.append({"method": args.method, "anatomy": args.anatomy, "train_anatomy": train_anatomy,
                         "mask_type": mask_type, "condition": condition, "volume": name,
                         "acquisition": acquisitions[name], "slice": sl, "accel": r,
                         **m, "TIME": times[i], "hardware": hw})
        path = out_dir / f"{args.method}__{setting}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summary = {r: {k: np.nanmean([row[k] for row in rows if row["accel"] == r]) for k in METRICS + ["TIME"]}
                   for r in accels}
        print(path.name, json.dumps({r: {k: round(v, 4) for k, v in d.items()} for r, d in summary.items()}))

        if args.save_recon:
            rec_dir = cfg["results_dir"] / args.anatomy / "recon" / args.method / setting
            rec_dir.mkdir(parents=True, exist_ok=True)
            for r in accels:
                for name in sorted({ds.items[i][0] for i in items}):
                    idx = sorted((ds.items[i][1], i) for i in items if ds.items[i][0] == name and ds.items[i][2] == r)
                    np.save(rec_dir / f"{name}_R{r}.npy", np.stack([recons[i] for _, i in idx]).astype(np.float16))


def _target(ds, name, sl):
    import h5py
    with h5py.File(ds.vol_dir / f"{name}.h5", "r") as f:
        return f["target"][sl]


def _sense_target(cfg, anatomy, name, sl):
    import h5py
    from .reference import ref_path
    with h5py.File(ref_path(cfg, anatomy, name), "r") as f:
        return f["target"][sl], float(f.attrs["max"])


if __name__ == "__main__":
    main()
