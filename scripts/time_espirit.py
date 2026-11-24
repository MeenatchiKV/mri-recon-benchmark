"""Median ESPIRiT time per slice (GPU, synchronised, 3 warm-up slices) on test slices, per acceleration."""
import json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.config import load_config
from bench.espirit import espirit_maps
from bench.masks import num_low_frequencies

cfg = load_config(None); es = cfg["espirit"]; dev = torch.device("cuda")
test = json.load(open(cfg["data_dir"] / "knee" / "split.json"))["test"]
out = {}
for accel in cfg["masks"]["accelerations"]:
    n_low = num_low_frequencies(cfg["preprocess"]["crop"]["knee"][1], cfg["masks"]["center_fractions"][accel])
    times = []
    for name in test[:5]:
        with h5py.File(cfg["data_dir"] / "knee" / "volumes" / f"{name}.h5", "r") as f:
            ks = f["kspace"][::4]
        for k in ks:
            kt = torch.from_numpy(k).to(dev); torch.cuda.synchronize(); t = time.perf_counter()
            espirit_maps(kt, n_low, es["calib_readout"], es["kernel"], es["thresh"], es["crop"])
            torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    times = times[3:]
    out[accel] = dict(n_low=n_low, n=len(times), median=float(np.median(times)), iqr=[float(np.percentile(times, 25)), float(np.percentile(times, 75))])
    print(accel, out[accel], flush=True)
p = ROOT / "results" / "knee" / "tables" / "espirit_time.json"
json.dump(out, open(p, "w"), indent=1); print(p)
