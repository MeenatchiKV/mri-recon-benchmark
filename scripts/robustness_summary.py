"""Robustness summary on the every-4th-slice subset: volume-mean metric per method x condition, and the drop
from the same method's nominal result on the same slices. Writes results/<anat>/tables/robustness_<metric>.csv."""
import sys, glob, os
import pandas as pd
anat = sys.argv[1] if len(sys.argv) > 1 else "knee"
d = f"results/{anat}/per_slice"
rows = []
for f in glob.glob(f"{d}/*.csv"):
    m, s = os.path.basename(f)[:-4].split("__")
    if m == "scoremri" and s != "random_nominal":
        continue
    x = pd.read_csv(f)
    x = x[x.slice % 4 == 0]
    x["setting"] = s
    rows.append(x)
df = pd.concat(rows)
order = ["random_nominal"] + [f"random_{c}" for c in ["noise-0.005","noise-0.01","noise-0.02","noise-0.04","motion-1",
         "motion-2","motion-4","dropout-0.05","dropout-0.1","dropout-0.2","spike-2","spike-5","spike-10"]] + ["equispaced_nominal"]
for metric in ["SSIM", "PSNR", "SSIM_sref"]:
    v = df.groupby(["setting", "method", "accel", "volume"])[metric].mean().groupby(["setting", "method", "accel"]).mean()
    out = []
    for r in sorted(df.accel.unique()):
        t = v.xs(r, level="accel").unstack("setting")
        if r == 12:
            t = t[["random_nominal_R12"]]
        else:
            t = t[[c for c in order if c in t]]
        t.columns = [f"R{r}:{c.replace('random_', '')}" for c in t.columns]
        out.append(t)
    tab = pd.concat(out, axis=1).round(4)
    tab.to_csv(f"results/{anat}/tables/robustness_{metric}.csv")
    print(metric); print(tab.to_string()); print()
