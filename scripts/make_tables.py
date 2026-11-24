"""Write the paper's LaTeX tables straight from the result files, so every number in the paper has one source.

  python scripts/make_tables.py            -> paper/tables/*.tex

Tables
  models.tex           implementation source, parameters, training budget, best epoch, GPU hours
  nominal_R{4,8}.tex   volume-level mean and 95 % bootstrap CI per metric, RSS reference (bench.stats summary)
  reference.tex        SSIM / PSNR against the RSS and the SENSE reference, with mean ranks
  robustness.tex       SSIM (RSS reference) under each k-space perturbation and shift, R = 4 and 8
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "knee"
TAB = RES / "tables"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["zf", "grappa", "sense", "cs", "unet", "varnet", "modl", "rim", "swinmr", "kcd", "scoremri"]
NAME = {"zf": "Zero-filled", "grappa": "GRAPPA", "sense": "CG-SENSE", "cs": "CS (L1-wavelet)", "unet": "U-Net",
        "varnet": "E2E-VarNet", "modl": "MoDL", "rim": "RIM", "swinmr": "SwinMR", "kcd": "KCD",
        "scoremri": "score-MRI"}
FAMILY = {"zf": "Classical", "grappa": "Classical", "sense": "Classical", "cs": "Classical",
          "unet": "Image-domain CNN", "varnet": "Unrolled", "modl": "Unrolled", "rim": "Unrolled",
          "swinmr": "Transformer", "kcd": "Diffusion", "scoremri": "Diffusion"}
OPT = {"adam": "Adam", "rmsprop": "RMSprop"}
# metric, direction (+1 higher is better), decimals
METRICS = [("NMSE", -1, 4), ("PSNR", 1, 2), ("SSIM", 1, 3), ("HFEN", -1, 3), ("VIF", 1, 3), ("LPIPS", -1, 3)]


def signed(v):
    r = int(round(v))
    return f"{r:+d}" if r else "0"


def bold_best(values, direction, fmt):
    best = max(values, key=lambda v: direction * v)
    return [("\\textbf{" + fmt(v) + "}") if v == best else fmt(v) for v in values]


def write(name, body):
    (OUT / name).write_text(body, encoding="utf-8")
    print("wrote", OUT / name)


def nominal(accel, ref="rss"):
    s = pd.read_csv(TAB / f"random_nominal_ref-{ref}" / "all" / "summary.csv")
    s = s[(s.accel == accel) & s.method.isin(ORDER)].set_index("method").loc[[m for m in ORDER if m in set(s.method)]]
    cols = {}
    for m, d, k in METRICS:
        f = (lambda v, k=k: f"{v:.{k}f}")
        means = bold_best(list(s[f"{m}_mean"]), d, f)
        cols[m] = [f"{mu} \\scriptsize[{lo:.{k}f}, {hi:.{k}f}]"
                   for mu, lo, hi in zip(means, s[f"{m}_ci_lo"], s[f"{m}_ci_hi"])]
    rows = []
    for i, meth in enumerate(s.index):
        t = s.loc[meth, "TIME_median_s"]
        rows.append(" & ".join([NAME[meth]] + [cols[m][i] for m, _, _ in METRICS] + [f"{t:.3g}"]) + " \\\\")
    up, down = "$\\uparrow$", "$\\downarrow$"
    head = " & ".join(["Method"] + [m + " " + (up if d > 0 else down) for m, d, _ in METRICS]
                      + ["Time (s)"]) + " \\\\"
    write(f"nominal_R{accel}.tex", "\n".join([
        "\\begin{tabular}{l" + "c" * (len(METRICS) + 1) + "}", "\\toprule", head, "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}"]))


def reference():
    rows = []
    data = {}
    for ref, suffix in [("rss", ""), ("sense", "")]:
        s = pd.read_csv(TAB / f"random_nominal_ref-{ref}" / "all" / "summary.csv")
        r = pd.read_csv(TAB / f"random_nominal_ref-{ref}" / "all" / "ranking.csv")
        data[ref] = (s.set_index(["method", "accel"]), r.set_index(["method", "accel"]))
    for meth in ORDER:
        cells = [NAME[meth]]
        for accel in (4, 8):
            for ref in ("rss", "sense"):
                s, r = data[ref]
                cells.append(f"{s.loc[(meth, accel), 'SSIM_mean']:.3f}")
                cells.append(f"{r.loc[(meth, accel), 'mean_rank_equal_weights']:.2f}")
        rows.append(" & ".join(cells) + " \\\\")
    write("reference.tex", "\n".join([
        "\\begin{tabular}{l" + "cc" * 4 + "}", "\\toprule",
        " & \\multicolumn{4}{c}{$R=4$} & \\multicolumn{4}{c}{$R=8$} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        " & \\multicolumn{2}{c}{RSS ref.} & \\multicolumn{2}{c}{SENSE ref.} & \\multicolumn{2}{c}{RSS ref.}"
        " & \\multicolumn{2}{c}{SENSE ref.} \\\\",
        "Method" + " & SSIM & rank" * 4 + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))


def models():
    src = {"unet": "fastMRI (official)", "varnet": "fastMRI (official)", "modl": "authors' design, re-implemented",
           "rim": "Lonning et al. ConvGRU RIM", "swinmr": "SwinIR/SwinMR blocks, reduced width",
           "kcd": "authors' KSC-Diff code, multi-coil", "scoremri": "official score-MRI code, $nf=64$"}
    rows = []
    for m in ORDER[4:]:
        run = ROOT / "runs" / "knee" / m
        c = json.load(open(run / "config.json"))
        log = pd.read_csv(run / "log.csv")
        tr = c["train"]
        if "steps" in tr:
            budget, best = f"{tr['steps'] // 1000}k it.", "final (EMA)"
        else:
            budget = f"{tr['epochs']} ep."
            best = str(int(log.loc[(log.val_ssim_R4 + log.val_ssim_R8).idxmax(), "epoch"]) + 1)
        hours = log.seconds.sum() / 3600
        rows.append(f"{NAME[m]} & {src[m]} & {c['params'] / 1e6:.2f} & {OPT[tr.get('optimizer', 'adam')]}"
                    f" & {tr['lr']:g} & {budget} & {best} & {hours:.1f} \\\\")
    write("models.tex", "\n".join([
        "\\begin{tabular}{llcccccc}", "\\toprule",
        "Method & Implementation & Params (M) & Optim. & LR & Budget & Best epoch & GPU h \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}"]))


def robustness():
    t = pd.read_csv(TAB / "robustness_SSIM.csv", index_col=0)
    groups = [("Nominal", ["nominal"]), ("Noise $\\sigma$", ["noise-0.005", "noise-0.01", "noise-0.02", "noise-0.04"]),
              ("Motion (px)", ["motion-1", "motion-2", "motion-4"]),
              ("Dropout", ["dropout-0.05", "dropout-0.1", "dropout-0.2"]), ("Spikes", ["spike-2", "spike-5", "spike-10"]),
              ("Equisp.", ["equispaced_nominal"])]
    cond = [c for _, cs in groups for c in cs]
    label = lambda c: c.split("-")[-1] if "-" in c and not c.startswith("equi") else "--"
    out = []
    for accel in (4, 8):
        cols = [f"R{accel}:{c}" for c in cond]
        sub = t.loc[[m for m in ORDER if m in t.index and m != "scoremri"], cols]   # score-MRI: own table
        rows = []
        for m in sub.index:
            vals = [f"{v:.3f}" if pd.notna(v) else "--" for v in sub.loc[m]]
            rows.append(" & ".join([NAME[m]] + vals) + " \\\\")
        out += [f"\\multicolumn{{{len(cond) + 1}}}{{l}}{{\\textit{{$R={accel}$}}}} \\\\", *rows, "\\midrule"]
    head1 = " & ".join([""] + [f"\\multicolumn{{{len(cs)}}}{{c}}{{{g}}}" for g, cs in groups]) + " \\\\"
    head2 = " & ".join(["Method"] + [label(c) for c in cond]) + " \\\\"
    write("robustness.tex", "\n".join([
        "\\begin{tabular}{l" + "c" * len(cond) + "}", "\\toprule", head1, head2, "\\midrule", *out[:-1],
        "\\bottomrule", "\\end{tabular}"]))


def scoremri_subset():
    """All 11 methods on the every-16th-slice subset that score-MRI was evaluated on: SSIM change vs the same
    method's clean result on those slices (volume means), and score-MRI's rank among the 11 methods."""
    conds = [("noise-0.01", "noise .01"), ("noise-0.04", "noise .04"), ("motion-2", "motion 2"),
             ("motion-4", "motion 4"), ("dropout-0.2", "dropout 20\\,\\%"), ("spike-2", "spikes 2"),
             ("spike-10", "spikes 10"), ("equispaced_nominal", "equisp.")]
    ps = RES / "per_slice"

    def vol_mean(m, setting, accel):
        d = pd.read_csv(ps / f"{m}__{setting}.csv")
        d = d[(d.slice % 16 == 0) & (d.accel == accel)]
        return d.groupby("volume").SSIM.mean().mean()

    out = []
    for accel in (4, 8):
        rows, ranks = [], {}
        table = {m: {c: vol_mean(m, f"random_{c}" if not c.startswith("equi") else c, accel) for c, _ in conds}
                 | {"nominal": vol_mean(m, "random_nominal", accel)} for m in ORDER}
        for c, _ in conds:
            order = sorted(ORDER, key=lambda m: -table[m][c])
            ranks[c] = order.index("scoremri") + 1
        for m in ORDER:
            base = table[m]["nominal"]
            cells = [NAME[m], f"{base:.3f}"] + [signed((table[m][c] - base) * 100) for c, _ in conds]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append(" & ".join(["\\textit{score-MRI rank (of 11)}", ""] + [str(ranks[c]) for c, _ in conds]) + " \\\\")
        out += [f"\\multicolumn{{{len(conds) + 2}}}{{l}}{{\\textit{{$R={accel}$}}}} \\\\", *rows, "\\midrule"]
    head = " & ".join(["Method", "clean SSIM"] + [lab for _, lab in conds]) + " \\\\"
    write("scoremri_subset.tex", "\n".join([
        "\\begin{tabular}{lc" + "c" * len(conds) + "}", "\\toprule", head, "\\midrule", *out[:-1],
        "\\bottomrule", "\\end{tabular}"]))


def ranking():
    """Weight sensitivity of the overall ranking (nominal, RSS reference): mean metric rank with equal weights,
    probability of ranking first and 95 % rank interval over 10 000 Dirichlet(1) metric-weight vectors."""
    r = pd.read_csv(TAB / "random_nominal_ref-rss" / "all" / "ranking.csv").set_index(["method", "accel"])
    rows = []
    for m in sorted(ORDER, key=lambda m: r.loc[(m, 4), "mean_rank_equal_weights"]):
        cells = [NAME[m]]
        for accel in (4, 8):
            x = r.loc[(m, accel)]
            cells += [f"{x.mean_rank_equal_weights:.2f}", f"{x.p_rank1_random_weights:.2f}",
                      f"{x['rank_p2.5']:.0f}--{x['rank_p97.5']:.0f}"]
        rows.append(" & ".join(cells) + " \\\\")
    tau = {a: r.xs(a, level="accel").kendall_tau_median.iloc[0] for a in (4, 8)}
    write("ranking.tex", "\n".join([
        "\\begin{tabular}{lcccccc}", "\\toprule",
        " & \\multicolumn{3}{c}{$R=4$} & \\multicolumn{3}{c}{$R=8$} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Method & mean rank & $P(\\text{rank }1)$ & 95\\,\\% range & mean rank & $P(\\text{rank }1)$ & 95\\,\\% range \\\\",
        "\\midrule", *rows, "\\midrule",
        f"\\multicolumn{{7}}{{l}}{{median Kendall $\\tau$ vs equal-weight ranking: {tau[4]:.2f} ($R=4$), "
        f"{tau[8]:.2f} ($R=8$)}} \\\\", "\\bottomrule", "\\end{tabular}"]))


if __name__ == "__main__":
    models()
    for r in (4, 8):
        nominal(r)
    reference()
    robustness()
    scoremri_subset()
    ranking()
