"""Paper figures, drawn straight from the result files (python scripts/make_figures.py -> paper/figures/*.pdf).

Colour encodes the method FAMILY (reference data-viz palette, documented order); marker / label identifies the
method, so identity never depends on colour alone (11 methods exceed the 8 distinguishable hues).
"""
import json
from pathlib import Path

import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results" / "knee"
TAB = RES / "tables"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["zf", "grappa", "sense", "cs", "unet", "varnet", "modl", "rim", "swinmr", "kcd", "scoremri"]
NAME = {"zf": "Zero-filled", "grappa": "GRAPPA", "sense": "CG-SENSE", "cs": "CS", "unet": "U-Net",
        "varnet": "E2E-VarNet", "modl": "MoDL", "rim": "RIM", "swinmr": "SwinMR", "kcd": "KCD",
        "scoremri": "score-MRI"}
FAMILY = {"zf": "Classical", "grappa": "Classical", "sense": "Classical", "cs": "Classical",
          "unet": "Image-domain CNN", "varnet": "Unrolled", "modl": "Unrolled", "rim": "Unrolled",
          "swinmr": "Transformer", "kcd": "Diffusion", "scoremri": "Diffusion"}
FAM_COLOR = {"Classical": "#2a78d6", "Image-domain CNN": "#eb6834", "Unrolled": "#1baf7a",
             "Transformer": "#eda100", "Diffusion": "#e87ba4"}
MARKER = {"zf": "o", "grappa": "s", "sense": "^", "cs": "D", "unet": "o", "varnet": "o", "modl": "s",
          "rim": "^", "swinmr": "o", "kcd": "o", "scoremri": "s"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#d95926", "#f3c9b6", "#ecebe8", "#b9d3f2", "#2a78d6"])

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# ----------------------------------------------------------------------------------------------- convergence
def convergence():
    models = ["unet", "varnet", "modl", "rim", "swinmr", "kcd"]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6))
    for ax, m in zip(axes.flat, models):
        log = pd.read_csv(ROOT / "runs" / "knee" / m / "log.csv")
        ep = log.epoch + 1
        best = (log.val_ssim_R4 + log.val_ssim_R8).idxmax()
        for col, c, lab in [("val_ssim_R4", "#2a78d6", "$R=4$"), ("val_ssim_R8", "#eb6834", "$R=8$")]:
            ax.plot(ep, log[col], color=c, label=lab)
            ax.plot(ep[best], log[col][best], "o", ms=4, color=c, markeredgecolor="white", markeredgewidth=0.8)
        ax.axvline(40.5, color=INK2, lw=0.6, ls=":")
        ax.set_title(NAME[m], color=INK)
        ax.set_xlim(0, 51)
        ax.set_xlabel("epoch")
    axes[0, 0].set_ylabel("validation SSIM")
    axes[1, 0].set_ylabel("validation SSIM")
    axes[0, 0].legend(frameon=False, loc="lower right")
    # score-MRI: denoising score-matching loss per 500 iterations (log scale)
    ax = axes[1, 2]
    log = pd.read_csv(ROOT / "runs" / "knee" / "scoremri" / "log.csv")
    ax.plot(log.step / 1000, log.train_loss, color="#e87ba4")
    ax.set_yscale("log")
    ax.set_title("score-MRI", color=INK)
    ax.set_xlabel("iteration ($\\times 10^3$)")
    ax.set_ylabel("training loss")
    axes[1, 3].axis("off")
    axes[1, 3].text(0, 0.5, "Dots: selected checkpoint\n(best mean validation SSIM).\nDotted line: learning-rate\n"
                    "drop after epoch 40.\nscore-MRI: final EMA weights\nafter $10^5$ iterations.",
                    color=INK2, va="center", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_convergence")


# ---------------------------------------------------------------------------------------------- significance
def significance(ref="rss"):
    pw = pd.read_csv(TAB / f"random_nominal_ref-{ref}" / "all" / "pairwise.csv")
    rank = pd.read_csv(TAB / f"random_nominal_ref-{ref}" / "all" / "ranking.csv")
    metrics = ["SSIM", "PSNR", "LPIPS"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7))
    for i, accel in enumerate((4, 8)):
        order = list(rank[(rank.accel == accel) & rank.method.isin(ORDER)]
                     .sort_values("mean_rank_equal_weights").method)
        idx = {m: k for k, m in enumerate(order)}
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            mat = np.full((len(order), len(order)), np.nan)
            sig = np.zeros_like(mat, bool)
            for _, r in pw[(pw.accel == accel) & (pw.metric == metric)].iterrows():
                if r.method_a in idx and r.method_b in idx:
                    a, b = idx[r.method_a], idx[r.method_b]
                    mat[a, b], mat[b, a] = r.rank_biserial, -r.rank_biserial
                    sig[a, b] = sig[b, a] = bool(r.significant_05) if "significant_05" in r else r.p_holm < 0.05
            im = ax.imshow(mat, cmap=DIVERGING, vmin=-1, vmax=1)
            for a in range(len(order)):
                for b in range(len(order)):
                    if a != b and not sig[a, b]:
                        ax.plot(b, a, marker="x", ms=3, mew=0.8, color=INK, ls="")
            ax.set_xticks(range(len(order)), [NAME[m] for m in order], rotation=90)
            ax.set_yticks(range(len(order)), [NAME[m] for m in order] if j == 0 else [])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{metric}, $R={accel}$", color=INK)
            if i == 0:
                ax.set_xticklabels([])
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("rank-biserial r  (row better > 0)")
    cb.outline.set_visible(False)
    save(fig, f"fig_significance_{ref}")


# ----------------------------------------------------------------------------------------------- time-quality
LABEL_OFFSET = {"cs": (5, -9), "sense": (-12, 7), "modl": (5, -8)}


def time_quality():
    s = pd.read_csv(TAB / "random_nominal_ref-rss" / "all" / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, accel in zip(axes, (4, 8)):
        d = s[(s.accel == accel) & s.method.isin(ORDER)].set_index("method")
        for m in ORDER:
            r = d.loc[m]
            c = FAM_COLOR[FAMILY[m]]
            ax.errorbar(r.TIME_median_s, r.SSIM_mean, yerr=[[r.SSIM_mean - r.SSIM_ci_lo], [r.SSIM_ci_hi - r.SSIM_mean]],
                        fmt=MARKER[m], ms=5, color=c, mec="white", mew=0.8, elinewidth=0.8, capsize=0)
            ax.annotate(NAME[m], (r.TIME_median_s, r.SSIM_mean), xytext=LABEL_OFFSET.get(m, (4, 2)),
                        textcoords="offset points",
                        fontsize=6.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("median reconstruction time per slice (s)")
        ax.set_title(f"$R={accel}$", color=INK)
    axes[0].set_ylabel("SSIM (RSS reference)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAM_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "fig_time_quality")


# ------------------------------------------------------------------------------------------------ robustness
def robustness():
    """SSIM change vs the same method's clean result on the same slices (x100). Sequential scale on the size of
    the loss (all changes except the equispaced shift are losses); every cell is labelled with its value."""
    t = pd.read_csv(TAB / "robustness_SSIM.csv", index_col=0)
    cond = ["noise-0.005", "noise-0.01", "noise-0.02", "noise-0.04", "motion-1", "motion-2", "motion-4",
            "dropout-0.05", "dropout-0.1", "dropout-0.2", "spike-2", "spike-5", "spike-10", "equispaced_nominal"]
    lab = ["noise $\sigma$=.005", ".01", ".02", ".04", "motion 1 px", "2", "4", "dropout 5 %", "10 %", "20 %",
           "spikes 2", "5", "10", "equispaced"]
    meths = [m for m in ORDER if m in t.index and pd.notna(t.loc[m, "R4:noise-0.01"])]
    seq = LinearSegmentedColormap.from_list("loss", ["#f4f3f0", "#f3c9b6", "#eb6834", "#a8350f"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [14, 14, 1.3], "wspace": 0.06})

    def cells(ax, d, xt):
        im = ax.imshow(np.clip(-d, 0, None), cmap=seq, vmin=0, vmax=0.55, aspect="auto")
        for a in range(d.shape[0]):
            for b in range(d.shape[1]):
                v = round(d[a, b] * 100)
                ax.text(b, a, f"{v:+d}" if v else "0", ha="center", va="center", fontsize=5.5,
                        color="white" if -d[a, b] > 0.33 else INK)
        ax.set_xticks(range(len(xt)), xt, rotation=90)
        ax.tick_params(length=0)
        ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        return im

    for ax, accel in zip(axes[:2], (4, 8)):
        base = t.loc[meths, f"R{accel}:nominal"]
        d = np.stack([t.loc[meths, f"R{accel}:{c}"] - base for c in cond], 1)
        im = cells(ax, d, lab)
        ax.set_yticks(range(len(meths)), [NAME[m] for m in meths] if accel == 4 else [])
        ax.set_title(f"$R={accel}$", color=INK)
    d12 = (t.loc[meths, "R12:nominal_R12"] - t.loc[meths, "R8:nominal"]).values[:, None]
    cells(axes[2], d12, ["$R=12$ vs $R=8$"])
    axes[2].set_yticks([])
    axes[2].set_title("shift", color=INK)
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("SSIM loss vs clean")
    cb.outline.set_visible(False)
    fig.suptitle(r"Change in SSIM ($\times 100$) relative to each method's clean result on the same slices",
                 color=INK, fontsize=8.5, y=1.0)
    save(fig, "fig_robustness")


# ----------------------------------------------------------------------------------------------- qualitative
def qualitative(volume="file1000495", sl=None, accel=8, roi=(130, 120, 80), err_max=0.2):
    """One CORPD test slice at R = 8. Fixed display window [0, 99.5th percentile of the reference] for every
    image; error maps |x - ref| / window on a shared scale, shown only inside the anatomy mask used by the
    metrics (outside it, grey). Numbers: that slice's SSIM / PSNR (RSS reference, anatomy mask)."""
    from bench.metrics import foreground_mask
    with h5py.File(ROOT / "data" / "knee" / "volumes" / f"{volume}.h5", "r") as f:
        n = f["target"].shape[0]
        if sl is None:   # representative slice: central half of the volume, VarNet SSIM closest to its test median
            v = pd.read_csv(RES / "per_slice" / "varnet__random_nominal.csv")
            v = v[v.accel == accel]
            c = v[(v.volume == volume) & v.slice.between(n // 4, 3 * n // 4)]
            sl = int(c.loc[(c.SSIM - v.SSIM.median()).abs().idxmin(), "slice"])
        ref = f["target"][sl].astype(np.float32)
        vol_max = float(np.max(f["target"][()]))
    mask = foreground_mask(ref, vol_max)
    ps = {m: pd.read_csv(RES / "per_slice" / f"{m}__random_nominal.csv") for m in ORDER}
    vmax = np.percentile(ref, 99.5)
    y0, x0, w = roi
    cmap_err = plt.get_cmap("magma").copy()
    cmap_err.set_bad("#d9d8d4")
    blocks = [["ref", "zf", "grappa", "sense", "cs", "unet"], ["varnet", "modl", "rim", "swinmr", "kcd", "scoremri"]]
    fig, axes = plt.subplots(7, 6, figsize=(7.2, 8.2),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.03, "height_ratios": [1, 1, 1, 0.22, 1, 1, 1]})
    for a in axes[3]:
        a.axis("off")
    for bi, block in enumerate(blocks):
        r0 = 4 * bi
        for j, m in enumerate(block):
            img = ref if m == "ref" else np.load(RES / "recon" / m / "random_nominal" /
                                                  f"{volume}_R{accel}.npy")[sl].astype(np.float32)
            a_full, a_roi, a_err = axes[r0, j], axes[r0 + 1, j], axes[r0 + 2, j]
            a_full.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            a_full.add_patch(Rectangle((x0, y0), w, w, fill=False, ec="#eda100", lw=0.8))
            a_roi.imshow(img[y0:y0 + w, x0:x0 + w], cmap="gray", vmin=0, vmax=vmax)
            if m == "ref":
                a_full.set_title("Reference (RSS)", color=INK, fontsize=7)
                a_err.imshow(np.where(mask, 1.0, np.nan), cmap="gray", vmin=0, vmax=1.6)
                a_err.text(160, 160, "anatomy mask", color=INK, fontsize=6, ha="center", va="center")
            else:
                r = ps[m][(ps[m].volume == volume) & (ps[m].slice == sl) & (ps[m].accel == accel)].iloc[0]
                a_full.set_title(NAME[m], color=INK, fontsize=7)
                err = np.where(mask, np.abs(img - ref) / vmax, np.nan)
                im = a_err.imshow(err, cmap=cmap_err, vmin=0, vmax=err_max)
                a_err.text(4, 4, f"{r.SSIM:.3f} / {r.PSNR:.1f} dB", color=INK, fontsize=5.5, va="top",
                           bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
            for a in (a_full, a_roi, a_err):
                a.set_xticks([])
                a.set_yticks([])
                a.grid(False)
                for s_ in a.spines.values():
                    s_.set_visible(False)
    cb = fig.colorbar(im, ax=axes[-1, :], orientation="horizontal", fraction=0.04, pad=0.02, aspect=50,
                      ticks=[0, err_max / 2, err_max])
    cb.set_label("|error| / display maximum", fontsize=6.5)
    cb.outline.set_visible(False)
    save(fig, f"fig_qualitative_R{accel}")
    return volume, sl


def background(volume="file1000495", sl=24, accel=4):
    """Problem illustration: sampling mask, zero-filled image, the two fully sampled references and their
    difference inside the anatomy. Each reference is scaled by its own volume maximum, as in the metrics."""
    from bench.masks import eval_mask
    from bench.metrics import foreground_mask
    with h5py.File(ROOT / "data" / "knee" / "volumes" / f"{volume}.h5", "r") as f:
        rss = f["target"][sl].astype(np.float32)
        rss_max = float(np.max(f["target"][()]))
    with h5py.File(ROOT / "data" / "knee" / "ref_sense" / f"{volume}.h5", "r") as f:
        sense = f["target"][sl].astype(np.float32)
        sense_max = float(f.attrs["max"])
    zf = np.load(RES / "recon" / "zf" / "random_nominal" / f"{volume}_R{accel}.npy")[sl].astype(np.float32)
    mask, _ = eval_mask(volume, sl, rss.shape[1], accel, {4: 0.08, 8: 0.04}[accel])
    a, b = rss / rss_max, sense / sense_max
    fg = foreground_mask(rss, rss_max)
    vmax = np.percentile(a, 99.5)
    cmap_err = plt.get_cmap("magma").copy()
    cmap_err.set_bad("#d9d8d4")
    panels = [(np.tile(mask, (rss.shape[0], 1)), "gray", (0, 1), f"(a) mask, $R={accel}$"),
              (zf / rss_max, "gray", (0, vmax), "(b) zero-filled"),
              (a, "gray", (0, vmax), "(c) RSS reference"),
              (b, "gray", (0, vmax), "(d) SENSE reference"),
              (np.where(fg, np.abs(a - b) / vmax, np.nan), cmap_err, (0, 0.2), "(e) |(c) - (d)|")]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.75), gridspec_kw={"wspace": 0.04})
    for ax, (img, cm, (lo, hi), title) in zip(axes, panels):
        if title.startswith("(a)"):   # vector stripes: a raster of 320 one-pixel lines aliases when scaled
            ax.imshow(np.zeros_like(img), cmap="gray", vmin=0, vmax=1)
            for x in np.flatnonzero(mask):
                ax.axvspan(x - 0.5, x + 0.5, color="white", lw=0)
            ax.add_patch(Rectangle((-0.5, -0.5), img.shape[1], img.shape[0], fill=False, ec=INK2, lw=0.6))
        else:
            im = ax.imshow(img, cmap=cm, vmin=lo, vmax=hi)
        ax.set_title(title, color=INK, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    cb = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.03, ticks=[0, 0.1, 0.2])
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    save(fig, "fig_background")


PERTURB_ROWS = [("nominal", "Clean"), ("noise-0.04", "Noise\n$\\sigma = 0.04$"), ("motion-4", "Motion\n4 px"),
                ("dropout-0.2", "Line dropout\n20 %"), ("spike-2", "RF spikes\n2")]


def perturbation_examples(volume="file1000495", sl=24, accel=4, crop=(60, 60, 200)):
    """Same test slice reconstructed by ten methods under four k-space perturbations (scripts/perturb_examples.py
    writes the reconstructions; their metrics equal the benchmark's stored per-slice values). One display window
    [0, 99.5th percentile of the reference] for every image; numbers are this slice's SSIM / PSNR (RSS reference)."""
    methods = ["zf", "grappa", "sense", "cs", "unet", "varnet", "modl", "rim", "swinmr", "kcd"]
    rec = RES / "recon_perturb"
    met = pd.read_csv(rec / "metrics.csv")
    with h5py.File(ROOT / "data" / "knee" / "volumes" / f"{volume}.h5", "r") as f:
        ref = f["target"][sl].astype(np.float32)
    vmax = np.percentile(ref, 99.5)
    y0, x0, w = crop
    ncol = len(methods) + 1
    fig, axes = plt.subplots(len(PERTURB_ROWS), ncol, figsize=(7.2, 3.45),
                             gridspec_kw={"hspace": 0.04, "wspace": 0.03})
    for r, (cond, label) in enumerate(PERTURB_ROWS):
        for c in range(ncol):
            a = axes[r, c]
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            for s_ in a.spines.values():
                s_.set_visible(False)
            if c == 0:
                if r == 0:
                    a.imshow(ref[y0:y0 + w, x0:x0 + w], cmap="gray", vmin=0, vmax=vmax)
                    a.set_title("Reference", fontsize=6.5, color=INK, pad=2)
                elif r == 1:   # full field of view, with the crop shown in every other panel outlined
                    a.imshow(ref, cmap="gray", vmin=0, vmax=vmax)
                    a.add_patch(Rectangle((x0, y0), w, w, fill=False, ec="#eda100", lw=0.8))
                    a.text(0.5, -0.04, "full FOV", transform=a.transAxes, ha="center", va="top", fontsize=5.5,
                           color=INK2)
                else:
                    a.axis("off")
                a.text(-0.08, 0.5, label, transform=a.transAxes, ha="right", va="center", fontsize=6.5,
                       color=INK, linespacing=1.2)
                continue
            m = methods[c - 1]
            img = np.load(rec / f"{m}__{cond}.npy").astype(np.float32)
            a.imshow(img[y0:y0 + w, x0:x0 + w], cmap="gray", vmin=0, vmax=vmax)
            row = met[(met.method == m) & (met.condition == cond.replace("-", ":", 1))].iloc[0]
            a.text(0.03, 0.03, f"{row.SSIM:.2f} / {row.PSNR:.1f}", transform=a.transAxes, fontsize=4.8,
                   color=INK, va="bottom", ha="left",
                   bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
            if r == 0:
                a.set_title(NAME[m], fontsize=6.5, color=INK, pad=2)
    save(fig, "fig_perturbation_examples")


def pipeline():
    """Benchmark pipeline as a snake-shaped flow diagram (row 1 left->right, row 2 right->left, row 3 left->right)."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(0, 100.8); ax.set_ylim(0, 47); ax.axis("off")
    BOX, EDGE = "#f4f3f0", INK2

    def box(x, y, w, h, text, fill=BOX, edge=EDGE, ls="-", bold=None, size=6.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=edge,
                                    lw=0.9, ls=ls))
        if bold:
            ax.text(x + w / 2, y + h - 2.2, bold, ha="center", va="top", fontsize=size + 0.4,
                    fontweight="bold", color=INK)
            ax.text(x + w / 2, y + (h - 4.4) / 2, text, ha="center", va="center", fontsize=size, color=INK,
                    linespacing=1.25)
        else:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
                    linespacing=1.25)

    def arrow(p, q, conn="arc3"):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9,
                                                        mutation_scale=8, shrinkA=0, shrinkB=0,
                                                        connectionstyle=conn))

    # row 1: data preparation
    y1, h1, w1 = 34, 12, 18
    xs = [0.5, 20.9, 41.3, 61.7, 82.1]
    box(xs[0], y1, w1, h1, "fastMRI knee, 15 coils\n50 volumes, split\n30/5/15 by volume", bold="Raw k-space")
    box(xs[1], y1, w1, h1, "crop 320×320\nSVD to 8 virtual coils\nscale from centre", bold="Preprocess")
    box(xs[2], y1, w1, h1, "noise, motion, line\ndropout, RF spikes\n(robustness only)", bold="Perturbation",
        ls=(0, (3, 2)))
    box(xs[3], y1, w1, h1, "random or equispaced\nR = 4, 8; 12 unseen\nfixed per slice", bold="Mask")
    box(xs[4], y1, w1, h1, "identical data, mask\nand ACS lines\nfor all methods", bold="Undersampled y")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow((a + w1, y1 + h1 / 2), (b, y1 + h1 / 2))

    # row 2: references | methods | ESPIRiT
    y2, h2 = 15.5, 14
    box(0.5, y2, 18, h2, "RSS (primary) and\nSENSE magnitude\n+ foreground mask", bold="References")
    box(82.1, y2, 18, h2, "from ACS lines only;\nre-estimated under\nperturbation", bold="ESPIRiT maps")
    mx, mw = 20.5, 60.0
    box(mx, y2, mw, h2, "", fill="#ffffff")
    ax.text(mx + mw / 2, y2 + h2 - 1.6, "11 reconstruction methods, five families", ha="center", va="top",
            fontsize=6.6, fontweight="bold", color=INK)
    fams = [("Classical", "ZF, GRAPPA,\nCG-SENSE, CS", 12.4), ("Image-domain CNN", "U-Net", 10.4),
            ("Unrolled", "E2E-VarNet,\nMoDL, RIM", 10.8), ("Transformer", "SwinMR", 11.4),
            ("Diffusion", "KCD,\nscore-MRI", 9.6)]
    fx, gap = mx + 1.2, (mw - 2.4 - sum(f[2] for f in fams)) / (len(fams) - 1)
    for fam, members, fw in fams:
        c = FAM_COLOR[fam]
        ax.add_patch(FancyBboxPatch((fx, y2 + 1.2), fw, 8.2, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=c + "22", ec=c, lw=1.0))
        ax.text(fx + fw / 2, y2 + 8.7, fam.replace("Image-domain CNN", "Image CNN"), ha="center", va="top",
                fontsize=5.8, fontweight="bold", color=INK)
        ax.text(fx + fw / 2, y2 + 3.9, members, ha="center", va="center", fontsize=5.6, color=INK,
                linespacing=1.2)
        fx += fw + gap
    arrow((xs[4] + w1 / 2, y1), (xs[4] + w1 / 2, y2 + h2))                      # y -> ESPIRiT
    arrow((xs[4] + 2, y1), (mx + mw - 6, y2 + h2), "arc3,rad=0")               # y -> methods
    arrow((82.1, y2 + h2 / 2), (mx + mw, y2 + h2 / 2))                          # ESPIRiT -> methods
    arrow((xs[1] + 4, y1), (9.5, y2 + h2), "arc3,rad=0")                        # preprocess -> references

    # row 3: evaluation
    y3, h3 = 0.5, 11.5
    box(0.5, y3, 31, h3, "NMSE, PSNR, SSIM, HFEN, VIF, LPIPS\ninside the foreground mask",
        bold="Six metrics per slice")
    box(35.5, y3, 25, h3, "15 test volumes\n(540 slices)", bold="Per-volume means")
    box(64.5, y3, 35.6, h3, "Wilcoxon + Holm, effect sizes,\nbootstrap CIs, weight-sensitivity\nof the ranking",
        bold="Statistics")
    arrow((9.5, y2), (9.5, y3 + h3))                                             # references -> metrics
    arrow((mx + 6, y2), (24, y3 + h3), "arc3,rad=0")                             # methods -> metrics
    arrow((31.5, y3 + h3 / 2), (35.5, y3 + h3 / 2))
    arrow((60.5, y3 + h3 / 2), (64.5, y3 + h3 / 2))
    save(fig, "fig_pipeline")


if __name__ == "__main__":
    pipeline()
    background()
    convergence()
    significance("rss")
    time_quality()
    robustness()
    print("qualitative:", qualitative())
    qualitative(sl=24, accel=4)          # same slice as the R = 8 figure (chosen by the rule above)
    perturbation_examples()              # needs results/knee/recon_perturb from scripts/perturb_examples.py
