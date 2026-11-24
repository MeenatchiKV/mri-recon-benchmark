"""Volume-level statistics for the benchmark (addresses the pseudoreplication concern: the unit of
analysis is the test VOLUME, never the slice).

For one anatomy and one setting (default: random masks, nominal condition):
  summary.csv      per method x R: mean, SD and 95 % bootstrap CI over test volumes for every metric;
                   median (IQR) reconstruction time per slice
  pairwise.csv     every method pair, per R and metric: two-sided Wilcoxon signed-rank test on the
                   per-volume means, Holm correction within each (R, metric) family, matched-pairs
                   rank-biserial correlation (oriented so that > 0 means the first method is better),
                   median paired difference with bootstrap 95 % CI
  mixed.csv        sensitivity check on slice-level data: linear mixed model metric ~ method + (1 | volume)
  ranking.csv      mean rank over the quality metrics (equal weights), plus a weight-sensitivity analysis:
                   10 000 Dirichlet(1) weight vectors -> how often each method ranks first, its mean / 95 %
                   range of rank, and the Kendall tau of each re-weighted ranking with the equal-weight one

Usage:  python -m bench.stats --anatomy knee [--setting random_nominal]
"""
import argparse
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon
from statsmodels.stats.multitest import multipletests

from .config import load_config
from .metrics import HIGHER_IS_BETTER, METRICS

N_BOOT = 10000


def load(cfg, anatomy, setting):
    files = sorted((cfg["results_dir"] / anatomy / "per_slice").glob(f"*__{setting}.csv"))
    if not files:
        raise SystemExit(f"no per-slice results for {anatomy} / {setting}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def per_volume(df, metrics):
    return df.groupby(["method", "accel", "volume"])[metrics].mean().reset_index()


def bootstrap_ci(x, rng, stat=np.mean):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    return np.percentile(stat(x[idx], axis=1), [2.5, 97.5])


def summary(df, vol, metrics, rng):
    rows = []
    for (method, accel), g in vol.groupby(["method", "accel"]):
        row = {"method": method, "accel": accel, "n_volumes": len(g)}
        for m in metrics:
            lo, hi = bootstrap_ci(g[m].values, rng)
            row |= {f"{m}_mean": g[m].mean(), f"{m}_sd": g[m].std(ddof=1), f"{m}_ci_lo": lo, f"{m}_ci_hi": hi}
        t = df[(df.method == method) & (df.accel == accel)]["TIME"].dropna()
        row |= {"TIME_median_s": t.median(), "TIME_q1_s": t.quantile(0.25), "TIME_q3_s": t.quantile(0.75),
                "hardware": df[df.method == method]["hardware"].iloc[0]}
        rows.append(row)
    return pd.DataFrame(rows)


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation of paired differences d (zeros dropped)."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().values
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def pairwise(vol, metrics, rng):
    rows = []
    for accel, ga in vol.groupby("accel"):
        wide = {m: ga.pivot(index="volume", columns="method", values=m) for m in metrics}
        methods = sorted(ga.method.unique())
        for m in metrics:
            fam = []
            for a, b in combinations(methods, 2):
                pair = wide[m][[a, b]].dropna()
                d = (pair[a] - pair[b]).values
                if not HIGHER_IS_BETTER[m]:
                    d = -d                                   # > 0 always means "a is better"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                lo, hi = bootstrap_ci(d, rng, np.median)
                fam.append({"accel": accel, "metric": m, "method_a": a, "method_b": b, "n_volumes": len(d),
                            "median_diff_a_better": np.median(d), "diff_ci_lo": lo, "diff_ci_hi": hi,
                            "rank_biserial": rank_biserial(d), "p": p})
            if not fam:   # fewer than two methods
                continue
            p_holm = multipletests([r["p"] for r in fam], method="holm")[1]
            for r, ph in zip(fam, p_holm):
                r["p_holm"] = ph
                r["significant_0.05"] = ph < 0.05
            rows.extend(fam)
    return pd.DataFrame(rows)


def mixed(df, metrics):
    import statsmodels.formula.api as smf
    rows = []
    for accel, g in df.groupby("accel"):
        ref = "zf" if "zf" in set(g.method) else sorted(g.method.unique())[0]
        for m in metrics:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.mixedlm(f"{m} ~ C(method, Treatment('{ref}'))", g, groups=g["volume"]).fit()
            except np.linalg.LinAlgError:   # e.g. a single test volume (smoke runs)
                print(f"mixed model skipped for R{accel} {m}: singular design")
                continue
            for name, coef in fit.params.items():
                if name.startswith("C(method"):
                    method = name.split("[T.")[1].rstrip("]")
                    lo, hi = fit.conf_int().loc[name]
                    rows.append({"accel": accel, "metric": m, "method": method, "reference": ref,
                                 "effect_vs_reference": coef, "ci_lo": lo, "ci_hi": hi, "p": fit.pvalues[name]})
    return pd.DataFrame(rows)


def ranking(vol, metrics, rng):
    rows = []
    for accel, ga in vol.groupby("accel"):
        means = ga.groupby("method")[metrics].mean()
        ranks = pd.DataFrame({m: means[m].rank(ascending=not HIGHER_IS_BETTER[m]) for m in metrics})
        base = ranks.mean(axis=1)
        w = rng.dirichlet(np.ones(len(metrics)), size=N_BOOT)
        weighted = ranks.values @ w.T                                 # (methods, N_BOOT)
        overall = np.argsort(np.argsort(weighted, axis=0), axis=0) + 1
        taus = [kendalltau(base.values, weighted[:, i]).statistic for i in range(N_BOOT)]
        for j, method in enumerate(ranks.index):
            rows.append({"accel": accel, "method": method, **{f"rank_{m}": ranks.loc[method, m] for m in metrics},
                         "mean_rank_equal_weights": base[method],
                         "p_rank1_random_weights": float(np.mean(overall[j] == 1)),
                         "rank_mean_random_weights": float(overall[j].mean()),
                         "rank_p2.5": float(np.percentile(overall[j], 2.5)),
                         "rank_p97.5": float(np.percentile(overall[j], 97.5)),
                         "kendall_tau_median": float(np.median(taus))})
    return pd.DataFrame(rows).sort_values(["accel", "mean_rank_equal_weights"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--anatomy", required=True)
    ap.add_argument("--setting", default="random_nominal")
    ap.add_argument("--reference", choices=["rss", "sense", "whole"], default="rss",
                    help="rss: RSS reference in the anatomy mask (primary); sense: SENSE-combined reference in the "
                         "same mask; whole: RSS reference over the whole image (fastMRI convention)")
    ap.add_argument("--acquisition", help="only this contrast, e.g. CORPD_FBK or CORPDFS_FBK")
    ap.add_argument("--no-mixed", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(0)
    df = load(cfg, args.anatomy, args.setting)
    suffix = {"rss": "", "sense": "_sref", "whole": "_whole"}[args.reference]
    if suffix:
        df = df.drop(columns=METRICS).rename(columns={f"{m}{suffix}": m for m in METRICS})
    if args.acquisition:
        df = df[df.acquisition == args.acquisition]
    if df.method.isin(["full_rss", "full_sense"]).any():   # ceiling rows: summary only, never ranked or tested
        ceil = df[df.method.isin(["full_rss", "full_sense"])]
        df = df[~df.method.isin(["full_rss", "full_sense"])]
    else:
        ceil = None
    base = (cfg["results_dir"] / args.anatomy / "tables" /
            f"{args.setting}_ref-{args.reference}" / (args.acquisition or "all"))
    if ceil is not None:
        base.mkdir(parents=True, exist_ok=True)
        summary(ceil, per_volume(ceil, METRICS), METRICS, rng).to_csv(base / "ceiling.csv", index=False)

    # Slow samplers may be evaluated on a fixed slice subset (evaluate --slice-stride). Main tables: methods
    # with full coverage. If any method has partial coverage, a second set compares ALL methods on the
    # (volume, slice, R) items common to every method, so paired tests always use identical items.
    keys = ["volume", "slice", "accel"]
    n_items = df.groupby("method").size()
    full = n_items[n_items == n_items.max()].index
    analyses = {"": df[df.method.isin(full)]}
    if len(full) < len(n_items):
        print("partial coverage:", n_items[n_items < n_items.max()].to_dict(), "-> also writing common_subset/")
        common = set.intersection(*(set(map(tuple, g[keys].values)) for _, g in df.groupby("method")))
        analyses["common_subset"] = df[[tuple(r) in common for r in df[keys].values]]
    for sub, d in analyses.items():
        out = base / sub
        out.mkdir(parents=True, exist_ok=True)
        vol = per_volume(d, METRICS)
        summary(d, vol, METRICS, rng).to_csv(out / "summary.csv", index=False)
        pairwise(vol, METRICS, rng).to_csv(out / "pairwise.csv", index=False)
        ranking(vol, METRICS, rng).to_csv(out / "ranking.csv", index=False)
        if not args.no_mixed:
            mixed(d, METRICS).to_csv(out / "mixed.csv", index=False)
        print(f"tables written to {out} ({d.method.nunique()} methods, {len(d) // d.method.nunique()} items each)")


if __name__ == "__main__":
    main()
