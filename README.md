# mri-recon-benchmark

Controlled benchmark of 11 accelerated MRI reconstruction methods (classical, unrolled, transformer, diffusion)
on fastMRI multi-coil knee data, with volume-level statistics and k-space robustness tests.

This repository contains the code, configuration, data split, per-slice results and training logs behind the
paper *MRI Reconstruction from Classical to Generative Deep Learning: Review and Experimental Benchmark on
fastMRI Knee* (under review). Trained weights are published as a GitHub release.

## Methods

| Family | Method | Implementation |
|---|---|---|
| Classical | Zero-filled | inverse FFT + root-sum-of-squares (RSS) |
| Classical | GRAPPA | `pygrappa`, kernel size and Tikhonov weight tuned on validation |
| Classical | CG-SENSE | `sigpy.mri.app.SenseRecon`, ESPIRiT maps, weight tuned on validation |
| Classical | CS (L1-wavelet) | `sigpy.mri.app.L1WaveletRecon`, ESPIRiT maps, weight tuned on validation |
| Image-domain CNN | U-Net | official `fastmri.models.Unet`, fastMRI baseline recipe |
| Unrolled | E2E-VarNet | official `fastmri.models.VarNet`, full paper size (12 cascades, 18 channels), learned sensitivities |
| Unrolled | MoDL | shared 5-layer CNN, K = 10 iterations with 10 CG steps, ESPIRiT maps, initialised from K = 1 |
| Unrolled | RIM | ConvGRU recurrent inference machine, 8 steps, ESPIRiT SENSE model |
| Transformer | SwinMR | residual Swin transformer blocks with SwinMR losses (reduced width for a 6 GB GPU) |
| Diffusion | KCD | k-space cold diffusion, extended to multi-coil with ESPIRiT maps; sampler steps chosen on validation |
| Diffusion | score-MRI | official NCSN++ / VE-SDE / predictor-corrector code, retrained on the training split (nf = 64) |

## Protocol

* **Data.** 50 multi-coil knee volumes from the fastMRI training set (26 CORPD_FBK, 24 CORPDFS_FBK), split by
  volume into 30 / 5 / 15 (train / validation / test), stratified by contrast: 1049 / 182 / 540 slices.
  The split and volume list are in [`data/knee/split.json`](data/knee/split.json) and
  [`data/knee/volumes.json`](data/knee/volumes.json).
* **Preprocessing.** Crop to 320 x 320, SVD coil compression to 8 virtual coils, one intensity scale per volume
  from the central 9 phase-encoding lines.
* **Sampling.** fastMRI random Cartesian masks, R = 4 (8 % centre) and R = 8 (4 % centre), fixed per test slice;
  R = 12 and equispaced masks as distribution shifts.
* **Shared calibration.** ESPIRiT maps from the ACS lines only, identical for every method that uses maps.
* **References.** RSS (primary) and a SENSE-combined reference (maps from the central 32 lines of the fully
  sampled data).
* **Metrics.** NMSE, PSNR, SSIM, HFEN, VIF and LPIPS inside an anatomy mask; whole-image values are kept as
  `*_whole`.
* **Statistics.** Per-volume means; Wilcoxon signed-rank tests with Holm correction, rank-biserial effect sizes,
  bootstrap 95 % confidence intervals, a mixed-effects check, and a metric-weight sensitivity analysis of the
  ranking (no composite score).
* **Robustness.** 13 perturbations applied in k-space (receiver noise, rigid motion, silent line dropout,
  RF spikes), with ESPIRiT maps re-estimated from the perturbed data.

All settings are in [`configs/default.yaml`](configs/default.yaml).

## Setup

Python 3.11 and a CUDA GPU (the experiments used an RTX 4050 Laptop GPU with 6 GB).

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from [pytorch.org](https://pytorch.org) if the default
wheel does not fit.

score-MRI uses the official code of Chung & Ye (2022). Clone it next to this repository so that the folder
`../score-MRI-main` exists:

```bash
git clone https://github.com/HJ-harry/score-MRI ../score-MRI-main
```

**Data.** fastMRI cannot be redistributed. Request access at [fastmri.med.nyu.edu](https://fastmri.med.nyu.edu),
download the knee multi-coil training set, and point `raw_dirs.knee` in `configs/default.yaml` to it. Only the
50 volumes listed in `data/knee/volumes.json` are used.

## Running the benchmark

[`run_all.sh`](run_all.sh) runs every step in order; the PowerShell scripts in [`scripts/`](scripts) are the
Windows versions that were used for the paper. The main steps are:

```bash
python -m bench.preprocess --anatomy knee            # volumes, split
python -m bench.espirit --anatomy knee               # ESPIRiT maps from the ACS lines
python -m bench.reference --anatomy knee             # SENSE reference images
python -m bench.tune --anatomy knee                  # classical weights, validation only

python -m bench.train --model varnet --anatomy knee  # likewise unet, rim, swinmr, kcd (see run_all.sh for MoDL)
python -m bench.train_score --anatomy knee           # score-MRI
python -m bench.select_steps --model kcd --anatomy knee --steps 25 50 100 200 1000

python -m bench.evaluate --method varnet --anatomy knee --save-recon
python -m bench.evaluate --method varnet --anatomy knee --slice-stride 4 \
    --conditions noise:0.04 motion:4 dropout:0.2 spike:10      # any of bench.perturb.CONDITIONS
python -m bench.stats --anatomy knee                 # add --reference sense / whole for the other references
```

Training all models takes about a week on the 6 GB laptop GPU; evaluation about a day.

## Reproducing the tables and figures

The per-slice results of every method and condition are in [`results/knee/per_slice`](results/knee/per_slice),
the statistics in [`results/knee/tables`](results/knee/tables) and the training logs in [`runs/knee`](runs/knee).
From these files alone:

```bash
python scripts/robustness_summary.py
python scripts/make_tables.py        # LaTeX tables -> paper/tables/
python scripts/make_figures.py       # figures -> paper/figures/
```

The convergence, significance, speed-quality and robustness figures need only the released files. The image
figures (reference, qualitative and perturbation examples) also need the preprocessed fastMRI data and the saved
reconstructions (`bench.evaluate --save-recon`, `scripts/perturb_examples.py`).

## Repository layout

```
configs/default.yaml      every setting in one place
bench/preprocess.py       raw k-space -> cropped, coil-compressed, scaled volumes + split
bench/masks.py            fastMRI masks, deterministic test masks
bench/espirit.py          GPU ESPIRiT from the ACS lines
bench/reference.py        SENSE-combined reference images
bench/data.py             slice dataset: identical inputs for every method
bench/classical.py        zero-filled, GRAPPA, CG-SENSE, CS
bench/models/             U-Net, E2E-VarNet, MoDL, RIM, SwinMR, KCD, score-MRI
bench/train.py            training with a per-epoch convergence log
bench/train_score.py      score-MRI training
bench/select_steps.py     diffusion sampler steps chosen on validation
bench/tune.py             validation tuning of the classical methods
bench/perturb.py          k-space perturbations
bench/evaluate.py         per-slice metrics and timing for any method and condition
bench/metrics.py          metrics and anatomy mask
bench/stats.py            volume-level statistics, rankings, weight sensitivity
scripts/                  run scripts, tables, figures, timing
results/knee/             per-slice results, statistics, tuned classical parameters
runs/knee/                training logs, model configurations, sampler settings
data/knee/                volume list and split (no image data)
```
