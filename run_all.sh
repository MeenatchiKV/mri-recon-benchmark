#!/usr/bin/env bash
# Full knee benchmark, in order. Every step is resumable or skips finished work.
# Laptop budget (RTX 4050, 6 GB): preprocessing + maps ~2 h, tuning ~3 h, training ~7 days, evaluation ~1 day.
# The PowerShell scripts in scripts/ are the Windows equivalents used for the paper.
set -euo pipefail
cd "$(dirname "$0")"
A=knee

python -m bench.preprocess --anatomy $A --workers 3
python -m bench.espirit --anatomy $A
python -m bench.espirit --anatomy $A --accel 12 --split test        # unseen-acceleration test only
python -m bench.reference --anatomy $A
python -m bench.tune --anatomy $A

for m in unet varnet rim swinmr; do
  python -m bench.train --model $m --anatomy $A --resume
done
# MoDL: K = 1 network first, then K = 10 initialised from it (as recommended by the authors)
python -m bench.train --model modl --anatomy $A --tag k1 --model-args '{"iterations": 1}' --resume
python -m bench.train --model modl --anatomy $A --init-from runs/$A/modl_k1 --resume
# KCD validates with its sampler on 40 validation items per epoch
python -m bench.train --model kcd --anatomy $A --resume --max-val 40
python -m bench.train_score --anatomy $A --resume

# diffusion sampler steps, chosen on the validation volumes
python -m bench.select_steps --model kcd --anatomy $A --steps 25 50 100 200 1000
python -m bench.select_steps --model scoremri --anatomy $A --steps 100 200 500

PERTURB="noise:0.005 noise:0.01 noise:0.02 noise:0.04 motion:1 motion:2 motion:4 dropout:0.05 dropout:0.1 dropout:0.2 spike:2 spike:5 spike:10"
# fully sampled ceilings (RSS and SENSE combination of the full k-space)
for m in full_rss full_sense; do python -m bench.evaluate --method $m --anatomy $A; done
for m in zf grappa sense cs unet varnet modl rim swinmr kcd; do
  python -m bench.evaluate --method $m --anatomy $A --save-recon
  python -m bench.evaluate --method $m --anatomy $A --slice-stride 4 --conditions $PERTURB
  python -m bench.evaluate --method $m --anatomy $A --slice-stride 4 --mask-type equispaced
  python -m bench.evaluate --method $m --anatomy $A --slice-stride 4 --accels 12
done
# score-MRI (~2 min per slice): all test slices nominal, perturbations on every 16th slice
python -m bench.evaluate --method scoremri --anatomy $A --save-recon --cache
python -m bench.evaluate --method scoremri --anatomy $A --slice-stride 16 --cache --conditions $PERTURB
python -m bench.evaluate --method scoremri --anatomy $A --slice-stride 16 --cache --mask-type equispaced
python -m bench.evaluate --method scoremri --anatomy $A --slice-stride 16 --cache --accels 12

for ref in rss sense whole; do
  python -m bench.stats --anatomy $A --reference $ref
done
python scripts/robustness_summary.py
python scripts/make_tables.py
python scripts/make_figures.py
