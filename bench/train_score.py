"""Train score-MRI's NCSN++ on the training volumes of one anatomy (no test or validation data, and no
public checkpoint, which was trained on fastMRI knee volumes that overlap our data).

Training follows the official recipe (losses.get_step_fn: continuous VE-SDE denoising score matching,
Adam lr 2e-4, 5000 warm-up steps, gradient clipping 1, EMA 0.999, random horizontal flips) on the
fully sampled RSS magnitude images, each scaled to [0, 1].

Convergence evidence: training loss every 500 steps, and every `--val-every` steps the EMA model
reconstructs a fixed set of validation slices at R = 4 and 8 with a short sampler (log.csv).
The EMA weights with the best validation SSIM are saved as best.pt (loadable by bench.evaluate).

Usage:  python -m bench.train_score --anatomy knee [--steps 100000] [--nf 64]
"""
import argparse
import csv
import json
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import load_config
from .data import SliceDataset
from .models.scoremri import ScoreMRI
from .score_import import load_score_mri
from .train import to_device, validate


class MagnitudeSlices(Dataset):
    def __init__(self, cfg, anatomy):
        names = json.load(open(cfg["data_dir"] / anatomy / "split.json"))["train"]
        info = {v["name"]: v for v in json.load(open(cfg["data_dir"] / anatomy / "volumes.json"))}
        self.dir = cfg["data_dir"] / anatomy / "volumes"
        self.items = [(n, s) for n in names for s in range(info[n]["slices"])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, s = self.items[i]
        with h5py.File(self.dir / f"{name}.h5", "r") as f:
            img = f["target"][s]
        img = img / (img.max() + 1e-12)
        if np.random.rand() < 0.5:
            img = img[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(img))[None].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--anatomy", required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--nf", type=int, default=64)
    ap.add_argument("--val-every", type=int, default=10000)
    ap.add_argument("--val-slices", type=int, default=8)
    ap.add_argument("--val-n-steps", type=int, default=100, help="reverse steps of the validation sampler")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    s = load_score_mri()
    device = torch.device("cuda")
    size = cfg["preprocess"]["crop"][args.anatomy][0]
    wrapper = ScoreMRI(nf=args.nf, image_size=size).to(device)
    model = wrapper.score_model
    hp = dict(ScoreMRI.default_train, steps=args.steps)
    scfg = wrapper.cfg
    scfg.optim.lr, scfg.optim.warmup, scfg.optim.grad_clip = hp["lr"], hp["warmup"], hp["grad_clip"]

    run_dir = cfg["runs_dir"] / args.anatomy / ("scoremri" + (f"_{args.tag}" if args.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    json.dump({"model": "scoremri", "model_args": {"nf": args.nf, "image_size": size}, "train": hp,
               "params": n_params, "anatomy": args.anatomy, "gpu": torch.cuda.get_device_name()},
              open(run_dir / "config.json", "w"), indent=1)
    print(f"score-MRI NCSN++ nf={args.nf}: {n_params / 1e6:.1f} M parameters -> {run_dir}")

    ema = s.ema.ExponentialMovingAverage(model.parameters(), decay=hp["ema_rate"])
    optimizer = s.losses.get_optimizer(scfg, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
    step_fn = s.losses.get_step_fn(wrapper.sde(), train=True, optimize_fn=s.losses.optimization_manager(scfg),
                                   reduce_mean=False, continuous=True, likelihood_weighting=False)
    best = -1.0
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        ema.load_state_dict(ck["ema"])
        state["step"], best = ck["step"], ck["best"]

    train_dl = DataLoader(MagnitudeSlices(cfg, args.anatomy), batch_size=hp["batch_size"], shuffle=True,
                          num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    val_ds = SliceDataset(cfg, args.anatomy, "val")
    val_ds = Subset(val_ds, np.linspace(0, len(val_ds) - 1, args.val_slices * len(val_ds.accels)).astype(int).tolist())
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=0)

    log_path = run_dir / "log.csv"
    accels = cfg["masks"]["accelerations"]
    if not (args.resume and log_path.exists()):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "seconds"] + [f"val_{m}_R{r}" for r in accels
                                                                         for m in ("ssim", "psnr")])
    losses, t0 = [], time.time()
    model.train()
    while state["step"] < hp["steps"]:
        for x in train_dl:
            loss = step_fn(state, x.to(device))
            losses.append(loss.item())
            step = state["step"]
            if step % 500 == 0 or step % args.val_every == 0 or step >= hp["steps"]:
                row = [step, float(np.mean(losses)), round(time.time() - t0, 1)]
                if step % args.val_every == 0 and step > 0 or step >= hp["steps"]:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    wrapper.n_steps = args.val_n_steps
                    val = validate(wrapper, val_dl, device, cfg["metrics"]["foreground_threshold"])
                    mean_ssim = float(np.mean([v[0] for v in val.values()]))
                    if mean_ssim > best:
                        best = mean_ssim
                        torch.save({"model": wrapper.state_dict(), "step": step, "val_ssim": mean_ssim},
                                   run_dir / "best.pt")
                    ema.restore(model.parameters())
                    model.train()
                    row += [round(float(val[r][i]), 5) for r in accels for i in (0, 1)]
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "ema": ema.state_dict(), "step": step, "best": best}, run_dir / "last.pt")
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                print(row, flush=True)
                losses, t0 = [], time.time()
            if state["step"] >= hp["steps"]:
                break


if __name__ == "__main__":
    main()
