"""Train one learned model on one anatomy, with random masks at every R in the config (one model covers
R = 4 and 8, as in the fastMRI baselines).

Every epoch logs training loss and validation SSIM / PSNR per R to runs/<anatomy>/<model>/log.csv
(convergence evidence for the paper); the checkpoint with the best mean validation SSIM is kept.

Usage:  python -m bench.train --model varnet --anatomy knee [--epochs 50] [--model-args '{"chans": 18}']
Smoke:  python -m bench.train --model varnet --anatomy knee --epochs 1 --max-steps 20 --max-val 10
"""
import argparse
import csv
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import load_config
from .data import SliceDataset
from .metrics import foreground_mask, psnr, ssim
from .models import build_model


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_optimizer(name, params, lr, weight_decay):
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def validate(model, loader, device, fg_threshold):
    """Mean foreground SSIM / PSNR per R (the primary metrics of the benchmark). Iterative samplers
    (KCD) use their short `val_steps` sampler here; the full sampler is used at evaluation."""
    model.eval()
    full_steps = getattr(model, "sample_steps", None)
    if hasattr(model, "val_steps"):
        model.sample_steps = model.val_steps
    scores = {}
    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch)[0].float().cpu().numpy()
        ref = batch["target"][0].cpu().numpy()
        dr = float(batch["max_value"][0])
        r = int(batch["accel"][0])
        fg = foreground_mask(ref, dr, fg_threshold)
        scores.setdefault(r, []).append((ssim(ref, pred, dr, fg), psnr(ref, pred, dr, fg)))
    if full_steps is not None:
        model.sample_steps = full_steps
    model.train()
    return {r: np.mean(v, axis=0) for r, v in sorted(scores.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--model", required=True)
    ap.add_argument("--anatomy", required=True)
    ap.add_argument("--model-args", default="{}")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--max-steps", type=int, help="steps per epoch (smoke test)")
    ap.add_argument("--max-val", type=int, help="validation items (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", help="run directory whose best.pt initialises the weights (e.g. MoDL K=10 "
                                        "from K=1, as in the reference code)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = json.loads(args.model_args)
    model = build_model(args.model, **model_args).to(device)
    if args.init_from:
        model.load_state_dict(torch.load(f"{args.init_from}/best.pt", map_location=device)["model"])
        print(f"initialised from {args.init_from}/best.pt")
    hp = dict(model.default_train)
    if args.epochs:
        hp["epochs"] = args.epochs
    if args.lr:
        hp["lr"] = args.lr
    needs_maps = args.model in ("modl", "rim", "kcd")

    run_dir = cfg["runs_dir"] / args.anatomy / (args.model + (f"_{args.tag}" if args.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    json.dump({"model": args.model, "model_args": model_args, "train": hp, "init_from": args.init_from, "params": n_params,
               "anatomy": args.anatomy, "seed": args.seed, "accelerations": cfg["masks"]["accelerations"],
               "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"},
              open(run_dir / "config.json", "w"), indent=1)
    print(f"{args.model}: {n_params / 1e6:.2f} M parameters -> {run_dir}")

    train_ds = SliceDataset(cfg, args.anatomy, "train", train=True, with_maps=needs_maps)
    val_ds = SliceDataset(cfg, args.anatomy, "val", train=False, with_maps=needs_maps)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.workers,
                          worker_init_fn=train_ds.worker_init, pin_memory=True, persistent_workers=args.workers > 0)
    if args.max_val:  # spread the smoke-test validation over the whole val set
        val_ds = Subset(val_ds, np.linspace(0, len(val_ds) - 1, args.max_val).astype(int).tolist())
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers)

    opt = make_optimizer(hp["optimizer"], model.parameters(), hp["lr"], hp["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, hp["lr_step"], hp["lr_gamma"])
    start, best = 0, -1.0
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]

    log_path = run_dir / "log.csv"
    accels = cfg["masks"]["accelerations"]
    header = ["epoch", "lr", "train_loss", "seconds", "peak_gb"] + [f"val_{m}_R{r}" for r in accels
                                                                    for m in ("ssim", "psnr")]
    if not (args.resume and log_path.exists()):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    model.train()
    for epoch in range(start, hp["epochs"]):
        t0, losses = time.time(), []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(train_dl):
            if args.max_steps and step >= args.max_steps:
                break
            batch = to_device(batch, device)
            pred = model(batch)
            loss = model.loss(batch, pred)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if getattr(model, "clip_grad_value", None):
                torch.nn.utils.clip_grad_value_(model.parameters(), model.clip_grad_value)
            opt.step()
            losses.append(loss.item())
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {np.mean(losses[-100:]):.5f}", flush=True)
        sched.step()
        val = validate(model, val_dl, device, cfg["metrics"]["foreground_threshold"])
        mean_ssim = float(np.mean([v[0] for v in val.values()]))
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        row = [epoch, opt.param_groups[0]["lr"], float(np.mean(losses)), round(time.time() - t0, 1), round(peak, 2)]
        row += [round(float(val[r][i]), 5) for r in accels for i in (0, 1)]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(dict(zip(header, row)), flush=True)
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "epoch": epoch, "best": max(best, mean_ssim)}
        if mean_ssim > best:
            best = mean_ssim
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_ssim": mean_ssim}, run_dir / "best.pt")
        torch.save(state, run_dir / "last.pt")


if __name__ == "__main__":
    main()
