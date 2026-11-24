from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("data_dir", "runs_dir", "results_dir"):
        cfg[key] = _resolve(cfg[key])
    cfg["raw_dirs"] = {k: _resolve(v) for k, v in cfg["raw_dirs"].items()}
    cfg["test_raw_dirs"] = {k: _resolve(v) for k, v in cfg.get("test_raw_dirs", {}).items()}
    cfg["masks"]["center_fractions"] = {int(k): v for k, v in cfg["masks"]["center_fractions"].items()}
    return cfg


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()
