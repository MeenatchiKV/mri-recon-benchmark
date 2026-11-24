"""Stream a fastMRI .tar.xz archive (local file or the signed download URL from the fastMRI email) and keep
only a contrast-stratified subset of volumes, without ever storing the full archive on disk.

Reading stops as soon as every quota is filled. Re-running skips volumes that are already saved
(the stream itself restarts from the beginning).

Usage:
  python -m bench.fetch_subset --anatomy knee  --source "<knee_multicoil_val URL or .tar.xz path>"
  python -m bench.fetch_subset --anatomy brain --source "<brain_multicoil_val URL or .tar.xz path>"
"""
import argparse
import io
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import h5py

from .config import load_config

# Test-set contrast mix, chosen to follow the contrasts present in the training volumes
QUOTAS = {
    "knee": {"CORPD_FBK": 10, "CORPDFS_FBK": 10},
    "brain": {"AXT2": 10, "AXT1POST": 5, "AXT1": 2, "AXFLAIR": 2, "AXT1PRE": 1},
}


def open_stream(source):
    if str(source).startswith(("http://", "https://")):
        return urllib.request.urlopen(source)
    return open(source, "rb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--anatomy", required=True, choices=list(QUOTAS))
    ap.add_argument("--source", required=True, help="fastMRI *_multicoil_val .tar.xz: signed URL or local path")
    ap.add_argument("--max-gb", type=float, default=400, help="stop after reading this much (uncompressed)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["test_raw_dirs"][args.anatomy])
    out_dir.mkdir(parents=True, exist_ok=True)

    quota = dict(QUOTAS[args.anatomy])
    have = Counter()
    for f in out_dir.glob("*.h5"):
        with h5py.File(f, "r") as h:
            have[h.attrs.get("acquisition")] += 1
    print("already saved:", dict(have))

    read_gb = 0.0
    with open_stream(args.source) as stream, tarfile.open(fileobj=stream, mode="r|xz") as tar:
        for member in tar:
            if all(have[a] >= n for a, n in quota.items()):
                break
            if not member.isfile() or not member.name.endswith(".h5"):
                continue
            read_gb += member.size / 1e9
            if read_gb > args.max_gb:
                print("reached --max-gb, stopping")
                break
            name = Path(member.name).name
            data = tar.extractfile(member).read()
            with h5py.File(io.BytesIO(data), "r") as h:
                acq = h.attrs.get("acquisition")
                ok = "kspace" in h and "reconstruction_rss" in h   # fully sampled reference present
            if not ok or acq not in quota or have[acq] >= quota[acq] or (out_dir / name).exists():
                continue
            (out_dir / name).write_bytes(data)
            have[acq] += 1
            print(f"saved {name} ({acq}), {dict(have)}, read {read_gb:.1f} GB", flush=True)
    missing = {a: n - have[a] for a, n in quota.items() if have[a] < n}
    print("done" if not missing else f"quota not filled: {missing}")


if __name__ == "__main__":
    main()
