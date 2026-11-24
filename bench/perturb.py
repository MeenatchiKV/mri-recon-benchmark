"""Acquisition perturbations applied in k-space (where these effects physically occur), before
reconstruction, on every test slice. The reference image is always the clean fully sampled RSS.
All perturbations are deterministic per (volume, slice, R, condition).

  noise:<level>    complex Gaussian receiver noise on the acquired samples, sigma = level * volume max
                   (orthonormal FFT, so sigma is also the per-coil image-domain noise std)
  motion:<pixels>  rigid in-plane translation during the scan: with sequential PE ordering, the lines after a
                   random onset (30-70 % of the scan) are acquired with the object shifted by up to <pixels>
                   (Fourier shift theorem -> linear phase on those lines) -> ghosting
  dropout:<frac>   a fraction of the acquired non-ACS lines is lost (zeroed) but the mask still marks them
                   as sampled, i.e. silent data corruption
  spike:<count>    herringbone (RF spike) artefact: <count> acquired k-space samples, in all coils, get an added
                   spike of 500x the median acquired magnitude with random phase

Distribution shifts handled by bench.evaluate rather than here: R = 12 (unseen at training), equispaced masks
(models trained on random masks), and cross-anatomy (knee model on brain data and vice versa).
"""
import zlib

import numpy as np

from .masks import acs_slice


def _rng(sample, condition):
    return np.random.default_rng(zlib.crc32(f"{sample['volume']}/{sample['slice']}/{sample['accel']}/{condition}".encode()))


def apply(sample, condition):
    """Returns a new masked k-space (C, H, W) complex64 for the given condition string."""
    k = sample["masked_kspace"].copy()
    mask = sample["mask"]
    if condition == "nominal":
        return k
    kind, level = condition.split(":")
    level = float(level)
    rng = _rng(sample, condition)
    c, h, w = k.shape

    if kind == "noise":
        sigma = level * sample["max_value"]
        noise = (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) * sigma / np.sqrt(2)
        return (k + noise * mask).astype(np.complex64)

    if kind == "motion":
        onset = int(rng.uniform(0.3, 0.7) * w)
        dy, dx = rng.uniform(-1, 1, size=2)
        dy, dx = dy * level, dx * level
        ky = (np.arange(h) - h // 2)[:, None] / h
        kx = (np.arange(w) - w // 2)[None, :] / w
        phase = np.exp(-2j * np.pi * (ky * dy + kx * dx))
        moved = np.zeros(w, bool)
        moved[onset:] = True
        return np.where(moved[None, None, :], k * phase[None], k).astype(np.complex64)

    if kind == "dropout":
        acs = np.zeros(w, bool)
        acs[acs_slice(w, sample["num_low"])] = True
        candidates = np.flatnonzero((mask > 0) & ~acs)
        lost = rng.choice(candidates, size=int(round(level * len(candidates))), replace=False)
        k[..., lost] = 0
        return k

    if kind == "spike":
        sampled = np.flatnonzero(mask > 0)
        amp = 500 * np.median(np.abs(k[..., sampled]))
        for _ in range(int(level)):
            row, col = rng.integers(h), rng.choice(sampled)
            k[:, row, col] += amp * np.exp(2j * np.pi * rng.uniform())
        return k

    raise ValueError(condition)


CONDITIONS = ["nominal",
              "noise:0.005", "noise:0.01", "noise:0.02", "noise:0.04",
              "motion:1", "motion:2", "motion:4",
              "dropout:0.05", "dropout:0.1", "dropout:0.2",
              "spike:2", "spike:5", "spike:10"]
