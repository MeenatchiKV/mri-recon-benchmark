"""Image-quality metrics on magnitude images, all with the fastMRI data-range convention
(data_range = maximum of the reference volume).

NMSE, PSNR, SSIM   fastMRI evaluation (Zbontar et al. 2018); SSIM with the 7x7 window used by fastMRI
HFEN               Ravishankar & Bresler 2011: ||LoG(x) - LoG(ref)|| / ||LoG(ref)||, sigma = 1.5
VIF                pixel-domain VIF (Sheikh & Bovik 2006), 4 scales
LPIPS              Zhang et al. 2018, AlexNet; images scaled to [-1, 1] by the data range

Primary metrics are computed inside an anatomy mask derived from the reference only, so that the
different background handling of RSS-combining methods (ZF, GRAPPA, U-Net, VarNet, SwinMR) and
sensitivity-combining methods (SENSE, CS, MoDL, RIM) does not decide the comparison. Whole-image metrics
(fastMRI convention) are returned alongside with the suffix _whole.
"""
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.metrics import structural_similarity

METRICS = ["NMSE", "PSNR", "SSIM", "HFEN", "VIF", "LPIPS"]
HIGHER_IS_BETTER = {"NMSE": False, "PSNR": True, "SSIM": True, "HFEN": False, "VIF": True, "LPIPS": False,
                    "TIME": False}


def foreground_mask(ref, data_range, threshold=0.1):
    m = gaussian_filter(ref, 3) > threshold * data_range
    m = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=5))
    m = ndi.binary_dilation(m, iterations=3)
    return m if m.sum() >= 64 else np.ones_like(m)   # near-empty edge slices fall back to the whole image


def nmse(ref, x, m):
    return float(np.sum((ref - x)[m] ** 2) / np.sum(ref[m] ** 2))


def psnr(ref, x, data_range, m=None):
    m = np.ones(ref.shape, bool) if m is None else m
    return float(10 * np.log10(data_range ** 2 / np.mean((ref - x)[m] ** 2)))


def ssim(ref, x, data_range, m=None):
    _, s = structural_similarity(ref, x, data_range=data_range, full=True)
    return float(s.mean() if m is None else s[m].mean())


def hfen(ref, x, m, sigma=1.5):
    lr, lx = gaussian_laplace(ref, sigma), gaussian_laplace(x, sigma)
    return float(np.linalg.norm((lx - lr)[m]) / np.linalg.norm(lr[m]))


def vif(ref, x, sigma_nsq=2.0, eps=1e-10):
    ref, x = ref.astype(np.float64), x.astype(np.float64)
    num, den = 0.0, 0.0
    for scale in range(1, 5):
        sd = (2 ** (4 - scale + 1) + 1) / 5.0
        if scale > 1:
            ref = gaussian_filter(ref, sd)[::2, ::2]
            x = gaussian_filter(x, sd)[::2, ::2]
        mu1, mu2 = gaussian_filter(ref, sd), gaussian_filter(x, sd)
        s1 = np.maximum(gaussian_filter(ref * ref, sd) - mu1 ** 2, 0)
        s2 = np.maximum(gaussian_filter(x * x, sd) - mu2 ** 2, 0)
        s12 = gaussian_filter(ref * x, sd) - mu1 * mu2
        g = s12 / (s1 + eps)
        sv = s2 - g * s12
        g[s1 < eps] = 0
        sv[s1 < eps] = s2[s1 < eps]
        s1[s1 < eps] = 0
        g[s2 < eps] = 0
        sv[s2 < eps] = 0
        sv[g < 0] = s2[g < 0]
        g[g < 0] = 0
        sv[sv <= eps] = eps
        num += np.sum(np.log10(1 + g ** 2 * s1 / (sv + sigma_nsq)))
        den += np.sum(np.log10(1 + s1 / sigma_nsq))
    return float(num / den)


class Evaluator:
    def __init__(self, device="cuda", hfen_sigma=1.5, lpips_net="alex", foreground_threshold=0.1):
        import lpips
        self.device = device
        self.hfen_sigma, self.fg_thr = hfen_sigma, foreground_threshold
        self.lpips = lpips.LPIPS(net=lpips_net, verbose=False).to(device).eval()

    @torch.no_grad()
    def lpips_score(self, ref, x, data_range):
        def prep(a):
            a = torch.from_numpy(np.clip(a / data_range, 0, 1) * 2 - 1).float()
            return a[None, None].repeat(1, 3, 1, 1).to(self.device)
        return float(self.lpips(prep(ref), prep(x)).item())

    def _all(self, ref, x, dr, m):
        return {"NMSE": nmse(ref, x, m), "PSNR": psnr(ref, x, dr, m), "SSIM": ssim(ref, x, dr, m),
                "HFEN": hfen(ref, x, m, self.hfen_sigma), "VIF": vif(ref * m, x * m),
                "LPIPS": self.lpips_score(ref * m, x * m, dr)}

    def __call__(self, ref, x, data_range, sense_ref=None, sense_range=None):
        """Metrics vs the RSS reference (anatomy mask, and whole image as *_whole) and, if given, vs the
        SENSE-combined reference (*_sref, same anatomy mask, data range = that reference's volume max)."""
        ref = np.asarray(ref, np.float32)
        x = np.asarray(x, np.float32)
        fg = foreground_mask(ref, data_range, self.fg_thr)
        out = self._all(ref, x, data_range, fg)
        whole = self._all(ref, x, data_range, np.ones_like(fg))
        out.update({f"{k}_whole": v for k, v in whole.items()})
        if sense_ref is not None:
            sref = self._all(np.asarray(sense_ref, np.float32), x, sense_range, fg)
            out.update({f"{k}_sref": v for k, v in sref.items()})
        out["fg_fraction"] = float(fg.mean())
        return out
