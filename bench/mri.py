"""Centred orthonormal FFTs and the multi-coil SENSE forward model, for numpy and torch.

Conventions used everywhere in this project:
  k-space     (..., C, H, W) complex, H = readout (fully sampled), W = phase encode (undersampled)
  mask        (W,) bool/float, broadcast over coils and readout
  sens maps   (..., C, H, W) complex, normalised so that sum_c |S_c|^2 = 1 inside the support
  image       (..., H, W) complex coil-combined image, or (..., H, W) real magnitude

Forward model:  y = M F S x + n,   A = M F S,   A^H y = sum_c conj(S_c) F^H (M y_c)
"""
import numpy as np
import torch


# ----------------------------------------------------------------------------- numpy
def fft2c_np(x, axes=(-2, -1)):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def ifft2c_np(k, axes=(-2, -1)):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=axes), axes=axes, norm="ortho"), axes=axes)


def rss_np(x, axis=-3):
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=axis))


def center_crop_np(x, shape):
    """Centre crop the last two axes to `shape`, zero-padding any axis that is smaller."""
    pad = [(0, 0)] * (x.ndim - 2)
    for n, t in zip(x.shape[-2:], shape):
        extra = max(t - n, 0)
        pad.append((extra // 2, extra - extra // 2))
    x = np.pad(x, pad)
    h, w = x.shape[-2:]
    th, tw = shape
    top, left = (h - th) // 2, (w - tw) // 2
    return x[..., top:top + th, left:left + tw]


# ----------------------------------------------------------------------------- torch
def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(k):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def rss(x, dim=-3):
    return torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=dim))


def sens_expand(x, maps):
    """Coil-combined image (..., H, W) -> coil k-space (..., C, H, W)."""
    return fft2c(maps * x.unsqueeze(-3))


def sens_reduce(k, maps):
    """Coil k-space (..., C, H, W) -> coil-combined image (..., H, W)."""
    return torch.sum(torch.conj(maps) * ifft2c(k), dim=-3)


def forward_op(x, maps, mask):
    """A x = M F S x."""
    return sens_expand(x, maps) * mask


def adjoint_op(y, maps, mask):
    """A^H y = S^H F^H M y."""
    return sens_reduce(y * mask, maps)


def normal_op(x, maps, mask):
    """A^H A x."""
    return adjoint_op(forward_op(x, maps, mask), maps, mask)


def conjugate_gradient(normal, rhs, x0, n_iter=10, tol=1e-10):
    """Solve normal(x) = rhs with CG; batched over the leading dimension, differentiable."""
    x = x0
    r = rhs - normal(x)
    p = r
    rs = _cdot(r, r)
    for _ in range(n_iter):
        ap = normal(p)
        alpha = rs / (_cdot(p, ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = _cdot(r, r)
        if bool((rs_new.abs() < tol).all()):
            break
        p = r + (rs_new / (rs + 1e-30)) * p
        rs = rs_new
    return x


def _cdot(a, b):
    """Batched <a, b> over all but the first dim, shaped for broadcasting."""
    d = torch.sum(torch.conj(a) * b, dim=tuple(range(1, a.ndim))).real
    return d.reshape((-1,) + (1,) * (a.ndim - 1))


def complex_to_chan(x):
    """(B, H, W) complex -> (B, 2, H, W) real."""
    return torch.stack([x.real, x.imag], dim=1)


def chan_to_complex(x):
    """(B, 2, H, W) real -> (B, H, W) complex."""
    return torch.complex(x[:, 0], x[:, 1])
