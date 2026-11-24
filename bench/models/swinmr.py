"""SwinMR (Huang et al. 2022, "Swin transformer for fast MRI"), a SwinIR-style network:
shallow conv -> residual Swin transformer blocks (RSTB, shifted 8x8 windows) -> conv -> image residual.

As in SwinMR the network maps the zero-filled magnitude image to the reconstructed magnitude image
(here the RSS of the zero-filled coil images) without a data-consistency layer, and is trained with
  alpha * Charbonnier(image) + beta * Charbonnier(|FFT| of image) + gamma * VGG perceptual loss.
The width / depth defaults are smaller than the SwinMR paper so the model trains on 320 x 320 and
384 x 384 images on a 6 GB GPU; they are reported with the results. Gradient checkpointing per block.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..mri import ifft2c, rss


def window_partition(x, ws):
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, c)


def window_reverse(win, ws, h, w):
    b = win.shape[0] // ((h // ws) * (w // ws))
    x = win.view(b, h // ws, w // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, ws, heads):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.bias_table = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + ws - 1
        self.register_buffer("rel_index", rel[..., 0] * (2 * ws - 1) + rel[..., 1], persistent=False)

    def forward(self, x, mask=None):
        bw, n, c = x.shape
        q, k, v = self.qkv(x).reshape(bw, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self.bias_table[self.rel_index.view(-1)].view(n, n, -1).permute(2, 0, 1)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(bw // nw, nw, self.heads, n, n) + mask[None, :, None]
            attn = attn.view(bw, self.heads, n, n)
        x = (attn.softmax(-1) @ v).transpose(1, 2).reshape(bw, n, c)
        return self.proj(x)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift, mlp_ratio=2.0):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def _mask(self, h, w, device):
        if not self.shift:
            return None
        img = torch.zeros(1, h, w, 1, device=device)
        cnt = 0
        for hs in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
            for wsl in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
                img[:, hs, wsl, :] = cnt
                cnt += 1
        win = window_partition(img, self.ws).squeeze(-1)
        m = win[:, None, :] - win[:, :, None]
        return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)

    def forward(self, x):                          # x: (B, H, W, C)
        b, h, w, c = x.shape
        y = self.norm1(x)
        if self.shift:
            y = torch.roll(y, (-self.shift, -self.shift), (1, 2))
        y = window_reverse(self.attn(window_partition(y, self.ws), self._mask(h, w, x.device)), self.ws, h, w)
        if self.shift:
            y = torch.roll(y, (self.shift, self.shift), (1, 2))
        x = x + y
        return x + self.mlp(self.norm2(x))


class RSTB(nn.Module):
    def __init__(self, dim, depth, heads, ws, grad_checkpoint):
        super().__init__()
        self.blocks = nn.ModuleList(SwinBlock(dim, heads, ws, 0 if i % 2 == 0 else ws // 2) for i in range(depth))
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x):                          # x: (B, C, H, W)
        y = x.permute(0, 2, 3, 1)
        for blk in self.blocks:
            y = checkpoint(blk, y, use_reentrant=False) if self.grad_checkpoint and self.training else blk(y)
        return x + self.conv(y.permute(0, 3, 1, 2))


class SwinMR(nn.Module):
    default_train = {"optimizer": "adam", "lr": 2e-4, "weight_decay": 0.0, "epochs": 50, "lr_step": 40,
                     "lr_gamma": 0.5}

    def __init__(self, embed_dim=96, depths=(6, 6, 6, 6), heads=(6, 6, 6, 6), window=8, alpha=15.0,
                 beta=0.1, gamma=0.0025, grad_checkpoint=True):
        super().__init__()
        self.window = window
        self.conv_first = nn.Conv2d(1, embed_dim, 3, padding=1)
        self.layers = nn.ModuleList(RSTB(embed_dim, d, h, window, grad_checkpoint) for d, h in zip(depths, heads))
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.conv_last = nn.Conv2d(embed_dim, 1, 3, padding=1)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self._vgg_holder = []   # a plain list keeps the frozen loss network out of state_dict / parameters()

    def forward(self, batch):
        zf = rss(ifft2c(batch["masked_kspace"]))
        scale = zf.amax(dim=(-2, -1), keepdim=True) + 1e-11
        x = (zf / scale).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            f = self.conv_first(x)
            y = f
            for layer in self.layers:
                y = layer(y)
            y = self.norm(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            out = x + self.conv_last(self.conv_body(y) + f)
        self._scale = scale
        return out.float().squeeze(1) * scale

    def _vgg_features(self, x):
        if not self._vgg_holder:
            from torchvision.models import VGG19_Weights, vgg19
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].eval().to(x.device)
            for p in vgg.parameters():
                p.requires_grad_(False)
            self._vgg_holder.append(vgg)
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return self._vgg_holder[0]((x.repeat(1, 3, 1, 1) - mean) / std)

    def loss(self, batch, pred):
        def charbonnier(a, b, eps=1e-9):
            return torch.sqrt((a - b) ** 2 + eps).mean()
        p, t = pred / self._scale, batch["target"] / self._scale
        loss = self.alpha * charbonnier(p, t)
        loss = loss + self.beta * charbonnier(torch.fft.fft2(p).abs(), torch.fft.fft2(t).abs())
        if self.gamma:
            loss = loss + self.gamma * F.l1_loss(self._vgg_features(p.unsqueeze(1)),
                                                 self._vgg_features(t.unsqueeze(1)))
        return loss
