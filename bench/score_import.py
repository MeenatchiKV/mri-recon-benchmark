"""Load the official score-MRI code (Chung & Ye 2022, ../score-MRI-main) on a machine without a CUDA compiler.

score-MRI's NCSN++ calls a custom CUDA kernel (op.upfirdn2d) that is compiled at import time. The same repo
ships the pure-PyTorch implementation `upfirdn2d_native`, which computes the identical operation; we register
a stand-in `op` module that always uses it. Nothing else in the official code is changed.
"""
import importlib
import sys
import types
from pathlib import Path

from .config import ROOT

SCORE_ROOT = (ROOT.parent / "score-MRI-main").resolve()
_loaded = None


def _native_op_module():
    src = (SCORE_ROOT / "op" / "upfirdn2d.py").read_text()
    start = src.index("def upfirdn2d_native(")
    ns = {}
    exec("import torch\nfrom torch.nn import functional as F\n" + src[start:], ns)   # official native code
    native = ns["upfirdn2d_native"]

    def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
        return native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

    op = types.ModuleType("op")
    op.upfirdn2d = upfirdn2d
    op.FusedLeakyReLU = op.fused_leaky_relu = None   # not used by NCSN++
    return op


def load_score_mri():
    """Returns a namespace with the official modules: config, mutils, sde_lib, losses, sampling, ema."""
    global _loaded
    if _loaded is None:
        sys.modules.setdefault("op", _native_op_module())
        if str(SCORE_ROOT) not in sys.path:
            sys.path.insert(0, str(SCORE_ROOT))
        ns = types.SimpleNamespace()
        ns.config_module = importlib.import_module("configs.ve.fastmri_knee_320_ncsnpp_continuous")
        ns.mutils = importlib.import_module("models.utils")
        importlib.import_module("models.ncsnpp")   # registers the architecture
        ns.sde_lib = importlib.import_module("sde_lib")
        ns.losses = importlib.import_module("losses")
        ns.sampling = importlib.import_module("sampling")
        ns.ema = importlib.import_module("models.ema")
        _loaded = ns
    return _loaded
