"""Centralized CPU / macOS compatibility shim for the Affordance-Driven HOI
Diffusion stage (Stage B) of OpenHOI.

Like Stage A's openhoi_cpu.py, this keeps the upstream source unmodified by
installing fallbacks *before* the `lib.*` modules are imported:

  1. A pure-PyTorch `pytorch3d` stub (only `ops.knn.knn_points` and
     `structures.Meshes` are imported by the inference path -- the functions
     that use them are not called by plain diffusion sampling, so a light
     stub avoids the painful Apple-Silicon pytorch3d build).
  2. `.cuda()` -> no-op on CPU (the builders and helpers call `.cuda()`).
  3. `torch.load` defaults to `map_location="cpu"` (checkpoints were saved on GPU).

Call `install()` before importing anything under `lib/`.
"""

from __future__ import annotations

import sys
import types

import torch

_INSTALLED = False


# --------------------------------------------------------------------------- #
# pytorch3d stub
# --------------------------------------------------------------------------- #

class _KNNResult:
    def __init__(self, dists, idx):
        self.dists = dists
        self.idx = idx


def _knn_points(p1, p2, lengths1=None, lengths2=None, K=1, **_kwargs):
    """Pure-torch stand-in for pytorch3d.ops.knn_points.

    p1:(B,N,3) p2:(B,M,3) -> squared dists & indices of K nearest, shape (B,N,K).
    """
    dist = torch.cdist(p1, p2)
    dists, idx = dist.topk(K, dim=-1, largest=False)
    return _KNNResult(dists ** 2, idx)


class _Meshes:
    """Minimal placeholder; the inference path only constructs it, never uses
    its rasterization methods."""

    def __init__(self, verts=None, faces=None, **_kwargs):
        self.verts = verts
        self.faces = faces

    def verts_padded(self):
        return self.verts

    def faces_padded(self):
        return self.faces


def _install_pytorch3d_stub() -> None:
    try:
        import pytorch3d  # noqa: F401  (real install available)
        return
    except ImportError:
        pass

    pkg = types.ModuleType("pytorch3d")
    pkg.__path__ = []

    ops = types.ModuleType("pytorch3d.ops")
    ops.__path__ = []
    knn = types.ModuleType("pytorch3d.ops.knn")
    knn.knn_points = _knn_points
    ops.knn = knn
    ops.knn_points = _knn_points

    structures = types.ModuleType("pytorch3d.structures")
    structures.__path__ = []
    structures.Meshes = _Meshes
    structures.join_meshes_as_scene = lambda *a, **k: None
    structures.join_meshes_as_batch = lambda *a, **k: None

    pkg.ops = ops
    pkg.structures = structures
    for name, mod in {
        "pytorch3d": pkg,
        "pytorch3d.ops": ops,
        "pytorch3d.ops.knn": knn,
        "pytorch3d.structures": structures,
    }.items():
        sys.modules.setdefault(name, mod)


# --------------------------------------------------------------------------- #
# runtime patches
# --------------------------------------------------------------------------- #

def _is_cuda_arg(x) -> bool:
    return (isinstance(x, str) and "cuda" in x) or \
           (isinstance(x, torch.device) and x.type == "cuda")


def _remap_to_args(args, kwargs):
    """Rewrite any cuda device in .to()/.cuda() args to cpu."""
    args = tuple("cpu" if _is_cuda_arg(a) else a for a in args)
    if _is_cuda_arg(kwargs.get("device")):
        kwargs = {**kwargs, "device": "cpu"}
    return args, kwargs


def _install_runtime_patches() -> None:
    if not torch.cuda.is_available():
        # Force CPU everywhere: some deps (e.g. sentence-transformers) otherwise
        # auto-select MPS, producing cross-device (mps vs cpu) tensor mismatches.
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            mps.is_available = lambda: False

        if not getattr(torch.Tensor.cuda, "_openhoi_cpu", False):
            def _tensor_cuda(self, *a, **k):
                return self
            _tensor_cuda._openhoi_cpu = True
            torch.Tensor.cuda = _tensor_cuda
        if not getattr(torch.nn.Module.cuda, "_openhoi_cpu", False):
            def _module_cuda(self, *a, **k):
                return self
            _module_cuda._openhoi_cpu = True
            torch.nn.Module.cuda = _module_cuda

        # Hard-coded .to("cuda") / .to(device=cuda) -> cpu (upstream sprinkles these).
        if not getattr(torch.Tensor.to, "_openhoi_cpu", False):
            _orig_tensor_to = torch.Tensor.to

            def _tensor_to(self, *a, **k):
                a, k = _remap_to_args(a, k)
                return _orig_tensor_to(self, *a, **k)
            _tensor_to._openhoi_cpu = True
            torch.Tensor.to = _tensor_to

        if not getattr(torch.nn.Module.to, "_openhoi_cpu", False):
            _orig_module_to = torch.nn.Module.to

            def _module_to(self, *a, **k):
                a, k = _remap_to_args(a, k)
                return _orig_module_to(self, *a, **k)
            _module_to._openhoi_cpu = True
            torch.nn.Module.to = _module_to

        # CPU-only: checkpoints were saved on GPU; default them onto CPU.
        if not getattr(torch.load, "_openhoi_cpu", False):
            _orig_load = torch.load

            def _load(f, *a, **k):
                k.setdefault("map_location", "cpu")
                return _orig_load(f, *a, **k)

            _load._openhoi_cpu = True
            torch.load = _load


def _install_numpy_aliases() -> None:
    # This code predates NumPy 1.24, which removed np.bool/np.int/etc.
    import numpy as np
    for name, builtin in {"bool": bool, "int": int, "float": float,
                          "object": object, "str": str, "complex": complex}.items():
        if not hasattr(np, name):
            setattr(np, name, builtin)


def install() -> torch.device:
    global _INSTALLED
    if not _INSTALLED:
        _install_pytorch3d_stub()
        _install_numpy_aliases()
        _install_runtime_patches()
        _INSTALLED = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
