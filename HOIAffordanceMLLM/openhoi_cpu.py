"""Centralized CPU / macOS compatibility shim for HOIAffordanceMLLM.

The upstream code targets A100 GPUs and imports several CUDA-only packages
(``knn_cuda``, ``pointnet2_ops``, ``deepspeed``), sprinkles ``.cuda()`` calls
through the model, and hard-codes weight paths under ``/root/tmp``.

Rather than editing those files, this single module reproduces the required
behaviour on CPU by:

  1. Registering pure-PyTorch fallbacks for the CUDA-only imports in
     ``sys.modules`` (so the unmodified ``import`` statements just work).
  2. Monkeypatching ``Tensor.cuda`` / ``Module.cuda`` to no-ops when CUDA is
     absent (so hard-coded ``.cuda()`` calls stay on CPU).
  3. Redirecting ``torch.load`` reads of ``/root/tmp/...`` to the local weight
     directory (``$OPENHOI_DATA_ROOT``, default ``~/openhoi-data``).

Usage (must run *before* ``import llava``)::

    import openhoi_cpu
    openhoi_cpu.install()
    from llava.model import LISAForCausalLM   # now imports cleanly on CPU

``install()`` is idempotent and a no-op when a real CUDA stack is present, so
importing it on a Linux/GPU box does not change behaviour.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def data_root() -> Path:
    """Local directory holding the downloaded weights / datasets."""
    return Path(
        os.environ.get("OPENHOI_DATA_ROOT", str(Path.home() / "openhoi-data"))
    ).expanduser()


# Specific remaps where the on-disk layout differs from the hard-coded path.
def _specific_remaps() -> dict[str, Path]:
    root = data_root()
    return {
        # HF stores uni3d under modelzoo/uni3d-b/model.pt, not directly under uni3d/.
        "/root/tmp/uni3d/model.pt": root / "uni3d" / "modelzoo" / "uni3d-b" / "model.pt",
    }


def remap_path(path):
    """Redirect a hard-coded ``/root/tmp/...`` path into ``data_root()``."""
    if not isinstance(path, (str, os.PathLike)):
        return path
    text = os.fspath(path)
    specific = _specific_remaps()
    if text in specific:
        return str(specific[text])
    prefix = "/root/tmp/"
    if text.startswith(prefix):
        return str(data_root() / text[len(prefix):])
    return path


# --------------------------------------------------------------------------- #
# Device helper
# --------------------------------------------------------------------------- #

def get_device() -> torch.device:
    """Preferred inference device. CPU on a Mac unless CUDA is available.

    MPS is intentionally *not* selected by default: the 7B forward pass is
    numerically unstable on MPS for this model. Set ``OPENHOI_ALLOW_MPS=1`` to
    opt in.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        os.environ.get("OPENHOI_ALLOW_MPS", "").lower() in {"1", "true", "yes"}
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Pure-PyTorch geometry ops (replacements for knn_cuda / pointnet2_ops)
# --------------------------------------------------------------------------- #

def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(b, n, 1)
    dist += torch.sum(dst ** 2, -1).view(b, 1, m)
    return dist


def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Indices of the ``nsample`` nearest neighbours in ``xyz`` for ``new_xyz``.

    xyz: (B, N, C), new_xyz: (B, S, C) -> (B, S, nsample)
    """
    _, group_idx = torch.topk(
        _square_distance(new_xyz, xyz), nsample, dim=-1, largest=False, sorted=False
    )
    return group_idx


class KNN:
    """Drop-in replacement for ``knn_cuda.KNN`` used by DGCNN_Propagation.

    Matches the knn_cuda calling convention: inputs are ``(B, C, N)`` and the
    returned index tensor is ``(B, k, S)`` so that ``idx.shape[1] == k``.
    """

    def __init__(self, k: int, transpose_mode: bool = False) -> None:
        self.k = k
        self.transpose_mode = transpose_mode

    def __call__(self, ref: torch.Tensor, query: torch.Tensor):
        if self.transpose_mode:
            ref, query = query, ref
        idx = knn_point(self.k, ref.transpose(1, 2), query.transpose(1, 2))  # (B, S, k)
        return None, idx.transpose(1, 2).contiguous()  # (B, k, S)


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative FPS. xyz: (B, N, 3) -> (B, npoint) long indices."""
    device = xyz.device
    b, n, _ = xyz.shape
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.full((b, n), 1e10, device=device)
    # Deterministic seed point keeps CPU runs reproducible.
    farthest = torch.zeros(b, dtype=torch.long, device=device)
    batch_indices = torch.arange(b, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather columns. features: (B, C, N), idx: (B, S) -> (B, C, S)."""
    idx_expanded = idx.long().unsqueeze(1).expand(-1, features.shape[1], -1)
    return torch.gather(features, 2, idx_expanded)


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #

_INSTALLED = False


def _register_module(name: str, module: types.ModuleType) -> None:
    sys.modules.setdefault(name, module)


def _install_import_stubs() -> None:
    # knn_cuda -------------------------------------------------------------- #
    try:
        import knn_cuda  # noqa: F401  (real package available)
    except ImportError:
        m = types.ModuleType("knn_cuda")
        m.KNN = KNN
        _register_module("knn_cuda", m)

    # pointnet2_ops.pointnet2_utils ---------------------------------------- #
    try:
        import pointnet2_ops  # noqa: F401
    except ImportError:
        pkg = types.ModuleType("pointnet2_ops")
        pkg.__path__ = []  # advertise as a package so submodule import works
        utils = types.ModuleType("pointnet2_ops.pointnet2_utils")
        utils.furthest_point_sample = furthest_point_sample
        utils.gather_operation = gather_operation
        pkg.pointnet2_utils = utils
        _register_module("pointnet2_ops", pkg)
        _register_module("pointnet2_ops.pointnet2_utils", utils)

    # deepspeed (import-time only; training path is unused on CPU) ---------- #
    # Note: the stub carries a valid __spec__ so transformers' feature probe
    # (importlib.util.find_spec("deepspeed")) does not raise; its subsequent
    # importlib.metadata lookup then fails, so transformers correctly treats
    # deepspeed as unavailable.
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        import importlib.machinery as _machinery

        m = types.ModuleType("deepspeed")
        m.__spec__ = _machinery.ModuleSpec("deepspeed", loader=None)
        _register_module("deepspeed", m)


def _install_runtime_patches() -> None:
    # torch.empty scalar guard --------------------------------------------- #
    # transformers 4.3x low_cpu_mem_usage loading calls
    # ``torch.empty(*param.size(), dtype=...)``; for a 0-d parameter (e.g.
    # Uni3D's ``logit_scale``) that becomes ``torch.empty()`` and raises. Make
    # the zero-arg call return a 0-d tensor. (Version-compat, not CUDA-specific.)
    if not getattr(torch.empty, "_openhoi_cpu", False):
        _orig_empty = torch.empty

        def _empty(*size, **kwargs):
            if len(size) == 0:
                return _orig_empty((), **kwargs)
            return _orig_empty(*size, **kwargs)

        _empty._openhoi_cpu = True
        torch.empty = _empty

    if torch.cuda.is_available():
        return

    # .cuda() -> stay on CPU ----------------------------------------------- #
    if not getattr(torch.Tensor.cuda, "_openhoi_cpu", False):
        def _tensor_cuda(self, *args, **kwargs):
            return self
        _tensor_cuda._openhoi_cpu = True
        torch.Tensor.cuda = _tensor_cuda

    if not getattr(torch.nn.Module.cuda, "_openhoi_cpu", False):
        def _module_cuda(self, *args, **kwargs):
            return self
        _module_cuda._openhoi_cpu = True
        torch.nn.Module.cuda = _module_cuda

    # torch.load path remapping -------------------------------------------- #
    if not getattr(torch.load, "_openhoi_cpu", False):
        _orig_load = torch.load

        def _load(f, *args, **kwargs):
            kwargs.setdefault("map_location", "cpu")
            return _orig_load(remap_path(f), *args, **kwargs)

        _load._openhoi_cpu = True
        torch.load = _load


def install() -> torch.device:
    """Install CPU fallbacks (idempotent). Returns the chosen inference device."""
    global _INSTALLED
    if not _INSTALLED:
        _install_import_stubs()
        _install_runtime_patches()
        _INSTALLED = True
    return get_device()
