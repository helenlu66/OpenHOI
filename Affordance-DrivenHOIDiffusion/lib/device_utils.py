"""Device selection for Affordance-Driven HOI Diffusion (CPU / MPS / CUDA)."""

from __future__ import annotations

import os

import torch


def get_inference_device() -> torch.device:
    override = os.environ.get("OPENHOI_DEVICE", "").strip().lower()
    if override:
        if override == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if override == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if override == "cpu":
            return torch.device("cpu")
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
