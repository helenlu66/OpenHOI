"""Cross-platform device helpers for HOIAffordanceMLLM inference."""

from __future__ import annotations

import os

import torch


def get_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if os.environ.get("OPENHOI_FORCE_CPU", "").lower() in {"1", "true", "yes"}:
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device("cpu")


def use_deepspeed() -> bool:
    if os.environ.get("OPENHOI_USE_DEEPSPEED", "").lower() in {"0", "false", "no"}:
        return False
    return torch.cuda.is_available()


def dict_to_device(input_dict, device: torch.device):
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            tensor = value
            if device.type == "mps" and tensor.dtype == torch.float64:
                tensor = tensor.to(torch.float32)
            input_dict[key] = tensor.to(device, non_blocking=False)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            input_dict[key] = [
                (
                    tensor.to(torch.float32)
                    if device.type == "mps" and tensor.dtype == torch.float64
                    else tensor
                ).to(device, non_blocking=False)
                for tensor in value
            ]
    return input_dict


def empty_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
