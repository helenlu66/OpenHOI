"""CPU/CUDA-compatible KNN used by HOIAffordanceMLLM on macOS and Ubuntu."""

from __future__ import annotations

import torch


def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(src.shape[0], src.shape[1], 1)
    dist += torch.sum(dst ** 2, -1).view(dst.shape[0], 1, dst.shape[1])
    return dist


def _knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    _, group_idx = torch.topk(
        _square_distance(new_xyz, xyz), nsample, dim=-1, largest=False, sorted=False
    )
    return group_idx


class KNN:
    """Drop-in fallback when knn_cuda is unavailable (e.g. macOS without CUDA)."""

    def __init__(self, k: int, transpose_mode: bool = False) -> None:
        self.k = k
        self.transpose_mode = transpose_mode

    def __call__(self, coor_k: torch.Tensor, coor_q: torch.Tensor):
        if self.transpose_mode:
            coor_k, coor_q = coor_q, coor_k
        device = coor_k.device
        # Large KNN matmuls are unstable on MPS; compute on CPU and move indices back.
        if device.type == "mps":
            coor_k = coor_k.cpu()
            coor_q = coor_q.cpu()
        idx = _knn_point(self.k, coor_k.transpose(1, 2), coor_q.transpose(1, 2))
        return None, idx.transpose(1, 2).contiguous().to(device)
