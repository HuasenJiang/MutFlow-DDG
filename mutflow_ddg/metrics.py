from __future__ import annotations

import torch
from torch import Tensor


def _correlation(left: Tensor, right: Tensor) -> Tensor:
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if denominator <= 1e-12:
        return left.new_tensor(float("nan"))
    return (left * right).sum() / denominator


def _rank(values: Tensor) -> Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(
        values.numel(), dtype=torch.float32, device=values.device
    )
    return ranks


def regression_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    prediction = prediction.detach().float().cpu()
    target = target.detach().float().cpu()
    error = prediction - target
    return {
        "mae": error.abs().mean().item(),
        "mse": error.square().mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "pcc": _correlation(prediction, target).item(),
        "spearman": _correlation(_rank(prediction), _rank(target)).item(),
    }


def symmetry_metrics(direct: Tensor, reverse: Tensor) -> dict[str, float]:
    direct = direct.detach().float().cpu()
    reverse = reverse.detach().float().cpu()
    return {
        "antisymmetry_mae": (direct + reverse).abs().mean().item(),
        "direct_reverse_pcc": _correlation(direct, reverse).item(),
    }
