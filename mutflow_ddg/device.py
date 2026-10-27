from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(name)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device={name!r} was requested, but CUDA is not available to PyTorch"
        )

    index = device.index if device.index is not None else 0
    device_count = torch.cuda.device_count()
    if index < 0 or index >= device_count:
        raise RuntimeError(
            f"device=cuda:{index} was requested, but PyTorch only sees "
            f"{device_count} CUDA device(s)"
        )
    return torch.device(f"cuda:{index}")


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    index = device.index if device.index is not None else 0
    return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
