"""Checkpoint / tensor serialization helpers.

Checkpoints are flat dicts containing at least ``model_state`` plus whatever
metadata the producing stage wants to record (optimizer state, step count,
configs). Everything stored here must stay loadable with
``torch.load(..., weights_only=True)``, i.e. tensors and plain Python
containers of primitives only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save ``model`` (+ optional ``optimizer`` state and extra metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model_state": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def save_tensors(path: Path | str, **tensors: torch.Tensor) -> Path:
    """Save named tensors (e.g. a generated dataset) to a ``.pt`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, path)
    return path


def load_tensors(path: Path | str) -> dict[str, torch.Tensor]:
    """Load tensors saved by :func:`save_tensors`."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)
