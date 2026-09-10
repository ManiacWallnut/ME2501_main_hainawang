"""Linear models: factories, config extraction, checkpoint reconstruction.

For now the codebase only needs ``torch.nn.Linear``; later assignments can
grow this module (e.g. MLP factories) while keeping the same
``model_config`` / ``model_from_config`` interface used by the checkpoint
format.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path

import torch
from torch import nn

from ..utils.io import load_checkpoint


def build_linear_model(in_features: int, out_features: int = 1, bias: bool = True) -> nn.Linear:
    """A plain ``torch.nn.Linear`` mapping ``in_features -> out_features``."""
    return nn.Linear(in_features, out_features, bias=bias)


def model_config(model: nn.Module) -> dict[str, Any]:
    """Extract a JSON-friendly config sufficient to rebuild a Linear model."""
    if not isinstance(model, nn.Linear):
        raise TypeError(f"model_config currently supports nn.Linear, got {type(model).__name__}")
    return {
        "kind": "linear",
        "in_features": model.in_features,
        "out_features": model.out_features,
        "bias": model.bias is not None,
    }


def model_from_config(config: dict[str, Any]) -> nn.Module:
    """Rebuild a model from the config recorded in a checkpoint."""
    if config.get("kind", "linear") != "linear":
        raise ValueError(f"unknown model kind: {config.get('kind')!r}")
    return build_linear_model(
        config["in_features"], config["out_features"], config.get("bias", True)
    )


def model_from_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild + load a model from a checkpoint; also returns the checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path)
    model = model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    if device is not None:
        model = model.to(device)
    return model, checkpoint
