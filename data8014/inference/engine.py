"""The ``InferenceEngine``: run a (checkpointed) model over a dataloader.

Deliberately separate from the ``Trainer``: inference never needs gradients or
optimizer state, and in real systems it has its own optimal configurations
(batch size, parallelism, KV caches, ...) which are out of scope here. The
engine also owns checkpoint loading so downstream stages only deal with a
ready-to-run model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models.linear import model_from_checkpoint
from ..utils.plotting import savefig


@dataclass
class InferenceOutput:
    """Model predictions over a dataloader, plus optional inputs / labels."""

    predictions: torch.Tensor
    labels: torch.Tensor | None = None
    inputs: torch.Tensor | None = None


class InferenceEngine:
    """Loads a checkpoint (or takes a model) and predicts on a dataloader."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.checkpoint: dict | None = None

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Path | str, device: torch.device
    ) -> "InferenceEngine":
        """Build an engine whose model is loaded from a ``.pt`` checkpoint."""
        model, checkpoint = model_from_checkpoint(checkpoint_path, device)
        engine = cls(model, device)
        engine.checkpoint = checkpoint
        return engine

    @torch.no_grad()
    def run(
        self,
        dataloader: DataLoader,
        *,
        collect_inputs: bool = False,
        collect_labels: bool = False,
    ) -> InferenceOutput:
        """Predict on every batch; tensors are gathered on CPU."""
        predictions, inputs, labels = [], [], []
        for batch in dataloader:
            x = batch[0].to(self.device, non_blocking=True)
            predictions.append(self.model(x).detach().cpu())
            if collect_labels:
                labels.append(batch[1].cpu())
            if collect_inputs:
                inputs.append(x.cpu())
        return InferenceOutput(
            predictions=torch.cat(predictions),
            labels=torch.cat(labels) if labels else None,
            inputs=torch.cat(inputs) if inputs else None,
        )


def plot_true_vs_predicted(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    out_path: Path | str,
    title: str = "Inference: predictions vs true labels",
) -> Path:
    """Scatter true labels (x axis) against predictions (y axis).

    The identity line visualizes accuracy; the MSE and the mean signed error
    ("bias") are annotated in the title.
    """
    labels = labels.flatten().cpu()
    predictions = predictions.flatten().cpu()
    mse = float(((predictions - labels) ** 2).mean())
    bias = float((predictions - labels).mean())

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.scatter(labels, predictions, s=6, alpha=0.25, edgecolors="none")
    lo = float(min(labels.min(), predictions.min()))
    hi = float(max(labels.max(), predictions.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.0, label="y = x")
    ax.set_xlabel("true label y")
    ax.set_ylabel("predicted label")
    ax.set_title(f"{title}\nMSE {mse:.5f} | bias {bias:+.5f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return savefig(fig, out_path)
