"""The ``EvalRunner``: collect evaluation metrics for a checkpointed model.

Wraps an :class:`~data8014.inference.engine.InferenceEngine` (which owns
checkpoint loading and plain prediction) and adds gradient-sensitive
diagnostics that a pure forward pass cannot provide. For Assignment 0 the
requested evaluation is: per-example test loss and per-example gradient norm
(norm of the gradient of each example's loss w.r.t. the model parameters), the
correlation between the two, and histograms of these metrics over batches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..inference.engine import InferenceEngine
from ..utils.plotting import savefig


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation coefficient (NaN when either input is constant)."""
    xm = x - x.mean()
    ym = y - y.mean()
    denom = torch.sqrt((xm * xm).sum()) * torch.sqrt((ym * ym).sum())
    if float(denom) == 0.0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def per_example_gradient_norms(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """``||grad_theta L_i||_2`` for each example ``i`` in the batch.

    Model-agnostic: computed with ``torch.autograd.grad`` on one example at a
    time (cheap for small models, and correct for any differentiable module).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    norms = []
    for i in range(x.shape[0]):
        prediction = model(x[i : i + 1])
        loss_i = ((prediction - y[i : i + 1]) ** 2).sum()
        grads = torch.autograd.grad(loss_i, params)
        total = torch.zeros((), device=x.device)
        for grad in grads:
            total = total + grad.square().sum()
        norms.append(torch.sqrt(total))
    return torch.stack(norms).detach()


@dataclass
class EvalResult:
    """Everything collected during one evaluation run."""

    per_example_loss: torch.Tensor  # [n]
    per_example_grad_norm: torch.Tensor  # [n]
    batch_mean_loss: torch.Tensor  # [num_batches]
    batch_mean_grad_norm: torch.Tensor  # [num_batches]
    batch_correlation: torch.Tensor  # [num_batches], may contain NaN
    summary: dict[str, float | int] = field(default_factory=dict)

    def summary_str(self) -> str:
        lines = []
        for key, value in self.summary.items():
            text = f"{value:.6g}" if isinstance(value, float) else str(value)
            lines.append(f"{key}: {text}")
        return "\n".join(lines)


class EvalRunner:
    """Loads a checkpoint (via an InferenceEngine) and collects metrics."""

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path | str, device: torch.device) -> "EvalRunner":
        return cls(InferenceEngine.from_checkpoint(checkpoint_path, device))

    def run(self, dataloader: DataLoader) -> EvalResult:
        """Evaluate per-example loss / gradient norm and their correlation."""
        model = self.engine.model
        losses, grad_norms = [], []
        batch_mean_loss, batch_mean_gnorm, batch_corr = [], [], []
        for x, y in dataloader:
            x = x.to(self.engine.device, non_blocking=True)
            y = y.to(self.engine.device, non_blocking=True)
            with torch.enable_grad():
                batch_gnorm = per_example_gradient_norms(model, x, y)
            with torch.no_grad():
                predictions = model(x)
                batch_loss = ((predictions - y) ** 2).flatten()
            losses.append(batch_loss.cpu())
            grad_norms.append(batch_gnorm.cpu())
            batch_mean_loss.append(float(batch_loss.mean()))
            batch_mean_gnorm.append(float(batch_gnorm.mean()))
            batch_corr.append(pearson(batch_loss, batch_gnorm))

        all_losses = torch.cat(losses)
        all_gnorms = torch.cat(grad_norms)
        correlations = torch.tensor(batch_corr, dtype=torch.float64)
        summary: dict[str, float | int] = {
            "num_examples": int(all_losses.numel()),
            "num_batches": len(batch_mean_loss),
            "mean_per_example_loss": float(all_losses.mean()),
            "mean_per_example_grad_norm": float(all_gnorms.mean()),
            "overall_correlation_loss_vs_grad_norm": pearson(all_losses, all_gnorms),
            "mean_batch_correlation": float(torch.nanmean(correlations))
            if correlations.numel()
            else float("nan"),
        }
        return EvalResult(
            per_example_loss=all_losses,
            per_example_grad_norm=all_gnorms,
            batch_mean_loss=torch.tensor(batch_mean_loss),
            batch_mean_grad_norm=torch.tensor(batch_mean_gnorm),
            batch_correlation=correlations,
            summary=summary,
        )


def plot_eval_result(result: EvalResult, out_dir: Path | str) -> list[Path]:
    """Histograms of the metrics over batches + per-example scatter."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []

    # Histograms over batches: mean loss, mean grad norm, and the per-batch
    # loss-vs-grad-norm correlation (NaN batches dropped).
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    axes[0].hist(result.batch_mean_loss.tolist(), bins=20, edgecolor="black")
    axes[0].set_xlabel("batch mean per-example loss")
    axes[0].set_ylabel("batches")
    axes[0].set_title("per-batch mean loss")
    axes[1].hist(result.batch_mean_grad_norm.tolist(), bins=20, edgecolor="black")
    axes[1].set_xlabel("batch mean per-example grad norm")
    axes[1].set_ylabel("batches")
    axes[1].set_title("per-batch mean grad norm")
    valid_corr = [c for c in result.batch_correlation.tolist() if c == c]
    axes[2].hist(valid_corr, bins=20, edgecolor="black")
    axes[2].set_xlabel("per-batch corr(loss, grad norm)")
    axes[2].set_ylabel("batches")
    overall = result.summary.get("overall_correlation_loss_vs_grad_norm", float("nan"))
    axes[2].set_title(f"per-batch correlation\n(overall corr = {overall:.4f})")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    figures.append(savefig(fig, out_dir / "eval_batch_metric_histograms.png"))

    # Bonus diagnostic: per-example loss vs per-example gradient norm.
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(
        result.per_example_loss, result.per_example_grad_norm, s=5, alpha=0.25, edgecolors="none"
    )
    ax.set_xlabel("per-example loss")
    ax.set_ylabel("per-example gradient norm")
    ax.set_title(f"loss vs grad norm (corr = {overall:.4f})")
    ax.grid(True, alpha=0.3)
    figures.append(savefig(fig, out_dir / "eval_loss_vs_grad_norm.png"))
    return figures
