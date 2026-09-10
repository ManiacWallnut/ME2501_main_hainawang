"""A minimal but realistic ``Trainer`` abstraction.

A trainer takes a dataloader (``torch.utils.data.DataLoader``) and a model
(``torch.nn.Module``) and outputs a trained model. The salient abstraction is
:meth:`Trainer.train_step`: a *single* optimization step given one batch and
the current model state, returning relevant outputs (predictions, loss,
gradient norm, step time). The full :meth:`Trainer.train` loop additionally

* logs step-window metrics every ``log_every_steps`` steps to a JSON-lines
  file: train loss, gradient norm, memory usage, time per step, dataloader
  throughput, sample throughput, and model-flop utilization (MFU),
* computes the validation loss every ``val_every_steps`` steps and at every
  epoch end,
* checkpoints model + optimizer + metadata (``.pt``) every epoch.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models.linear import model_config
from ..utils.io import save_checkpoint
from .logging import MetricsLogger
from .metrics import (
    estimate_flops_per_step,
    estimate_peak_flops,
    gradient_norm,
    memory_usage_mb,
    mfu,
    reset_memory_stats,
)

#: Sentinel stored in checkpoints instead of ``None`` so the file stays
#: loadable with ``torch.load(..., weights_only=True)``.
AUTO_PEAK_FLOPS = -1.0


@dataclass
class TrainerConfig:
    """Hyperparameters of the training loop (not of the model)."""

    num_epochs: int = 20
    log_every_steps: int = 20
    val_every_steps: int = 200
    checkpoint_every_epochs: int = 1
    #: Peak FLOP/s used for MFU; ``None`` -> estimate via GEMM microbenchmark.
    peak_flops: float | None = None


class Trainer:
    """Trains a model with SGD-style updates, metrics logging and checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        config: TrainerConfig,
        logger: MetricsLogger,
        checkpoint_path: Path | str | None = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.config = config
        self.logger = logger
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.global_step = 0

    # ------------------------------------------------------------------ steps

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
        """One optimization step on a single batch.

        Returns the batch predictions plus step metrics (loss, gradient norm,
        wall-clock step time, batch size).
        """
        x, y = (t.to(self.device, non_blocking=True) for t in batch)
        start = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.model(x)
        loss = self.loss_fn(predictions, y)
        loss.backward()
        grad_norm = gradient_norm(self.model)
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        step_time = time.perf_counter() - start
        return {
            "predictions": predictions.detach(),
            "batch_size": int(x.shape[0]),
            "loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "step_time": step_time,
        }

    @torch.no_grad()
    def validation_loss(self, loader: DataLoader) -> float:
        """Mean loss of the current model over an entire dataloader."""
        was_training = self.model.training
        self.model.eval()
        total, count = 0.0, 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            loss = self.loss_fn(self.model(x), y)
            total += float(loss) * x.shape[0]
            count += x.shape[0]
        if was_training:
            self.model.train()
        return total / max(count, 1)

    # ------------------------------------------------------------------- loop

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        """Run the full training loop; returns a summary dict."""
        peak_flops = (
            self.config.peak_flops
            if self.config.peak_flops is not None
            else estimate_peak_flops(str(self.device))
        )
        self.logger.log(
            "meta",
            self.global_step,
            device=str(self.device),
            peak_flops=peak_flops,
            num_batches_per_epoch=len(train_loader),
        )
        run_start = time.perf_counter()

        for epoch in range(self.config.num_epochs):
            self.model.train()
            window = self._new_window()
            iterator = iter(train_loader)
            while True:
                fetch_start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                window["fetch_time_s"] += time.perf_counter() - fetch_start
                window["fetch_samples"] += int(batch[0].shape[0])

                outputs = self.train_step(batch)
                self.global_step += 1
                step = self.global_step
                batch_flops = estimate_flops_per_step(self.model, outputs["batch_size"])
                window["loss"].append(outputs["loss"])
                window["grad_norm"].append(outputs["grad_norm"])
                window["step_time"].append(outputs["step_time"])
                window["sample_throughput"].append(
                    outputs["batch_size"] / max(outputs["step_time"], 1e-12)
                )
                window["mfu"].append(
                    mfu(batch_flops, outputs["step_time"], peak_flops)
                )

                if step % self.config.log_every_steps == 0:
                    self._log_train_window(epoch, step, window)
                    window = self._new_window()
                if val_loader is not None and step % self.config.val_every_steps == 0:
                    self._log_validation(epoch, step, val_loader)

            if window["loss"]:  # flush the partial window at epoch end
                self._log_train_window(epoch, self.global_step, window)
            if val_loader is not None:
                self._log_validation(epoch, self.global_step, val_loader)
            if self.checkpoint_path is not None and (
                (epoch + 1) % max(self.config.checkpoint_every_epochs, 1) == 0
                or epoch == self.config.num_epochs - 1
            ):
                self.save_checkpoint()

        summary: dict[str, Any] = {
            "steps": self.global_step,
            "epochs": self.config.num_epochs,
            "peak_flops": peak_flops,
            "wall_time_s": time.perf_counter() - run_start,
            "final_train_loss": self.validation_loss(train_loader),
        }
        if val_loader is not None:
            summary["final_val_loss"] = self.validation_loss(val_loader)
        self.logger.log("summary", self.global_step, **summary)
        if self.checkpoint_path is not None:
            self.save_checkpoint(extra={"summary": summary})
        return summary

    # --------------------------------------------------------------- internals

    @staticmethod
    def _new_window() -> dict[str, Any]:
        return defaultdict(list, fetch_time_s=0.0, fetch_samples=0)

    def _log_train_window(self, epoch: int, step: int, window: dict[str, Any]) -> None:
        fetch_time = window["fetch_time_s"]
        record = {
            "epoch": epoch,
            "train_loss": _mean(window["loss"]),
            "grad_norm": _mean(window["grad_norm"]),
            "step_time_s": _mean(window["step_time"]),
            "memory_mb": memory_usage_mb(self.device),
            "dataloader_throughput_sps": (
                window["fetch_samples"] / fetch_time if fetch_time > 0 else None
            ),
            "sample_throughput_sps": _mean(window["sample_throughput"]),
            "mfu": _mean(window["mfu"]),
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        reset_memory_stats(self.device)
        self.logger.log("train", step, **record)

    def _log_validation(self, epoch: int, step: int, val_loader: DataLoader) -> None:
        val_loss = self.validation_loss(val_loader)
        self.logger.log("val", step, epoch=epoch, val_loss=val_loss)

    def save_checkpoint(self, extra: dict[str, Any] | None = None) -> Path:
        """Write the ``.pt`` checkpoint to ``self.checkpoint_path``."""
        if self.checkpoint_path is None:
            raise ValueError("Trainer was constructed without a checkpoint_path")
        config_dict = asdict(self.config)
        if config_dict.get("peak_flops") is None:
            config_dict["peak_flops"] = AUTO_PEAK_FLOPS
        return save_checkpoint(
            self.checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            extra={
                "step": self.global_step,
                "model_config": model_config(self.model),
                "trainer_config": config_dict,
                **(extra or {}),
            },
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
