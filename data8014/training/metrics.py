"""Step-level training metrics.

* **gradient norm**: L2 norm over all parameter gradients (measured after
  ``loss.backward()``, before ``optimizer.step()``).
* **memory usage**: CUDA peak tensor memory (MB); on CPU the process RSS (MB)
  via :mod:`psutil`, or NaN if psutil is unavailable.
* **FLOPs / MFU**: forward+backward FLOPs of a stack of ``nn.Linear`` layers
  under the standard "backward = 2x forward" convention (so 3x forward),
  divided by measured peak GEMM throughput to obtain model-flop utilization.
"""
from __future__ import annotations

import time
from functools import lru_cache

import torch
from torch import nn


def gradient_norm(model: nn.Module) -> float:
    """Total L2 norm of all parameter gradients."""
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum())
    return total ** 0.5


def memory_usage_mb(device: torch.device) -> float:
    """Peak memory (MB): CUDA allocated tensors, or process RSS on CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2 ** 20
    try:
        import psutil
    except ImportError:
        return float("nan")
    return psutil.Process().memory_info().rss / 2 ** 20


def reset_memory_stats(device: torch.device) -> None:
    """Reset the CUDA peak-memory counter (no-op elsewhere)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def estimate_flops_per_step(model: nn.Module, batch_size: int) -> int:
    """Forward+backward FLOPs of the Linear layers in ``model``.

    A ``b x in @ in x out`` matmul costs ``2 * b * in * out`` FLOPs
    (multiply-adds counted as two ops); the backward pass is conventionally
    counted as twice the forward, giving the standard 3x-forward rule used in
    MFU accounting.
    """
    forward = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            forward += 2 * batch_size * module.in_features * module.out_features
    return 3 * forward


@lru_cache(maxsize=None)
def estimate_peak_flops(
    device: str = "cpu",
    dtype_str: str = "float32",
    size: int = 2048,
    reps: int = 5,
) -> float:
    """Estimate peak matmul FLOP/s on ``device`` via a GEMM microbenchmark.

    This measures *achievable* GEMM throughput (typically 60-90% of the
    datasheet peak), so MFU values computed against it are conservative.
    Override with an explicit datasheet number via ``TrainerConfig.peak_flops``
    if exact peak accounting matters. Cached per process.
    """
    device_obj = torch.device(device)
    dtype = getattr(torch, dtype_str)
    a = torch.randn(size, size, device=device_obj, dtype=dtype)
    b = torch.randn(size, size, device=device_obj, dtype=dtype)
    for _ in range(2):  # warm up allocator / kernels
        a @ b
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)
    start = time.perf_counter()
    for _ in range(reps):
        a @ b
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)
    seconds = max((time.perf_counter() - start) / reps, 1e-12)
    return 2 * size ** 3 / seconds


def mfu(flops: float, step_time_s: float, peak_flops: float) -> float:
    """Model-flop utilization: FLOPs executed / (step time x peak FLOP/s)."""
    return flops / (max(step_time_s, 1e-12) * peak_flops)
