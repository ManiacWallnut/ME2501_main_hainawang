"""Small matplotlib helpers (Agg backend, no GUI required)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Mapping, Sequence


def ema(values: Sequence[float], alpha: float = 0.9) -> list[float]:
    """Exponential moving average; the first value seeds the average."""
    if not values:
        return []
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * out[-1] + (1.0 - alpha) * float(value))
    return out


def savefig(fig: plt.Figure, path: Path | str, dpi: int = 150) -> Path:
    """Save a figure, creating parent directories, then close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_series(
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    out_path: Path | str,
    *,
    xlabel: str = "step",
    ylabel: str = "",
    title: str | None = None,
    logx: bool = False,
    logy: bool = False,
    smooth: float | None = 0.9,
) -> Path:
    """Plot one or more ``name -> (steps, values)`` series into ``out_path``.

    Unless ``smooth=None``, an EMA curve is drawn on top of the raw series.
    Sparse series (< 2 points, e.g. validation loss logged every few steps)
    are drawn as marker lines instead of raw+EMA.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for label, (steps, values) in series.items():
        if len(values) < 2 or smooth is None:
            ax.plot(steps, values, label=label, marker="o", markersize=3, linewidth=1.0)
        else:
            ax.plot(steps, values, label=f"{label} (raw)", linewidth=0.8, alpha=0.35)
            ax.plot(steps, ema(values, smooth), label=f"{label} (EMA {smooth:g})", linewidth=1.8)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return savefig(fig, out_path)
