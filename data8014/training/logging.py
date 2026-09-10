"""Local JSON-lines metrics logging and plot generation.

Every record is one line of JSON with at least ``{"tag": ..., "step": ...}``;
``tag`` distinguishes record kinds (``train`` step window, ``val`` validation,
``meta``/``summary`` run-level info). Plots are rendered from the log file, so
figures can always be regenerated without re-training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.plotting import plot_series

#: Per-metric figure styles (losses get a combined train+val figure instead).
TRAIN_METRIC_PLOTS: dict[str, dict[str, Any]] = {
    "grad_norm": {"ylabel": "gradient L2 norm", "logy": True},
    "memory_mb": {"ylabel": "memory (MB)"},
    "step_time_s": {"ylabel": "time per step (s)"},
    "dataloader_throughput_sps": {"ylabel": "dataloader throughput (samples/s)", "logy": True},
    "sample_throughput_sps": {"ylabel": "sample throughput (samples/s)", "logy": True},
    "mfu": {"ylabel": "model-flop utilization", "logy": True},
}


class MetricsLogger:
    """Append-only JSON-lines logger with an optional console echo."""

    def __init__(self, path: Path | str, echo: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, tag: str, step: int, **fields: Any) -> None:
        record = {"tag": tag, "step": step}
        record.update(fields)
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        if self.echo:
            printable = {k: v for k, v in fields.items() if isinstance(v, (int, float))}
            body = " ".join(f"{k}={v:.4g}" for k, v in printable.items())
            print(f"[{tag}] step {step} {body}")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def load_records(path: Path | str) -> list[dict[str, Any]]:
    """Read back every JSON-lines record from a log file."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_training_log(log_path: Path | str, out_dir: Path | str) -> list[Path]:
    """Render one PNG per metric (train/validation losses share a figure)."""
    records = load_records(log_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = [r for r in records if r.get("tag") == "train"]
    val = [r for r in records if r.get("tag") == "val"]

    figures: list[Path] = []
    series: dict[str, tuple[list[float], list[float]]] = {}
    if train:
        series["train loss"] = ([r["step"] for r in train], [r["train_loss"] for r in train])
    if val:
        series["validation loss"] = ([r["step"] for r in val], [r["val_loss"] for r in val])
    if series:
        figures.append(
            plot_series(
                series,
                out_dir / "train_val_loss.png",
                ylabel="MSE loss",
                title="Training / validation loss",
            )
        )

    for key, style in TRAIN_METRIC_PLOTS.items():
        subset = [r for r in train if isinstance(r.get(key), (int, float))]
        if not subset:
            continue
        figures.append(
            plot_series(
                {key: ([r["step"] for r in subset], [r[key] for r in subset])},
                out_dir / f"train_{key}.png",
                title=key,
                **style,
            )
        )
    return figures
