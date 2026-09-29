"""HW1 Part 1(a): MSP optimization convergence along the T-axis.

Runs complete-regime MSP at D=16, theta=0.1, sigma=0, N=8000 for at least
three seeds; records held-out eval loss and recovery error after every outer
iteration (including T=0); plots each raw metric on semilogy and log-log axes;
and compares power-law vs exponential fits to the transient above the final
finite-sample / LASSO floor.

Because every MSP iterate is orthogonal, ISTA initialized at zero solves the
validation LASSO exactly in one step: Z = soft(X_val A^T, lambda/2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data8014.dictlearn import (
    DLConfig,
    eval_loss,
    fit_decay_models,
    generate_dataset,
    msp,
    random_orthogonal,
    recovery_summary,
)
from data8014.dictlearn.paths import FIGURE_DIR, LOG_DIR, ensure_directories
from data8014.utils.plotting import savefig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--log", type=Path, default=LOG_DIR / "dl_part1a_msp_convergence.jsonl")
    parser.add_argument("--figure", type=Path, default=FIGURE_DIR / "dl_part1a_msp_convergence.png")
    return parser.parse_args()


def metrics_at(a_true: np.ndarray, a_hat: np.ndarray, x_val: np.ndarray, lam: float) -> dict[str, Any]:
    recovery = recovery_summary(a_true, a_hat)
    return {
        **recovery,
        "recovery_error": 1.0 - recovery["mean_s"],
        # One ISTA step is exact because a_hat is orthogonal.
        "eval_loss": eval_loss(a_hat, x_val, lam=lam, ista_steps=1),
    }


def aggregate(records: list[dict[str, Any]], seeds: list[int], iters: int) -> dict[str, np.ndarray]:
    steps = np.arange(iters + 1)
    out: dict[str, np.ndarray] = {"steps": steps}
    for key in ("recovery_error", "eval_loss"):
        values = np.empty((len(seeds), iters + 1))
        for i, seed in enumerate(seeds):
            by_step = {r["iteration"]: r[key] for r in records if r["seed"] == seed}
            values[i] = [by_step[t] for t in steps]
        out[f"{key}_by_seed"] = values
        out[f"{key}_mean"] = values.mean(axis=0)
        out[f"{key}_std"] = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(steps, dtype=float)
    return out


def plot_results(agg: dict[str, np.ndarray], fits: dict[str, Any], path: Path, args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    specs = [
        ("eval_loss", "held-out eval loss"),
        ("recovery_error", r"recovery error $1-\mathrm{mean}(s_k)$"),
    ]
    for col, (key, label) in enumerate(specs):
        steps = agg["steps"]
        mean, std = agg[f"{key}_mean"], agg[f"{key}_std"]
        positive = mean[mean > 0.0]
        plot_floor = max(float(positive.min()) * 0.5, 1e-16)
        lower = np.maximum(mean - std, plot_floor)
        upper = np.maximum(mean + std, plot_floor)

        ax = axes[0, col]
        ax.semilogy(steps, np.maximum(mean, plot_floor), color="tab:blue", label="mean")
        ax.fill_between(steps, lower, upper, color="tab:blue", alpha=0.2, label="±1 std")
        ax.axhline(fits[key]["floor"], color="tab:orange", linestyle="--", linewidth=1.0, label="tail floor")
        ax.set_xlabel("MSP iterations T")
        ax.set_ylabel(label)
        ax.set_title(f"{label} (semilogy)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1, col]
        valid = steps > 0
        ax.loglog(steps[valid], np.maximum(mean[valid], plot_floor), color="tab:blue", label="mean")
        ax.fill_between(steps[valid], lower[valid], upper[valid], color="tab:blue", alpha=0.2, label="±1 std")
        ax.axhline(fits[key]["floor"], color="tab:orange", linestyle="--", linewidth=1.0)
        ax.set_xlabel("MSP iterations T")
        ax.set_ylabel(label)
        ax.set_title(f"{label} (log-log)")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"HW1 Part 1(a): MSP convergence (D={args.dim}, N={args.n}, "
        rf"$\theta$={args.theta}, $\sigma$={args.sigma}; {len(args.seeds)} seeds)"
    )
    savefig(fig, path)


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 3:
        print("WARNING: the assignment requires at least three seeds for reported results")
    ensure_directories()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for seed in args.seeds:
        cfg = DLConfig(
            D=args.dim,
            m=args.dim,
            theta=args.theta,
            sigma=args.sigma,
            regime="complete",
            seed=seed,
        )
        data = generate_dataset(cfg, n=args.n, n_val=args.n_val)
        a0 = random_orthogonal(args.dim, np.random.SeedSequence([2026, seed, 1]))

        initial = {"tag": "iteration", "seed": seed, "iteration": 0, **metrics_at(data.A, a0, data.X_val, args.lam)}
        records.append(initial)

        def track(iteration: int, a_hat: np.ndarray, seed: int = seed) -> dict[str, Any]:
            record = {
                "tag": "iteration",
                "seed": seed,
                "iteration": iteration,
                **metrics_at(data.A, a_hat, data.X_val, args.lam),
            }
            records.append(record)
            return record

        msp(data.X, args.iters, a0, callback=track)
        final = records[-1]
        print(
            f"seed {seed}: recovery_error={final['recovery_error']:.6g}, "
            f"eval_loss={final['eval_loss']:.6g}, duplicates={final['dup_matches']}"
        )

    agg = aggregate(records, args.seeds, args.iters)
    fits = {
        key: fit_decay_models(agg["steps"], agg[f"{key}_mean"])
        for key in ("recovery_error", "eval_loss")
    }
    per_seed_fits: dict[str, list[dict[str, Any]]] = {}
    parameter_spread: dict[str, dict[str, Any]] = {}
    for key in ("recovery_error", "eval_loss"):
        per_seed_fits[key] = [
            {"seed": seed, **fit_decay_models(agg["steps"], agg[f"{key}_by_seed"][i])}
            for i, seed in enumerate(args.seeds)
        ]
        parameter_spread[key] = {}
        for model, parameter in (("power", "alpha"), ("exponential", "rho")):
            values = [f[model][parameter] for f in per_seed_fits[key] if f[model] is not None]
            parameter_spread[key][parameter] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    summary = {
        "tag": "summary",
        "config": {
            "D": args.dim,
            "N": args.n,
            "N_val": args.n_val,
            "theta": args.theta,
            "sigma": args.sigma,
            "lambda": args.lam,
            "iterations": args.iters,
            "seeds": args.seeds,
            "validation_ista_steps": 1,
        },
        "fits": fits,
        "per_seed_fits": per_seed_fits,
        "fit_parameter_spread": parameter_spread,
    }
    with args.log.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"tag": "meta", **summary["config"]}) + "\n")
        for record in records:
            file.write(json.dumps(record) + "\n")
        file.write(json.dumps(summary) + "\n")

    plot_results(agg, fits, args.figure, args)
    for key, fit in fits.items():
        print(f"{key}: floor={fit['floor']:.6g}, power={fit['power']}, exponential={fit['exponential']}")
        print(f"  across-seed fit spread: {parameter_spread[key]}")
    print(f"figure -> {args.figure}")
    print(f"log -> {args.log}")


if __name__ == "__main__":
    main()
