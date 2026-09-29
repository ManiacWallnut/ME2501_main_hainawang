"""HW1 Part 1(b): converged MSP error versus sample count N.

For each seed, generate one N_max dataset and use prefixes so A*, validation
data and initialization are fixed across N. MSP stops by signed/permutation-
invariant iterate stationarity (not by ground-truth recovery). Fit
E(N)-E_inf = c N^{-beta} on the smallest two-thirds of N and predict the
largest N, for both recovery error (E_inf=0) and eval loss (E_inf measured with
A* on the same validation set).
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
    fit_power_law_fixed_floor,
    generate_dataset,
    msp_until_stationary,
    power_law_prediction,
    random_orthogonal,
    recovery_summary,
)
from data8014.dictlearn.paths import FIGURE_DIR, LOG_DIR, ensure_directories
from data8014.utils.plotting import savefig

DEFAULT_SAMPLE_COUNTS = [100, 178, 316, 562, 1000, 1780, 3160, 5620, 10000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sample-counts", type=int, nargs="+", default=DEFAULT_SAMPLE_COUNTS)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--min-iters", type=int, default=20)
    parser.add_argument("--stationarity-tol", type=float, default=1e-12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log", type=Path, default=LOG_DIR / "dl_part1b_msp_sample_size.jsonl")
    parser.add_argument("--figure", type=Path, default=FIGURE_DIR / "dl_part1b_msp_sample_size.png")
    return parser.parse_args()


def aggregate(
    records: list[dict[str, Any]], seeds: list[int], sample_counts: list[int]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"sample_counts": np.asarray(sample_counts)}
    for key in ("recovery_error", "eval_loss", "eval_excess"):
        values = np.empty((len(seeds), len(sample_counts)))
        for i, seed in enumerate(seeds):
            by_n = {r["N"]: r[key] for r in records if r["seed"] == seed}
            values[i] = [by_n[n] for n in sample_counts]
        out[f"{key}_by_seed"] = values
        out[f"{key}_mean"] = values.mean(axis=0)
        out[f"{key}_std"] = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(sample_counts))
    return out


def extrapolation_report(observed: float, predicted: float) -> dict[str, float]:
    absolute = abs(predicted - observed)
    return {
        "observed": float(observed),
        "predicted": float(predicted),
        "absolute_error": float(absolute),
        "relative_error": float(absolute / max(abs(observed), np.finfo(float).tiny)),
    }


def plot_results(
    agg: dict[str, np.ndarray], fits: dict[str, dict[str, Any]], path: Path, args: argparse.Namespace
) -> None:
    n = agg["sample_counts"].astype(float)
    n_dense = np.geomspace(n.min(), n.max(), 300)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    specs = [
        ("recovery_error", "recovery error", "recovery"),
        ("eval_excess", r"eval loss $-\ \mathcal{E}_\infty$", "eval"),
    ]
    for ax, (key, ylabel, fit_key) in zip(axes, specs):
        mean, std = agg[f"{key}_mean"], agg[f"{key}_std"]
        valid = mean > 0.0
        ax.errorbar(
            n[valid], mean[valid], yerr=np.minimum(std[valid], mean[valid] * 0.95),
            fmt="o", capsize=3, label="mean ± std", color="tab:blue",
        )
        fit = fits[fit_key]
        if fit["beta"] is not None:
            excess_fit = fit["c"] * n_dense ** (-fit["beta"])
            ax.plot(n_dense, excess_fit, "--", color="tab:orange", label=fr"fit $\beta={fit['beta']:.3f}$")
        ax.axvline(n[-1], color="gray", linestyle=":", linewidth=0.9, label="prediction target")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training samples N")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"HW1 Part 1(b): converged MSP sample-size scaling "
        rf"(D={args.dim}, $\theta$={args.theta}; {len(args.seeds)} seeds)"
    )
    savefig(fig, path)


def main() -> None:
    args = parse_args()
    sample_counts = sorted(set(args.sample_counts))
    if len(sample_counts) < 6:
        raise ValueError("Part 1(b) requires at least six N values")
    if sample_counts[-1] / sample_counts[0] < 100:
        raise ValueError("Part 1(b) requires N to span at least two orders of magnitude")
    if len(args.seeds) < 3:
        print("WARNING: the assignment requires at least three seeds for reported results")

    ensure_directories()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    floors: dict[int, float] = {}

    for seed in args.seeds:
        cfg = DLConfig(
            D=args.dim,
            m=args.dim,
            theta=args.theta,
            sigma=args.sigma,
            regime="complete",
            seed=seed,
        )
        # One draw at N_max; all smaller datasets are strict prefixes.
        data = generate_dataset(cfg, n=sample_counts[-1], n_val=args.n_val)
        a0 = random_orthogonal(args.dim, np.random.SeedSequence([2026, seed, 2]))
        floor = eval_loss(data.A, data.X_val, lam=args.lam, ista_steps=1)
        floors[seed] = floor

        for n in sample_counts:
            a_hat, convergence = msp_until_stationary(
                data.X[:n],
                a0,
                max_iters=args.max_iters,
                min_iters=args.min_iters,
                tol=args.stationarity_tol,
                patience=args.patience,
            )
            recovery = recovery_summary(data.A, a_hat)
            held_out = eval_loss(a_hat, data.X_val, lam=args.lam, ista_steps=1)
            record = {
                "tag": "result",
                "seed": seed,
                "N": n,
                "eval_floor": floor,
                "eval_loss": held_out,
                "eval_excess": held_out - floor,
                "recovery_error": 1.0 - recovery["mean_s"],
                **recovery,
                **convergence,
            }
            records.append(record)
            print(
                f"seed={seed} N={n:5d}: T={convergence['iterations']:4d} "
                f"stationary={convergence['converged']} "
                f"recovery_error={record['recovery_error']:.4g} "
                f"eval_excess={record['eval_excess']:.4g}"
            )

    agg = aggregate(records, args.seeds, sample_counts)
    n = agg["sample_counts"].astype(float)
    n_fit = int(np.floor(2 * len(n) / 3))
    recovery_fit = fit_power_law_fixed_floor(n, agg["recovery_error_mean"], floor=0.0, n_fit=n_fit)
    eval_floor_mean = float(np.mean(list(floors.values())))
    eval_fit = fit_power_law_fixed_floor(n, agg["eval_loss_mean"], floor=eval_floor_mean, n_fit=n_fit)

    per_seed_fits: dict[str, list[dict[str, Any]]] = {"recovery": [], "eval": []}
    for i, seed in enumerate(args.seeds):
        per_seed_fits["recovery"].append(
            {"seed": seed, **fit_power_law_fixed_floor(n, agg["recovery_error_by_seed"][i], floor=0.0, n_fit=n_fit)}
        )
        per_seed_fits["eval"].append(
            {"seed": seed, **fit_power_law_fixed_floor(n, agg["eval_loss_by_seed"][i], floor=floors[seed], n_fit=n_fit)}
        )

    target = float(n[-1])
    recovery_prediction = power_law_prediction(target, recovery_fit)
    eval_prediction = power_law_prediction(target, eval_fit)
    extrapolation = {
        "recovery": extrapolation_report(float(agg["recovery_error_mean"][-1]), recovery_prediction),
        "eval_loss_raw": extrapolation_report(float(agg["eval_loss_mean"][-1]), eval_prediction),
        "eval_excess": extrapolation_report(
            float(agg["eval_excess_mean"][-1]), eval_prediction - eval_floor_mean
        ),
    }
    fits = {"recovery": recovery_fit, "eval": eval_fit}
    beta_spread: dict[str, dict[str, Any]] = {}
    for key in ("recovery", "eval"):
        betas = [fit["beta"] for fit in per_seed_fits[key] if fit["beta"] is not None]
        beta_spread[key] = {
            "mean": float(np.mean(betas)) if betas else None,
            "std": float(np.std(betas, ddof=1)) if len(betas) > 1 else 0.0,
            "n": len(betas),
        }
    summary = {
        "tag": "summary",
        "n_fit": n_fit,
        "fit_sample_counts": sample_counts[:n_fit],
        "prediction_target_N": sample_counts[-1],
        "fits": fits,
        "per_seed_fits": per_seed_fits,
        "beta_spread": beta_spread,
        "extrapolation": extrapolation,
        "capped_runs": int(sum(not r["converged"] for r in records)),
    }

    with args.log.open("w", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "tag": "meta",
                    "D": args.dim,
                    "theta": args.theta,
                    "sigma": args.sigma,
                    "lambda": args.lam,
                    "N_val": args.n_val,
                    "sample_counts": sample_counts,
                    "seeds": args.seeds,
                    "validation_ista_steps": 1,
                    "stationarity_tol": args.stationarity_tol,
                    "stationarity_patience": args.patience,
                    "max_iters": args.max_iters,
                }
            )
            + "\n"
        )
        for record in records:
            file.write(json.dumps(record) + "\n")
        file.write(json.dumps(summary) + "\n")

    plot_results(agg, fits, args.figure, args)
    print("--- fits / extrapolation ---")
    print(json.dumps(summary, indent=2))
    print(f"figure -> {args.figure}")
    print(f"log -> {args.log}")


if __name__ == "__main__":
    main()
