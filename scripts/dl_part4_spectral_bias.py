"""HW1 Part 4 (bonus): spectral ordering in per-atom dictionary recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from data8014.dictlearn import (
    SpectralDLConfig,
    altmin_block,
    generate_spectral_dataset,
    msp_step,
    random_orthogonal,
    recovery_scores,
)
from data8014.dictlearn.paths import FIGURE_DIR, LOG_DIR, ensure_directories
from data8014.utils.plotting import savefig

ALPHAS = [0.75, 1.0, 1.25]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--c", type=float, default=0.45)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--inner-steps", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--log", type=Path, default=LOG_DIR / "dl_part4_spectral_bias.jsonl")
    return parser.parse_args()


def run_one(method: str, alpha: float, seed: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = SpectralDLConfig(
        D=args.dim, c=args.c, alpha=alpha, sigma=args.sigma, seed=seed
    )
    data = generate_spectral_dataset(cfg, args.n)
    a = random_orthogonal(args.dim, np.random.SeedSequence([2026, seed, 4]))
    recovery_time = np.full(args.dim, -1, dtype=int)
    scores = recovery_scores(data.A, a)

    if method == "altmin":
        z = np.zeros((args.n, args.dim))
        for iteration in range(1, args.iters + 1):
            a, z = altmin_block(data.X, a, z, args.lam, L=args.inner_steps)
            scores = recovery_scores(data.A, a)
            newly = (recovery_time < 0) & (scores > args.threshold)
            recovery_time[newly] = iteration
    elif method == "msp":
        for iteration in range(1, args.iters + 1):
            a = msp_step(data.X, a)
            scores = recovery_scores(data.A, a)
            newly = (recovery_time < 0) & (scores > args.threshold)
            recovery_time[newly] = iteration
    else:
        raise ValueError(method)

    empirical_var = data.Z.var(axis=0)
    return [
        {
            "tag": "atom",
            "method": method,
            "alpha": alpha,
            "seed": seed,
            "atom": k + 1,
            "p": float(data.p[k]),
            "v": float(data.v[k]),
            "tau": float(data.tau[k]),
            "empirical_variance": float(empirical_var[k]),
            "recovery_time": int(recovery_time[k]) if recovery_time[k] >= 0 else None,
            "final_s": float(scores[k]),
        }
        for k in range(args.dim)
    ]


def plot_panel(ax: plt.Axes, records: list[dict[str, Any]], method: str, alpha: float, budget: int) -> None:
    subset = [r for r in records if r["method"] == method and np.isclose(r["alpha"], alpha)]
    seeds = sorted({r["seed"] for r in subset})
    for seed in seeds:
        rows = sorted([r for r in subset if r["seed"] == seed], key=lambda r: r["tau"], reverse=True)
        tau = np.asarray([r["tau"] for r in rows])
        recovered = np.asarray([r["recovery_time"] is not None for r in rows])
        times = np.asarray([r["recovery_time"] if r["recovery_time"] is not None else budget * 1.15 for r in rows])
        ax.scatter(tau[recovered], times[recovered], s=22, alpha=0.55, color="tab:blue")
        ax.scatter(tau[~recovered], times[~recovered], s=34, alpha=0.75, color="tab:red", marker="^")
    ax.axhline(budget, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"per-atom energy $\tau_k$")
    ax.set_ylabel(r"recovery time $t_k$")
    ax.set_title(f"{method}, alpha={alpha:g}")
    ax.grid(True, which="both", alpha=0.3)


def make_plots(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=True)
    for ax, alpha in zip(axes, ALPHAS):
        plot_panel(ax, records, "altmin", alpha, args.iters)
    fig.suptitle("HW1 Part 4.2: recovery time vs per-atom energy (alternating minimization)")
    savefig(fig, FIGURE_DIR / "dl_part4_ordering.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    plot_panel(axes[0], records, "altmin", 1.0, args.iters)
    plot_panel(axes[1], records, "msp", 1.0, args.iters)
    fig.suptitle("HW1 Part 4.3: does the algorithm change spectral ordering?")
    savefig(fig, FIGURE_DIR / "dl_part4_algorithm.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    subset = [r for r in records if r["method"] == "altmin" and np.isclose(r["alpha"], 1.0)]
    for seed in sorted({r["seed"] for r in subset}):
        rows = sorted([r for r in subset if r["seed"] == seed], key=lambda r: r["atom"])
        ax.plot([r["atom"] for r in rows], [r["empirical_variance"] for r in rows], "o-", alpha=0.45)
    reference = sorted([r for r in subset if r["seed"] == min(args.seeds)], key=lambda r: r["atom"])
    ax.plot([r["atom"] for r in reference], [r["tau"] for r in reference], "k--", linewidth=1.5, label=r"theory $\tau_k$")
    ax.set_yscale("log")
    ax.set_xlabel("atom k")
    ax.set_ylabel("code-coordinate variance")
    ax.set_title("Part 4 generator check: empirical variance varies across atoms")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    savefig(fig, FIGURE_DIR / "dl_part4_variances.png")


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method, alphas in (("altmin", ALPHAS), ("msp", [1.0])):
        summary[method] = {}
        for alpha in alphas:
            subset = [r for r in records if r["method"] == method and np.isclose(r["alpha"], alpha)]
            times = [r["recovery_time"] for r in subset if r["recovery_time"] is not None]
            per_seed_ranges = {}
            for seed in args.seeds:
                seed_times = [r["recovery_time"] for r in subset if r["seed"] == seed and r["recovery_time"] is not None]
                per_seed_ranges[str(seed)] = [min(seed_times), max(seed_times)] if seed_times else None
            summary[method][str(alpha)] = {
                "expected_sparsity": float(sum(r["p"] for r in subset if r["seed"] == args.seeds[0])),
                "recovered": len(times),
                "total": len(subset),
                "time_min": min(times) if times else None,
                "time_max": max(times) if times else None,
                "per_seed_ranges": per_seed_ranges,
            }
    return summary


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 3:
        print("WARNING: final reported quantities require at least three seeds")
    ensure_directories()
    records: list[dict[str, Any]] = []
    with args.log.open("w", encoding="utf-8") as log:
        for alpha in ALPHAS:
            for seed in args.seeds:
                result = run_one("altmin", alpha, seed, args)
                records.extend(result)
                for row in result:
                    log.write(json.dumps(row) + "\n")
                recovered = sum(r["recovery_time"] is not None for r in result)
                print(f"altmin alpha={alpha} seed={seed}: recovered {recovered}/{args.dim}")
        for seed in args.seeds:
            result = run_one("msp", 1.0, seed, args)
            records.extend(result)
            for row in result:
                log.write(json.dumps(row) + "\n")
            recovered = sum(r["recovery_time"] is not None for r in result)
            print(f"MSP alpha=1 seed={seed}: recovered {recovered}/{args.dim}")
        summary = summarize(records, args)
        log.write(json.dumps({"tag": "summary", "results": summary}) + "\n")
    make_plots(records, args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
