"""HW1 Part 2: density cost, identifiability transition and held-out loss.

Complete regime, D=16, N=8000, sigma=0, at least three seeds. For the 2.1
sweep, record T*(theta), the first MSP iteration at which at least 95% of atoms
have s_k>0.9 (for D=16 this means all atoms). For the 2.2 sweep, optimize to
signed/permutation-invariant stationarity (or a common cap), then report the
fraction recovered. Part 2.3 reports held-out loss for the union of both theta
grids. Records are flushed incrementally and --resume skips completed runs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data8014.dictlearn import (
    DLConfig,
    dictionary_alignment_error,
    eval_loss,
    generate_dataset,
    msp_step,
    random_orthogonal,
    recovery_scores,
    recovery_summary,
)
from data8014.dictlearn.paths import FIGURE_DIR, LOG_DIR, ensure_directories
from data8014.utils.plotting import savefig

COST_THETAS = [0.2, 0.4, 0.6, 0.7, 0.8, 0.85]
TRANSITION_THETAS = [0.88, 0.90, 0.91, 0.92, 0.93, 0.95, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--max-iters", type=int, default=20000)
    parser.add_argument("--stationarity-tol", type=float, default=1e-12)
    parser.add_argument("--stationarity-patience", type=int, default=5)
    parser.add_argument("--min-iters", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log", type=Path, default=LOG_DIR / "dl_part2_phase_transition.jsonl")
    parser.add_argument("--figure-cost", type=Path, default=FIGURE_DIR / "dl_part2_Tstar_vs_theta.png")
    parser.add_argument("--figure-transition", type=Path, default=FIGURE_DIR / "dl_part2_transition.png")
    return parser.parse_args()


def load_completed(path: Path) -> dict[tuple[int, float, str], dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[tuple[int, float, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record.get("tag") == "result":
                completed[(record["seed"], float(record["theta"]), record["sweep"])] = record
    return completed


def run_one(theta: float, seed: int, sweep: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = DLConfig(
        D=args.dim,
        m=args.dim,
        theta=theta,
        sigma=0.0,
        regime="complete",
        seed=seed,
    )
    data = generate_dataset(cfg, n=args.n, n_val=args.n_val)
    a = random_orthogonal(args.dim, np.random.SeedSequence([2026, seed, 3]))
    target_fraction = 0.95
    t_star: int | None = None
    stationary_count = 0
    last_alignment_error = float("inf")
    start = time.perf_counter()

    for iteration in range(1, args.max_iters + 1):
        a_next = msp_step(data.X, a)
        last_alignment_error = max(0.0, dictionary_alignment_error(a, a_next))
        a = a_next
        scores = recovery_scores(data.A, a)
        fraction = float((scores > 0.9).mean())
        if t_star is None and fraction >= target_fraction:
            t_star = iteration

        # T* is a first-passage statistic; continue to stationarity so Part 2.3
        # evaluates a converged learned dictionary rather than the threshold-crossing iterate.
        if iteration >= args.min_iters and last_alignment_error <= args.stationarity_tol:
            stationary_count += 1
        else:
            stationary_count = 0
        if stationary_count >= args.stationarity_patience:
            break

    recovery = recovery_summary(data.A, a)
    held_out = eval_loss(a, data.X_val, lam=args.lam, ista_steps=1)
    return {
        "tag": "result",
        "sweep": sweep,
        "seed": seed,
        "theta": theta,
        "iterations": iteration,
        "stationary": stationary_count >= args.stationarity_patience,
        "alignment_error": last_alignment_error,
        "t_star": t_star,
        "t_star_censored": t_star is None,
        "eval_loss": held_out,
        "kurtosis": 3.0 * (1.0 / theta - 1.0),
        "wall_time_s": time.perf_counter() - start,
        **recovery,
    }


def mean_std(records: list[dict[str, Any]], key: str, thetas: list[float]) -> tuple[np.ndarray, np.ndarray]:
    mean, std = [], []
    for theta in thetas:
        values = np.asarray([r[key] for r in records if np.isclose(r["theta"], theta)], dtype=float)
        mean.append(values.mean())
        std.append(values.std(ddof=1) if len(values) > 1 else 0.0)
    return np.asarray(mean), np.asarray(std)


def plot_cost(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for i, theta in enumerate(COST_THETAS):
        subset = [r for r in records if r["sweep"] == "cost" and np.isclose(r["theta"], theta)]
        observed = [r["t_star"] for r in subset if r["t_star"] is not None]
        censored = [r for r in subset if r["t_star"] is None]
        rng = np.random.default_rng(100 + i)
        if observed:
            jitter = rng.uniform(-0.006, 0.006, len(observed))
            ax.scatter(theta + jitter, observed, color="tab:blue", alpha=0.75, s=28)
            ax.errorbar(
                theta,
                np.mean(observed),
                yerr=np.std(observed, ddof=1) if len(observed) > 1 else 0.0,
                fmt="o",
                color="black",
                capsize=3,
                zorder=4,
            )
        if censored:
            ax.scatter([theta] * len(censored), [args.max_iters] * len(censored), marker="^", color="tab:red", label="censored" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel(r"code density $\theta$")
    ax.set_ylabel(r"$T_\star(\theta)$ (MSP iterations)")
    ax.set_title(r"HW1 Part 2.1: cost of density ($s_k>0.9$ for at least 95% atoms)")
    ax.grid(True, which="both", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    savefig(fig, args.figure_cost)


def plot_transition(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    all_thetas = sorted(set(COST_THETAS + TRANSITION_THETAS))
    # If a theta appeared in both sweeps, prefer transition records; current grids are disjoint.
    selected: list[dict[str, Any]] = []
    for theta in all_thetas:
        candidates = [r for r in records if np.isclose(r["theta"], theta)]
        transition = [r for r in candidates if r["sweep"] == "transition"]
        selected.extend(transition or candidates)

    frac_mean, frac_std = mean_std(selected, "frac_gt_0.9", all_thetas)
    loss_mean, loss_std = mean_std(selected, "eval_loss", all_thetas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax1.errorbar(all_thetas, frac_mean, yerr=frac_std, fmt="o-", capsize=3)
    ax1.set_xlabel(r"code density $\theta$")
    ax1.set_ylabel(r"fraction of atoms with $s_k>0.9$")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Recovery / identifiability")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(all_thetas, loss_mean, yerr=loss_std, fmt="o-", capsize=3, color="tab:orange")
    ax2.set_xlabel(r"code density $\theta$")
    ax2.set_ylabel("held-out eval loss")
    ax2.set_title("Can reconstruction loss see the transition?")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"HW1 Parts 2.2–2.3 (D={args.dim}, N={args.n}; {len(args.seeds)} seeds)")
    savefig(fig, args.figure_transition)


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 3:
        print("WARNING: the assignment requires at least three seeds")
    ensure_directories()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    existing = load_completed(args.log) if args.resume else {}
    if not args.resume:
        args.log.write_text("", encoding="utf-8")

    jobs = [
        *( (theta, seed, "cost") for theta in COST_THETAS for seed in args.seeds ),
        *( (theta, seed, "transition") for theta in TRANSITION_THETAS for seed in args.seeds ),
    ]
    with args.log.open("a", encoding="utf-8") as file:
        if not existing:
            file.write(json.dumps({"tag": "meta", "D": args.dim, "N": args.n, "N_val": args.n_val, "lambda": args.lam, "seeds": args.seeds, "max_iters": args.max_iters}) + "\n")
            file.flush()
        for theta, seed, sweep in jobs:
            key = (seed, float(theta), sweep)
            if key in existing:
                print(f"skip completed: {sweep} theta={theta} seed={seed}")
                continue
            record = run_one(theta, seed, sweep, args)
            file.write(json.dumps(record) + "\n")
            file.flush()
            print(
                f"{sweep:10s} theta={theta:.2f} seed={seed}: T={record['iterations']} "
                f"T*={record['t_star']} frac={record['frac_gt_0.9']:.3f} "
                f"loss={record['eval_loss']:.4g} time={record['wall_time_s']:.2f}s"
            )

    completed = load_completed(args.log)
    records = list(completed.values())
    expected = len(COST_THETAS + TRANSITION_THETAS) * len(args.seeds)
    if len(records) != expected:
        print(f"partial run: {len(records)}/{expected} results; rerun with --resume")
        return

    plot_cost(records, args)
    plot_transition(records, args)
    transition_records = [r for r in records if r["sweep"] == "transition"]
    frac_mean, frac_std = mean_std(transition_records, "frac_gt_0.9", TRANSITION_THETAS)
    first_below = next((theta for theta, value in zip(TRANSITION_THETAS, frac_mean) if value < 0.95), None)
    summary = {
        "tag": "summary",
        "transition_theta_first_mean_below_0.95": first_below,
        "transition_fraction_mean": dict(zip(map(str, TRANSITION_THETAS), frac_mean.tolist())),
        "transition_fraction_std": dict(zip(map(str, TRANSITION_THETAS), frac_std.tolist())),
        "censored_cost_runs": sum(r["t_star_censored"] for r in records if r["sweep"] == "cost"),
        "nonstationary_transition_runs": sum(not r["stationary"] for r in transition_records),
    }
    with args.log.open("a", encoding="utf-8") as file:
        file.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"figures -> {args.figure_cost}, {args.figure_transition}")


if __name__ == "__main__":
    main()
