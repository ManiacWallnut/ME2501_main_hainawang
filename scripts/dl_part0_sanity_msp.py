"""HW1 Part 0 sanity check: MSP on the complete regime.

Settings from the assignment: sigma = 0, theta = 0.2, D = 16, m = D = 16,
N = 8000, complete regime. MSP should reach s_k > 0.99 for essentially every
atom within about 30 iterations.

For each seed the script generates the dataset, runs MSP for ``--iters``
iterations from a fresh random orthogonal initialization, records the recovery
metrics after every iteration, and writes:

* a JSONL log (one record per seed x iteration) under ``outputs/logs/``;
* a figure of mean / min recovery vs iteration under ``outputs/figures/``;
* a pass/fail verdict per seed (min_k s_k > 0.99 at the final iteration).

Run:  python scripts/dl_part0_sanity_msp.py            (5 seeds by default)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data8014.dictlearn.paths import FIGURE_DIR, LOG_DIR, ensure_directories
from data8014.dictlearn import DLConfig, generate_dataset, msp, recovery_summary
from data8014.utils.plotting import savefig

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--atoms", type=int, default=16)
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--log", type=Path, default=LOG_DIR / "dl_part0_sanity_msp.jsonl")
    parser.add_argument("--figure", type=Path, default=FIGURE_DIR / "dl_part0_sanity_msp.png")
    return parser.parse_args()


def orthogonal_init(dim: int, seed: int) -> np.ndarray:
    """Random orthogonal initialization for MSP (fresh stream per seed)."""
    rng = np.random.default_rng(np.random.SeedSequence([2026, seed]))
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q


def main() -> None:
    args = parse_args()
    ensure_directories()
    args.log.parent.mkdir(parents=True, exist_ok=True)

    curves_mean: list[tuple[list[float], list[float]]] = []
    curves_min: list[tuple[list[float], list[float]]] = []
    verdicts: list[dict] = []

    with args.log.open("w", encoding="utf-8") as log_file:
        for seed in args.seeds:
            cfg = DLConfig(
                D=args.dim, m=args.atoms, theta=args.theta,
                sigma=args.sigma, regime="complete", seed=seed,
            )
            data = generate_dataset(cfg, n=args.n, with_val=False)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 777]))

            def track(t: int, a_hat: np.ndarray, seed: int = seed) -> dict:
                summary = recovery_summary(data.A, a_hat)
                summary.update({"tag": "msp_iter", "seed": seed, "iter": t})
                log_file.write(json.dumps(summary) + "\n")
                return summary

            _, trace = msp(data.X, args.iters, orthogonal_init(args.dim, seed), callback=track)

            iters = [rec["iter"] for rec in trace]
            means = [rec["mean_s"] for rec in trace]
            mins = [rec["min_s"] for rec in trace]
            curves_mean.append((iters, means))
            curves_min.append((iters, mins))

            passed = mins[-1] > args.threshold
            first_pass = next((t for t, m in zip(iters, mins) if m > args.threshold), None)
            verdict = {
                "tag": "seed_summary",
                "seed": seed,
                "passed": passed,
                "first_iter_min_s_above_threshold": first_pass,
                "final_mean_s": means[-1],
                "final_min_s": mins[-1],
                "final_frac_gt_0.99": trace[-1]["frac_gt_0.99"],
                "final_dup_matches": trace[-1]["dup_matches"],
                "final_mean_s_onetoone": trace[-1]["mean_s_onetoone"],
            }
            verdicts.append(verdict)
            log_file.write(json.dumps(verdict) + "\n")
            print(
                f"seed {seed}: min_k s_k = {mins[-1]:.4f} at iter {args.iters} "
                f"(first iter above {args.threshold}: {first_pass}) -> "
                f"{'PASS' if passed else 'FAIL'}"
            )

    # Figure: per-seed mean and min recovery vs iteration.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for iters, means in curves_mean:
        ax.plot(iters, means, color="tab:blue", alpha=0.6, linewidth=1.2)
    for iters, mins in curves_min:
        ax.plot(iters, mins, color="tab:red", alpha=0.6, linewidth=1.2, linestyle="--")
    ax.axhline(args.threshold, color="k", linewidth=0.8, linestyle=":")
    ax.text(0.2, args.threshold, f"threshold s_k = {args.threshold}", va="bottom", fontsize=8)
    ax.set_xlabel("MSP iteration")
    ax.set_ylabel(r"recovery $s_k$")
    ax.set_title(
        f"MSP sanity check (complete, D={args.dim}, $\\theta$={args.theta}, "
        f"N={args.n}, $\\sigma$={args.sigma})\n"
        f"blue: mean s_k, red dashed: min_k s_k, one line per seed"
    )
    ax.grid(True, alpha=0.3)
    savefig(fig, args.figure)
    print(f"figure -> {args.figure}")

    all_pass = all(v["passed"] for v in verdicts)
    n_atoms = args.atoms * len(args.seeds)
    frac = sum(v["final_frac_gt_0.99"] * args.atoms for v in verdicts) / n_atoms
    print(
        f"\nVERDICT: {'ALL SEEDS PASS' if all_pass else 'SOME SEEDS FAIL'} - "
        f"fraction of atoms with s_k > {args.threshold} across seeds: {frac:.3f}"
    )


if __name__ == "__main__":
    main()
