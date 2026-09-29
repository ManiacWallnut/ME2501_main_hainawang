"""HW1 Part 0(d) warm-up: convergence of ISTA for the inner LASSO.

Overcomplete regime, N = 2000, D = 16, m = 32, theta = 0.1, lambda = 0.6,
sigma = 0; the dictionary in the LASSO (HW1 Eq. 4) is fixed to the ground
truth A, and ISTA starts from Z_0 = 0.

The script tracks the objective gap f(Z_t) - f* together with

* the worst-case proximal-gradient bound of book Theorem 8.6,
  f(Z_k) - f* <= L ||Z_0 - Z*||^2 / (2k),  L = 2||A||_2^2
  (with Z_0 = 0 the bound is fully explicit once Z* is estimated), and
* the *support identification* iteration: the first t from which the
  entrywise nonzero pattern of Z_t equals that of the LASSO solution Z*
  forever. This is the structural property the O(1/k) convex bound cannot
  see: once the support is identified, the remaining dynamics are (piecewise)
  linear and the tail decays geometrically - faster than any 1/k bound.

f* is estimated with a long FISTA run (relative objective change over the
last quarter below --fstar-tol; reached machine precision in practice).

Outputs: JSONL log + two-panel figure (log-log gap vs Theorem-8.6 bound;
semilog-y tail with fitted geometric rate rho).
Run:  python scripts/dl_part0_warmup_ista.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data8014.dictlearn import (
    DLConfig,
    fista,
    generate_dataset,
    ista_step_size,
    soft_threshold,
)
from data8014.dictlearn.paths import FIGURE_DIR, LOG_DIR, ensure_directories
from data8014.utils.plotting import savefig

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--atoms", type=int, default=32)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--ista-iters", type=int, default=6000)
    parser.add_argument("--fstar-iters", type=int, default=40000)
    parser.add_argument("--fstar-tol", type=float, default=1e-12)
    parser.add_argument("--log", type=Path, default=LOG_DIR / "dl_part0_warmup_ista.jsonl")
    parser.add_argument("--figure", type=Path, default=FIGURE_DIR / "dl_part0_warmup_ista.png")
    return parser.parse_args()


def ista_pass(
    x: np.ndarray,
    a: np.ndarray,
    lam: float,
    n_steps: int,
    eta: float,
    supp_star: np.ndarray | None = None,
):
    """ISTA from Z_0 = 0, recording per-iterate diagnostics.

    Returns ``(z, objectives, nnz, matches)`` where entry ``t`` refers to the
    state after ``t`` steps (entry 0 is Z_0 = 0); ``matches[t]`` says whether
    the entrywise support at ``t`` equals ``supp_star`` (if provided).
    """
    z = np.zeros((x.shape[0], a.shape[0]))
    objectives = np.empty(n_steps + 1)
    nnz = np.empty(n_steps + 1, dtype=np.int64)
    matches = np.zeros(n_steps + 1, dtype=bool) if supp_star is not None else None
    for t in range(n_steps + 1):
        if t > 0:
            grad = 2.0 * (z @ a - x) @ a.T
            z = soft_threshold(z - eta * grad, eta * lam)
        objectives[t] = float(np.linalg.norm(x - z @ a, "fro") ** 2 + lam * np.abs(z).sum())
        support = np.abs(z) > 1e-12
        nnz[t] = int(support.sum())
        if matches is not None:
            matches[t] = np.array_equal(support, supp_star)
    return z, objectives, nnz, matches


def first_stable_match(matches: np.ndarray) -> int | None:
    """First index from which ``matches`` stays True forever, else None."""
    stable = np.where(matches)[0]
    for i in stable:
        if matches[i:].all():
            return int(i)
    return None


def fit_linear_rate(t: np.ndarray, gaps: np.ndarray) -> float:
    """Least squares on (t, log gap); returns the per-step factor exp(slope)."""
    keep = gaps > 0
    if keep.sum() < 2:
        return float("nan")
    slope = np.polyfit(t[keep].astype(float), np.log(gaps[keep]), 1)[0]
    return float(np.exp(slope))


def main() -> None:
    args = parse_args()
    ensure_directories()
    args.log.parent.mkdir(parents=True, exist_ok=True)

    cfg = DLConfig(
        D=args.dim, m=args.atoms, theta=args.theta,
        sigma=args.sigma, regime="overcomplete", seed=args.seed,
    )
    data = generate_dataset(cfg, n=args.n, with_val=False)
    a = data.A  # dictionary fixed to the ground truth
    eta = ista_step_size(a)  # 1/(2||A||_2^2)
    print(f"||A||_2 = {np.linalg.norm(a, 2):.4f}, eta = {eta:.4f}, threshold = {eta * args.lam:.4f}")

    # --- f* via long FISTA -----------------------------------------------
    _, info_star = fista(data.X, a, args.lam, args.fstar_iters, track_objective=True)
    obj_traj = np.array(info_star["objective"])
    quarter = obj_traj[-max(args.fstar_iters // 4, 1):]
    rel_change = float((quarter.max() - quarter.min()) / max(quarter.min(), 1e-300))
    f_star = float(obj_traj[-1])
    if rel_change > args.fstar_tol:
        print(f"WARNING: f* estimate not converged to tolerance (rel change {rel_change:.2e})")
    print(f"f* = {f_star:.10f} (rel change over last quarter: {rel_change:.2e})")

    # --- ISTA pass 1: objective gap + final iterate; pass 2: support matches
    z, objectives, nnz, _ = ista_pass(data.X, a, args.lam, args.ista_iters, eta)
    supp_star = np.abs(z) > 1e-12
    _, _, _, matches = ista_pass(data.X, a, args.lam, args.ista_iters, eta, supp_star=supp_star)

    gaps = objectives - f_star
    t_axis = np.arange(len(gaps))
    t_support = first_stable_match(matches)
    print(f"nnz(Z*) = {int(supp_star.sum())} / {supp_star.size} entries")

    # Theoretical bound (book Theorem 8.6): L ||Z_0 - Z*||_F^2 / (2k); Z_0 = 0.
    lipschitz = 2.0 * np.linalg.norm(a, ord=2) ** 2
    bound = lipschitz * float(np.linalg.norm(z, "fro") ** 2) / (2.0 * np.clip(t_axis, 1, None))

    # Geometric fit on the tail after support identification.
    rho = float("nan")
    if t_support is not None:
        sel = slice(max(t_support, 1), None)
        rho = fit_linear_rate(t_axis[sel], gaps[sel])
    bound_at = int(t_support or 1)
    print(
        f"support identified from t = {t_support}; geometric tail rho = {rho:.5f}; "
        f"gap/bound at t={bound_at}: {gaps[bound_at] / bound[bound_at]:.3g}"
    )

    with args.log.open("w", encoding="utf-8") as f:
        meta = {
            "tag": "meta",
            "config": {
                "D": args.dim, "m": args.atoms, "N": args.n, "theta": args.theta,
                "lam": args.lam, "sigma": args.sigma, "seed": args.seed,
                "ista_iters": args.ista_iters,
            },
            "eta": eta,
            "f_star": f_star,
            "fstar_rel_change": rel_change,
            "lipschitz": lipschitz,
            "nnz_star": int(supp_star.sum()),
            "t_support_identified": t_support,
            "tail_geometric_rate_rho": rho,
        }
        f.write(json.dumps(meta) + "\n")
        for t in range(len(gaps)):
            f.write(
                json.dumps(
                    {
                        "tag": "iter",
                        "t": int(t),
                        "gap": float(gaps[t]),
                        "bound": float(bound[t]),
                        "nnz": int(nnz[t]),
                        "support_matches_star": bool(matches[t]),
                    }
                )
                + "\n"
            )

    # --- figure -----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    s = 1  # skip t = 0 (Theorem-8.6 bound undefined there)
    ax1.loglog(t_axis[s:], gaps[s:], linewidth=1.4, label=r"observed $f(Z_t)-f^\star$")
    ax1.loglog(
        t_axis[s:], bound[s:], "--", linewidth=1.2,
        label=r"Theorem 8.6 bound $L\|Z_0-Z^\star\|_F^2/(2k)$",
    )
    if t_support is not None:
        ax1.axvline(t_support, color="gray", linestyle=":", linewidth=1.0)
        ax1.text(t_support * 1.15, gaps[s] * 0.6, " support\n identified", fontsize=8, color="gray")
    ax1.set_xlabel("ISTA iteration $k$")
    ax1.set_ylabel("objective gap")
    ax1.set_title("log-log: observed gap vs O(1/k) worst-case bound")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.semilogy(t_axis[s:], gaps[s:], linewidth=1.4, label="observed gap")
    if np.isfinite(rho) and t_support is not None:
        tail = np.arange(t_support, len(gaps))
        ax2.semilogy(
            tail, gaps[t_support] * rho ** (tail - t_support), "--", linewidth=1.2,
            label=fr"geometric fit $\rho^k$, $\rho={rho:.4f}$",
        )
        ax2.axvline(t_support, color="gray", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("ISTA iteration $k$")
    ax2.set_title("semilogy: geometric tail after support identification")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(
        fr"ISTA on the inner LASSO (overcomplete, $m$={args.atoms}, $D$={args.dim}, "
        fr"$\theta$={args.theta}, $\lambda$={args.lam}, $\sigma$={args.sigma})"
    )
    savefig(fig, args.figure)
    print(f"figure -> {args.figure}")
    print(f"log -> {args.log}")


if __name__ == "__main__":
    main()
