"""HW1 Part 3: real-data dictionary learning on PCA-whitened MNIST.

Phases are independently runnable/resumable:

  preprocess -- download MNIST, spectrum, k-dimensional PCA + whitening;
  taxis      -- N=12000, 400 alternating-minimization blocks, record every 5;
  naxis      -- N grid, stop by 50-block objective change or 1000-block cap;
  atoms      -- inverse-whiten and display seed-0 T-axis dictionaries;
  all        -- run all phases in order (default).

All arrays in learning/inference are float32. Long jobs checkpoint under
outputs/data/ and append JSONL records under outputs/logs/.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from data8014.dictlearn import altmin_block, fit_decay_models, ista, lasso_objective
from data8014.dictlearn.mnist import (
    PCAWhitening,
    explained_variance_counts,
    fit_pca_whitening,
    load_mnist,
)
from data8014.dictlearn.paths import DATA_DIR, FIGURE_DIR, LOG_DIR, OUTPUT_DATA_DIR, ensure_directories
from data8014.utils.plotting import savefig

NAXIS_COUNTS = [200, 400, 800, 1600, 3200, 6400, 12800]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "preprocess", "taxis", "naxis", "atoms"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--multipliers", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--lam", type=float, default=2.0)
    parser.add_argument("--inner-steps", type=int, default=30)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--baseline-eval-steps", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--taxis-n", type=int, default=12000)
    parser.add_argument("--taxis-blocks", type=int, default=400)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--naxis-counts", type=int, nargs="+", default=NAXIS_COUNTS)
    parser.add_argument("--naxis-cap", type=int, default=1000)
    parser.add_argument("--objective-window", type=int, default=50)
    parser.add_argument("--objective-tol", type=float, default=1e-5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def processed_path(args: argparse.Namespace) -> Path:
    return args.processed or OUTPUT_DATA_DIR / f"dl_part3_mnist_k{args.k}.npz"


def random_dictionary(m: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([2026, seed, m, k]))
    a = rng.standard_normal((m, k)).astype(np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def train_order(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([8014, seed])).permutation(n)


def reconstruction_metrics(
    x: np.ndarray,
    a: np.ndarray,
    lam: float,
    n_steps: int,
    z0: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    z, _ = ista(x, a, lam, n_steps=n_steps, z0=z0)
    loss = float(np.linalg.norm(x - z @ a, "fro") ** 2 / x.shape[0])
    nnz = float(np.count_nonzero(z) / z.size)
    return z, loss, nnz


def preprocess(args: argparse.Namespace) -> None:
    path = processed_path(args)
    ensure_directories()
    train_x, _, test_x, _ = load_mnist(args.data_dir)
    transform, spectrum = fit_pca_whitening(train_x, args.k)
    counts = explained_variance_counts(spectrum)
    train_w = transform.transform(train_x)
    test_w = transform.transform(test_x[: args.test_size])
    covariance = train_w.T @ train_w / train_w.shape[0]
    whiten_error = float(np.linalg.norm(covariance - np.eye(args.k), "fro") / np.sqrt(args.k))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train=train_w,
        test=test_w,
        mean=transform.mean,
        components=transform.components,
        top_eigenvalues=transform.eigenvalues,
        spectrum=spectrum,
        whiten_error=np.asarray(whiten_error),
    )

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    indices = np.arange(1, len(spectrum) + 1)
    positive = spectrum > 0
    ax.loglog(indices[positive], spectrum[positive])
    ax.axvline(args.k, color="tab:orange", linestyle="--", label=f"chosen k={args.k}")
    ax.set_xlabel("principal direction index")
    ax.set_ylabel("covariance eigenvalue")
    ax.set_title("MNIST covariance spectrum")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    savefig(fig, FIGURE_DIR / "dl_part3_spectrum.png")
    print(f"variance counts: {counts}; whitening relative Frobenius error={whiten_error:.3e}")
    print(f"processed data -> {path}")


def load_processed(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], PCAWhitening]:
    path = processed_path(args)
    if not path.exists():
        raise FileNotFoundError(f"run --phase preprocess first: {path}")
    data = np.load(path)
    arrays = {key: data[key] for key in data.files}
    transform = PCAWhitening(
        mean=arrays["mean"], components=arrays["components"], eigenvalues=arrays["top_eigenvalues"]
    )
    return arrays, transform


def checkpoint_path(kind: str, m: int, seed: int, n: int | None = None) -> Path:
    suffix = f"_N{n}" if n is not None else ""
    return OUTPUT_DATA_DIR / f"dl_part3_{kind}_m{m}_seed{seed}{suffix}.npz"


def load_jsonl(path: Path, tag: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line and json.loads(line).get("tag") == tag]


def taxis(args: argparse.Namespace) -> None:
    arrays, _ = load_processed(args)
    train, test = arrays["train"], arrays["test"]
    log_path = LOG_DIR / "dl_part3_taxis.jsonl"
    if not args.resume:
        log_path.write_text("", encoding="utf-8")
    existing = load_jsonl(log_path, "checkpoint") if args.resume else []
    completed = {(r["m"], r["seed"], r["block"]) for r in existing}

    with log_path.open("a", encoding="utf-8") as log:
        for multiplier in args.multipliers:
            m = multiplier * args.k
            for seed in args.seeds:
                state_path = checkpoint_path("taxis", m, seed)
                order = train_order(len(train), seed)
                x = train[order[: args.taxis_n]]
                a0 = random_dictionary(m, args.k, seed)
                baseline_z, baseline_loss, baseline_nnz = reconstruction_metrics(
                    test, a0, args.lam, args.baseline_eval_steps
                )
                del baseline_z

                if args.resume and state_path.exists():
                    state = np.load(state_path)
                    block = int(state["block"])
                    a, z, z_val = state["a"], state["z"], state["z_val"]
                else:
                    block = 0
                    a = a0.copy()
                    z = np.zeros((len(x), m), dtype=np.float32)
                    z_val = np.zeros((len(test), m), dtype=np.float32)

                start = time.perf_counter()
                while block < args.taxis_blocks:
                    a, z = altmin_block(x, a, z, args.lam, L=args.inner_steps)
                    block += 1
                    if block % args.record_every == 0:
                        z_val, held_out, nnz = reconstruction_metrics(
                            test, a, args.lam, args.eval_steps, z0=z_val
                        )
                        train_obj = lasso_objective(z, a, x, args.lam)
                        record = {
                            "tag": "checkpoint",
                            "m": m,
                            "seed": seed,
                            "block": block,
                            "train_objective": train_obj,
                            "train_objective_per_sample": train_obj / len(x),
                            "held_out_loss": held_out,
                            "random_loss": baseline_loss,
                            "loss_ratio": held_out / baseline_loss,
                            "code_nnz_fraction": nnz,
                            "random_code_nnz_fraction": baseline_nnz,
                            "wall_time_s": time.perf_counter() - start,
                        }
                        if (m, seed, block) not in completed:
                            log.write(json.dumps(record) + "\n")
                            log.flush()
                        print(
                            f"T-axis m={m} seed={seed} block={block}: "
                            f"val={held_out:.4g} ratio={record['loss_ratio']:.3f} obj/N={record['train_objective_per_sample']:.4g}"
                        )
                    if block % args.checkpoint_every == 0 or block == args.taxis_blocks:
                        np.savez(state_path, block=block, a=a, z=z, z_val=z_val)

    plot_taxis(log_path, args)
    summarize_taxis(log_path, args)


def grouped_records(records: list[dict[str, Any]], m: int, seeds: list[int], xs: list[int], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty((len(seeds), len(xs)))
    for i, seed in enumerate(seeds):
        lookup = {r.get("block", r.get("N")): r[key] for r in records if r["m"] == m and r["seed"] == seed}
        values[i] = [lookup[x] for x in xs]
    return values.mean(axis=0), values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(xs))


def plot_taxis(log_path: Path, args: argparse.Namespace) -> None:
    records = load_jsonl(log_path, "checkpoint")
    blocks = list(range(args.record_every, args.taxis_blocks + 1, args.record_every))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for multiplier in args.multipliers:
        m = multiplier * args.k
        for row, key in enumerate(("held_out_loss", "train_objective_per_sample")):
            mean, std = grouped_records(records, m, args.seeds, blocks, key)
            for col, scale in enumerate(("semilogy", "loglog")):
                ax = axes[row, col]
                line = getattr(ax, scale)(blocks, mean, label=f"m={m}")[0]
                ax.fill_between(blocks, np.maximum(mean - std, 1e-12), mean + std, alpha=0.18, color=line.get_color())
                if row == 0:
                    baseline = np.mean([
                        next(r["random_loss"] for r in records if r["m"] == m and r["seed"] == seed)
                        for seed in args.seeds
                    ])
                    ax.axhline(
                        baseline,
                        color=line.get_color(),
                        linestyle=":",
                        linewidth=1.0,
                        label=f"m={m} random",
                    )
                ax.set_xlabel("blocks")
                ax.set_ylabel("held-out loss" if row == 0 else "training objective / sample")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend()
    axes[0, 0].set_title("held-out (semilogy)")
    axes[0, 1].set_title("held-out (log-log)")
    axes[1, 0].set_title("training objective (semilogy)")
    axes[1, 1].set_title("training objective (log-log)")
    fig.suptitle(f"HW1 Part 3 T-axis: PCA-whitened MNIST, k={args.k}, N={args.taxis_n}")
    savefig(fig, FIGURE_DIR / "dl_part3_taxis.png")


def summarize_taxis(log_path: Path, args: argparse.Namespace) -> None:
    records = load_jsonl(log_path, "checkpoint")
    summary: dict[str, Any] = {}
    blocks = np.arange(args.record_every, args.taxis_blocks + 1, args.record_every)
    for multiplier in args.multipliers:
        m = multiplier * args.k
        final = [
            next(r for r in records if r["m"] == m and r["seed"] == seed and r["block"] == args.taxis_blocks)
            for seed in args.seeds
        ]
        summary[str(m)] = {}
        for key in ("held_out_loss", "train_objective_per_sample", "loss_ratio", "code_nnz_fraction", "random_loss"):
            values = np.asarray([r[key] for r in final])
            summary[str(m)][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        summary[str(m)]["convergence_fits"] = {}
        summary[str(m)]["per_seed_convergence_fits"] = {}
        for key in ("held_out_loss", "train_objective_per_sample"):
            mean_curve, _ = grouped_records(records, m, args.seeds, blocks.tolist(), key)
            summary[str(m)]["convergence_fits"][key] = fit_decay_models(blocks, mean_curve)
            summary[str(m)]["per_seed_convergence_fits"][key] = [
                {
                    "seed": seed,
                    **fit_decay_models(
                        blocks,
                        np.asarray([
                            next(r[key] for r in records if r["m"] == m and r["seed"] == seed and r["block"] == int(block))
                            for block in blocks
                        ]),
                    ),
                }
                for seed in args.seeds
            ]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"tag": "summary", "final_block": args.taxis_blocks, "results": summary}) + "\n")
    print(json.dumps(summary, indent=2))


def naxis_run_one(
    x: np.ndarray,
    test: np.ndarray,
    a0: np.ndarray,
    m: int,
    seed: int,
    n: int,
    baseline_loss: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    state_path = checkpoint_path("naxis", m, seed, n)
    if args.resume and state_path.exists():
        state = np.load(state_path)
        block = int(state["block"])
        a, z = state["a"], state["z"]
        objectives = state["objectives"].tolist()
    else:
        block = 0
        a = a0.copy()
        z = np.zeros((n, m), dtype=np.float32)
        objectives: list[float] = []
    converged = False
    start = time.perf_counter()
    while block < args.naxis_cap:
        a, z = altmin_block(x, a, z, args.lam, L=args.inner_steps)
        block += 1
        objective = lasso_objective(z, a, x, args.lam) / n
        objectives.append(objective)
        if len(objectives) > args.objective_window:
            old = objectives[-args.objective_window - 1]
            relative = abs(objectives[-1] - old) / max(abs(old), 1e-12)
            if relative < args.objective_tol:
                converged = True
                break
        if block % args.checkpoint_every == 0:
            np.savez(state_path, block=block, a=a, z=z, objectives=np.asarray(objectives))
    np.savez(state_path, block=block, a=a, z=z, objectives=np.asarray(objectives))
    _, held_out, nnz = reconstruction_metrics(test, a, args.lam, args.baseline_eval_steps)
    return {
        "tag": "result",
        "m": m,
        "seed": seed,
        "N": n,
        "blocks": block,
        "converged": converged,
        "train_objective_per_sample": objectives[-1],
        "held_out_loss": held_out,
        "random_loss": baseline_loss,
        "loss_ratio": held_out / baseline_loss,
        "code_nnz_fraction": nnz,
        "wall_time_s": time.perf_counter() - start,
    }


def scaling_model(n: np.ndarray, floor: float, c: float, beta: float) -> np.ndarray:
    return floor + c * n ** (-beta)


def fit_free_floor(n: np.ndarray, values: np.ndarray, n_fit: int) -> dict[str, float]:
    x, y = n[:n_fit].astype(float), values[:n_fit].astype(float)
    upper_floor = float(y.min() * (1.0 - 1e-8))
    p0 = [max(0.0, upper_floor * 0.8), max(y[0] - upper_floor * 0.8, 1e-4) * x[0] ** 0.5, 0.5]
    params, covariance = curve_fit(
        scaling_model,
        x,
        y,
        p0=p0,
        bounds=([0.0, 0.0, 1e-6], [upper_floor, np.inf, 5.0]),
        maxfev=50000,
    )
    prediction = scaling_model(x, *params)
    ss_res = float(np.square(y - prediction).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    stderr = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "floor": float(params[0]), "c": float(params[1]), "beta": float(params[2]),
        "floor_stderr": float(stderr[0]), "c_stderr": float(stderr[1]), "beta_stderr": float(stderr[2]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def naxis(args: argparse.Namespace) -> None:
    arrays, _ = load_processed(args)
    train, test = arrays["train"], arrays["test"]
    counts = sorted(args.naxis_counts)
    log_path = LOG_DIR / "dl_part3_naxis.jsonl"
    if not args.resume:
        log_path.write_text("", encoding="utf-8")
    records = load_jsonl(log_path, "result") if args.resume else []
    completed = {(r["m"], r["seed"], r["N"]) for r in records}

    with log_path.open("a", encoding="utf-8") as log:
        for multiplier in args.multipliers:
            m = multiplier * args.k
            for seed in args.seeds:
                order = train_order(len(train), seed)
                a0 = random_dictionary(m, args.k, seed)
                _, baseline_loss, _ = reconstruction_metrics(test, a0, args.lam, args.baseline_eval_steps)
                for n in counts:
                    if (m, seed, n) in completed:
                        continue
                    result = naxis_run_one(train[order[:n]], test, a0, m, seed, n, baseline_loss, args)
                    log.write(json.dumps(result) + "\n")
                    log.flush()
                    records.append(result)
                    print(
                        f"N-axis m={m} seed={seed} N={n}: blocks={result['blocks']} "
                        f"conv={result['converged']} loss={result['held_out_loss']:.4g} ratio={result['loss_ratio']:.3f}"
                    )
    expected = len(args.multipliers) * len(args.seeds) * len(counts)
    if len({(r["m"], r["seed"], r["N"]) for r in records}) < expected:
        print("partial N-axis run; rerun with --resume")
        return
    summarize_naxis(records, counts, log_path, args)


def summarize_naxis(records: list[dict[str, Any]], counts: list[int], log_path: Path, args: argparse.Namespace) -> None:
    n = np.asarray(counts, dtype=float)
    n_fit = math.floor(2 * len(n) / 3) + (1 if len(n) % 3 else 0)  # 5 of 7
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    summaries: dict[str, Any] = {}
    for multiplier in args.multipliers:
        m = multiplier * args.k
        mean, std = grouped_records(records, m, args.seeds, counts, "held_out_loss")
        fit = fit_free_floor(n, mean, n_fit)
        prediction = float(scaling_model(np.asarray([n[-1]]), fit["floor"], fit["c"], fit["beta"])[0])
        error = abs(prediction - mean[-1])
        per_seed = []
        for seed in args.seeds:
            values = np.asarray([next(r["held_out_loss"] for r in records if r["m"] == m and r["seed"] == seed and r["N"] == count) for count in counts])
            per_seed.append({"seed": seed, **fit_free_floor(n, values, n_fit)})
        selected = [r for r in records if r["m"] == m]
        final_n = [r for r in selected if r["N"] == counts[-1]]
        summaries[str(m)] = {
            "fit": fit,
            "prediction": prediction,
            "observed": float(mean[-1]),
            "absolute_error": float(error),
            "relative_error": float(error / mean[-1]),
            "per_seed_fits": per_seed,
            "beta_mean": float(np.mean([f["beta"] for f in per_seed])),
            "beta_std": float(np.std([f["beta"] for f in per_seed], ddof=1)),
            "converged_runs": int(sum(r["converged"] for r in selected)),
            "capped_runs": int(sum(not r["converged"] for r in selected)),
            "final_loss_ratio_mean": float(np.mean([r["loss_ratio"] for r in final_n])),
            "final_loss_ratio_std": float(np.std([r["loss_ratio"] for r in final_n], ddof=1)),
            "final_code_nnz_fraction_mean": float(np.mean([r["code_nnz_fraction"] for r in final_n])),
            "final_code_nnz_fraction_std": float(np.std([r["code_nnz_fraction"] for r in final_n], ddof=1)),
        }
        container = ax.errorbar(n, mean, yerr=std, fmt="o", capsize=3, label=f"m={m} mean±std")
        color = container.lines[0].get_color()
        capped = [r for r in selected if not r["converged"]]
        if capped:
            ax.scatter(
                [r["N"] for r in capped],
                [r["held_out_loss"] for r in capped],
                marker="^",
                s=42,
                facecolors="none",
                edgecolors=color,
                linewidths=1.1,
                label=f"m={m} capped run",
                zorder=4,
            )
        dense = np.geomspace(n.min(), n.max(), 300)
        ax.plot(dense, scaling_model(dense, fit["floor"], fit["c"], fit["beta"]), "--", color=color, label=fr"m={m} fit $\beta={fit['beta']:.3f}$")
        baseline = float(np.mean([r["random_loss"] for r in final_n]))
        ax.axhline(baseline, color=color, linestyle=":", linewidth=1.0, label=f"m={m} random")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training samples N")
    ax.set_ylabel("held-out loss")
    ax.set_title("HW1 Part 3 N-axis: MNIST sample-size scaling")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    savefig(fig, FIGURE_DIR / "dl_part3_naxis.png")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"tag": "summary", "n_fit": n_fit, "summaries": summaries}) + "\n")
    print(json.dumps(summaries, indent=2))


def atoms(args: argparse.Namespace) -> None:
    _, transform = load_processed(args)
    for multiplier in args.multipliers:
        m = multiplier * args.k
        state_path = checkpoint_path("taxis", m, args.seeds[0])
        if not state_path.exists():
            raise FileNotFoundError(f"missing T-axis dictionary: {state_path}")
        a = np.load(state_path)["a"]
        pixels = transform.atoms_to_pixels(a).reshape(m, 28, 28)
        cols = 16
        rows = math.ceil(m / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.05, rows * 1.05))
        axes = np.asarray(axes).reshape(-1)
        for i, ax in enumerate(axes):
            ax.axis("off")
            if i >= m:
                continue
            atom = pixels[i]
            scale = np.max(np.abs(atom))
            ax.imshow(atom / max(scale, 1e-12), cmap="gray", vmin=-1, vmax=1)
        fig.suptitle(f"MNIST dictionary atoms mapped to pixel space (k={args.k}, m={m})")
        savefig(fig, FIGURE_DIR / f"dl_part3_atoms_m{m}.png", dpi=180)


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 3 and args.phase in ("all", "taxis", "naxis"):
        print("WARNING: final reported quantities require at least three seeds")
    ensure_directories()
    phases = ["preprocess", "taxis", "naxis", "atoms"] if args.phase == "all" else [args.phase]
    for phase in phases:
        print(f"=== Part 3 phase: {phase} ===")
        globals()[phase](args)


if __name__ == "__main__":
    main()
