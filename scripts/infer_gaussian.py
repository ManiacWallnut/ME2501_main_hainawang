"""Assignment 0 (inference part): run the checkpoint on fresh Gaussian data.

Regenerates a new Gaussian dataset with the *same* ground-truth theta_star as
the training stage, lets the ``InferenceEngine`` load the checkpoint and
predict labels, saves the test set for the evaluation stage, and plots true
labels (x axis) vs predictions (y axis).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from data8014.config import (
    FIGURE_DIR,
    GAUSSIAN_CKPT_PATH,
    GAUSSIAN_TEST_SET_PATH,
    THETA_STAR_PATH,
    ensure_directories,
    get_device,
    make_generator,
)
from data8014.data.gaussian import DEFAULT_NOISE_STD, GaussianConfig, make_gaussian_dataset
from data8014.inference.engine import InferenceEngine, plot_true_vs_predicted
from data8014.utils.io import load_tensors, save_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5_000, help="size of the regenerated test set")
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--seed", type=int, default=3_000, help="seed of the new draw")
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--checkpoint", type=Path, default=GAUSSIAN_CKPT_PATH)
    parser.add_argument("--theta", type=Path, default=THETA_STAR_PATH,
                        help="theta_star saved by scripts/train_gaussian.py")
    parser.add_argument("--out", type=Path, default=GAUSSIAN_TEST_SET_PATH,
                        help="where to save the test set for the evaluation stage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    device = get_device()
    print(f"device: {device}")

    theta_star = load_tensors(args.theta)["theta_star"]
    engine = InferenceEngine.from_checkpoint(args.checkpoint, device)
    print(f"loaded checkpoint {args.checkpoint} (step {engine.checkpoint.get('step')})")

    test_set = make_gaussian_dataset(
        GaussianConfig(args.n, theta_star.numel(), args.noise_std),
        theta_star,
        make_generator(args.seed),
    )
    X, y = test_set.tensors
    save_tensors(args.out, X=X, y=y)
    print(f"test set ({args.n} examples, same theta_star) saved -> {args.out}")

    output = engine.run(DataLoader(test_set, batch_size=args.batch_size), collect_labels=True)
    mse = float(((output.predictions - output.labels) ** 2).mean())
    print(f"test MSE: {mse:.6f} (noise variance is {args.noise_std ** 2:.4f})")

    figure = plot_true_vs_predicted(
        output.labels, output.predictions, FIGURE_DIR / "inference_true_vs_pred.png"
    )
    print(f"figure -> {figure}")


if __name__ == "__main__":
    main()
