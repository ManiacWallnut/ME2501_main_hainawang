"""Assignment 0 (training part): SGD on a synthetic Gaussian dataset.

Generates X in R^{10,000 x 10}, a ground-truth theta_star in R^10, noise
eps ~ N(0, 0.01) and labels y = X theta_star + eps; trains a ``torch.nn.Linear``
model with ``torch.optim.SGD`` and squared-error loss through the ``Trainer``
abstraction; saves the checkpoint (``.pt``), a JSONL metrics log, and metric
plots. ``theta_star`` is saved so later stages can regenerate data with the
same ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data8014.config import (
    GAUSSIAN_CKPT_PATH,
    LOG_DIR,
    THETA_STAR_PATH,
    ensure_directories,
    get_device,
    make_generator,
    seed_everything,
)
from data8014.data.gaussian import DEFAULT_NOISE_STD, GaussianConfig, make_gaussian_dataset, sample_theta_star
from data8014.models.linear import build_linear_model
from data8014.training.logging import MetricsLogger, plot_training_log
from data8014.training.trainer import Trainer, TrainerConfig
from data8014.utils.io import save_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--val-size", type=int, default=1_000,
                        help="extra held-out examples drawn with the same theta_star")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD,
                        help="N(0, 0.01) means std=0.1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta-seed", type=int, default=1_000)
    parser.add_argument("--checkpoint", type=Path, default=GAUSSIAN_CKPT_PATH)
    parser.add_argument("--log", type=Path, default=LOG_DIR / "train_gaussian.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    device = get_device()
    seed_everything(args.seed)
    print(f"device: {device}")

    # Ground truth theta_star, handed off to the inference / evaluation stages.
    theta_star = sample_theta_star(args.dim, make_generator(args.theta_seed))
    save_tensors(THETA_STAR_PATH, theta_star=theta_star)
    print(f"theta_star saved -> {THETA_STAR_PATH} (||theta_star|| = {theta_star.norm():.4f})")

    train_set = make_gaussian_dataset(
        GaussianConfig(args.train_size, args.dim, args.noise_std),
        theta_star,
        make_generator(args.seed * 10 + 1),
    )
    val_set = make_gaussian_dataset(
        GaussianConfig(args.val_size, args.dim, args.noise_std),
        theta_star,
        make_generator(args.seed * 10 + 2),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1_024)

    model = build_linear_model(args.dim, out_features=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    config = TrainerConfig(num_epochs=args.epochs)
    with MetricsLogger(args.log) as logger:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            config=config,
            logger=logger,
            checkpoint_path=args.checkpoint,
        )
        summary = trainer.train(train_loader, val_loader=val_loader)

    print("--- training summary ---")
    for key, value in summary.items():
        print(f"{key}: {value:.6g}" if isinstance(value, float) else f"{key}: {value}")
    learned = model.weight.detach().flatten().cpu()
    distance = float((learned - theta_star).norm())
    print(f"||theta_hat - theta_star||_2 = {distance:.6f}")
    print(f"checkpoint -> {args.checkpoint}")

    figures = plot_training_log(args.log, Path("outputs/figures"))
    print("metric plots:")
    for figure in figures:
        print(f"  {figure}")


if __name__ == "__main__":
    main()
