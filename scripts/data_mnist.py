"""Assignment 0 (data part): download MNIST and visualize augmented samples.

Inverts the colors of MNIST training images, takes a random crop, and saves
the 10,000th / 20,000th / 30,000th / 40,000th / 50,000th training images
under these augmentations (positions are 1-indexed), both as individual PNGs
and as a two-row comparison figure with the originals.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import transforms
from torchvision.utils import save_image

from data8014.config import DATA_DIR, FIGURE_DIR, ensure_directories, seed_everything
from data8014.data.mnist import (
    DEFAULT_CROP_SIZE,
    build_augmentation,
    fetch_samples,
    get_mnist,
    save_augmentation_figure,
)

#: Default positions asked for by the assignment (1-indexed).
DEFAULT_POSITIONS = [10_000, 20_000, 30_000, 40_000, 50_000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positions", type=int, nargs="+", default=DEFAULT_POSITIONS,
        help="1-indexed training-set positions to visualize (default: %(default)s)",
    )
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--seed", type=int, default=0, help="seed for the random crops")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    seed_everything(args.seed)

    original_set = get_mnist(args.data_dir, train=True, transform=transforms.ToTensor())
    augmented_set = get_mnist(
        args.data_dir, train=True, transform=build_augmentation(crop_size=args.crop_size)
    )
    print(f"MNIST train examples: {len(original_set)} (expect 60,000)")

    originals = fetch_samples(original_set, args.positions)
    augmented = fetch_samples(augmented_set, args.positions)

    for position, (image, label) in zip(args.positions, augmented):
        out_path = args.out_dir / f"mnist_augmented_{position:06d}.png"
        save_image(image, out_path)
        print(f"position {position}: label={label} -> {out_path.name}")

    combined = save_augmentation_figure(
        originals, augmented, args.positions, args.out_dir / "mnist_augmented_samples.png"
    )
    print(f"comparison figure -> {combined}")


if __name__ == "__main__":
    main()
