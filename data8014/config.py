"""Global configuration: repo paths, device selection, seeding.

All artifacts produced by scripts (checkpoints, logs, figures, generated
datasets) live under ``outputs/`` at the repository root so the package itself
stays free of generated state. Raw third-party datasets (e.g. MNIST) are
downloaded under ``data/``. Checkpoints / generated data / logs are gitignored,
figures are tracked so they can be referenced by the writeup.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Raw datasets downloaded from the internet (gitignored).
DATA_DIR = REPO_ROOT / "data"

#: Generated artifacts (subsets gitignored, figures tracked).
OUTPUT_DIR = REPO_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"

#: Artifacts handed off between pipeline stages (produce in one script,
#: consume in the next): the shared ground-truth parameter, the trained
#: checkpoint, and the regenerated Gaussian test set.
THETA_STAR_PATH = OUTPUT_DATA_DIR / "theta_star.pt"
GAUSSIAN_CKPT_PATH = CHECKPOINT_DIR / "gaussian_linear.pt"
GAUSSIAN_TEST_SET_PATH = OUTPUT_DATA_DIR / "gaussian_test.pt"


def ensure_directories() -> None:
    """Create every output directory if it does not exist yet."""
    for path in (DATA_DIR, CHECKPOINT_DIR, OUTPUT_DATA_DIR, LOG_DIR, FIGURE_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)


def get_device(index: int = 0) -> torch.device:
    """Pick the best available torch device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_generator(seed: int) -> torch.Generator:
    """A dedicated torch RNG so datasets stay reproducible given a seed."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_everything(seed: int) -> None:
    """Seed python / torch RNGs for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
