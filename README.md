# DATA 8014 — Semester Codebase

A single, semester-long AI research codebase for **DATA 8014 (Fall 2026)**,
structured around four subsystems: *data*, *training*, *inference*, and
*evaluation*. Assignment 0 builds the skeleton and exercises every subsystem
on MNIST augmentation and a synthetic Gaussian linear-regression task.

## Layout

```
.
├── data8014/                  # the package: all reusable codebase code
│   ├── config.py              # repo paths, device selection, seeding
│   ├── data/                  # dataset construction (gaussian, mnist)
│   ├── models/                # model factories + checkpoint reconstruction
│   ├── training/              # Trainer, step metrics, JSONL logging, plots
│   ├── inference/             # InferenceEngine + prediction plots
│   ├── evaluation/            # EvalRunner + evaluation plots
│   └── utils/                 # checkpoint IO, plotting helpers
├── scripts/                   # one-off experiment entry points (outside the package)
│   ├── data_mnist.py          # A0 data part:  download MNIST + augmented samples
│   ├── train_gaussian.py      # A0 training:   Trainer on the Gaussian dataset
│   ├── infer_gaussian.py      # A0 inference:  checkpoint -> predictions + plot
│   └── eval_gaussian.py       # A0 evaluation: loss/grad-norm correlation
├── outputs/                   # generated artifacts (checkpoints/logs/data gitignored,
│                              # figures tracked for the writeup)
├── data/                      # downloaded raw datasets (gitignored)
├── writeup/                   # LaTeX writeup (compiled separately)
└── docs/                      # assignment statements
```

## Setup

Requires Python >= 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                       # create .venv and install torch + the package
uv run python <script>        # run anything inside the managed environment
```

<details>
<summary>Notes recorded while setting up on this Windows machine (click)</summary>

- The NVIDIA driver (560.76, CUDA 12.6) matches the **cu126** wheels, so
  `pyproject.toml` pins torch/torchvision to the official cu126 index (the
  explicit-index equivalent of `uv add torch --torch-backend=auto` on uv
  builds that lack the flag).
- `download-r2.pytorch.org` (where the index redirects downloads) is not
  reachable from this network, so the two wheels were fetched once from
  `download.pytorch.org` into `wheels/` (gitignored) and installed manually:
  `uv pip install --python .venv wheels/*.whl matplotlib psutil` plus
  `uv pip install --python .venv -e . --no-deps`. On this machine run scripts
  with `uv run --no-sync python scripts/...` (or `.venv/Scripts/python.exe`)
  so uv does not try to re-download torch from the unreachable mirror.
- The venv must be built on the **uv-managed CPython** (`uv python install
  3.12`, here 3.12.13). Venvs built on the system Anaconda Python fail to
  load torch (`c10.dll` DLL-initialization error).

</details>

## Assignment 0 pipeline

```bash
# 1. Data: download MNIST, save augmented samples (invert + random crop 24x24)
uv run python scripts/data_mnist.py

# 2. Training: SGD on the Gaussian dataset; checkpoint + metrics + plots
uv run python scripts/train_gaussian.py

# 3. Inference: load checkpoint, regenerate data with the same theta_star, plot
uv run python scripts/infer_gaussian.py

# 4. Evaluation: per-example loss & gradient-norm correlation, histograms
uv run python scripts/eval_gaussian.py
```

All stages are wired through files under `outputs/`:

| Artifact | Producer | Consumer |
| --- | --- | --- |
| `outputs/data/theta_star.pt` | `train_gaussian.py` | `infer_gaussian.py` |
| `outputs/checkpoints/gaussian_linear.pt` | `train_gaussian.py` | `infer_gaussian.py`, `eval_gaussian.py` |
| `outputs/data/gaussian_test.pt` | `infer_gaussian.py` | `eval_gaussian.py` |
| `outputs/logs/train_gaussian.jsonl` | trainer | `data8014.training.logging.plot_training_log` |
| `outputs/figures/*.png` | all stages | writeup |

## Design notes & conventions

- **Package vs. experiments.** All reusable code lives in the `data8014`
  package; `scripts/` holds thin, one-off entry points for each assignment
  task, and can import the package because it is installed by `uv`.
- **`Trainer`.** The salient abstraction is `Trainer.train_step(batch)` — one
  optimization step given a batch and the current model state, returning
  predictions and step metrics. The full loop logs train/val loss, gradient
  norm, memory, time per step, dataloader/sample throughput and MFU to a
  JSONL file; plots are regenerated from the log, never from memory.
- **`InferenceEngine`.** Owns checkpoint loading and prediction only — no
  gradients, no optimizer — mirroring how inference differs from training in
  real systems (own batch sizes, no backward).
- **`EvalRunner`.** Wraps an `InferenceEngine` and adds gradient-sensitive
  diagnostics: per-example gradient norms via `torch.autograd.grad`, the
  correlation between per-example loss and gradient norm, and per-batch
  histograms.
- **Noise convention.** `N(0, 0.01)` is read as variance 0.01 (std 0.1);
  configurable via `GaussianConfig.noise_std`.
- **Validation data.** The 10,000-example training set is kept exactly as
  specified; an extra 1,000-example validation set is drawn with the same
  `theta_star` for periodic validation.
- **Positions are 1-indexed.** “The 10,000th training image” → dataset index
  9,999.
- **MFU accounting.** FLOPs per step use the standard 3×-forward convention
  for Linear layers; the peak FLOP/s denominator is measured by a cached GEMM
  microbenchmark (achievable throughput, hence conservative MFU). Override
  with a datasheet number via `TrainerConfig.peak_flops` if desired.
- **Reproducibility.** Every dataset draw takes an explicit `torch.Generator`
  seeded by CLI flags, so re-running a script reproduces its inputs exactly.

## Writeup

`writeup/writeup.tex` is a LaTeX skeleton with placeholders — the writeup
itself (structure description, plot explanations, AI-usage disclosure) is
written manually by the student, per course policy.
