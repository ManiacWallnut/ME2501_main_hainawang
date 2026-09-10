# Assignment 0: Setting Up the Codebase

**Data 8014 — Fall 2026**

------

In this course, the assignments are (open-ended) projects with both a *written component* and a *programming component*. Often these components will be interleaved throughout the assignment. To realistically mimic the engineering work you will do in large-scale AI research and engineering

> *we want you to do all programming within the same codebase.*

**In this assignment, you will build a skeleton of the codebase you will use for future assignments.** Don't worry too much — nothing is permanent, and you can change the codebase for each assignment — but egregious changes (such as deleting and completely restarting the codebase for each assignment) may result in reduced marks. The point is to learn how to build on top of the same codebase for a while, and grapple with the design choices you make now in the future.

**All programming and mathematical work for the assignment may be done with AI, but it is your responsibility to understand its outputs.** We also ask that the writeup is done by you and not AI, and at the end of the written part, you disclose the usage and any specifics (e.g., which models/settings you use for a particular part). Note that AI still does not have the best grasp of how to design long-standing abstractions (among other things) and works best in a clean codebase, so beware of using it uncritically here.

## 1. Instructions

An AI research codebase should have several things:

- *Data code*, including collecting, pre-processing, and post-processing datasets.
- *Training code*, which takes in a dataset and model class and trains the model on the dataset.
- *Inference code*, which runs your model on a new dataset and outputs the results.
- *Evaluation code*, which checks the performance of your model on a held-out dataset.

In this project you will build skeletons of all of these using the [PyTorch](https://pytorch.org/) package. We provide some abstractions you may use, but feel free to explore designs for yourself.

### Starting instructions.

Register at **[data8014.com](https://data8014.com/)**, open **Profile**, and link your GitHub account. Complete any prompts to join the `hku-data8014-students` GitHub organization. Use the website's **Create repository** interface, or ask the **AI TA**, to create your private repository in this organization. This repository will hold your semester-long codebase.

You can select it in a chat session and work on the code with the AI TA. Alternatively, clone it to your computer, develop locally using your preferred editor, and push your changes to GitHub.

When submitting a coding assignment on the website, provide your **repository name** and the **full commit SHA** identifying the version you want graded. Push that commit to GitHub before submitting; subsequent changes will not alter your submitted version.

If you are new to Git and GitHub, familiarize yourself with cloning, committing, pushing, and finding a commit SHA before starting the coursework.

We strongly recommend that you use [`uv`](https://docs.astral.sh/uv/) to manage your codebase. You should also learn the basics — ChatGPT knows `uv` well, so you can use it to help.

Set up the `pyproject.toml` file, as follows:

```toml
[project]
name = "Add your project name here"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"  # Feel free to modify this
dependencies = []

[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["Add your project name here"]
```

Once you fill out the project name[^1] and (optionally) description, you should make a project directory: a subdirectory of the Git directory whose name is the project name. For example, if your Git directory is `~/data8014` and the project name in the `pyproject.toml` is `codebase` then the project directory should be `~/data8014/codebase`. All of your main codebase should go in this folder, while experiments and one-off scripts can go outside.

[^1]: For this, you should use a single word or multiple words separated by underscores, with no spaces or non-alphanumeric characters.

Install PyTorch in your project by navigating to the Git directory in the command line and running the following command:

```bash
uv add torch --torch-backend=auto
```

### Data code.

PyTorch has some great abstractions for data already, including `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`. For now, your goal is to:

- Download a copy of the MNIST dataset. It should have 60,000 train examples and 10,000 test examples.
- Write a script which inverts the colors of MNIST images, takes a random crop, and shows the 10,000th, 20,000th, 30,000th, 40,000th, and 50,000th training image under these "augmentations". Hint: you can use `torchvision` transformations and a dataloader's `transform` property for this.
- Save these outputs and be prepared to put them in the project writeup.

### Training code.

A key and common abstraction for an AI codebase is a `Trainer`. A trainer is basically a way to take in a dataloader (a `torch.utils.data.DataLoader`) and model (a `torch.nn.Module`) and output a trained model. Usually the salient abstraction is one which takes a single training step, given a batch, and the current state of the model, and returns any relevant outputs, such as the predictions and other metrics. For now, your goal is to:

- Create a fake (multi-dimensional) Gaussian dataset with 10,000 inputs X∈R10,000×10**X**∈R10,000×10 of dimension 10, a true parameter θ⋆∈R10**θ**⋆∈R10 of dimension 10, noise ε∈R10,000∼N(0,0.01)**ε**∈R10,000∼N(0,0.01) and labels y=Xθ⋆+ε∈R10,000**y**=**X****θ**⋆+**ε**∈R10,000.
- Create a `Trainer`-type abstraction, and train a linear model (`torch.nn.Linear`) using stochastic gradient descent (`torch.optim.SGD`) on this data with squared error loss, saving the saved model checkpoint in `.pt` format.
- Implement several crucial metrics: train and validation loss (you need only validate every few steps), gradient norm, memory usage, time per step, dataloader throughput, sample throughput, and model-flop utilization (MFU). Log these throughout the training (you may use [WandB](https://wandb.ai/site/) or just a local text file logger). Save plots of these metrics and put them in the project writeup.

### Inference code.

Another abstraction is that of an inference engine (for which you might have the class `InferenceEngine` or `Inferencer`). This object is a way to take in a dataloader and model and output the predictions of this model on the dataloader. While in the simple case of linear models this looks identical to the training procedure, in practice an inference engine often has different optimal configurations (such as batch size and computational parallelism) compared to a trainer, so it is helpful to represent this distinction in code. Also, in certain cases such as language models, inference occurs via iterated stateful predictions, which is a mechanism not implemented in a training engine. These two distinctions are out of scope for this project. For now, your goal is to:

- Create an `InferenceEngine`-type abstraction, which loads a checkpoint and runs it on a dataloader to get predicted labels.
- Regenerate a new Gaussian dataset with the same ground truth parameter θ⋆**θ**⋆ as in the previous paragraph, and let the model predict the corresponding labels.
- Plot the true labels on the x*x* axis and predictions on the y*y* axis, visualizing the accuracy and bias of your trained model. Prepare to add this plot to the project writeup.

### Evaluation code.

The last abstraction is that of an evaluation runner. It may use the inference engine along with a trained model and assess many qualities of the model. A basic example of an evaluation is to compute the test loss, though we will have some more interesting ones in the future. For now, your goal is to:

- Create an `EvalRunner`-type abstraction, which loads a checkpoint and runs it on a dataloader to collect a set of evaluation metrics.
- Use the test Gaussian dataset from the previous part, and compute per-example test loss and gradient norm, and return as a metric how correlated the two quantities are with each other.
- Plot a histogram of these metrics over batches and prepare to add this plot to the project writeup.

### Writeup.

Congratulations! You have completed Assignment 0. Prepare a writeup in LaTeX, compile it to PDF, and include:

- The URL of your private repository in the `hku-data8014-students` GitHub organization;
- A brief description of your code structure and any abstractions you introduced;
- The required plots and accompanying explanations for each part of the assignment;
- A disclosure of your AI usage. Using AI is permitted; please explain where and how you used it.

Push your final code to GitHub. Then open Assignment 0 on [https://data8014.com](https://data8014.com/), upload your PDF writeup, and enter your repository name and the full commit SHA of the version you want graded. You can obtain the SHA by running `git rev-parse HEAD` in your local repository.

Click **Submit current work** and verify that a submission receipt appears. **Uploading files or sending chat messages alone does not submit your assignment. Your submission preserves the specified code version and writeup**; subsequent GitHub pushes will not change it.

Submit by the deadline displayed on the assignment page.