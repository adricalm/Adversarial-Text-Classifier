# Adversarial Text Classifier

Prompt-injection detection with a small Transformer encoder classifier trained from scratch. The release dataset is **`data/final_dataset/final_binary_dataset.csv`** (plus `final_dataset_stats.json` for counts).

## Setup

From the repository root (example with [uv](https://github.com/astral-sh/uv)):

```bash
uv sync
```

CUDA-enabled PyTorch is pinned via `pyproject.toml`. Use CPU wheels from PyPI instead if you do not use NVIDIA CUDA.

## Training and manual testing

1. **`notebooks/binary_transformer.ipynb`** — First code cell sets the working directory to the repo root. It loads **`data/final_dataset/final_binary_dataset.csv`**, fits a BPE tokenizer to **`data/tokenizer.json`**, and writes **`best_checkpoint.pt`** in the repo root (artifacts are gitignored).

2. **`notebooks/manual_prompt_test.ipynb`** — Loads **`best_checkpoint.pt`** and **`data/tokenizer.json`** for ad hoc prompts.

After a fresh clone you need either a local training run or a copied checkpoint + tokenizer. Model code lives in **`transformer.py`**, **`tokenizer.py`**, **`self_attention.py`**.

## Public repository note

`final_binary_dataset.csv` contains full prompt text. Confirm licensing and privacy expectations before a **public** repository.
