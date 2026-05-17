# Adversarial Text Classifier

Prompt-injection detection with a small Transformer encoder classifier trained from scratch on a custom English binary dataset (HF + Kaggle → hand-edited merge + curated JSONL).

## Setup

From the repository root, install dependencies (example using [uv](https://github.com/astral-sh/uv)):

```bash
uv sync
```

The project pins a CUDA-enabled PyTorch index in `pyproject.toml`. Adjust or use the default CPU wheels if you do not use NVIDIA CUDA.

## Data pipeline

Place large inputs under `data/raw/` (not tracked in Git):

- `data/raw/hf/train.parquet`, `data/raw/hf/test.parquet` — Hugging Face splits.
- `data/raw/kaggle/Prompt_INJECTION_And_Benign_DATASET.jsonl` — Kaggle-style JSONL.
- `data/raw/curated_prompt_injection_dataset_v1.jsonl` — your curated English rows (required for the final merge).

**1. Derived merge** (English-only HF + filtered Kaggle) — writes `data/derived/` including `merged_binary_dataset.csv` and diagnostics:

```bash
python scripts/build_merged_dataset.py
```

Hand-edit or replace the derived table as needed; the training pipeline expects the hand-finished base as **`data/derived/updated_merged.csv`**.

**2. Final training dataset** — merges `updated_merged.csv` with the curated JSONL and writes the version-controlled release files:

- `data/final_dataset/final_binary_dataset.csv`
- `data/final_dataset/final_dataset_stats.json`

```bash
python scripts/build_final_dataset.py
```

Intermediate and bulky paths (`data/raw/`, `data/derived/`, `data/corpus.txt`, `data/tokenizer.json`) are listed in `.gitignore`. Only the **final** CSV + stats JSON are intended as reproducible artifacts in Git.

## Training and manual testing

1. Open **`notebooks/binary_transformer.ipynb`**. The first cell changes the working directory to the repo root so paths work when the notebook lives under `notebooks/`. It loads **`data/final_dataset/final_binary_dataset.csv`**, trains the BPE tokenizer to **`data/tokenizer.json`**, and writes **`best_checkpoint.pt`** in the repo root (checkpoint files are gitignored).

2. Open **`notebooks/manual_prompt_test.ipynb`** to load **`best_checkpoint.pt`** and **`data/tokenizer.json`** and run ad hoc strings.

If you clone a fresh copy, you must run the training notebook (or copy a checkpoint + tokenizer from elsewhere). Checkpoints are not committed.

## Public repository note

`final_binary_dataset.csv` contains full prompt text derived from public/adapted sources and manual curation. Confirm licensing and privacy expectations before publishing a **public** repository.
