"""
Evaluate all three models on the test split that exactly mirrors pretrained_test.ipynb:
  - TORCH_SEED = 10  (matches pretrained_test.ipynb, NOT the TF-IDF notebook which uses 1)
  - Combined dataset: MAIN_CSVS + FINETUNE_CSVS
  - 70 / 15 / 15 split, shuffled with random_state=TORCH_SEED
  - test_comb = combined_df.iloc[int(0.85 * n):]

Prints accuracy, precision, recall, F1, and confusion matrix for each model.
"""

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "Models"
DATA_DIR   = BASE_DIR / "data" / "final_dataset"

PRETRAINED_MODEL    = "protectai/deberta-v3-base-prompt-injection-v2"
TFIDF_CKPT          = MODELS_DIR / "best_checkpoint_TF_IDF.pt"
FINETUNED_CKPT      = MODELS_DIR / "finetuned_deberta-v2_best.pt"
VECTORIZER_CACHE    = MODELS_DIR / "tfidf_vectorizer.pkl"

# ── Reproducibility (must match pretrained_test.ipynb exactly) ───────────────
TORCH_SEED   = 10
TRAIN_SPLIT  = 0.7
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
TEXT_COL     = "text"
LABEL_COL    = "label"
LABEL_NAME   = {0: "SAFE", 1: "INJECTION"}

torch.manual_seed(TORCH_SEED)
random.seed(TORCH_SEED)
np.random.seed(TORCH_SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps"  if torch.backends.mps.is_available()
    else "cpu"
)

# ── Dataset CSV lists (identical to pretrained_test.ipynb) ───────────────────
MAIN_CSVS = [
    "final_binary_dataset.csv",
    "clanker_dataset_hard_1.csv",
    "clanker_dataset_hard_2.csv",
    "clanker_dataset_hard_3.csv",
    "clanker_dataset_4.csv",
    "clanker_dataset_medium_5.csv",
    "clanker_dataset_long_6.csv",
    "clanker_dataset_long_7.csv",
    "clanker_dataset_long_8.csv",
    "clanker_hard_examples_v9.csv",
    "clanker_hard_examples_v10.csv",
    "clanker_hard_examples_v11.csv",
    "clanker_hard_examples_v12.csv",
]

FINETUNE_CSVS = [
    "context_finetune_dataset.csv",
    "context_finetune_dataset_v2.csv",
    "context_finetune_dataset_v3.csv",
    "context_finetune_dataset_v4.csv",
    "context_finetune_dataset_v5.csv",
    "context_finetune_dataset_v6.csv",
    "context_finetune_dataset_v7.csv",
    "context_finetune_dataset_v8.csv",
]


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_csvs(names: list[str]) -> pd.DataFrame:
    frames = []
    for name in names:
        path = DATA_DIR / name
        if not path.exists():
            print(f"  [warn] missing {path.name}, skipping")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.dropna(subset=[TEXT_COL])
        df[TEXT_COL]  = df[TEXT_COL].astype(str)
        df[LABEL_COL] = df[LABEL_COL].astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


ADVERSARIAL_CSV = "adversarial_tfidf_hard.csv"


def _append_adversarial(test_comb: pd.DataFrame) -> pd.DataFrame:
    path = DATA_DIR / ADVERSARIAL_CSV
    if not path.exists():
        print(f"  [warn] adversarial file not found, skipping: {path}")
        return test_comb
    adv_df = pd.read_csv(path, encoding="utf-8-sig")
    adv_df = adv_df.dropna(subset=[TEXT_COL])
    adv_df[TEXT_COL]  = adv_df[TEXT_COL].astype(str)
    adv_df[LABEL_COL] = adv_df[LABEL_COL].astype(int)
    test_comb = pd.concat([test_comb, adv_df[[TEXT_COL, LABEL_COL]]], ignore_index=True)
    print(f"  + {len(adv_df)} rows appended from {ADVERSARIAL_CSV}")
    return test_comb


def build_test_split() -> pd.DataFrame:
    """Reproduce test_comb from pretrained_test.ipynb (seed=10), then append adversarial rows.

    The notebook shuffles df and df_ft independently before concatenating,
    so we must mirror that exact sequence to get the same test rows.
    """
    # Step 1: load and shuffle each group independently (matches notebook)
    df    = _load_csvs(MAIN_CSVS).sample(frac=1, random_state=TORCH_SEED).reset_index(drop=True)
    df_ft = _load_csvs(FINETUNE_CSVS).sample(frac=1, random_state=TORCH_SEED).reset_index(drop=True)

    # Step 2: concat using common columns, then shuffle the combined frame
    common_cols = [c for c in df.columns if c in df_ft.columns]
    combined_df = pd.concat(
        [df[common_cols], df_ft[common_cols]], ignore_index=True
    ).sample(frac=1, random_state=TORCH_SEED).reset_index(drop=True)

    n = len(combined_df)
    test_comb = combined_df.iloc[int((1 - TEST_SPLIT) * n):]
    test_comb = _append_adversarial(test_comb)

    print(f"Test set: {len(test_comb)} samples  "
          f"(SAFE={(test_comb[LABEL_COL]==0).sum()}, "
          f"INJECTION={(test_comb[LABEL_COL]==1).sum()})")
    return test_comb


# ── Model helpers ─────────────────────────────────────────────────────────────

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def load_tfidf():
    ckpt       = torch.load(TFIDF_CKPT, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    input_dim  = state_dict["linear.weight"].shape[1]
    model      = LogisticRegressionModel(input_dim)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    if VECTORIZER_CACHE.exists():
        with open(VECTORIZER_CACHE, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at {VECTORIZER_CACHE}. "
            "Start the backend once to auto-generate it, or run inference.py directly."
        )
    return model, vectorizer


def load_deberta(checkpoint: Path | None = None):
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_tfidf(
    model: LogisticRegressionModel,
    vectorizer: TfidfVectorizer,
    texts: list[str],
    batch_size: int = 512,
) -> list[int]:
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="TF-IDF"):
        batch   = texts[i : i + batch_size]
        x       = vectorizer.transform(batch).toarray()
        x_t     = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits = model(x_t)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


def predict_deberta(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int = 32,
) -> list[int]:
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="DeBERTa"):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(name: str, targets: list[int], preds: list[int]) -> None:
    acc  = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average="binary", zero_division=0)
    rec  = recall_score(targets, preds, average="binary", zero_division=0)
    f1   = f1_score(targets, preds, average="binary", zero_division=0)
    cm   = confusion_matrix(targets, preds)

    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"            pred SAFE  pred INJ")
    print(f"  true SAFE     {cm[0,0]:5d}     {cm[0,1]:5d}")
    print(f"  true INJ      {cm[1,0]:5d}     {cm[1,1]:5d}")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Seed  : {TORCH_SEED}  (matches pretrained_test.ipynb)\n")

    test_df = build_test_split()
    texts   = test_df[TEXT_COL].tolist()
    targets = test_df[LABEL_COL].astype(int).tolist()

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    print("\nLoading TF-IDF model…")
    tfidf_model, vectorizer = load_tfidf()
    tfidf_preds = predict_tfidf(tfidf_model, vectorizer, texts)
    report("TF-IDF + Logistic Regression", targets, tfidf_preds)

    # ── Tokenizer (shared by both DeBERTa models) ────────────────────────────
    print("\nLoading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    # ── Pretrained DeBERTa ───────────────────────────────────────────────────
    print("\nLoading pretrained DeBERTa…")
    pt_model   = load_deberta(checkpoint=None)
    pt_preds   = predict_deberta(pt_model, tokenizer, texts)
    report("Pretrained DeBERTa-v3  (protectai/deberta-v3-base-prompt-injection-v2)",
           targets, pt_preds)

    # ── Finetuned DeBERTa ────────────────────────────────────────────────────
    print("\nLoading finetuned DeBERTa…")
    ft_model   = load_deberta(checkpoint=FINETUNED_CKPT)
    ft_preds   = predict_deberta(ft_model, tokenizer, texts)
    report(f"Finetuned DeBERTa-v3  ({FINETUNED_CKPT.name})", targets, ft_preds)


if __name__ == "__main__":
    main()
