"""
Model loading and inference with gradient-based explanations.

TF-IDF:   signed attribution  = grad × input  (pos = toward INJECTION)
DeBERTa:  gradient norm per token              (always positive, shows sensitivity)
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent          # repo root
MODELS_DIR = BASE_DIR / "Models"
DATA_DIR = BASE_DIR / "data" / "final_dataset"
VECTORIZER_CACHE = MODELS_DIR / "tfidf_vectorizer.pkl"

PRETRAINED_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
TORCH_SEED = 1
TRAIN_SPLIT = 0.7

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


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _load_csvs(names: list[str]) -> pd.DataFrame:
    frames = []
    for name in names:
        path = DATA_DIR / name
        if not path.exists():
            logger.warning("Dataset not found, skipping: %s", path)
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
        frames.append(df[["text", "label"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["text", "label"])


def _fit_and_cache_vectorizer() -> TfidfVectorizer:
    """Reproduce the exact training split used in TF_IDF_Training.ipynb and fit vectorizer.

    Must mirror the three-shuffle sequence from the notebook:
      1. shuffle df   (seed=TORCH_SEED)
      2. shuffle df_ft (seed=TORCH_SEED)
      3. concat, then shuffle combined (seed=TORCH_SEED)
    Skipping either of the individual shuffles produces a different train_comb
    and therefore different IDF values, breaking the saved model weights.
    """
    logger.info("Fitting TF-IDF vectorizer from training data (one-time setup)…")

    # Steps 1 & 2 — shuffle each group individually before concat (matches notebook)
    df_main = _load_csvs(MAIN_CSVS).sample(frac=1, random_state=TORCH_SEED).reset_index(drop=True)
    df_ft   = _load_csvs(FINETUNE_CSVS).sample(frac=1, random_state=TORCH_SEED).reset_index(drop=True)

    # Step 3 — concat then shuffle combined
    common_cols = ["text", "label"]
    combined = (
        pd.concat([df_main[common_cols], df_ft[common_cols]], ignore_index=True)
        .sample(frac=1, random_state=TORCH_SEED)
        .reset_index(drop=True)
    )

    n = len(combined)
    train_texts = combined.iloc[: int(TRAIN_SPLIT * n)]["text"].tolist()

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 1))
    vectorizer.fit(train_texts)

    with open(VECTORIZER_CACHE, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Vectorizer saved to %s", VECTORIZER_CACHE)
    return vectorizer


class Predictor:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        logger.info("Loading TF-IDF model…")
        self.tfidf_model, self.tfidf_vectorizer = self._load_tfidf()
        logger.info("Loading tokenizer…")
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        logger.info("Loading pretrained DeBERTa…")
        self.pretrained_model = self._load_pretrained_deberta()
        logger.info("Loading finetuned DeBERTa…")
        self.finetuned_model = self._load_finetuned_deberta()
        logger.info("All models loaded.")

    # ------------------------------------------------------------------ #
    # Model loading helpers                                                #
    # ------------------------------------------------------------------ #

    def _load_tfidf(self) -> tuple[LogisticRegressionModel, TfidfVectorizer]:
        ckpt_path = MODELS_DIR / "best_checkpoint_TF_IDF.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

        input_dim = state_dict["linear.weight"].shape[1]
        num_classes = state_dict["linear.weight"].shape[0]
        model = LogisticRegressionModel(input_dim, num_classes)
        model.load_state_dict(state_dict)
        model.eval()

        if not VECTORIZER_CACHE.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {VECTORIZER_CACHE}. "
                "Generate it from the training notebook and place it in Models/."
            )
        with open(VECTORIZER_CACHE, "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    def _load_pretrained_deberta(self) -> AutoModelForSequenceClassification:
        model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)
        model.to(self.device).eval()
        return model

    def _load_finetuned_deberta(self) -> AutoModelForSequenceClassification:
        ckpt_path = MODELS_DIR / "finetuned_deberta-v2_best.pt"
        model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Checkpoint is a raw OrderedDict state dict (no wrapper key)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def _tfidf_predict(self, text: str) -> dict:
        x_np = self.tfidf_vectorizer.transform([text]).toarray()  # (1, 10000)
        x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        logits = self.tfidf_model(x)
        pred = int(logits.argmax(dim=1).item())
        probs = torch.softmax(logits, dim=1)
        confidence = float(probs[0, pred].item())

        logits[0, pred].backward()
        # L2 norm of gradient per feature (scalar features → absolute value)
        attrs = x.grad.abs().squeeze(0).detach().numpy()  # (10000,)

        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        nonzero_idx = x_np[0].nonzero()[0]

        items = [
            {"word": feature_names[i], "score": float(attrs[i])}
            for i in nonzero_idx
        ]
        items.sort(key=lambda w: w["score"], reverse=True)

        return {
            "label": "INJECTION" if pred == 1 else "SAFE",
            "confidence": confidence,
            "items": items[:30],
        }

    def _deberta_predict(self, text: str, model: AutoModelForSequenceClassification) -> dict:
        enc = self.tokenizer(
            text, truncation=True, max_length=512, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        # Standard forward pass for correct prediction (matches notebook's predict_text)
        with torch.no_grad():
            std_logits = model(**enc).logits
            pred = int(std_logits.argmax(dim=1).item())
            probs = torch.softmax(std_logits, dim=1)
            confidence = float(probs[0, pred].item())

        # Separate inputs_embeds pass solely for gradient extraction
        embeds = model.deberta.embeddings.word_embeddings(enc["input_ids"])
        embeds = embeds.detach().requires_grad_(True)
        grad_logits = model(inputs_embeds=embeds, attention_mask=enc["attention_mask"]).logits
        grad_logits[0, pred].backward()

        # L2 norm of embedding gradient per token — how sensitive the model is to each token
        grad_norm = embeds.grad.norm(dim=-1).squeeze(0).tolist()  # (seq_len,)
        raw_tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        special = set(self.tokenizer.all_special_tokens)

        # Merge SentencePiece subword tokens (▁ marks word start); take max norm across pieces
        items = []
        current_word = ""
        current_scores: list[float] = []

        for token, score in zip(raw_tokens, grad_norm):
            if token in special:
                if current_word:
                    items.append({"word": current_word, "score": max(current_scores)})
                    current_word, current_scores = "", []
                continue
            if token.startswith("\u2581"):   # ▁ = SentencePiece word boundary
                if current_word:
                    items.append({"word": current_word, "score": max(current_scores)})
                current_word = token[1:] or token
                current_scores = [score]
            else:
                current_word += token
                current_scores.append(score)

        if current_word:
            items.append({"word": current_word, "score": max(current_scores)})

        return {
            "label": "INJECTION" if pred == 1 else "SAFE",
            "confidence": confidence,
            "items": items,
        }

    def predict_all(self, text: str) -> dict:
        return {
            "tfidf": self._tfidf_predict(text),
            "pretrained": self._deberta_predict(text, self.pretrained_model),
            "finetuned": self._deberta_predict(text, self.finetuned_model),
        }
