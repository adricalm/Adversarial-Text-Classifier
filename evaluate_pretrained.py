import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "protectai/deberta-v3-base-prompt-injection"
TEXT_COL = "text"
LABEL_COL = "label"


def safe_token_text(token: str) -> str:
    # Make tokenizer pieces printable in non-UTF8 terminals (e.g. Windows cp1252).
    return token.encode("ascii", errors="backslashreplace").decode("ascii")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.dropna(subset=[TEXT_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df.reset_index(drop=True)


@torch.no_grad()
def predict(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    frame: pd.DataFrame,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> tuple[list[int], list[float]]:
    model.eval()
    preds: list[int] = []
    probs_class1: list[float] = []
    texts = frame[TEXT_COL].tolist()

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        batch_preds = torch.argmax(probs, dim=-1)
        preds.extend(batch_preds.cpu().tolist())
        probs_class1.extend(probs[:, 1].cpu().tolist())

    return preds, probs_class1


def evaluate(
    frame: pd.DataFrame,
    preds: list[int],
) -> None:
    targets = frame[LABEL_COL].astype(int).tolist()
    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average="binary", zero_division=0)
    recall = recall_score(targets, preds, average="binary", zero_division=0)
    f1 = f1_score(targets, preds, average="binary", zero_division=0)
    cm = confusion_matrix(targets, preds)

    print("\n[Full dataset]")
    print(f"accuracy : {acc:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall   : {recall:.4f}")
    print(f"f1       : {f1:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)


def token_gradient_saliency(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    label: int,
    device: torch.device,
    max_length: int,
    top_k: int,
) -> dict:
    model.eval()
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    special_tokens_mask = encoded["special_tokens_mask"].squeeze(0).bool()

    embeds = model.get_input_embeddings()(input_ids).detach()
    embeds.requires_grad_(True)

    output = model(inputs_embeds=embeds, attention_mask=attention_mask)
    target = torch.tensor([label], dtype=torch.long, device=device)
    loss = F.cross_entropy(output.logits, target)

    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm = embeds.grad.detach().norm(dim=-1).squeeze(0).cpu()
    token_ids = input_ids.squeeze(0).cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    valid_positions = [i for i in range(len(tokens)) if not bool(special_tokens_mask[i])]
    sorted_positions = sorted(valid_positions, key=lambda i: float(grad_norm[i]), reverse=True)
    top_positions = sorted_positions[:top_k]

    top_tokens = [
        {"token": tokens[i], "score": float(grad_norm[i]), "position": int(i)}
        for i in top_positions
    ]
    max_score = float(grad_norm[top_positions[0]]) if top_positions else 0.0
    mean_score = float(grad_norm[valid_positions].mean()) if valid_positions else 0.0

    pred = int(torch.argmax(output.logits, dim=-1).item())
    pred_prob = float(torch.softmax(output.logits, dim=-1)[0, pred].item())

    return {
        "pred": pred,
        "pred_prob": pred_prob,
        "loss": float(loss.item()),
        "max_grad": max_score,
        "mean_grad": mean_score,
        "top_tokens": top_tokens,
    }


def run_gradient_study(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    frame: pd.DataFrame,
    device: torch.device,
    max_length: int,
    top_k: int,
    max_samples: int,
) -> None:
    n = min(len(frame), max_samples)
    aggregate = defaultdict(lambda: {"sum": 0.0, "count": 0})
    per_sample: list[dict] = []

    for i in range(n):
        row = frame.iloc[i]
        text = str(row[TEXT_COL])
        label = int(row[LABEL_COL])
        result = token_gradient_saliency(
            model=model,
            tokenizer=tokenizer,
            text=text,
            label=label,
            device=device,
            max_length=max_length,
            top_k=top_k,
        )
        per_sample.append({"idx": i, "label": label, "text": text, **result})
        for tok in result["top_tokens"]:
            aggregate[tok["token"]]["sum"] += tok["score"]
            aggregate[tok["token"]]["count"] += 1

    print(f"\n[Gradient Study | full dataset | first {n} samples]")
    avg_max = float(np.mean([x["max_grad"] for x in per_sample])) if per_sample else 0.0
    avg_mean = float(np.mean([x["mean_grad"] for x in per_sample])) if per_sample else 0.0
    print(f"average(max token grad norm): {avg_max:.6f}")
    print(f"average(mean token grad norm): {avg_mean:.6f}")

    ranked_tokens = sorted(
        (
            (tok, vals["sum"] / vals["count"], vals["count"])
            for tok, vals in aggregate.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:15]
    print("\nTop tokens by average gradient saliency:")
    for tok, score, count in ranked_tokens:
        safe_tok = safe_token_text(tok)
        print(f"  {safe_tok:<18} avg_grad={score:.6f} seen={count}")

    hardest = sorted(per_sample, key=lambda x: x["max_grad"], reverse=True)[:5]
    print("\nMost gradient-sensitive examples:")
    for item in hardest:
        status = "correct" if item["pred"] == item["label"] else "wrong"
        snippet = item["text"][:180].replace("\n", " ")
        print(
            f"- idx={item['idx']} label={item['label']} pred={item['pred']} ({status}) "
            f"max_grad={item['max_grad']:.6f} loss={item['loss']:.6f} prob={item['pred_prob']:.4f}"
        )
        print(f"  text: {snippet}...")
        print(
            "  top tokens: "
            + ", ".join(
                f"{safe_token_text(t['token'])}({t['score']:.4f})"
                for t in item["top_tokens"]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrained prompt-injection model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/final_dataset/final_binary_dataset.csv"),
        help="Path to final binary CSV.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max length.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument(
        "--gradient-samples",
        type=int,
        default=77,
        help="How many samples to include in gradient study (from start of dataset).",
    )
    parser.add_argument(
        "--top-k-tokens",
        type=int,
        default=8,
        help="Top-k tokens to print per sample for gradient saliency.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    df = load_dataset(args.dataset)
    print(f"Loaded dataset: {args.dataset.resolve()} ({len(df)} rows)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    preds, _ = predict(
        model=model,
        tokenizer=tokenizer,
        frame=df,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    evaluate(df, preds)

    run_gradient_study(
        model=model,
        tokenizer=tokenizer,
        frame=df,
        device=device,
        max_length=args.max_length,
        top_k=args.top_k_tokens,
        max_samples=args.gradient_samples,
    )


if __name__ == "__main__":
    main()
