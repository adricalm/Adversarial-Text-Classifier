# Adversarial Text Classifier

Prompt-injection detection using three classifiers: a TF-IDF logistic regression baseline, a pretrained DeBERTa-v3 model, and a fine-tuned version of it. Includes a FastAPI backend and React frontend for live inference.

## Setup

From the repo root (requires [uv](https://github.com/astral-sh/uv)):

```bash
uv sync
```

> CUDA-enabled PyTorch is pinned in `pyproject.toml`. Use CPU wheels if you don't have an NVIDIA GPU.

Model checkpoints are **not** included in the repo. You need to either train them (see below) or obtain them separately and place them in `Models/`:

| File | Description |
|---|---|
| `Models/best_checkpoint_TF_IDF.pt` | TF-IDF logistic regression weights |
| `Models/tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `Models/finetuned_deberta-v2_best.pt` | Fine-tuned DeBERTa weights |

---

## Notebooks

### 1. Pretrained DeBERTa — `pretrained/pretrained_test.ipynb`

The main experiment notebook. Covers:
- Loading the dataset (`data/final_dataset/`)
- Evaluating the pretrained `protectai/deberta-v3-base-prompt-injection-v2` model
- Fine-tuning it on the combined dataset
- Gradient-based token attribution analysis

Run all cells in order. The fine-tuned checkpoint is saved to `Models/finetuned_deberta-v2_best.pt`.

### 2. TF-IDF Baseline — `TF_IDF/TF_IDF_Training.ipynb`

Trains a logistic regression classifier on TF-IDF features. Run all cells to produce:
- `Models/best_checkpoint_TF_IDF.pt`
- `Models/tfidf_vectorizer.pkl`

---

## Backend (FastAPI)

Serves all three models (TF-IDF, pretrained DeBERTa, fine-tuned DeBERTa) with gradient-based explanations.

```bash
cd Backend
pip install -r requirements.txt
uvicorn main:app --reload
```

API runs at `http://localhost:8000`. Endpoints:
- `GET /health` — checks if models are loaded
- `POST /predict` — `{"text": "your prompt here"}`

> Requires all model files in `Models/` to be present before starting.

---

## Frontend (React + Vite)

```bash
cd Frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. Make sure the backend is running first.
