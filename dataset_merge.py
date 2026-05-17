"""Build merged English binary dataset from Hugging Face parquets (Lingua) + Kaggle JSONL (context + attack rules).

CLI entrypoints live in `scripts/build_merged_dataset.py` and `scripts/build_final_dataset.py`.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

TEXT_COL = "text"
LABEL_COL = "label"
SOURCE_COL = "source"
ORIGIN_ID_COL = "origin_id"
ATTACK_TYPE_COL = "attack_type"
CONTEXT_COL = "context"

POSITIVE_ATTACK_TYPES = frozenset({"jailbreaking", "data_leakage"})

_LANGUAGE_TOKEN = re.compile(r"\b([A-Za-z]+)-language\b", re.IGNORECASE)


def context_indicates_non_english(context: str | None) -> bool:
    """True if context mentions a non-English *-language tag (e.g. French-language)."""
    if not context:
        return False
    for m in _LANGUAGE_TOKEN.finditer(context):
        if m.group(1).lower() != "english":
            return True
    return False


def build_lang_detector():
    return LanguageDetectorBuilder.from_all_languages().build()


def split_english_and_holdout(
    frame: pd.DataFrame,
    text_col: str,
    label_col: str,
    *,
    detector,
    min_chars: int,
    holdout_path: Path,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_idx: list[int] = []
    holdout_rows: list[dict[str, Any]] = []

    for i, row in frame.iterrows():
        raw_text = row[text_col]
        text = str(raw_text).strip()

        if len(text) < min_chars:
            holdout_rows.append(
                {
                    text_col: raw_text,
                    label_col: row[label_col],
                    "detected_language": "",
                    "filter_reason": "too_short_for_detection",
                }
            )
            continue

        lang = detector.detect_language_of(text)
        if lang is None:
            holdout_rows.append(
                {
                    text_col: raw_text,
                    label_col: row[label_col],
                    "detected_language": "",
                    "filter_reason": "ambiguous_or_unknown",
                }
            )
        elif lang != Language.ENGLISH:
            holdout_rows.append(
                {
                    text_col: raw_text,
                    label_col: row[label_col],
                    "detected_language": lang.name,
                    "filter_reason": "non_english",
                }
            )
        else:
            keep_idx.append(i)

    kept = frame.loc[keep_idx].reset_index(drop=True)
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(holdout_path, index=False, encoding="utf-8-sig")

    if verbose:
        print(f"English rows kept: {len(kept)} / {len(frame)}")
        if len(holdout_df):
            print("Holdout breakdown:\n", holdout_df["filter_reason"].value_counts())

    return kept, holdout_df


def load_english_parquet_subset(
    parquet_dir: Path,
    *,
    detector,
    min_chars: int,
    non_english_holdout_path: Path,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    verbose: bool = True,
) -> pd.DataFrame:
    df = pd.concat(
        [
            pd.read_parquet(parquet_dir / "train.parquet"),
            pd.read_parquet(parquet_dir / "test.parquet"),
        ],
        axis=0,
    ).reset_index(drop=True)
    if verbose:
        print(f"Merged HF parquets: {len(df)} rows")

    df[label_col] = df[label_col].astype(int)
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)

    kept, _ = split_english_and_holdout(
        df,
        text_col,
        label_col,
        detector=detector,
        min_chars=min_chars,
        holdout_path=non_english_holdout_path,
        verbose=verbose,
    )

    out = kept.copy()
    out[SOURCE_COL] = "hf"
    out[ORIGIN_ID_COL] = out.index.astype(str)
    out[ATTACK_TYPE_COL] = ""
    out[CONTEXT_COL] = ""
    return out


def load_jsonl_binary_subset(
    jsonl_path: Path,
    *,
    excluded_malicious_path: Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_json(jsonl_path, lines=True)
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    stats = {
        "jsonl_total": len(raw),
        "jsonl_non_english_context": 0,
        "jsonl_benign_kept": 0,
        "jsonl_positive_kept": 0,
        "jsonl_malicious_excluded_attack": 0,
        "jsonl_unknown_label": 0,
        "jsonl_missing_prompt": 0,
    }

    for _, rec in raw.iterrows():
        context = rec.get("context")
        if context_indicates_non_english(str(context) if context is not None else ""):
            stats["jsonl_non_english_context"] += 1
            continue

        label_raw = str(rec.get("label", "")).strip().lower()
        attack_type = str(rec.get("attack_type", "")).strip().lower()
        prompt = rec.get("prompt")
        if prompt is None or (isinstance(prompt, float) and pd.isna(prompt)):
            stats["jsonl_missing_prompt"] += 1
            continue
        rid = rec.get("id", "")

        if label_raw == "benign":
            rows.append(
                {
                    TEXT_COL: prompt,
                    LABEL_COL: 0,
                    SOURCE_COL: "kaggle",
                    ORIGIN_ID_COL: rid,
                    ATTACK_TYPE_COL: attack_type,
                    CONTEXT_COL: context if context is not None else "",
                }
            )
            stats["jsonl_benign_kept"] += 1
        elif label_raw == "malicious":
            if attack_type in POSITIVE_ATTACK_TYPES:
                rows.append(
                    {
                        TEXT_COL: prompt,
                        LABEL_COL: 1,
                        SOURCE_COL: "kaggle",
                        ORIGIN_ID_COL: rid,
                        ATTACK_TYPE_COL: attack_type,
                        CONTEXT_COL: context if context is not None else "",
                    }
                )
                stats["jsonl_positive_kept"] += 1
            else:
                stats["jsonl_malicious_excluded_attack"] += 1
                excluded.append(
                    {
                        "id": rid,
                        "prompt": prompt,
                        "attack_type": attack_type,
                        "context": context,
                        "label": label_raw,
                    }
                )
        else:
            stats["jsonl_unknown_label"] += 1

    out = pd.DataFrame(rows)
    if excluded_malicious_path is not None and excluded:
        excluded_malicious_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(excluded).to_csv(
            excluded_malicious_path, index=False, encoding="utf-8-sig"
        )
        if verbose:
            print(
                f"Wrote excluded malicious (non-target attack types) to {excluded_malicious_path.resolve()} ({len(excluded)} rows)"
            )

    if verbose:
        print("JSONL summary:", stats)

    return out, stats


def build_merged_dataframe(
    hf_df: pd.DataFrame,
    kaggle_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.concat([hf_df, kaggle_df], axis=0, ignore_index=True)
    return merged


def write_merged_csv(
    merged: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False, encoding="utf-8-sig")


def compute_merge_statistics(
    merged: pd.DataFrame,
    jsonl_stage_stats: dict[str, int],
) -> dict[str, Any]:
    """Summarize merged rows: source counts, labels overall/per source, Kaggle positive attack mix."""
    total = len(merged)
    by_src = merged[SOURCE_COL].value_counts()
    hf_n = int(by_src.get("hf", 0))
    kg_n = int(by_src.get("kaggle", 0))

    def label_counts(frame: pd.DataFrame) -> dict[int, int]:
        vc = frame[LABEL_COL].value_counts().sort_index()
        return {int(k): int(v) for k, v in vc.items()}

    per_source = {
        "hf": label_counts(merged[merged[SOURCE_COL] == "hf"]),
        "kaggle": label_counts(merged[merged[SOURCE_COL] == "kaggle"]),
    }
    overall = label_counts(merged)

    kaggle_pos = merged[
        (merged[SOURCE_COL] == "kaggle") & (merged[LABEL_COL] == 1)
    ]
    kaggle_pos_n = len(kaggle_pos)
    atk_vc = kaggle_pos[ATTACK_TYPE_COL].value_counts()
    jb = int(atk_vc.get("jailbreaking", 0))
    dl = int(atk_vc.get("data_leakage", 0))

    def share(n: int) -> float:
        return round(n / kaggle_pos_n, 6) if kaggle_pos_n else 0.0

    return {
        "row_counts": {"total": total, "hf": hf_n, "kaggle": kg_n},
        "labels": {"overall": overall, "per_source": per_source},
        "kaggle_positive_attack_types": {
            "counts": {"jailbreaking": jb, "data_leakage": dl},
            "total_kaggle_positives": kaggle_pos_n,
            "share_within_kaggle_positives": {
                "jailbreaking": share(jb),
                "data_leakage": share(dl),
            },
        },
        "jsonl_pipeline": dict(jsonl_stage_stats),
    }


def write_merge_stats(stats: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


# --- Final dataset: base CSV (updated_merged) + curated JSONL ----------------------------

FINAL_DATASET_EXTRA_COLS = (
    "category",
    "language",
    "difficulty",
    "length_bucket",
    "notes",
    "suggested_split",
)

FINAL_DATASET_COLUMNS: tuple[str, ...] = (
    TEXT_COL,
    LABEL_COL,
    SOURCE_COL,
    ORIGIN_ID_COL,
    ATTACK_TYPE_COL,
    CONTEXT_COL,
) + FINAL_DATASET_EXTRA_COLS


def prepare_base_csv_for_final(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure base rows match the final rectangular schema; drop unknown columns with a warning."""
    core = (
        TEXT_COL,
        LABEL_COL,
        SOURCE_COL,
        ORIGIN_ID_COL,
        ATTACK_TYPE_COL,
        CONTEXT_COL,
    )
    missing = [c for c in core if c not in df.columns]
    if missing:
        raise ValueError(f"Base CSV missing required columns: {missing}")

    unknown = [c for c in df.columns if c not in FINAL_DATASET_COLUMNS]
    if unknown:
        warnings.warn(f"Dropping unexpected base CSV columns: {unknown}", stacklevel=2)

    out = df.loc[:, list(core)].copy()

    for c in FINAL_DATASET_EXTRA_COLS:
        out[c] = df[c] if c in df.columns else ""

    for c in FINAL_DATASET_EXTRA_COLS:
        out[c] = out[c].fillna("").astype(str)

    return out[list(FINAL_DATASET_COLUMNS)]


def load_curated_jsonl_for_final(
    jsonl_path: Path,
    *,
    source_fixed: str = "curated",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load English-only curated JSONL rows into the final-dataset schema."""
    raw = pd.read_json(jsonl_path, lines=True)
    raw_n = len(raw)
    meta: dict[str, Any] = {
        "jsonl_path": str(jsonl_path.resolve()),
        "raw_line_count": raw_n,
        "dropped_non_english": 0,
        "dropped_bad_label": 0,
        "dropped_missing_prompt": 0,
    }

    if "language" not in raw.columns:
        raise ValueError("Curated JSONL must include a 'language' column")

    lang_mask = raw["language"].astype(str).str.lower() == "en"
    meta["dropped_non_english"] = int((~lang_mask).sum())
    filt = raw.loc[lang_mask].copy()

    rows: list[dict[str, Any]] = []
    for _, rec in filt.iterrows():
        prompt = rec.get("prompt")
        if prompt is None or (isinstance(prompt, float) and pd.isna(prompt)):
            meta["dropped_missing_prompt"] += 1
            continue
        label_raw = rec.get("label")
        try:
            label_i = int(label_raw)
        except (TypeError, ValueError):
            meta["dropped_bad_label"] += 1
            continue
        if label_i not in (0, 1):
            meta["dropped_bad_label"] += 1
            continue

        rid = rec.get("id", "")
        at = rec.get("attack_type", "")
        at_s = "" if at is None or (isinstance(at, float) and pd.isna(at)) else str(at)

        def _str_col(key: str) -> str:
            v = rec.get(key)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            return str(v)

        rows.append(
            {
                TEXT_COL: str(prompt).strip(),
                LABEL_COL: label_i,
                SOURCE_COL: source_fixed,
                ORIGIN_ID_COL: str(rid),
                ATTACK_TYPE_COL: at_s,
                CONTEXT_COL: "",
                "category": _str_col("category"),
                "language": _str_col("language"),
                "difficulty": _str_col("difficulty"),
                "length_bucket": _str_col("length_bucket"),
                "notes": _str_col("notes"),
                "suggested_split": _str_col("suggested_split"),
            }
        )

    out = pd.DataFrame(rows)
    if list(out.columns) != list(FINAL_DATASET_COLUMNS):
        out = out.reindex(columns=list(FINAL_DATASET_COLUMNS))

    kept = len(out)
    meta["rows_after_en_and_validation"] = kept
    if verbose:
        print("Curated JSONL:", meta)
        if kept != 180:
            print(
                f"Note: expected 180 curated rows after filtering; got {kept} (see drops above)."
            )

    return out, meta


def _vc_to_ordered_dict(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def compute_final_dataset_statistics(
    final_df: pd.DataFrame,
    jsonl_meta: dict[str, Any],
    base_text_set: set[str],
) -> dict[str, Any]:
    """Counts by source (hf / kaggle / curated), labels, curated-only breakdowns, overlap with base."""
    total = len(final_df)
    by_src = final_df[SOURCE_COL].value_counts()
    row_counts = {
        "total": total,
        "hf": int(by_src.get("hf", 0)),
        "kaggle": int(by_src.get("kaggle", 0)),
        "curated": int(by_src.get("curated", 0)),
    }

    def label_counts(frame: pd.DataFrame) -> dict[int, int]:
        if len(frame) == 0:
            return {}
        vc = frame[LABEL_COL].value_counts().sort_index()
        return {int(k): int(v) for k, v in vc.items()}

    per_source = {
        "hf": label_counts(final_df[final_df[SOURCE_COL] == "hf"]),
        "kaggle": label_counts(final_df[final_df[SOURCE_COL] == "kaggle"]),
        "curated": label_counts(final_df[final_df[SOURCE_COL] == "curated"]),
    }
    overall = label_counts(final_df)

    cur = final_df[final_df[SOURCE_COL] == "curated"]
    label_by_category: dict[str, dict[int, int]] = {}
    if len(cur):
        for cat, g in cur.groupby("category", dropna=False):
            if pd.isna(cat) or cat == "":
                key = "(empty)"
            else:
                key = str(cat)
            label_by_category[key] = {
                int(k): int(v) for k, v in g[LABEL_COL].value_counts().items()
            }

    curated_only: dict[str, Any] = {
        "row_count": len(cur),
        "by_category": _vc_to_ordered_dict(cur["category"])
        if len(cur) and "category" in cur.columns
        else {},
        "by_attack_type": _vc_to_ordered_dict(cur[ATTACK_TYPE_COL])
        if len(cur)
        else {},
        "by_difficulty": _vc_to_ordered_dict(cur["difficulty"])
        if len(cur) and "difficulty" in cur.columns
        else {},
        "by_length_bucket": _vc_to_ordered_dict(cur["length_bucket"])
        if len(cur) and "length_bucket" in cur.columns
        else {},
        "label_by_category": {
            k: {str(int(lb)): n for lb, n in v.items()}
            for k, v in label_by_category.items()
        },
    }

    dup = 0
    if len(cur):
        dup = int(
            cur[TEXT_COL]
            .astype(str)
            .isin(base_text_set)
            .sum()
        )

    return {
        "row_counts": row_counts,
        "labels": {"overall": overall, "per_source": per_source},
        "curated_only": curated_only,
        "jsonl_input": jsonl_meta,
        "duplicate_text_vs_base": dup,
    }


def write_final_dataset_stats(stats: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
