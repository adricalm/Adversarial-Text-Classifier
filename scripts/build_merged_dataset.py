"""Merge English-only HF parquets with filtered Kaggle JSONL into data/derived/.

Place inputs under:
  data/raw/hf/train.parquet, data/raw/hf/test.parquet
  data/raw/kaggle/Prompt_INJECTION_And_Benign_DATASET.jsonl
Outputs (UTF-8 with BOM for CSVs) and merge_stats.json: data/derived/

Hand-finished training CSV (HF + Kaggle + curated): build via
  python scripts/build_final_dataset.py
which writes data/final_dataset/.

Run from the repository root: python scripts/build_merged_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_merge import (
    LABEL_COL,
    SOURCE_COL,
    build_lang_detector,
    build_merged_dataframe,
    compute_merge_statistics,
    load_english_parquet_subset,
    load_jsonl_binary_subset,
    write_merge_stats,
    write_merged_csv,
)

DATA_DIR = ROOT / "data"
RAW_HF_DIR = DATA_DIR / "raw" / "hf"
RAW_KAGGLE_DIR = DATA_DIR / "raw" / "kaggle"
DERIVED_DIR = DATA_DIR / "derived"

JSONL_IN = RAW_KAGGLE_DIR / "Prompt_INJECTION_And_Benign_DATASET.jsonl"
NON_ENGLISH_PARQUET_HOLDOUT = DERIVED_DIR / "non_english_holdout.csv"
EXCLUDED_JSONL_MALICIOUS = DERIVED_DIR / "jsonl_excluded_malware.csv"
MERGED_CSV = DERIVED_DIR / "merged_binary_dataset.csv"
MERGE_STATS_JSON = DERIVED_DIR / "merge_stats.json"

MIN_CHARS_FOR_LANG = 12


def main() -> None:
    if not JSONL_IN.exists():
        raise FileNotFoundError(f"Kaggle JSONL not found: {JSONL_IN.resolve()}")
    for name in ("train.parquet", "test.parquet"):
        pq = RAW_HF_DIR / name
        if not pq.exists():
            raise FileNotFoundError(f"Hugging Face parquet not found: {pq.resolve()}")

    detector = build_lang_detector()

    hf_df = load_english_parquet_subset(
        RAW_HF_DIR,
        detector=detector,
        min_chars=MIN_CHARS_FOR_LANG,
        non_english_holdout_path=NON_ENGLISH_PARQUET_HOLDOUT,
        verbose=True,
    )
    print(f"Wrote HF non-English holdout to {NON_ENGLISH_PARQUET_HOLDOUT.resolve()}")

    kaggle_df, jsonl_stage_stats = load_jsonl_binary_subset(
        JSONL_IN,
        excluded_malicious_path=EXCLUDED_JSONL_MALICIOUS,
        verbose=True,
    )

    merged = build_merged_dataframe(hf_df, kaggle_df)
    write_merged_csv(merged, MERGED_CSV)

    stats = compute_merge_statistics(merged, jsonl_stage_stats)
    write_merge_stats(stats, MERGE_STATS_JSON)

    print(f"Merged rows: {len(merged)} -> {MERGED_CSV.resolve()}")
    print(f"Wrote statistics -> {MERGE_STATS_JSON.resolve()}")
    print("Label counts (overall):\n", merged[LABEL_COL].value_counts().sort_index())
    print("Rows by source:")
    print(merged[SOURCE_COL].value_counts())
    kpa = stats["kaggle_positive_attack_types"]
    print(
        "Kaggle positives by attack_type:",
        kpa["counts"],
        f"(total kaggle positives: {kpa['total_kaggle_positives']})",
    )


if __name__ == "__main__":
    main()
