"""Merge hand-edited base CSV with curated JSONL into data/final_dataset/ (training ground truth).

Reads:
  data/derived/updated_merged.csv
  data/raw/curated_prompt_injection_dataset_v1.jsonl
Writes:
  data/final_dataset/final_binary_dataset.csv
  data/final_dataset/final_dataset_stats.json

Run from the repository root: python scripts/build_final_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from dataset_merge import (
    LABEL_COL,
    TEXT_COL,
    compute_final_dataset_statistics,
    load_curated_jsonl_for_final,
    prepare_base_csv_for_final,
    write_final_dataset_stats,
    write_merged_csv,
)

DATA_DIR = ROOT / "data"
BASE_CSV = DATA_DIR / "derived" / "updated_merged.csv"
CURATED_JSONL = DATA_DIR / "raw" / "curated_prompt_injection_dataset_v1.jsonl"
FINAL_DIR = DATA_DIR / "final_dataset"
FINAL_CSV = FINAL_DIR / "final_binary_dataset.csv"
FINAL_STATS_JSON = FINAL_DIR / "final_dataset_stats.json"


def main() -> None:
    if not BASE_CSV.exists():
        raise FileNotFoundError(f"Base CSV not found: {BASE_CSV.resolve()}")
    if not CURATED_JSONL.exists():
        raise FileNotFoundError(f"Curated JSONL not found: {CURATED_JSONL.resolve()}")

    base_raw = pd.read_csv(BASE_CSV, encoding="utf-8-sig")
    base_raw = base_raw.dropna(subset=[TEXT_COL])
    base_raw[TEXT_COL] = base_raw[TEXT_COL].astype(str)
    base_raw[LABEL_COL] = base_raw[LABEL_COL].astype(int)
    base_text_set = set(base_raw[TEXT_COL])

    base_df = prepare_base_csv_for_final(base_raw)
    curated_df, jsonl_meta = load_curated_jsonl_for_final(CURATED_JSONL, verbose=True)

    final_df = pd.concat([base_df, curated_df], axis=0, ignore_index=True)
    write_merged_csv(final_df, FINAL_CSV)

    stats = compute_final_dataset_statistics(final_df, jsonl_meta, base_text_set)
    write_final_dataset_stats(stats, FINAL_STATS_JSON)

    print(f"Wrote final dataset -> {FINAL_CSV.resolve()} ({len(final_df)} rows)")
    print(f"Wrote statistics -> {FINAL_STATS_JSON.resolve()}")
    print("Row counts by source:", stats["row_counts"])
    print("Labels overall:", stats["labels"]["overall"])
    print("Duplicate curated texts also in base:", stats["duplicate_text_vs_base"])


if __name__ == "__main__":
    main()
