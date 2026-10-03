"""Inspect Synthetic Host Telemetry Dataset.

[SYNTHETIC DATA SOURCE NOTICE]
==============================
WARNING: ALL DATA INSPECTED BY THIS SCRIPT IS 100% SYNTHETIC.
This script is a read-only exploration script that loads synthetic_host_events.csv
and displays characteristics including row count, columns, data types,
synthetic label distribution, and sample events.
It does NOT represent authentic host telemetry or real endpoint sensor data.
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.host_dataset_generator import (
    ATTACK_PROCESS_PAIRS,
    BENIGN_ASSOCIATED_PARENTS,
    load_synthetic_host_events,
)


def inspect_host_dataset() -> None:
    target_path = PROJECT_ROOT / "data" / "raw" / "synthetic_host_events.csv"

    print("=" * 70)
    print("[SYNTHETIC DATASET INSPECTION NOTICE]")
    print("WARNING: All data inspected is 100% SYNTHETIC (NOT REAL TELEMETRY).")
    print("Generated for prototype testing and pipeline evaluation.")
    print("=" * 70)

    if not target_path.exists():
        print(f"\nError: Synthetic dataset not found at {target_path}")
        print("Run 'python modules/host_dataset_generator.py' to generate it first.")
        sys.exit(1)

    print(f"\nLoading synthetic host dataset from: {target_path} ...")
    df = load_synthetic_host_events(target_path)
    print("Synthetic dataset loaded successfully.\n")

    # 1. Total row count
    row_count = len(df)
    print("=" * 70)
    print(f"[SYNTHETIC] TOTAL ROW COUNT: {row_count:,}")
    print("=" * 70)

    # 2. Full list of column names
    print("\n" + "=" * 70)
    print(f"[SYNTHETIC] FULL LIST OF COLUMN NAMES ({len(df.columns)} columns):")
    print("=" * 70)
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx:2d}. {col}")

    # 3. Data types (dtypes)
    print("\n" + "=" * 70)
    print("[SYNTHETIC] DATA TYPES (dtypes):")
    print("=" * 70)
    print(df.dtypes.to_string())

    # 4. Unique values in 'label' column with counts
    print("\n" + "=" * 70)
    print("[SYNTHETIC] LABEL DISTRIBUTION ('label' column):")
    print("=" * 70)
    if "label" in df.columns:
        label_counts = df["label"].value_counts(dropna=False)
        print(label_counts.to_string())
    else:
        print("Warning: 'label' column not found in synthetic DataFrame.")

    # 5. Overlap metrics
    attack_pairs_set = set(ATTACK_PROCESS_PAIRS)
    benign_attack_pair_count = sum(
        1
        for is_mal, parent, proc in zip(df["is_malicious"], df["parent_process"], df["process_name"])
        if is_mal == 0 and (parent, proc) in attack_pairs_set
    )
    malicious_benign_parent_count = int(
        ((df["is_malicious"] == 1) & df["parent_process"].isin(BENIGN_ASSOCIATED_PARENTS)).sum()
    )

    print("\n" + "=" * 70)
    print("[SYNTHETIC] CONTROLLED OVERLAP METRICS:")
    print("=" * 70)
    n_benign_total = (df["is_malicious"] == 0).sum()
    n_mal_total = (df["is_malicious"] == 1).sum()
    pct_benign = (benign_attack_pair_count / n_benign_total) if n_benign_total else 0
    pct_mal = (malicious_benign_parent_count / n_mal_total) if n_mal_total else 0
    print(
        f"BENIGN rows using attack-associated process pair: {benign_attack_pair_count:,} "
        f"({pct_benign:.1%}) | "
        f"Malicious rows using benign-associated parent: {malicious_benign_parent_count:,} "
        f"({pct_mal:.1%})"
    )

    # 6. First 5 rows
    print("\n" + "=" * 70)
    print("[SYNTHETIC] FIRST 5 ROWS:")
    print("=" * 70)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head(5).to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    inspect_host_dataset()
