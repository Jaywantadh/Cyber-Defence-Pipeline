"""Inspect CICIDS2017 Dataset.

Read-only inspection script that loads Tuesday-WorkingHours.pcap_ISCX.csv
and displays dataset characteristics including row count, columns, data types,
label distribution, and sample rows.
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_loader import load_raw_data


def inspect_dataset() -> None:
    target_path = PROJECT_ROOT / "data" / "raw" / "Tuesday-WorkingHours.pcap_ISCX.csv"

    if not target_path.exists():
        print(f"Error: Target file not found at {target_path}")
        sys.exit(1)

    print(f"Loading raw dataset from: {target_path} ...")
    df = load_raw_data(target_path)
    print("Dataset loaded successfully.\n")

    # 1. Total row count
    row_count = len(df)
    print("=" * 60)
    print(f"TOTAL ROW COUNT: {row_count:,}")
    print("=" * 60)

    # 2. Full list of column names
    print("\n" + "=" * 60)
    print(f"FULL LIST OF COLUMN NAMES ({len(df.columns)} columns):")
    print("=" * 60)
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx:2d}. {col}")

    # 3. Data types (dtypes)
    print("\n" + "=" * 60)
    print("DATA TYPES (dtypes):")
    print("=" * 60)
    print(df.dtypes.to_string())

    # 4. Unique values in 'Label' column with counts
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION ('Label' column):")
    print("=" * 60)
    if "Label" in df.columns:
        label_counts = df["Label"].value_counts(dropna=False)
        print(label_counts.to_string())
    else:
        print("Warning: 'Label' column not found in DataFrame.")

    # 5. First 3 rows
    print("\n" + "=" * 60)
    print("FIRST 3 ROWS:")
    print("=" * 60)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head(3))


if __name__ == "__main__":
    inspect_dataset()
