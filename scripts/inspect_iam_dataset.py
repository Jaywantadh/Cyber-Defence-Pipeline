"""Inspect Synthetic IAM Telemetry Dataset.

[SYNTHETIC DATA SOURCE NOTICE]
==============================
WARNING: ALL DATA INSPECTED BY THIS SCRIPT IS 100% SYNTHETIC.
This script is a read-only exploration script that loads synthetic_iam_events.csv
and displays characteristics including row count, columns, data types,
synthetic label distribution, empirical class failure rates, and sample events.
It does NOT represent authentic IAM telemetry or real Active Directory logs.
"""

from pathlib import Path
import sys
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.iam_dataset_generator import load_synthetic_iam_events


def inspect_iam_dataset() -> None:
    target_path = PROJECT_ROOT / "data" / "raw" / "synthetic_iam_events.csv"

    print("=" * 70)
    print("[SYNTHETIC IAM DATASET INSPECTION NOTICE]")
    print("WARNING: All data inspected is 100% SYNTHETIC (NOT REAL TELEMETRY).")
    print("Generated for prototype testing and pipeline evaluation.")
    print("=" * 70)

    if not target_path.exists():
        print(f"\nError: Synthetic dataset not found at {target_path}")
        print("Run 'python modules/iam_dataset_generator.py' to generate it first.")
        sys.exit(1)

    print(f"\nLoading synthetic IAM dataset from: {target_path} ...")
    df = load_synthetic_iam_events(target_path)
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
        print(df["label"].value_counts(dropna=False).to_string())
    else:
        print("Warning: 'label' column not found in synthetic DataFrame.")

    # 5. Empirical class verification
    print("\n" + "=" * 70)
    print("[SYNTHETIC] EMPIRICAL CLASS DISTRIBUTIONS:")
    print("=" * 70)
    for lbl in ["BENIGN", "BRUTE_FORCE_LOGIN", "PASS_THE_HASH"]:
        sub_df = df[df["label"] == lbl]
        fail_rate = (sub_df["auth_result"] == "FAILURE").mean()
        most_common_logon = sub_df["logon_type"].mode()[0]
        most_common_auth = sub_df["auth_type"].mode()[0]
        print(
            f"Label: {lbl:<20} | Rows: {len(sub_df):>5,} | "
            f"FAILURE Rate: {fail_rate:.1%} | "
            f"Top Logon: {most_common_logon:<10} | "
            f"Top Auth: {most_common_auth}"
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
    inspect_iam_dataset()
