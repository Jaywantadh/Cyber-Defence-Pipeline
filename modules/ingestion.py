"""Ingestion Module.

Responsible for reading, parsing, and normalizing raw event/CSV data from the
data/raw directory into structured data formats suitable for downstream pipeline
processing.
"""

import pprint
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Ensure project root is in sys.path so modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_loader import load_raw_data


def normalize_events(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Cleans data quality issues and normalizes rows into consistent event dictionaries.

    Parameters
    ----------
    df : pd.DataFrame
        Raw flow-statistics DataFrame loaded from CICIDS2017 CSV.

    Returns
    -------
    List[Dict[str, Any]]
        List of normalized event dictionaries with keys 'event_id', 'features', and 'label'.
    """
    initial_rows = len(df)
    feature_cols = [col for col in df.columns if col != "Label"]

    # Work on a copy to avoid mutating the original DataFrame in place
    clean_df = df.copy()

    # 1. Clean data quality issues known in CICIDS2017:
    # Replace Infinity and -Infinity values with NaN across feature columns
    clean_df[feature_cols] = clean_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Drop any row that has NaN in any feature column after replacement
    clean_df = clean_df.dropna(subset=feature_cols)
    dropped_count = initial_rows - len(clean_df)
    print(f"Dropped {dropped_count:,} row(s) containing NaN or infinite values.")

    # 2. Convert features to plain float dictionary for each row
    features_df = clean_df[feature_cols].astype(float)
    records = features_df.to_dict(orient="records")
    labels = clean_df["Label"].tolist() if "Label" in clean_df.columns else [None] * len(clean_df)

    # Build standardized event dictionaries
    events = [
        {
            "event_id": idx,
            "features": rec,
            "label": lbl,
        }
        for idx, (rec, lbl) in enumerate(zip(records, labels))
    ]

    return events


if __name__ == "__main__":
    csv_path = PROJECT_ROOT / "data" / "raw" / "Tuesday-WorkingHours.pcap_ISCX.csv"
    if not csv_path.exists():
        print(f"Error: Target file not found at {csv_path}")
        sys.exit(1)

    print(f"Loading dataset from: {csv_path} ...")
    raw_df = load_raw_data(csv_path)
    print(f"Raw DataFrame loaded: {len(raw_df):,} rows, {len(raw_df.columns)} columns.")

    print("\nNormalizing events...")
    events = normalize_events(raw_df)
    print(f"Total normalized events produced: {len(events):,}")

    print("\n" + "=" * 60)
    print("SAMPLE NORMALIZED EVENT (first event):")
    print("=" * 60)
    pprint.pprint(events[0], sort_dicts=False, width=100)
