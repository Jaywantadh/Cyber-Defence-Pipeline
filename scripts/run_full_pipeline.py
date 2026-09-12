"""Full Pipeline Batch Runner.

Wired execution of all 6 pipeline stages at scale:
load_raw_data -> normalize_events -> detect -> propose_action -> compute_risk_score -> evaluate_gate -> log_decision.
"""

from collections import Counter
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.audit_log import init_db, log_decision
from modules.dataset_loader import load_raw_data
from modules.detection import detect
from modules.ingestion import normalize_events
from modules.network_agent import propose_action
from modules.risk_engine import compute_risk_score
from modules.safety_gate import evaluate_gate


def run_pipeline_on_event(
    event: Dict[str, Any],
    model: Any,
    errors_list: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Chains all pipeline stages for ONE normalized event:

    detect -> propose_action -> compute_risk_score -> evaluate_gate -> log_decision.

    Parameters
    ----------
    event : Dict[str, Any]
        Normalized event dict from ingestion.
    model : Any
        Trained threat detection classifier.
    errors_list : Optional[List[Dict[str, Any]]], optional
        List to collect any per-event processing exceptions.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing the gate_result with 'true_label', 'risk_level',
        and 'risk_score', or None if an exception occurred.
    """
    try:
        det_result = detect(model, event)
        act_result = propose_action(det_result, event)
        risk_result = compute_risk_score(act_result)
        gate_result = evaluate_gate(risk_result)
        log_decision(event, det_result, act_result, risk_result, gate_result)

        result = dict(gate_result)
        result["true_label"] = event.get("label")
        result["risk_level"] = risk_result.get("risk_level")
        result["risk_score"] = risk_result.get("risk_score")
        return result
    except Exception as exc:
        if errors_list is not None:
            errors_list.append(
                {
                    "event_id": event.get("event_id"),
                    "error": str(exc),
                }
            )
        return None


def run_batch(sample_size: int = 5000, random_state: int = 42) -> None:
    """Executes the pipeline over a stratified sample of the Tuesday dataset.

    Parameters
    ----------
    sample_size : int, optional
        Number of events to sample and run (default: 5000).
    random_state : int, optional
        Random seed for stratified sampling reproducibility (default: 42).
    """
    csv_path = PROJECT_ROOT / "data" / "raw" / "Tuesday-WorkingHours.pcap_ISCX.csv"
    model_path = PROJECT_ROOT / "models" / "threat_detector.pkl"

    if not csv_path.exists():
        print(f"Error: Dataset not found at {csv_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"Error: Trained model not found at {model_path}")
        sys.exit(1)

    print("=" * 70)
    print(f"STARTING BATCH RUN (sample_size={sample_size:,}, random_state={random_state})")
    print("=" * 70)

    # 1. Initialize audit database once
    init_db()

    # 2. Load model once
    print(f"Loading model from {model_path} ...")
    model = joblib.load(model_path)

    # 3. Load and normalize raw data
    print(f"Loading raw data from {csv_path} ...")
    raw_df = load_raw_data(csv_path)

    print("Normalizing events...")
    events = normalize_events(raw_df)
    total_available = len(events)
    print(f"Total normalized events available: {total_available:,}")

    # 4. Stratified sampling preserving real label proportions
    labels = [e["label"] for e in events]
    print(f"\nTaking stratified sample of {sample_size:,} events...")
    sampled_events, _, _, _ = train_test_split(
        events,
        labels,
        train_size=sample_size,
        stratify=labels,
        random_state=random_state,
    )

    sample_dist = Counter(e["label"] for e in sampled_events)
    print(f"Sample label distribution: {dict(sample_dist)}")

    # 5. Process events
    errors: List[Dict[str, Any]] = []
    processed_results: List[Dict[str, Any]] = []

    print(f"\nRunning pipeline on {len(sampled_events):,} events...")
    start_time = time.time()

    for idx, event in enumerate(sampled_events):
        res = run_pipeline_on_event(event, model, errors_list=errors)
        if res is not None:
            processed_results.append(res)

        if (idx + 1) % 500 == 0 or (idx + 1) == len(sampled_events):
            elapsed_chunk = time.time() - start_time
            rate = (idx + 1) / elapsed_chunk if elapsed_chunk > 0 else 0
            print(
                f"  Progress: [{idx + 1:5d}/{len(sampled_events):5d}] events processed "
                f"({rate:.1f} events/sec)",
                flush=True,
            )

    total_wall_clock = time.time() - start_time

    # 6. Summary Report
    print("\n" + "=" * 70)
    print("BATCH EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Sample Requested : {sample_size:,}")
    print(f"Successfully Processed : {len(processed_results):,}")
    print(f"Total Errors Encountered: {len(errors)}")

    if errors:
        print("\nFirst 5 Error Messages:")
        for err in errors[:5]:
            print(f"  Event #{err['event_id']}: {err['error']}")

    df_res = pd.DataFrame(processed_results)

    # Gate decision breakdown
    print("\n" + "-" * 40)
    print("BREAKDOWN BY GATE DECISION:")
    print("-" * 40)
    print(df_res["gate_decision"].value_counts().to_string())

    # Risk level breakdown
    print("\n" + "-" * 40)
    print("BREAKDOWN BY RISK LEVEL:")
    print("-" * 40)
    # Order categories logically if present
    risk_order = ["NONE", "LOW", "MEDIUM", "HIGH"]
    risk_counts = df_res["risk_level"].value_counts()
    for lvl in risk_order:
        if lvl in risk_counts:
            print(f"  {lvl:<8}: {risk_counts[lvl]:,}")

    # Confusion-style cross-tab: true_label vs gate_decision
    print("\n" + "=" * 70)
    print("CROSS-TABULATION: True Label vs Gate Decision")
    print("=" * 70)
    ct = pd.crosstab(
        df_res["true_label"],
        df_res["gate_decision"],
        margins=True,
        margins_name="Total",
    )
    print(ct.to_string())
    print("=" * 70)

    print(f"\nTotal Wall-Clock Time: {total_wall_clock:.2f} seconds ({len(processed_results)/total_wall_clock:.1f} events/sec)")
    print("=" * 70)


if __name__ == "__main__":
    run_batch(sample_size=5000, random_state=42)
