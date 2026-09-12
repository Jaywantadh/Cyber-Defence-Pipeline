"""Network Agent Module.

Responsible for inspecting threat detection outputs and proposing automated
response actions (e.g., ISOLATE_FLOW, FLAG_FOR_REVIEW, or NO_ACTION)
based on flow-level characteristics.
"""

from pathlib import Path
import sys
from typing import Any, Dict
import joblib

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_loader import load_raw_data
from modules.detection import detect, prepare_training_data
from modules.ingestion import normalize_events

# Named rate threshold for rule-based action proposal:
# Empirical quantile analysis of 'Flow Packets/s' across combined attack samples
# (FTP-Patator + SSH-Patator, n=13,835) from Tuesday-WorkingHours shows:
#   10th percentile:      2.60909 packets/s
#   25th percentile:      2.83943 packets/s
#   50th percentile (Q2): 5.88638 packets/s (median)
#   75th percentile: 25,000.00000 packets/s
#   90th percentile: 51,282.05128 packets/s
# HIGH_RATE_THRESHOLD is set to the 50th percentile (median = 5.88638 pkts/s).
# Flows at or above this median rate receive ISOLATE_FLOW; flows below receive FLAG_FOR_REVIEW.
HIGH_RATE_THRESHOLD: float = 5.88638


def propose_action(
    detection_result: Dict[str, Any],
    event: Dict[str, Any],
) -> Dict[str, Any]:
    """Proposes a candidate response action based on threat detection output and flow dynamics.

    Parameters
    ----------
    detection_result : Dict[str, Any]
        Output dictionary from detection.detect() containing 'event_id',
        'prediction' ("THREAT" | "BENIGN"), and 'confidence'.
    event : Dict[str, Any]
        Original normalized event dictionary containing 'event_id',
        'features', and 'label'.

    Returns
    -------
    Dict[str, Any]
        Proposed action dictionary containing:
        - If BENIGN: {"event_id": int, "action": "NO_ACTION", "reason": str}
        - If THREAT: {"event_id": int, "action": "ISOLATE_FLOW" | "FLAG_FOR_REVIEW",
                      "reason": str, "detection_confidence": float}
    """
    event_id = event["event_id"]
    prediction = detection_result.get("prediction")
    confidence = detection_result.get("confidence", 0.0)

    # 1. Benign traffic requires no remediation
    if prediction == "BENIGN":
        return {
            "event_id": event_id,
            "action": "NO_ACTION",
            "reason": "Traffic classified as benign",
        }

    # 2. Threat traffic: determine candidate action based on Flow Packets/s rate
    features = event.get("features", {})
    flow_packets_s = features.get("Flow Packets/s", 0.0)

    if flow_packets_s >= HIGH_RATE_THRESHOLD:
        action = "ISOLATE_FLOW"
        reason = (
            f"High-rate automated attack detected: Flow Packets/s of {flow_packets_s:,.2f} "
            f"is at or above median attack rate threshold ({HIGH_RATE_THRESHOLD:.5f} pkts/s)."
        )
    else:
        action = "FLAG_FOR_REVIEW"
        reason = (
            f"Low-rate suspicious flow detected: Flow Packets/s of {flow_packets_s:,.2f} "
            f"is below median attack rate threshold ({HIGH_RATE_THRESHOLD:.5f} pkts/s); requires security analyst review."
        )

    return {
        "event_id": event_id,
        "action": action,
        "reason": reason,
        "detection_confidence": confidence,
    }


if __name__ == "__main__":
    model_path = PROJECT_ROOT / "models" / "threat_detector.pkl"
    csv_path = PROJECT_ROOT / "data" / "raw" / "Tuesday-WorkingHours.pcap_ISCX.csv"

    if not model_path.exists():
        print(f"Error: Trained model not found at {model_path}")
        sys.exit(1)

    if not csv_path.exists():
        print(f"Error: Dataset not found at {csv_path}")
        sys.exit(1)

    print(f"Loading model from: {model_path} ...")
    model = joblib.load(model_path)

    print(f"Loading dataset from: {csv_path} ...")
    raw_df = load_raw_data(csv_path)

    print("Normalizing events...")
    events = normalize_events(raw_df)

    print("Splitting dataset to identify test set samples...")
    _, X_test, _, _ = prepare_training_data(events)

    # Select the exact same 3 test samples as detection.py
    sample_benign = None
    sample_ftp = None
    sample_ssh = None

    for idx in X_test.index:
        ev = events[idx]
        lbl = ev.get("label")
        if lbl == "BENIGN" and sample_benign is None:
            sample_benign = ev
        elif lbl == "FTP-Patator" and sample_ftp is None:
            sample_ftp = ev
        elif lbl == "SSH-Patator" and sample_ssh is None:
            sample_ssh = ev

        if sample_benign and sample_ftp and sample_ssh:
            break

    sample_targets = [
        ("BENIGN", sample_benign),
        ("FTP-Patator", sample_ftp),
        ("SSH-Patator", sample_ssh),
    ]

    print("\n" + "=" * 70)
    print("PIPELINE TEST: LOAD -> NORMALIZE -> DETECT -> PROPOSE ACTION")
    print("=" * 70)

    for category, sample_ev in sample_targets:
        if sample_ev is not None:
            det_result = detect(model, sample_ev)
            action_proposal = propose_action(det_result, sample_ev)

            print(f"\n[Sample Category: {category}]")
            print(f"  Event ID             : {sample_ev['event_id']}")
            print(f"  True Label           : {sample_ev['label']}")
            print(f"  Detection Result     : {det_result}")
            print(f"  Action Proposal      : {action_proposal}")
        else:
            print(f"\nWarning: Test sample for {category} could not be found.")
