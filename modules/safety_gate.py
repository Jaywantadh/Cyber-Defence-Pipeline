"""Safety Gate Module.

Responsible for policy enforcement, blast-radius containment, and ensuring
human-in-the-loop sign-off for high-risk actions prior to execution.
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
from modules.network_agent import propose_action
from modules.risk_engine import HIGH_RISK_THRESHOLD, compute_risk_score

# Synchronized threshold for autonomous execution boundary.
# Actions with risk scores at or above this bound require human approval.
RISK_AUTO_EXECUTE_BOUND: float = HIGH_RISK_THRESHOLD


def evaluate_gate(risk_result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates whether a proposed action can be automatically executed or blocked for human review.

    Safety Logic:
        Higher risk implies GREATER caution:
        - If action is "NO_ACTION": AUTO_EXECUTE
        - If risk_score >= RISK_AUTO_EXECUTE_BOUND: NEEDS_HUMAN_APPROVAL (blocked)
        - If risk_score < RISK_AUTO_EXECUTE_BOUND: AUTO_EXECUTE (allowed)

    Parameters
    ----------
    risk_result : Dict[str, Any]
        Output dictionary from risk_engine.compute_risk_score() containing
        'event_id', 'risk_score', 'risk_level', and 'action'.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'event_id', 'gate_decision' ("AUTO_EXECUTE" | "NEEDS_HUMAN_APPROVAL"),
        'final_action', and 'gate_reason'.
    """
    event_id = risk_result["event_id"]
    action = risk_result.get("action", "NO_ACTION")
    risk_score = risk_result.get("risk_score", 0.0)

    # 1. No action required
    if action == "NO_ACTION":
        return {
            "event_id": event_id,
            "gate_decision": "AUTO_EXECUTE",
            "final_action": "NO_ACTION",
            "gate_reason": "No action required",
        }

    # 2. Safety Gate evaluation: High risk requires human authorization
    if risk_score >= RISK_AUTO_EXECUTE_BOUND:
        return {
            "event_id": event_id,
            "gate_decision": "NEEDS_HUMAN_APPROVAL",
            "final_action": action,
            "gate_reason": (
                f"Risk score {risk_score} meets/exceeds high-risk bound "
                f"({RISK_AUTO_EXECUTE_BOUND}); autonomous execution blocked pending human approval."
            ),
        }
    else:
        return {
            "event_id": event_id,
            "gate_decision": "AUTO_EXECUTE",
            "final_action": action,
            "gate_reason": (
                f"Risk score {risk_score} below high-risk bound "
                f"({RISK_AUTO_EXECUTE_BOUND}); safe to execute automatically."
            ),
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

    # Identify the same 3 test set samples
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

    print("\n" + "=" * 80)
    print("6-STAGE PIPELINE: LOAD -> NORMALIZE -> DETECT -> ACTION -> RISK -> SAFETY GATE")
    print("=" * 80)

    for category, sample_ev in sample_targets:
        if sample_ev is not None:
            det_result = detect(model, sample_ev)
            action_proposal = propose_action(det_result, sample_ev)
            risk_result = compute_risk_score(action_proposal)
            gate_result = evaluate_gate(risk_result)

            print(f"\n[Sample Category: {category}]")
            print(f"  Event ID             : {sample_ev['event_id']}")
            print(f"  True Label           : {sample_ev['label']}")
            print(f"  Detection Result     : {det_result}")
            print(f"  Action Proposal      : {action_proposal}")
            print(f"  Risk Engine Result   : {risk_result}")
            print(f"  Safety Gate Result   : {gate_result}")
        else:
            print(f"\nWarning: Test sample for {category} could not be found.")
