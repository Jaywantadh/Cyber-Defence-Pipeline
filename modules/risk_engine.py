"""Risk Engine Module.

Responsible for computing a quantitative risk score and categorical risk level
for proposed network defense actions based on detection confidence and action severity.
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

# Action severity weights reflecting operational impact of autonomous actions
ACTION_SEVERITY_WEIGHTS: Dict[str, float] = {
    "ISOLATE_FLOW": 1.0,
    "TERMINATE_PROCESS": 1.0,
    "RESET_CREDENTIALS": 1.0,
    "FLAG_FOR_REVIEW": 0.5,
}

# Named risk score thresholds for risk level categorization
HIGH_RISK_THRESHOLD: float = 0.7
MEDIUM_RISK_THRESHOLD: float = 0.4
LOW_RISK_THRESHOLD: float = 0.0


def compute_risk_score(action_proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Computes a numeric risk score and categorical risk level for a proposed action.

    Formula:
        risk_score = detection_confidence * action_severity_weight (clamped to [0.0, 1.0])

    Risk Level Mapping:
        >= 0.7 -> "HIGH"
        >= 0.4 -> "MEDIUM"
        >  0.0 -> "LOW"
        else   -> "NONE"

    Parameters
    ----------
    action_proposal : Dict[str, Any]
        Output dictionary from network_agent.propose_action() containing 'event_id',
        'action' ("NO_ACTION" | "ISOLATE_FLOW" | "FLAG_FOR_REVIEW"), and optionally
        'detection_confidence'.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'event_id', 'risk_score', 'risk_level', and 'action'.
    """
    event_id = action_proposal["event_id"]
    action = action_proposal.get("action", "NO_ACTION")

    # Benign traffic requires no action and carries zero risk
    if action == "NO_ACTION":
        return {
            "event_id": event_id,
            "risk_score": 0.0,
            "risk_level": "NONE",
            "action": action,
        }

    detection_confidence = float(action_proposal.get("detection_confidence", 0.0))
    # Unrecognized actions default to 1.0 (fail-closed) so unknown actions are never treated as low/mid-risk.
    if action in ACTION_SEVERITY_WEIGHTS:
        severity_weight = ACTION_SEVERITY_WEIGHTS[action]
    else:
        print(
            f"Warning: Unrecognized action '{action}' encountered in risk engine (event_id={event_id}). "
            f"Defaulting to maximum severity weight (1.0) for fail-closed safety."
        )
        severity_weight = 1.0

    # Compute raw risk score and clamp to [0.0, 1.0]
    raw_score = detection_confidence * severity_weight
    clamped_score = max(0.0, min(1.0, raw_score))

    # Categorize risk level
    if clamped_score >= HIGH_RISK_THRESHOLD:
        risk_level = "HIGH"
    elif clamped_score >= MEDIUM_RISK_THRESHOLD:
        risk_level = "MEDIUM"
    elif clamped_score > LOW_RISK_THRESHOLD:
        risk_level = "LOW"
    else:
        risk_level = "NONE"

    return {
        "event_id": event_id,
        "risk_score": round(clamped_score, 4),
        "risk_level": risk_level,
        "action": action,
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

    print("\n" + "=" * 75)
    print("PIPELINE TEST: LOAD -> NORMALIZE -> DETECT -> PROPOSE ACTION -> RISK SCORE")
    print("=" * 75)

    for category, sample_ev in sample_targets:
        if sample_ev is not None:
            det_result = detect(model, sample_ev)
            action_proposal = propose_action(det_result, sample_ev)
            risk_result = compute_risk_score(action_proposal)

            print(f"\n[Sample Category: {category}]")
            print(f"  Event ID             : {sample_ev['event_id']}")
            print(f"  True Label           : {sample_ev['label']}")
            print(f"  Detection Result     : {det_result}")
            print(f"  Action Proposal      : {action_proposal}")
            print(f"  Risk Engine Result   : {risk_result}")
        else:
            print(f"\nWarning: Test sample for {category} could not be found.")

    print("\n" + "=" * 75)
    print("FAIL-CLOSED TEST: UNRECOGNIZED ACTION STRING DEFAULTS TO SEVERITY 1.0")
    print("=" * 75)
    unrecognized_proposal = {
        "event_id": 99999,
        "action": "CUSTOM_UNKNOWN_ACTION",
        "detection_confidence": 0.85,
    }
    unrecognized_risk = compute_risk_score(unrecognized_proposal)
    print(f"Action Proposal : {unrecognized_proposal}")
    print(f"Risk Result     : {unrecognized_risk}")
    assert unrecognized_risk["risk_score"] == 0.85, f"Expected risk_score 0.85, got {unrecognized_risk['risk_score']}"
    assert unrecognized_risk["risk_level"] == "HIGH"
    print("Confirmed: Unrecognized action received maximum severity weight 1.0 (fail-closed).")

