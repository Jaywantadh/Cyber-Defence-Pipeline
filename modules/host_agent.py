"""Host Agent Module.

[SYNTHETIC DATA SOURCE NOTICE]
==============================
WARNING: ALL DATA PROCESSED BY THIS MODULE IS SYNTHETIC.
This module inspects host threat detection outputs and proposes automated
host-level response actions (TERMINATE_PROCESS, FLAG_FOR_REVIEW, or NO_ACTION)
for simulated Windows endpoint events.

Note on Detection Confidence:
Model confidence returned by host_detection.detect() is NOT a calibrated probability
due to class_weight='balanced' training. It serves strictly as a relative decision
and risk-weighting signal, not a true likelihood.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Set
import joblib

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.host_dataset_generator import (
    ATTACK_PROCESS_PAIRS,
    generate_host_events,
    load_synthetic_host_events,
)
from modules.host_detection import detect, prepare_training_data, train_model
from modules.risk_engine import compute_risk_score
from modules.safety_gate import evaluate_gate

# High-severity office application parents and interactive script interpreters
OFFICE_APP_PARENTS: Set[str] = {"winword.exe", "excel.exe", "acrobat.exe"}
HIGH_SEVERITY_PROCESSES: Set[str] = {"powershell.exe", "cmd.exe"}


def propose_action(
    detection_result: Dict[str, Any],
    event: Dict[str, Any],
) -> Dict[str, Any]:
    """Proposes a candidate response action based on host detection output and event context.

    Parameters
    ----------
    detection_result : Dict[str, Any]
        Output dictionary from host_detection.detect() containing:
        - 'event_id': int or None
        - 'prediction': "THREAT" | "BENIGN"
        - 'confidence': float (relative decision signal, not calibrated probability)
    event : Dict[str, Any]
        Original host telemetry event dictionary containing:
        - 'event_id': int
        - 'process_name': str
        - 'parent_process': str
        - 'label': str ("BENIGN", "SUSPICIOUS_PROCESS_INJECTION", "PRIVILEGE_ESCALATION_ATTEMPT")
        - 'is_malicious': int (0 or 1)

    Returns
    -------
    Dict[str, Any]
        Proposed action dictionary containing:
        - If BENIGN: {"event_id": int, "action": "NO_ACTION", "reason": str}
        - If THREAT: {"event_id": int, "action": "TERMINATE_PROCESS" | "FLAG_FOR_REVIEW",
                      "reason": str, "detection_confidence": float}
    """
    event_id = event["event_id"]
    prediction = detection_result.get("prediction")
    confidence = detection_result.get("confidence", 0.0)

    # 1. Benign classification requires no host remediation
    if prediction == "BENIGN":
        return {
            "event_id": event_id,
            "action": "NO_ACTION",
            "reason": "Process activity classified as benign",
        }

    # 2. Threat classification: select action based on severity indicators and label type
    process_name = str(event.get("process_name", ""))
    parent_process = str(event.get("parent_process", ""))
    label = str(event.get("label", ""))

    # High-severity indicator: shell execution from office apps (macro/injection pattern)
    # or explicit SUSPICIOUS_PROCESS_INJECTION label
    is_office_macro_injection = (
        parent_process in OFFICE_APP_PARENTS and process_name in HIGH_SEVERITY_PROCESSES
    )
    is_high_severity = (label == "SUSPICIOUS_PROCESS_INJECTION") or is_office_macro_injection

    if is_high_severity:
        action = "TERMINATE_PROCESS"
        reason = (
            f"High-severity process injection pattern: '{process_name}' spawned by "
            f"'{parent_process}' indicates active payload execution; immediate process termination proposed."
        )
    else:
        action = "FLAG_FOR_REVIEW"
        reason = (
            f"Suspicious host activity: '{process_name}' spawned by '{parent_process}' "
            f"(reconnaissance/privilege escalation); flagged for security analyst review."
        )

    return {
        "event_id": event_id,
        "action": action,
        "reason": reason,
        "detection_confidence": confidence,
    }


if __name__ == "__main__":
    model_path = PROJECT_ROOT / "models" / "host_detector.pkl"
    csv_path = PROJECT_ROOT / "data" / "raw" / "synthetic_host_events.csv"

    print("=" * 75)
    print("[HOST AGENT PIPELINE TEST]")
    print("NOTE: Evaluating synthetic endpoint telemetry for prototype response actions.")
    print("=" * 75)

    # 1. Load or generate synthetic host events dataset
    if csv_path.exists():
        print(f"\nLoading synthetic host dataset from: {csv_path} ...")
        df = load_synthetic_host_events(csv_path)
    else:
        print("\nGenerating synthetic host dataset (50,000 events, 3% attacks)...")
        df = generate_host_events(n_events=50000, attack_ratio=0.03, random_state=42)
    print(f"Loaded {len(df):,} synthetic host events.")

    # 2. Extract feature columns alignment and load model
    X_train, _, y_train, _, feature_columns = prepare_training_data(df)
    if not model_path.exists():
        print(f"\nModel not found at {model_path}. Training now...")
        model = train_model(X_train, y_train, model_path=model_path)
    else:
        print(f"\nLoading trained model from: {model_path} ...")
        model = joblib.load(model_path)

    # 3. Select 3 test samples matching host_detection.py categories
    attack_pairs_set = set(ATTACK_PROCESS_PAIRS)

    # Sample 1: Clearly BENIGN (non-overlap pair, e.g. explorer.exe -> winword.exe)
    benign_non_overlap_mask = (df["is_malicious"] == 0) & (
        ~df.apply(lambda r: (r["parent_process"], r["process_name"]) in attack_pairs_set, axis=1)
    )
    sample_clearly_benign = df[benign_non_overlap_mask].iloc[0].to_dict()

    # Sample 2: Ambiguous overlap pair (e.g. winword.exe -> cmd.exe)
    ambiguous_overlap_mask = df.apply(
        lambda r: (r["parent_process"], r["process_name"]) in attack_pairs_set, axis=1
    )
    sample_ambiguous = df[ambiguous_overlap_mask].iloc[0].to_dict()

    # Sample 3: Clearly malicious (non-overlap attack pair, e.g. explorer.exe -> rundll32.exe)
    malicious_non_overlap_mask = (df["is_malicious"] == 1) & (
        ~df.apply(lambda r: (r["parent_process"], r["process_name"]) in attack_pairs_set, axis=1)
    )
    sample_clearly_malicious = df[malicious_non_overlap_mask].iloc[0].to_dict()

    # Sample 4: Reconnaissance / Privilege Escalation (illustrating FLAG_FOR_REVIEW)
    pe_mask = (df["label"] == "PRIVILEGE_ESCALATION_ATTEMPT") & (df["process_name"] == "whoami.exe")
    sample_priv_esc = df[pe_mask].iloc[0].to_dict()

    sample_targets = [
        ("Clearly BENIGN (non-overlap pair)", sample_clearly_benign),
        ("Ambiguous Overlap Pair (shared features between benign & attack)", sample_ambiguous),
        ("Clearly Malicious (non-overlap injection pair -> TERMINATE_PROCESS)", sample_clearly_malicious),
        ("Reconnaissance / Privilege Escalation (discovery command -> FLAG_FOR_REVIEW)", sample_priv_esc),
    ]

    print("\n" + "=" * 75)
    print("PIPELINE TEST: DETECT -> PROPOSE ACTION -> COMPUTE RISK SCORE -> SAFETY GATE")
    print("=" * 75)

    for category, sample_ev in sample_targets:
        det_result = detect(model, feature_columns, sample_ev)
        action_proposal = propose_action(det_result, sample_ev)
        risk_result = compute_risk_score(action_proposal)
        gate_result = evaluate_gate(risk_result)

        print(f"\n[Sample Category: {category}]")
        print(f"  Event ID             : {sample_ev['event_id']}")
        print(f"  Process Tree         : {sample_ev['parent_process']} -> {sample_ev['process_name']}")
        print(f"  True Label           : {sample_ev['label']} (is_malicious={sample_ev['is_malicious']})")
        print(f"  Detection Result     : {det_result}")
        print(f"  Action Proposal      : {action_proposal}")
        print(f"  Risk Engine Result   : {risk_result}")
        print(f"  Safety Gate Result   : {gate_result}")
    print("\n" + "=" * 75)
