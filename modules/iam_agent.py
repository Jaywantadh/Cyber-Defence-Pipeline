"""IAM Agent Module.

[SYNTHETIC DATA SOURCE NOTICE]
==============================
WARNING: ALL DATA PROCESSED BY THIS MODULE IS SYNTHETIC.
This module inspects IAM threat detection outputs and proposes automated
identity response actions (RESET_CREDENTIALS, FLAG_FOR_REVIEW, or NO_ACTION)
for simulated Windows authentication and logon telemetry.

Domain-Specific Severity Inversion Note:
----------------------------------------
In network and host agents, sustained high-volume attacks (e.g., brute-force floods)
or malicious code execution trees represent the highest severity (ISOLATE_FLOW,
TERMINATE_PROCESS).

In IAM/authentication security, this relationship is deliberately inverted:
- A SUCCESSFUL malicious authentication (such as PASS_THE_HASH) indicates that the
  adversary has bypassed authentication controls, obtained valid session tokens, and
  is actively inside the perimeter. This represents an active breach of trust that
  demands immediate, high-severity remediation (RESET_CREDENTIALS, severity 1.0).
- A FAILED authentication attempt (such as BRUTE_FORCE_LOGIN), while suspicious and
  detected with high model confidence, has not yet breached the identity boundary.
  The attempted compromise was contained at the gate, representing lower immediate
  operational severity (FLAG_FOR_REVIEW, severity 0.5).

Note on Detection Confidence:
Model confidence returned by iam_detection.detect() is NOT a calibrated probability
due to class_weight='balanced' training. It serves as a relative decision-weighting
signal rather than a true likelihood.
"""

from pathlib import Path
import sys
from typing import Any, Dict
import joblib

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.iam_dataset_generator import (
    generate_iam_events,
    load_synthetic_iam_events,
)
from modules.iam_detection import detect, prepare_training_data, train_model
from modules.risk_engine import compute_risk_score
from modules.safety_gate import evaluate_gate


def propose_action(
    detection_result: Dict[str, Any],
    event: Dict[str, Any],
) -> Dict[str, Any]:
    """Proposes a candidate response action based on IAM detection output and authentication result.

    Parameters
    ----------
    detection_result : Dict[str, Any]
        Output dictionary from iam_detection.detect() containing:
        - 'event_id': int or None
        - 'prediction': "THREAT" | "BENIGN"
        - 'confidence': float (relative decision signal, not calibrated probability)
    event : Dict[str, Any]
        Original IAM telemetry event dictionary containing:
        - 'event_id': int
        - 'auth_type': str ("Kerberos", "NTLM", "Negotiate")
        - 'logon_type': str ("Interactive", "Network", "Service", "Batch", "RemoteInteractive")
        - 'auth_result': str ("SUCCESS" or "FAILURE")
        - 'label': str ("BENIGN", "BRUTE_FORCE_LOGIN", "PASS_THE_HASH")
        - 'is_malicious': int (0 or 1)

    Returns
    -------
    Dict[str, Any]
        Proposed action dictionary containing:
        - If BENIGN: {"event_id": int, "action": "NO_ACTION", "reason": str}
        - If THREAT: {"event_id": int, "action": "RESET_CREDENTIALS" | "FLAG_FOR_REVIEW",
                      "reason": str, "detection_confidence": float}
    """
    event_id = event["event_id"]
    prediction = detection_result.get("prediction")
    confidence = detection_result.get("confidence", 0.0)

    # 1. Benign authentication requires no identity remediation
    if prediction == "BENIGN":
        return {
            "event_id": event_id,
            "action": "NO_ACTION",
            "reason": "Authentication activity classified as benign",
        }

    # 2. Threat authentication: select action based on whether authentication succeeded
    auth_type = str(event.get("auth_type", "Unknown"))
    logon_type = str(event.get("logon_type", "Unknown"))
    auth_result = str(event.get("auth_result", "FAILURE"))

    # Inversion rationale: A SUCCESSFUL threat means active unauthorized access (breach) -> RESET_CREDENTIALS.
    # A FAILED threat means the attempt was blocked -> FLAG_FOR_REVIEW.
    if auth_result == "SUCCESS":
        action = "RESET_CREDENTIALS"
        reason = (
            f"Successful unauthorized authentication detected ({auth_type}/{logon_type}/{auth_result}); "
            f"compromised credentials require immediate invalidation."
        )
    else:
        action = "FLAG_FOR_REVIEW"
        reason = (
            f"Failed suspicious authentication attempt ({auth_type}/{logon_type}/{auth_result}); "
            f"flagged for security analyst review."
        )

    return {
        "event_id": event_id,
        "action": action,
        "reason": reason,
        "detection_confidence": confidence,
    }


if __name__ == "__main__":
    model_path = PROJECT_ROOT / "models" / "iam_detector.pkl"
    csv_path = PROJECT_ROOT / "data" / "raw" / "synthetic_iam_events.csv"

    print("=" * 75)
    print("[IAM AGENT PIPELINE TEST]")
    print("NOTE: Evaluating synthetic IAM telemetry for prototype response actions.")
    print("=" * 75)

    # 1. Load or generate synthetic IAM events dataset
    if csv_path.exists():
        print(f"\nLoading synthetic IAM dataset from: {csv_path} ...")
        df = load_synthetic_iam_events(csv_path)
    else:
        print("\nGenerating synthetic IAM dataset (50,000 events, 3% attacks)...")
        df = generate_iam_events(n_events=50000, attack_ratio=0.03, random_state=42)
    print(f"Loaded {len(df):,} synthetic IAM events.")

    # 2. Extract feature columns alignment and load model
    X_train, _, y_train, _, feature_columns = prepare_training_data(df)
    if not model_path.exists():
        print(f"\nModel not found at {model_path}. Training now...")
        model = train_model(X_train, y_train, model_path=model_path)
    else:
        print(f"\nLoading trained model from: {model_path} ...")
        model = joblib.load(model_path)

    # 3. Select 4 test samples matching iam_detection.py test categories
    # Sample 1: Clearly BENIGN (SUCCESS / Interactive / Kerberos)
    sample_clearly_benign = df[
        (df["label"] == "BENIGN")
        & (df["auth_result"] == "SUCCESS")
        & (df["logon_type"] == "Interactive")
        & (df["auth_type"] == "Kerberos")
    ].iloc[0].to_dict()

    # Sample 2: BENIGN-but-FAILURE (natural ambiguous overlap: benign user mistyped credentials)
    sample_benign_failure = df[
        (df["label"] == "BENIGN") & (df["auth_result"] == "FAILURE")
    ].iloc[0].to_dict()

    # Sample 3: BRUTE_FORCE_LOGIN (typical network login attempt)
    sample_brute_force = df[df["label"] == "BRUTE_FORCE_LOGIN"].iloc[0].to_dict()

    # Sample 4: PASS_THE_HASH (typical successful NTLM network authentication)
    sample_pass_the_hash = df[
        (df["label"] == "PASS_THE_HASH") & (df["auth_result"] == "SUCCESS")
    ].iloc[0].to_dict()

    sample_targets = [
        ("1. Clearly BENIGN (SUCCESS / Interactive / Kerberos)", sample_clearly_benign),
        ("2. BENIGN-but-FAILURE (Natural Ambiguous Overlap)", sample_benign_failure),
        ("3. BRUTE_FORCE_LOGIN (Network / NTLM / High Failure)", sample_brute_force),
        ("4. PASS_THE_HASH (Network / NTLM / Successful Hash Auth)", sample_pass_the_hash),
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
        print(
            f"  Auth Attributes      : auth_type={sample_ev['auth_type']}, "
            f"logon_type={sample_ev['logon_type']}, "
            f"auth_result={sample_ev['auth_result']}"
        )
        print(f"  True Label           : {sample_ev['label']} (is_malicious={sample_ev['is_malicious']})")
        print(f"  Detection Result     : {det_result}")
        print(f"  Action Proposal      : {action_proposal}")
        print(f"  Risk Engine Result   : {risk_result}")
        print(f"  Safety Gate Result   : {gate_result}")
    print("\n" + "=" * 75)
