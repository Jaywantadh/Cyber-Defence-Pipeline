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
from modules.risk_engine import (
    ACTION_SEVERITY_WEIGHTS,
    HIGH_RISK_THRESHOLD,
    compute_risk_score,
)

# Named constant for maximum consequence policy enforcement.
# Actions carrying this severity weight always require human review regardless of detection confidence.
ALWAYS_REVIEW_SEVERITY: float = 1.0

# Synchronized threshold for autonomous execution boundary.
# Actions with risk scores at or above this bound require human approval.
RISK_AUTO_EXECUTE_BOUND: float = HIGH_RISK_THRESHOLD


def evaluate_gate(risk_result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates whether a proposed action can be automatically executed or blocked for human review.

    Safety Logic:
        Higher risk and severity imply GREATER caution:
        - If action is "NO_ACTION": AUTO_EXECUTE
        - If action severity weight == ALWAYS_REVIEW_SEVERITY (1.0): NEEDS_HUMAN_APPROVAL
          (maximum-severity actions always receive human review, regardless of confidence)
        - If risk_score >= RISK_AUTO_EXECUTE_BOUND (0.7): NEEDS_HUMAN_APPROVAL (blocked)
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
    event_id = risk_result.get("event_id", "UNKNOWN")
    action = risk_result.get("action", "NO_ACTION")

    # Fail-closed check: if 'risk_score' is missing entirely from upstream risk_result, force NEEDS_HUMAN_APPROVAL
    if "risk_score" not in risk_result or risk_result["risk_score"] is None:
        print(
            f"Warning: 'risk_score' key missing from risk_result (event_id={event_id}). "
            f"Defaulting to NEEDS_HUMAN_APPROVAL for fail-closed safety."
        )
        return {
            "event_id": event_id,
            "gate_decision": "NEEDS_HUMAN_APPROVAL",
            "final_action": action,
            "gate_reason": (
                "Upstream 'risk_score' was missing from risk_result dictionary; "
                "failing closed to NEEDS_HUMAN_APPROVAL as a safety default, not a real risk assessment."
            ),
        }

    risk_score = round(float(risk_result["risk_score"]), 4)

    # 1. No action required
    if action == "NO_ACTION":
        return {
            "event_id": event_id,
            "gate_decision": "AUTO_EXECUTE",
            "final_action": "NO_ACTION",
            "gate_reason": "No action required",
        }

    # 2. Maximum consequence policy override: actions with maximum severity always require human review
    if action not in ACTION_SEVERITY_WEIGHTS:
        print(
            f"Warning: Unrecognized action '{action}' encountered in safety gate (event_id={event_id}). "
            f"Defaulting to maximum severity weight (1.0) for fail-closed safety."
        )
        # Unrecognized actions represent novel or malformed proposals; defaulting to 1.0 (fail-closed)
        severity_weight = 1.0
    else:
        severity_weight = ACTION_SEVERITY_WEIGHTS[action]

    if severity_weight >= ALWAYS_REVIEW_SEVERITY:
        return {
            "event_id": event_id,
            "gate_decision": "NEEDS_HUMAN_APPROVAL",
            "final_action": action,
            "gate_reason": (
                f"Action '{action}' carries maximum severity weight ({ALWAYS_REVIEW_SEVERITY}); "
                f"requires human approval regardless of detection confidence, per policy: "
                f"the most consequential actions always receive human review."
            ),
        }

    # 3. Safety Gate evaluation: High risk requires human authorization
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

    print("\n" + "=" * 80)
    print("FAIL-CLOSED TEST: MISSING 'risk_score' KEY DEFAULTS TO NEEDS_HUMAN_APPROVAL")
    print("=" * 80)
    malformed_risk_result = {
        "event_id": 88888,
        "action": "FLAG_FOR_REVIEW",
        # 'risk_score' key deliberately omitted
    }
    forced_gate = evaluate_gate(malformed_risk_result)
    print(f"Input Dict  : {malformed_risk_result}")
    print(f"Gate Result : {forced_gate}")
    assert forced_gate["gate_decision"] == "NEEDS_HUMAN_APPROVAL"
    assert "missing" in forced_gate["gate_reason"].lower()
    print("Confirmed: Missing risk_score resulted in NEEDS_HUMAN_APPROVAL (fail-closed).")

    print("\n" + "=" * 80)
    print("ADVERSARIAL GATE-BYPASS CHECK (1,000 ADVERSARIAL INPUTS)")
    print("=" * 80)
    from collections import Counter
    import contextlib
    import io
    import random

    random.seed(42)
    known_actions = ["NO_ACTION", "FLAG_FOR_REVIEW", "TERMINATE_PROCESS", "ISOLATE_FLOW", "RESET_CREDENTIALS"]
    unknown_actions = ["CUSTOM_BYPASS", "MALICIOUS_OVERRIDE", "UNKNOWN_OP", "", "NULL_ACTION", "ADMIN_DISABLE_GATE"]
    all_test_actions = known_actions + unknown_actions

    eval_results = []
    failures = []
    for _ in range(1000):
        r_dict = {}
        if random.random() > 0.15:
            r_dict["event_id"] = random.randint(1000, 999999)
        r_prob = random.random()
        if r_prob > 0.25:
            r_dict["risk_score"] = round(random.uniform(0.0, 1.0), 4)
        elif r_prob > 0.1:
            r_dict["risk_score"] = None
        if random.random() > 0.15:
            r_dict["action"] = random.choice(all_test_actions)

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = evaluate_gate(r_dict)
            dec = res.get("gate_decision")
            eval_results.append((r_dict, dec))
            if dec not in ("AUTO_EXECUTE", "NEEDS_HUMAN_APPROVAL"):
                failures.append((r_dict, res, f"Invalid decision: {dec}"))
        except Exception as exc:
            eval_results.append((r_dict, "EXCEPTION"))
            failures.append((r_dict, None, f"Uncaught exception: {exc}"))

    # Categorize generated test cases into requirements (a) through (f)
    cat_a = [(c, d) for c, d in eval_results if c.get("action") in known_actions and isinstance(c.get("risk_score"), (int, float))]
    cat_b = [(c, d) for c, d in eval_results if "action" in c and c["action"] not in known_actions and isinstance(c.get("risk_score"), (int, float)) and c["risk_score"] < 0.7]
    cat_c = [(c, d) for c, d in eval_results if "action" in c and c["action"] not in known_actions and isinstance(c.get("risk_score"), (int, float)) and c["risk_score"] >= 0.7]
    cat_d = [(c, d) for c, d in eval_results if "risk_score" not in c or c.get("risk_score") is None]
    cat_e = [(c, d) for c, d in eval_results if "action" not in c]
    cat_f = [(c, d) for c, d in eval_results if "event_id" not in c]

    categories = [
        ("(a) Known action + valid risk_score", cat_a),
        ("(b) Unknown/unrecognized action + risk_score < 0.7", cat_b),
        ("(c) Unknown/unrecognized action + risk_score >= 0.7", cat_c),
        ("(d) Missing risk_score key (omitted or None)", cat_d),
        ("(e) Missing action key", cat_e),
        ("(f) Missing event_id key", cat_f),
    ]

    print("\n--- CATEGORY BREAKDOWN ACROSS 1,000 ADVERSARIAL CASES ---")
    for name, items in categories:
        counts = Counter(d for _, d in items)
        auto_cnt = counts.get("AUTO_EXECUTE", 0)
        human_cnt = counts.get("NEEDS_HUMAN_APPROVAL", 0)
        other_cnt = len(items) - auto_cnt - human_cnt
        print(f"  {name:52s} : Count = {len(items):3d} | AUTO_EXECUTE: {auto_cnt:3d} | NEEDS_HUMAN_APPROVAL: {human_cnt:3d}" + (f" | OTHER: {other_cnt}" if other_cnt else ""))

    print(f"\n--- CATEGORY (b) ISOLATION: UNRECOGNIZED ACTION + LOW RISK SCORE (< 0.7) [Total: {len(cat_b)}] ---")
    for idx, (inp, dec) in enumerate(cat_b, 1):
        print(f"  [{idx:03d}] Input: {inp} -> gate_decision: {dec}")

    # Verify category (b) fail-closed behavior
    cat_b_auto = [inp for inp, dec in cat_b if dec == "AUTO_EXECUTE"]
    if cat_b_auto:
        print(f"\nCRITICAL FAIL-OPEN GAP: {len(cat_b_auto)} category (b) cases returned AUTO_EXECUTE!")
    else:
        print(f"\nConfirmed: 100% of category (b) cases ({len(cat_b)}/{len(cat_b)}) returned NEEDS_HUMAN_APPROVAL (fail-closed).")

    if not failures and not cat_b_auto:
        print("\nPASSED: gate never returned an invalid state across 1000 adversarial inputs")
    else:
        print(f"\nFAILED: {len(failures)} adversarial cases failed:")
        for r_in, r_out, err in failures[:5]:
            print(f"  Input: {r_in} -> Output: {r_out} -> Error: {err}")
        assert False, f"{len(failures)} adversarial test cases failed"

