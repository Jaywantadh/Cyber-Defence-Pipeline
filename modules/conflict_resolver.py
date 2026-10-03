"""Conflict Resolver Module.

[PART I ARCHITECTURAL BOUNDARY & SCOPE NOTICE]
==============================================
SCOPE LIMITATION: This module resolves overlapping, competing remediation
proposals strictly between the HOST and IAM domains.
The NETWORK domain is explicitly excluded from entity-level conflict resolution
in this Phase I prototype because the underlying CICIDS2017 network flow dataset
does not contain private internal hostnames or endpoint identifiers that map
to the host telemetry or Active Directory entities.
This exclusion is an intentional, documented Part I architectural boundary,
not an oversight. Network actions remain coordinated at the batch and global
priority level (see modules/coordinator.py).

This module ingests independent gate evaluation results for the SAME endpoint
computer produced by the Host pipeline and IAM pipeline, resolving competing
actions and ordering execution according to defense-in-depth security policy.
"""

from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Union
import joblib

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.host_agent import propose_action as host_propose_action
from modules.host_dataset_generator import (
    generate_host_events,
    load_synthetic_host_events,
)
from modules.host_detection import (
    detect as host_detect,
    prepare_training_data as host_prepare_training_data,
    train_model as host_train_model,
)
from modules.iam_agent import propose_action as iam_propose_action
from modules.iam_dataset_generator import (
    generate_iam_events,
    load_synthetic_iam_events,
)
from modules.iam_detection import (
    detect as iam_detect,
    prepare_training_data as iam_prepare_training_data,
    train_model as iam_train_model,
)
from modules.joint_scenario_generator import generate_joint_scenarios
from modules.risk_engine import compute_risk_score
from modules.safety_gate import evaluate_gate

# Recognized, standard gate decision states
VALID_GATE_DECISIONS: Set[str] = {"AUTO_EXECUTE", "NEEDS_HUMAN_APPROVAL"}


def _priority_order(
    host_action: str,
    host_risk_score: float,
    iam_action: str,
    iam_risk_score: float,
) -> List[str]:
    """Determines the execution sequence of competing response actions between Host and IAM.

    Policy:
        1. IAM actions (credential revocation, account resets) represent 'upstream'
           identity containment. Cutting off an active session prevents the attacker
           from refreshing tokens, re-authenticating, or spawning new processes while
           endpoint remediation is underway. Therefore, IAM actions are ordered
           BEFORE Host actions of equal or lower risk score.
        2. Host actions take precedence ONLY when the Host risk score is strictly higher
           than the IAM risk score (e.g. an active code injection payload requiring
           immediate process interruption over an asynchronous account review).

    Parameters
    ----------
    host_action : str
        Proposed host response action (e.g. 'TERMINATE_PROCESS', 'FLAG_FOR_REVIEW').
    host_risk_score : float
        Calculated quantitative risk score for the host action.
    iam_action : str
        Proposed IAM response action (e.g. 'RESET_CREDENTIALS', 'FLAG_FOR_REVIEW').
    iam_risk_score : float
        Calculated quantitative risk score for the IAM action.

    Returns
    -------
    List[str]
        Ordered list of action strings to be executed sequentially.
    """
    if host_risk_score > iam_risk_score:
        return [host_action, iam_action]
    else:
        return [iam_action, host_action]


def resolve_conflict(
    host_gate_result: Dict[str, Any],
    iam_gate_result: Dict[str, Any],
    computer: str,
) -> Dict[str, Any]:
    """Resolves overlapping remediation proposals for the same computer entity across Host and IAM domains.

    Combines independent evaluation results into a single unified incident response decision.

    Parameters
    ----------
    host_gate_result : Dict[str, Any]
        Output dictionary from Host safety_gate evaluation containing 'final_action',
        'gate_decision', 'gate_reason', and optionally 'risk_score'.
    iam_gate_result : Dict[str, Any]
        Output dictionary from IAM safety_gate evaluation containing 'final_action',
        'gate_decision', 'gate_reason', and optionally 'risk_score'.
    computer : str
        The shared endpoint hostname referenced by both observations (e.g. 'PC-0004').

    Returns
    -------
    Dict[str, Any]
        Resolved incident dictionary containing:
        - 'computer': shared endpoint hostname
        - 'combined_gate_decision': 'AUTO_EXECUTE' | 'NEEDS_HUMAN_APPROVAL'
        - 'execution_order': list of actions to execute sequentially
        - 'combined_reason': explanation of the resolution
        - 'host_result': original host gate result dictionary
        - 'iam_result': original IAM gate result dictionary
    """
    host_action = host_gate_result.get("final_action", "NO_ACTION")
    iam_action = iam_gate_result.get("final_action", "NO_ACTION")

    host_decision = host_gate_result.get("gate_decision")
    iam_decision = iam_gate_result.get("gate_decision")

    host_reason = host_gate_result.get("gate_reason", "")
    iam_reason = iam_gate_result.get("gate_reason", "")

    # Extract risk scores with fallbacks
    host_risk = float(host_gate_result.get("risk_score", 0.0))
    iam_risk = float(iam_gate_result.get("risk_score", 0.0))

    # -------------------------------------------------------------------------
    # CASE 1: Neither domain proposes action (normal benign operations)
    # -------------------------------------------------------------------------
    if host_action == "NO_ACTION" and iam_action == "NO_ACTION":
        if host_decision not in VALID_GATE_DECISIONS or iam_decision not in VALID_GATE_DECISIONS:
            bad_states = []
            if host_decision not in VALID_GATE_DECISIONS:
                bad_states.append(f"Host returned unrecognized state '{host_decision}'")
            if iam_decision not in VALID_GATE_DECISIONS:
                bad_states.append(f"IAM returned unrecognized state '{iam_decision}'")
            return {
                "computer": computer,
                "combined_gate_decision": "NEEDS_HUMAN_APPROVAL",
                "execution_order": [],
                "combined_reason": f"Upstream domain returned an unrecognized state ({'; '.join(bad_states)}); failing closed to NEEDS_HUMAN_APPROVAL.",
                "host_result": host_gate_result,
                "iam_result": iam_gate_result,
            }
        return {
            "computer": computer,
            "combined_gate_decision": "AUTO_EXECUTE",
            "execution_order": [],
            "combined_reason": "No action required on either domain",
            "host_result": host_gate_result,
            "iam_result": iam_gate_result,
        }

    # -------------------------------------------------------------------------
    # CASE 2: Exactly one domain proposes action (single-domain incident)
    # Note: This is not a competing conflict; it represents a single-domain incident
    # evaluated in a joint multi-agent context.
    # -------------------------------------------------------------------------
    if host_action != "NO_ACTION" and iam_action == "NO_ACTION":
        if host_decision not in VALID_GATE_DECISIONS or iam_decision not in VALID_GATE_DECISIONS:
            combined_decision = "NEEDS_HUMAN_APPROVAL"
            combined_reason = f"Upstream domain returned an unrecognized state (Host='{host_decision}', IAM='{iam_decision}'); failing closed to NEEDS_HUMAN_APPROVAL."
        else:
            combined_decision = host_decision
            combined_reason = f"[HOST] {host_reason} | [IAM] No action required"
        return {
            "computer": computer,
            "combined_gate_decision": combined_decision,
            "execution_order": [host_action],
            "combined_reason": combined_reason,
            "host_result": host_gate_result,
            "iam_result": iam_gate_result,
        }

    if iam_action != "NO_ACTION" and host_action == "NO_ACTION":
        if host_decision not in VALID_GATE_DECISIONS or iam_decision not in VALID_GATE_DECISIONS:
            combined_decision = "NEEDS_HUMAN_APPROVAL"
            combined_reason = f"Upstream domain returned an unrecognized state (Host='{host_decision}', IAM='{iam_decision}'); failing closed to NEEDS_HUMAN_APPROVAL."
        else:
            combined_decision = iam_decision
            combined_reason = f"[IAM] {iam_reason} | [HOST] No action required"
        return {
            "computer": computer,
            "combined_gate_decision": combined_decision,
            "execution_order": [iam_action],
            "combined_reason": combined_reason,
            "host_result": host_gate_result,
            "iam_result": iam_gate_result,
        }

    # -------------------------------------------------------------------------
    # CASE 3: BOTH domains propose action (competing cross-domain conflict)
    # Core multi-agent resolution logic:
    # 1. Gate Decision: Validate both decisions are known. If either is unrecognized,
    #    missing, or an error state, fail closed to 'NEEDS_HUMAN_APPROVAL'.
    #    Otherwise, 'NEEDS_HUMAN_APPROVAL' if EITHER side requires review
    #    ("most cautious wins" policy floor).
    # 2. Execution Order: Ordered by _priority_order() rule (upstream IAM priority
    #    unless Host risk strictly exceeds IAM risk).
    # -------------------------------------------------------------------------
    if host_decision not in VALID_GATE_DECISIONS or iam_decision not in VALID_GATE_DECISIONS:
        combined_gate_decision = "NEEDS_HUMAN_APPROVAL"
        bad_states = []
        if host_decision not in VALID_GATE_DECISIONS:
            bad_states.append(f"Host returned unrecognized state '{host_decision}'")
        if iam_decision not in VALID_GATE_DECISIONS:
            bad_states.append(f"IAM returned unrecognized state '{iam_decision}'")
        combined_reason = (
            f"Upstream domain returned an unrecognized state ({'; '.join(bad_states)}); "
            f"failing closed to NEEDS_HUMAN_APPROVAL."
        )
    elif host_decision == "NEEDS_HUMAN_APPROVAL" or iam_decision == "NEEDS_HUMAN_APPROVAL":
        combined_gate_decision = "NEEDS_HUMAN_APPROVAL"
        combined_reason = f"[IAM] {iam_reason} | [HOST] {host_reason}"
    else:
        combined_gate_decision = "AUTO_EXECUTE"
        combined_reason = f"[IAM] {iam_reason} | [HOST] {host_reason}"

    execution_order = _priority_order(
        host_action=host_action,
        host_risk_score=host_risk,
        iam_action=iam_action,
        iam_risk_score=iam_risk,
    )

    return {
        "computer": computer,
        "combined_gate_decision": combined_gate_decision,
        "execution_order": execution_order,
        "combined_reason": combined_reason,
        "host_result": host_gate_result,
        "iam_result": iam_gate_result,
    }


if __name__ == "__main__":
    print("=" * 95)
    print("MULTI-AGENT CONFLICT RESOLVER: HOST & IAM JOINT INCIDENT RESOLUTION")
    print("[SCOPE NOTICE]: Entity-level conflict resolution active for shared Host and IAM telemetry.")
    print("=" * 95)

    host_model_path = PROJECT_ROOT / "models" / "host_detector.pkl"
    iam_model_path = PROJECT_ROOT / "models" / "iam_detector.pkl"

    # 1. Load or train Host model
    if host_model_path.exists():
        print(f"Loading Host model from: {host_model_path}")
        host_model = joblib.load(host_model_path)
    else:
        print("Host model not found. Training on synthetic host events...")
        host_csv_path = PROJECT_ROOT / "data" / "raw" / "synthetic_host_events.csv"
        if host_csv_path.exists():
            hdf = load_synthetic_host_events(host_csv_path)
        else:
            hdf = generate_host_events(n_events=50000, attack_ratio=0.03, random_state=42)
        h_X_train, _, h_y_train, _, _ = host_prepare_training_data(hdf)
        host_model = host_train_model(h_X_train, h_y_train, model_path=host_model_path)

    # 2. Load or train IAM model
    if iam_model_path.exists():
        print(f"Loading IAM model from: {iam_model_path}")
        iam_model = joblib.load(iam_model_path)
    else:
        print("IAM model not found. Training on synthetic IAM events...")
        iam_csv_path = PROJECT_ROOT / "data" / "raw" / "synthetic_iam_events.csv"
        if iam_csv_path.exists():
            idf = load_synthetic_iam_events(iam_csv_path)
        else:
            idf = generate_iam_events(n_events=50000, attack_ratio=0.03, random_state=42)
        i_X_train, _, i_y_train, _, _ = iam_prepare_training_data(idf)
        iam_model = iam_train_model(i_X_train, i_y_train, model_path=iam_model_path)

    host_feature_cols = list(host_model.feature_names_in_)
    iam_feature_cols = list(iam_model.feature_names_in_)

    # 3. Generate 200 joint scenarios
    print("\nGenerating 200 paired joint scenarios (20 shared computers)...")
    scenarios = generate_joint_scenarios(n_scenarios=200, random_state=42)

    # 4. Process scenarios independently and resolve conflicts
    resolutions: List[Dict[str, Any]] = []

    for sc in scenarios:
        # Run Host pipeline
        h_ev = sc["host_event"]
        h_det = host_detect(host_model, host_feature_cols, h_ev)
        h_act = host_propose_action(h_det, h_ev)
        h_risk = compute_risk_score(h_act)
        h_gate = evaluate_gate(h_risk)
        h_gate["risk_score"] = h_risk.get("risk_score", 0.0)

        # Run IAM pipeline
        i_ev = sc["iam_event"]
        i_det = iam_detect(iam_model, iam_feature_cols, i_ev)
        i_act = iam_propose_action(i_det, i_ev)
        i_risk = compute_risk_score(i_act)
        i_gate = evaluate_gate(i_risk)
        i_gate["risk_score"] = i_risk.get("risk_score", 0.0)

        # Resolve conflict
        res = resolve_conflict(h_gate, i_gate, sc["computer"])
        res["scenario_id"] = sc["scenario_id"]
        res["scenario_type"] = sc["scenario_type"]
        resolutions.append(res)

    # 5. Print summary counts by combined gate decision
    print("\n" + "=" * 95)
    print("RESOLVED JOINT SCENARIOS SUMMARY")
    print("=" * 95)
    decision_counts = Counter(r["combined_gate_decision"] for r in resolutions)
    for dec, cnt in decision_counts.items():
        pct = (cnt / len(resolutions)) * 100.0
        print(f"  {dec:<24}: {cnt:>4} ({pct:>5.1f}%)")

    # 6. Print all "BOTH_ATTACK" scenario resolutions in full
    both_attacks = [r for r in resolutions if r["scenario_type"] == "BOTH_ATTACK"]

    print("\n" + "=" * 125)
    print(f"'BOTH_ATTACK' SCENARIO RESOLUTIONS (Total: {len(both_attacks)} Multi-Domain Collisions)")
    print("=" * 125)
    header = (
        f"{'ID':<4} | {'Computer':<9} | {'Host Action (Risk)':<28} | "
        f"{'IAM Action (Risk)':<28} | {'Execution Order':<38} | {'Combined Decision'}"
    )
    print(header)
    print("-" * 125)

    for r in both_attacks:
        h_res = r["host_result"]
        i_res = r["iam_result"]
        h_str = f"{h_res.get('final_action')} ({h_res.get('risk_score', 0.0):.4f})"
        i_str = f"{i_res.get('final_action')} ({i_res.get('risk_score', 0.0):.4f})"
        order_str = str(r["execution_order"])
        print(
            f"{r['scenario_id']:<4} | "
            f"{r['computer']:<9} | "
            f"{h_str:<28} | "
            f"{i_str:<28} | "
            f"{order_str:<38} | "
            f"{r['combined_gate_decision']}"
        )

    print("=" * 125)

    print("\n" + "=" * 95)
    print("FAIL-CLOSED TEST: UPSTREAM 'ERROR' DECISION FORCES NEEDS_HUMAN_APPROVAL")
    print("=" * 95)
    error_host_gate = {
        "event_id": 101,
        "final_action": "FLAG_FOR_REVIEW",
        "gate_decision": "ERROR",
        "gate_reason": "Pipeline execution error in host subsystem",
        "risk_score": 0.0,
    }
    normal_iam_gate = {
        "event_id": 102,
        "final_action": "FLAG_FOR_REVIEW",
        "gate_decision": "AUTO_EXECUTE",
        "gate_reason": "Low risk, auto execution permitted",
        "risk_score": 0.35,
    }
    resolved_error = resolve_conflict(error_host_gate, normal_iam_gate, computer="PC-0099")
    print(f"Host Input Decision : {error_host_gate['gate_decision']}")
    print(f"IAM Input Decision  : {normal_iam_gate['gate_decision']}")
    print(f"Combined Decision   : {resolved_error['combined_gate_decision']}")
    print(f"Combined Reason     : {resolved_error['combined_reason']}")
    assert resolved_error["combined_gate_decision"] == "NEEDS_HUMAN_APPROVAL"
    assert "unrecognized state" in resolved_error["combined_reason"].lower()
    print("Confirmed: Upstream ERROR decision forces combined NEEDS_HUMAN_APPROVAL (fail-closed).")
    print("=" * 95)
