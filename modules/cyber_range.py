"""Simulated Cyber Range Module.

[SIMULATED CYBER RANGE SCOPE BOUNDARY NOTICE]
==============================================
NOTICE: This module is a SIMULATED cyber range designed to replay sequenced,
multi-stage attack campaigns (synthesized from historical network flow captures
and synthetic endpoint/authentication telemetry) through the full multi-agent
defense pipeline end-to-end.

DELIBERATE SCOPE BOUNDARY:
This is NOT a live interactive range with real attacker tooling, active
command-and-control infrastructure, or live packet capture. All telemetry streams
represent deterministic, offline replay sequences.
Furthermore, the Stage 1 NETWORK domain (derived from the CICIDS2017 dataset)
carries no internal hostname or endpoint identity linking it to the Stage 2/3
HOST and IAM domains. The linkage across domains is a narrative sequencing
device representing "the same incident timeline," not a literal shared-identity
correlation chain (identical to the Part I architectural boundary documented
in modules/conflict_resolver.py).
"""

from datetime import datetime, timezone
import inspect
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.audit_log import init_db, log_decision
from modules.conflict_resolver import resolve_conflict
from modules.dataset_loader import load_raw_data
from modules.detection import detect as network_detect
from modules.explainability import build_explainability_report, get_attck_mapping
from modules.host_agent import propose_action as host_propose_action
from modules.host_dataset_generator import (
    ATTACK_PROCESS_PAIRS,
    generate_host_events,
)
from modules.host_detection import detect as host_detect
from modules.iam_agent import propose_action as iam_propose_action
from modules.iam_dataset_generator import (
    generate_iam_events,
)
from modules.iam_detection import detect as iam_detect
from modules.ingestion import normalize_events
from modules.llm_reasoning import explain_incident
from modules.network_agent import propose_action as network_propose_action
from modules.risk_engine import ACTION_SEVERITY_WEIGHTS, compute_risk_score
from modules.safety_gate import evaluate_gate

# Module-level model and telemetry cache for performance
_MODELS_CACHE: Optional[Dict[str, Any]] = None
_CACHED_NETWORK_EVENTS: Optional[List[Dict[str, Any]]] = None


def load_range_models() -> Dict[str, Any]:
    """Loads and caches machine learning models and feature columns for all three domains.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing loaded models and feature column lists for
        NETWORK, HOST, and IAM domains.
    """
    global _MODELS_CACHE
    if _MODELS_CACHE is not None:
        return _MODELS_CACHE

    net_model_path = PROJECT_ROOT / "models" / "threat_detector.pkl"
    host_model_path = PROJECT_ROOT / "models" / "host_detector.pkl"
    iam_model_path = PROJECT_ROOT / "models" / "iam_detector.pkl"

    if not net_model_path.exists():
        raise FileNotFoundError(f"Network model not found at {net_model_path}")
    if not host_model_path.exists():
        raise FileNotFoundError(f"Host model not found at {host_model_path}")
    if not iam_model_path.exists():
        raise FileNotFoundError(f"IAM model not found at {iam_model_path}")

    net_model = joblib.load(net_model_path)
    host_model = joblib.load(host_model_path)
    iam_model = joblib.load(iam_model_path)

    net_cols = list(getattr(net_model, "feature_names_in_", []))
    host_cols = list(getattr(host_model, "feature_names_in_", []))
    iam_cols = list(getattr(iam_model, "feature_names_in_", []))

    _MODELS_CACHE = {
        "network": {"model": net_model, "features": net_cols},
        "host": {"model": host_model, "features": host_cols},
        "iam": {"model": iam_model, "features": iam_cols},
    }
    return _MODELS_CACHE


def _get_network_events(random_state: int = 42) -> List[Dict[str, Any]]:
    """Loads and caches normalized network telemetry events from CICIDS2017."""
    global _CACHED_NETWORK_EVENTS
    if _CACHED_NETWORK_EVENTS is not None:
        return _CACHED_NETWORK_EVENTS

    csv_path = PROJECT_ROOT / "data" / "raw" / "Tuesday-WorkingHours.pcap_ISCX.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CICIDS2017 CSV not found at {csv_path}")

    raw_df = load_raw_data(csv_path)
    # Stratified subset: retain all attack rows and sample representative benign rows
    attack_mask = raw_df["Label"] != "BENIGN"
    benign_mask = raw_df["Label"] == "BENIGN"

    attacks_df = raw_df[attack_mask]
    benigns_df = raw_df[benign_mask].sample(n=min(5000, len(raw_df[benign_mask])), random_state=random_state)
    sample_df = pd.concat([attacks_df, benigns_df], ignore_index=True)

    _CACHED_NETWORK_EVENTS = normalize_events(sample_df)
    return _CACHED_NETWORK_EVENTS


def generate_multi_stage_intrusion_campaign(
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """Generates a sequenced 3-stage advanced persistent threat (APT) intrusion campaign.

    Campaign Architecture:
    ----------------------
    - Stage 1 (NETWORK): Initial access / ingress via SSH brute-force attack (SSH-Patator)
      derived from the CICIDS2017 dataset.
    - Stage 2 (HOST): Execution and privilege escalation via suspicious process injection
      (e.g., winword.exe -> powershell.exe) on target endpoint 'PC-0004'.
    - Stage 3 (IAM): Lateral movement and credential theft via Pass-the-Hash authentication
      targeting the SAME endpoint 'PC-0004'.

    Architectural Scope Notice:
    ---------------------------
    Stage 1 (NETWORK) has no entity identifier linking it directly to Stages 2-3 because
    CICIDS2017 network flow records do not carry Windows hostnames. This is a narrative
    sequencing device representing "the same chronological incident timeline," not a literal
    shared-identity correlation chain. Stages 2 and 3 share a literal entity ('PC-0004').

    Parameters
    ----------
    random_state : int, default=42
        Seed for deterministic event selection.

    Returns
    -------
    List[Dict[str, Any]]
        List of 3 stage specification dictionaries.
    """
    # 1. Stage 1: Network SSH-Patator Event (high-throughput ingress flow)
    net_events = _get_network_events(random_state=random_state)
    ssh_events = [
        e for e in net_events
        if e.get("label") == "SSH-Patator"
        and float(e["features"].get("Flow Packets/s", 0.0)) >= 5.88638
    ]
    if not ssh_events:
        ssh_events = [e for e in net_events if e.get("label") == "SSH-Patator"]
    selected_ssh = ssh_events[random_state % len(ssh_events)]

    stage_1 = {
        "campaign_id": "INTRUSION-01",
        "stage_number": 1,
        "stage_name": "Initial Ingress: SSH Brute Force Reconnaissance",
        "source_agent": "NETWORK",
        "event": selected_ssh,
        "simulated_timestamp": "2026-10-05T08:00:00Z",
    }

    # 2. Stage 2: Host Suspicious Process Injection on PC-0004
    host_df = generate_host_events(n_events=1000, attack_ratio=0.1, random_state=random_state)
    inj_events = host_df[host_df["label"] == "SUSPICIOUS_PROCESS_INJECTION"]
    if inj_events.empty:
        raise ValueError("Could not find SUSPICIOUS_PROCESS_INJECTION events in host generator.")
    host_event = inj_events.iloc[random_state % len(inj_events)].to_dict()
    host_event["computer"] = "PC-0004"

    stage_2 = {
        "campaign_id": "INTRUSION-01",
        "stage_number": 2,
        "stage_name": "Endpoint Execution: Suspicious Process Injection",
        "source_agent": "HOST",
        "event": host_event,
        "simulated_timestamp": "2026-10-05T08:15:00Z",
    }

    # 3. Stage 3: IAM Pass-the-Hash targeting PC-0004
    iam_df = generate_iam_events(n_events=1000, attack_ratio=0.1, random_state=random_state)
    pth_events = iam_df[(iam_df["label"] == "PASS_THE_HASH") & (iam_df["auth_result"] == "SUCCESS")]
    if pth_events.empty:
        raise ValueError("Could not find successful PASS_THE_HASH events in IAM generator.")
    iam_event = pth_events.iloc[random_state % len(pth_events)].to_dict()
    iam_event["target_computer"] = "PC-0004"
    iam_event["computer"] = "PC-0004"

    stage_3 = {
        "campaign_id": "INTRUSION-01",
        "stage_number": 3,
        "stage_name": "Lateral Movement: Pass-the-Hash Credential Abuse",
        "source_agent": "IAM",
        "event": iam_event,
        "simulated_timestamp": "2026-10-05T08:30:00Z",
    }

    return [stage_1, stage_2, stage_3]


def generate_normal_operations_day(
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """Generates a 10-stage simulated day of routine enterprise operations.

    Features predominantly BENIGN events across Network, Host, and IAM domains,
    interspersed with deliberately included 'ambiguous overlap' rows from each domain
    to test false-positive resistance and safety-gate calibration under routine load.

    Parameters
    ----------
    random_state : int, default=42
        Seed for deterministic event selection.

    Returns
    -------
    List[Dict[str, Any]]
        List of 10 stage specification dictionaries.
    """
    stages: List[Dict[str, Any]] = []
    base_time_str = "2026-10-05T09:00:00Z"
    timestamps = [
        "2026-10-05T09:00:00Z",
        "2026-10-05T09:45:00Z",
        "2026-10-05T10:30:00Z",
        "2026-10-05T11:15:00Z",
        "2026-10-05T12:00:00Z",
        "2026-10-05T13:00:00Z",
        "2026-10-05T13:45:00Z",
        "2026-10-05T14:30:00Z",
        "2026-10-05T15:15:00Z",
        "2026-10-05T16:00:00Z",
    ]

    # --- Domain 1: NETWORK (4 stages: 3 routine benign, 1 ambiguous overlap) ---
    net_events = _get_network_events(random_state=random_state)
    benign_net = [e for e in net_events if e.get("label") == "BENIGN"]

    # Select 3 routine benign network events
    for i in range(3):
        idx = (random_state * 7 + i * 31) % len(benign_net)
        stages.append({
            "campaign_id": "NORMALDAY-01",
            "stage_number": len(stages) + 1,
            "stage_name": f"Routine Network Traffic: Flow #{i + 1}",
            "source_agent": "NETWORK",
            "event": benign_net[idx],
            "simulated_timestamp": timestamps[len(stages)],
        })

    # Ambiguous Network event: find benign event with highest Flow Packets/s or high activity
    ambiguous_net = max(
        benign_net[:500],
        key=lambda e: float(e["features"].get("Flow Packets/s", 0.0)),
    )
    stages.append({
        "campaign_id": "NORMALDAY-01",
        "stage_number": len(stages) + 1,
        "stage_name": "Ambiguous Network Flow: High-Throughput Burst",
        "source_agent": "NETWORK",
        "event": ambiguous_net,
        "simulated_timestamp": timestamps[len(stages)],
    })

    # --- Domain 2: HOST (3 stages: 2 routine benign, 1 ambiguous overlap) ---
    host_df = generate_host_events(n_events=2000, attack_ratio=0.03, random_state=random_state)
    attack_pairs_set = set(ATTACK_PROCESS_PAIRS)

    # Routine host benign (non-overlap pairs)
    benign_non_overlap = host_df[
        (host_df["is_malicious"] == 0)
        & (~host_df.apply(lambda r: (r["parent_process"], r["process_name"]) in attack_pairs_set, axis=1))
    ]
    for i in range(2):
        ev = benign_non_overlap.iloc[(random_state + i * 17) % len(benign_non_overlap)].to_dict()
        stages.append({
            "campaign_id": "NORMALDAY-01",
            "stage_number": len(stages) + 1,
            "stage_name": f"Routine Host Process Execution: {ev['parent_process']} -> {ev['process_name']}",
            "source_agent": "HOST",
            "event": ev,
            "simulated_timestamp": timestamps[len(stages)],
        })

    # Ambiguous host event: benign process matching an attack pair structure (e.g. administrative script launch)
    ambiguous_host_df = host_df[
        (host_df["is_malicious"] == 0)
        & (host_df.apply(lambda r: (r["parent_process"], r["process_name"]) in attack_pairs_set, axis=1))
    ]
    ambiguous_host_ev = ambiguous_host_df.iloc[random_state % len(ambiguous_host_df)].to_dict()
    stages.append({
        "campaign_id": "NORMALDAY-01",
        "stage_number": len(stages) + 1,
        "stage_name": f"Ambiguous Host Process: {ambiguous_host_ev['parent_process']} -> {ambiguous_host_ev['process_name']}",
        "source_agent": "HOST",
        "event": ambiguous_host_ev,
        "simulated_timestamp": timestamps[len(stages)],
    })

    # --- Domain 3: IAM (3 stages: 2 routine benign, 1 ambiguous overlap) ---
    iam_df = generate_iam_events(n_events=2000, attack_ratio=0.03, random_state=random_state)

    # Routine IAM benign (Kerberos interactive/network SUCCESS)
    benign_success = iam_df[(iam_df["label"] == "BENIGN") & (iam_df["auth_result"] == "SUCCESS")]
    for i in range(2):
        ev = benign_success.iloc[(random_state + i * 23) % len(benign_success)].to_dict()
        stages.append({
            "campaign_id": "NORMALDAY-01",
            "stage_number": len(stages) + 1,
            "stage_name": f"Routine Authentication: {ev['auth_type']} ({ev['logon_type']}) SUCCESS",
            "source_agent": "IAM",
            "event": ev,
            "simulated_timestamp": timestamps[len(stages)],
        })

    # Ambiguous IAM event: benign user authentication failure (mistyped password)
    benign_failure = iam_df[(iam_df["label"] == "BENIGN") & (iam_df["auth_result"] == "FAILURE")]
    ambiguous_iam_ev = benign_failure.iloc[random_state % len(benign_failure)].to_dict()
    stages.append({
        "campaign_id": "NORMALDAY-01",
        "stage_number": len(stages) + 1,
        "stage_name": f"Ambiguous Authentication: Benign User Logon FAILURE ({ambiguous_iam_ev['auth_type']})",
        "source_agent": "IAM",
        "event": ambiguous_iam_ev,
        "simulated_timestamp": timestamps[len(stages)],
    })

    return stages


def run_campaign(
    campaign_events: List[Dict[str, Any]],
    db_path: Union[str, Path] = "logs/audit.db",
) -> Dict[str, Any]:
    """Replays a campaign sequence through the end-to-end multi-agent defense pipeline.

    For each stage:
        1. Runs domain detection, action proposal, risk engine, and safety gate.
        2. Builds model-grounded explainability report (MITRE ATT&CK alignment + top features).
        3. Invokes LLM reasoning explanation if the event is escalated (NEEDS_HUMAN_APPROVAL),
           with fail-safe fallback error handling.
        4. Logs the immutable decision trail to the audit database with campaign identifiers.
        5. Detects cross-domain entity collisions on shared endpoints (HOST & IAM) and
           executes conflict resolution per security policy.

    Parameters
    ----------
    campaign_events : List[Dict[str, Any]]
        List of campaign stage dictionaries.
    db_path : str or Path, optional
        Target SQLite audit database path (default: 'logs/audit.db').

    Returns
    -------
    Dict[str, Any]
        Structured campaign report containing stage-by-stage results and overall summary.
    """
    if not campaign_events:
        raise ValueError("Cannot run campaign on an empty event list.")

    models_data = load_range_models()
    campaign_id = campaign_events[0].get("campaign_id", "UNKNOWN-CAMPAIGN")

    start_time = time.perf_counter()
    stage_reports: List[Dict[str, Any]] = []

    # Map to track domain evaluations by computer entity for conflict resolution
    entity_gate_results: Dict[str, Dict[str, Any]] = {}

    for stage in campaign_events:
        stage_num = stage["stage_number"]
        stage_name = stage["stage_name"]
        source_agent = stage["source_agent"]
        event = stage["event"]

        # 1. Execute Domain Defense Pipeline
        if source_agent == "NETWORK":
            net_model = models_data["network"]["model"]
            net_cols = models_data["network"]["features"]

            det_res = network_detect(net_model, event)
            act_prop = network_propose_action(det_res, event)
            risk_res = compute_risk_score(act_prop)
            gate_res = evaluate_gate(risk_res)
            exp_rep = build_explainability_report(
                det_res, event, net_model, net_cols, label=event.get("label")
            )

        elif source_agent == "HOST":
            host_model = models_data["host"]["model"]
            host_cols = models_data["host"]["features"]

            det_res = host_detect(host_model, host_cols, event)
            act_prop = host_propose_action(det_res, event)
            risk_res = compute_risk_score(act_prop)
            gate_res = evaluate_gate(risk_res)
            exp_rep = build_explainability_report(
                det_res, event, host_model, host_cols, label=event.get("label")
            )

        elif source_agent == "IAM":
            iam_model = models_data["iam"]["model"]
            iam_cols = models_data["iam"]["features"]

            det_res = iam_detect(iam_model, iam_cols, event)
            act_prop = iam_propose_action(det_res, event)
            risk_res = compute_risk_score(act_prop)
            gate_res = evaluate_gate(risk_res)
            exp_rep = build_explainability_report(
                det_res, event, iam_model, iam_cols, label=event.get("label")
            )
        else:
            raise ValueError(f"Unrecognized source_agent '{source_agent}' in campaign.")

        # Ensure risk_score is available on gate_res for downstream conflict resolver
        gate_res["risk_score"] = risk_res.get("risk_score", 0.0)

        # 2. Extract ATT&CK mapping
        mitre_mapping = exp_rep.get("attck_mapping")
        mitre_id = mitre_mapping.get("technique_id") if mitre_mapping else None

        # 3. LLM Incident Explanation (for escalated events)
        llm_explanation = None
        if gate_res.get("gate_decision") == "NEEDS_HUMAN_APPROVAL":
            try:
                gate_for_llm = dict(gate_res)
                gate_for_llm["source_agent"] = source_agent
                gate_for_llm["risk_score"] = risk_res.get("risk_score", 0.0)
                llm_res = explain_incident(gate_for_llm)
                llm_explanation = llm_res.get("explanation")
            except Exception as exc:
                llm_explanation = f"LLM reasoning bypassed or unavailable ({exc})"

        # 4. Log Decision to Immutable Audit Database
        log_decision(
            event=event,
            detection_result=det_res,
            action_proposal=act_prop,
            risk_result=risk_res,
            gate_result=gate_res,
            source_agent=source_agent,
            llm_explanation=llm_explanation,
            mitre_technique_id=mitre_id,
            db_path=db_path,
            campaign_id=campaign_id,
            campaign_stage=f"Stage {stage_num}: {stage_name}",
        )

        # 5. Determine Detection Accuracy (prediction matches true label ground truth)
        true_label = event.get("label")
        prediction = det_res.get("prediction")
        if true_label == "BENIGN":
            detection_correct = (prediction == "BENIGN")
        else:
            detection_correct = (prediction == "THREAT")

        # Entity tracking for cross-domain collision checks
        comp = event.get("computer") or event.get("target_computer")
        if comp and source_agent in ("HOST", "IAM"):
            if comp not in entity_gate_results:
                entity_gate_results[comp] = {}
            entity_gate_results[comp][source_agent] = gate_res

        proposed_action_str = act_prop.get("action", "NO_ACTION")
        severity_weight = float(ACTION_SEVERITY_WEIGHTS.get(proposed_action_str, 0.0))

        stage_record = {
            "stage_number": stage_num,
            "stage_name": stage_name,
            "source_agent": source_agent,
            "true_label": true_label,
            "prediction": prediction,
            "detection_correct": detection_correct,
            "proposed_action": proposed_action_str,
            "action_severity_weight": severity_weight,
            "action": gate_res.get("final_action"),
            "risk_score": risk_res.get("risk_score"),
            "risk_level": risk_res.get("risk_level"),
            "gate_decision": gate_res.get("gate_decision"),
            "mitre_mapping": mitre_mapping,
            "human_review_needed": (gate_res.get("gate_decision") == "NEEDS_HUMAN_APPROVAL"),
            "gate_reason": gate_res.get("gate_reason"),
            "llm_explanation": llm_explanation,
            "computer": comp,
            "conflict_resolution": None,
        }
        stage_reports.append(stage_record)

    # 6. Check for Cross-Domain Entity Collisions (Host & IAM on same computer)
    overall_conflict_resolution = None
    for comp, agents_dict in entity_gate_results.items():
        if "HOST" in agents_dict and "IAM" in agents_dict:
            overall_conflict_resolution = resolve_conflict(
                host_gate_result=agents_dict["HOST"],
                iam_gate_result=agents_dict["IAM"],
                computer=comp,
            )
            # Attach conflict resolution details to the affected stages
            for s in stage_reports:
                if s["computer"] == comp:
                    s["conflict_resolution"] = overall_conflict_resolution
            break

    # 7. Aggregate Summary Metrics
    total_stages = len(stage_reports)
    correctly_detected = sum(1 for s in stage_reports if s["detection_correct"])
    escalated_count = sum(1 for s in stage_reports if s["human_review_needed"])
    wall_clock_seconds = round(time.perf_counter() - start_time, 4)

    return {
        "campaign_id": campaign_id,
        "stages": stage_reports,
        "summary": {
            "total_stages": total_stages,
            "correctly_detected": correctly_detected,
            "escalated_count": escalated_count,
            "conflict_resolution": overall_conflict_resolution,
            "wall_clock_seconds": wall_clock_seconds,
        },
    }


if __name__ == "__main__":
    init_db()

    print("=" * 80)
    print("SIMULATED CYBER RANGE — CAMPAIGN REPLAY & MULTI-AGENT VERIFICATION")
    print("=" * 80)

    total_bench_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # 1. Multi-Stage Intrusion Campaign (INTRUSION-01)
    # -------------------------------------------------------------------------
    print("\n[CAMPAIGN 1: MULTI-STAGE INTRUSION CAMPAIGN (INTRUSION-01)]")
    print("Generating sequenced APT attack kill-chain (Network -> Host -> IAM)...")
    intrusion_events = generate_multi_stage_intrusion_campaign(random_state=42)

    intrusion_report = run_campaign(intrusion_events)

    print("\n" + "=" * 80)
    print("STAGE-BY-STAGE INTRUSION REPORT:")
    print("=" * 80)

    for stg in intrusion_report["stages"]:
        mitre = stg["mitre_mapping"]
        mitre_str = (
            f"{mitre['technique_id']} - {mitre['technique_name']} ({mitre['tactic']})"
            if mitre
            else "None (Unmapped / Benign)"
        )

        print(f"\n--- [Stage {stg['stage_number']}: {stg['stage_name']}] ---")
        print(f"  Source Agent           : {stg['source_agent']}")
        if stg.get("computer"):
            print(f"  Target Entity          : {stg['computer']}")
        print(f"  True Label             : {stg['true_label']}")
        print(f"  ML Prediction          : {stg['prediction']} (Correct: {stg['detection_correct']})")
        print(f"  Proposed Action        : {stg['proposed_action']}")
        print(f"  Action Severity Weight : {stg['action_severity_weight']:.1f}")
        print(f"  Final Action           : {stg['action']}")
        print(f"  Risk Evaluation        : {stg['risk_score']:.4f} ({stg['risk_level']})")
        print(f"  Safety Gate Decision   : {stg['gate_decision']}")
        print(f"  Human Review Needed    : {stg['human_review_needed']}")
        print(f"  ATT&CK Classification  : {mitre_str}")
        print(f"  Gate Reason            : {stg['gate_reason']}")
        if stg.get("llm_explanation"):
            print(f"  AI Incident Explanation: {stg['llm_explanation']}")

    # Cross-Domain Collision Resolution Report
    conflict_res = intrusion_report["summary"]["conflict_resolution"]
    if conflict_res:
        print("\n" + "-" * 80)
        print("CROSS-DOMAIN CONFLICT RESOLUTION (Host + IAM Collision on PC-0004):")
        print("-" * 80)
        print(f"  Shared Computer        : {conflict_res['computer']}")
        print(f"  Combined Gate Decision : {conflict_res['combined_gate_decision']}")
        print(f"  Execution Sequence     : {conflict_res['execution_order']}")
        print(f"  Combined Policy Reason : {conflict_res['combined_reason']}")

    int_sum = intrusion_report["summary"]
    print("\n" + "-" * 80)
    print("INTRUSION CAMPAIGN SUMMARY:")
    print(f"  Total Stages Processed : {int_sum['total_stages']}")
    print(f"  Correctly Detected     : {int_sum['correctly_detected']}/{int_sum['total_stages']} ({int_sum['correctly_detected']/int_sum['total_stages']*100:.1f}%)")
    print(f"  Escalated Incidents    : {int_sum['escalated_count']}/{int_sum['total_stages']}")
    print(f"  Conflict Resolution    : {'Enforced' if conflict_res else 'None'}")
    print(f"  Wall-Clock Time        : {int_sum['wall_clock_seconds']:.2f} seconds")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # 2. Normal Operations Day Campaign (NORMALDAY-01)
    # -------------------------------------------------------------------------
    print("\n\n[CAMPAIGN 2: NORMAL OPERATIONS DAY (NORMALDAY-01)]")
    print("Generating 10-stage routine daily operations across Network, Host, and IAM...")
    normal_events = generate_normal_operations_day(random_state=42)

    normal_report = run_campaign(normal_events)

    print("\n" + "=" * 80)
    print("STAGE-BY-STAGE NORMAL OPERATIONS REPORT:")
    print("=" * 80)

    for stg in normal_report["stages"]:
        mitre = stg["mitre_mapping"]
        mitre_str = (
            f"{mitre['technique_id']} - {mitre['technique_name']} ({mitre['tactic']})"
            if mitre
            else "None (Unmapped / Benign)"
        )
        print(f"\n--- [Stage {stg['stage_number']}: {stg['stage_name']}] ---")
        print(f"  Source Agent           : {stg['source_agent']}")
        print(f"  True Label             : {stg['true_label']}")
        print(f"  ML Prediction          : {stg['prediction']} (Correct: {stg['detection_correct']})")
        print(f"  Proposed Action        : {stg['proposed_action']}")
        print(f"  Action Severity Weight : {stg['action_severity_weight']:.1f}")
        print(f"  Final Action           : {stg['action']}")
        print(f"  Risk Evaluation        : {stg['risk_score']:.4f} ({stg['risk_level']})")
        print(f"  Safety Gate Decision   : {stg['gate_decision']}")
        print(f"  Human Review Needed    : {stg['human_review_needed']}")
        print(f"  ATT&CK Classification  : {mitre_str}")
        print(f"  Gate Reason            : {stg['gate_reason']}")
        if stg.get("llm_explanation"):
            print(f"  AI Incident Explanation: {stg['llm_explanation']}")

    # Formatted 10-Stage Table
    print("\n" + "=" * 115)
    print("NORMAL OPERATIONS DAY — 10-STAGE BREAKDOWN TABLE:")
    print("=" * 115)
    header = (
        f"{'Stg':<4} | {'Domain':<7} | {'True Label':<10} | {'ML Pred':<7} | "
        f"{'Proposed Action':<18} | {'Weight':<6} | {'Risk':<6} | {'Gate Decision'}"
    )
    print(header)
    print("-" * 115)
    for s in normal_report["stages"]:
        w = s["action_severity_weight"]
        r = s["risk_score"]
        print(
            f"{s['stage_number']:<4} | "
            f"{s['source_agent']:<7} | "
            f"{str(s['true_label']):<10} | "
            f"{s['prediction']:<7} | "
            f"{str(s['proposed_action']):<18} | "
            f"{w:<6.1f} | "
            f"{r:<6.4f} | "
            f"{s['gate_decision']}"
        )
    print("=" * 115)

    norm_sum = normal_report["summary"]
    print("\n" + "=" * 80)
    print("NORMAL OPERATIONS DAY SUMMARY:")
    print("=" * 80)
    print(f"  Total Stages Processed : {norm_sum['total_stages']}")
    print(f"  Correctly Detected     : {norm_sum['correctly_detected']}/{norm_sum['total_stages']} ({norm_sum['correctly_detected']/norm_sum['total_stages']*100:.1f}%)")
    print(f"  Escalated Incidents    : {norm_sum['escalated_count']}/{norm_sum['total_stages']}")
    print(f"  Conflict Resolution    : {norm_sum['conflict_resolution'] or 'None (No Multi-Domain Attack Collision)'}")
    print(f"  Wall-Clock Time        : {norm_sum['wall_clock_seconds']:.2f} seconds")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 3. Overall Benchmark
    # -------------------------------------------------------------------------
    total_elapsed = round(time.perf_counter() - total_bench_start, 2)
    print(f"\nTOTAL SIMULATED RANGE BENCHMARK TIME (BOTH CAMPAIGNS): {total_elapsed:.2f} seconds\n")
