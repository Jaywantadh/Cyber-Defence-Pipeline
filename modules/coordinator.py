"""Coordinator Module.

Orchestrates multi-agent defense pipelines across Network, Host, and IAM
telemetry feeds into a single unified stream and priority view.
"""

from collections import Counter
import inspect
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Union
import warnings
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.audit_log import init_db, log_decision
from modules.dataset_loader import load_raw_data
from modules.detection import detect as network_detect
from modules.host_agent import propose_action as host_propose_action
from modules.host_dataset_generator import generate_host_events
from modules.host_detection import detect as host_detect
from modules.iam_agent import propose_action as iam_propose_action
from modules.iam_dataset_generator import generate_iam_events
from modules.iam_detection import detect as iam_detect
from modules.ingestion import normalize_events
from modules.network_agent import propose_action as network_propose_action
from modules.risk_engine import compute_risk_score
from modules.safety_gate import evaluate_gate

logger = logging.getLogger(__name__)


def run_agent_pipeline(
    source_agent: str,
    event: Dict[str, Any],
    model: Any,
    detect_fn: Callable[..., Dict[str, Any]],
    propose_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    db_path: Union[str, Path] = "logs/audit.db",
) -> Dict[str, Any]:
    """Runs a single event through an agent's end-to-end defense pipeline.

    Chains:
        detect -> propose_action -> compute_risk_score -> evaluate_gate -> log_decision

    Parameters
    ----------
    source_agent : str
        Agent identifier ("NETWORK", "HOST", or "IAM").
    event : Dict[str, Any]
        Normalized telemetry event dictionary.
    model : Any
        Trained machine learning model for the agent.
    detect_fn : Callable
        Detection function for the agent. Can accept (model, event) or
        (model, feature_columns, event).
    propose_fn : Callable
        Action proposal function accepting (detection_result, event).
    db_path : str or Path, optional
        Path to SQLite audit log database (default: 'logs/audit.db').

    Returns
    -------
    Dict[str, Any]
        Complete decision dictionary with 'source_agent', 'event_id',
        'true_label', 'detection_result', 'proposed_action', 'risk_score',
        'risk_level', 'gate_decision', 'final_action', and 'gate_reason'.
    """
    event_id = event.get("event_id")
    try:
        # Flexible invocation of detect_fn supporting 2-arg or 3-arg signatures
        sig = inspect.signature(detect_fn)
        num_params = len(sig.parameters)
        if num_params >= 3:
            feature_cols = getattr(model, "feature_names_in_", None)
            if feature_cols is not None:
                feature_cols = list(feature_cols)
            det_result = detect_fn(model, feature_cols, event)
        elif num_params == 2:
            det_result = detect_fn(model, event)
        else:
            det_result = detect_fn(event)

        action_proposal = propose_fn(det_result, event)
        risk_result = compute_risk_score(action_proposal)
        gate_result = evaluate_gate(risk_result)

        log_decision(
            event=event,
            detection_result=det_result,
            action_proposal=action_proposal,
            risk_result=risk_result,
            gate_result=gate_result,
            source_agent=source_agent,
            db_path=db_path,
        )

        final_act = gate_result.get("final_action", action_proposal.get("action"))
        return {
            "event_id": event_id,
            "source_agent": source_agent,
            "true_label": event.get("label"),
            "detection_prediction": det_result.get("prediction"),
            "detection_confidence": det_result.get("confidence"),
            "action": final_act,
            "proposed_action": action_proposal.get("action"),
            "final_action": final_act,
            "risk_score": risk_result.get("risk_score"),
            "risk_level": risk_result.get("risk_level"),
            "gate_decision": gate_result.get("gate_decision"),
            "gate_reason": gate_result.get("gate_reason"),
        }
    except Exception as exc:
        logger.error(
            f"Error in {source_agent} pipeline for event {event_id}: {exc}",
            exc_info=True,
        )
        return {
            "event_id": event_id,
            "source_agent": source_agent,
            "true_label": event.get("label"),
            "error": str(exc),
            "gate_decision": "ERROR",
            "action": "NO_ACTION",
            "final_action": "NO_ACTION",
            "risk_score": 0.0,
            "risk_level": "NONE",
            "gate_reason": f"Pipeline execution error: {exc}",
        }


def interleave_by_source_round_robin(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sorts items by risk_score descending, interleaving ties round-robin across source_agents.

    When multiple items share the exact same risk_score, they are interleaved across
    source_agent values round-robin (NETWORK, HOST, IAM, NETWORK, HOST, IAM...) to
    prevent a single telemetry domain from monopolizing ties in the unified priority queue.

    Parameters
    ----------
    items : List[Dict[str, Any]]
        List of event result dictionaries.

    Returns
    -------
    List[Dict[str, Any]]
        Sorted and round-robin interleaved list of event results.
    """
    from collections import defaultdict

    # Group items by risk_score rounded to 4 decimal places
    score_groups = defaultdict(list)
    for item in items:
        score = round(float(item.get("risk_score", 0.0)), 4)
        score_groups[score].append(item)

    interleaved_all: List[Dict[str, Any]] = []
    # Process distinct scores in descending order
    for score in sorted(score_groups.keys(), reverse=True):
        group_items = score_groups[score]

        # Partition group items into buckets by source_agent
        source_order = ["NETWORK", "HOST", "IAM"]
        all_sources_in_group = list(
            dict.fromkeys(source_order + [it.get("source_agent", "") for it in group_items])
        )

        buckets: Dict[str, List[Dict[str, Any]]] = {src: [] for src in all_sources_in_group}
        for it in group_items:
            buckets[it.get("source_agent", "")].append(it)

        # Round-robin interleaving drain across sources
        while any(len(b) > 0 for b in buckets.values()):
            for src in all_sources_in_group:
                if buckets[src]:
                    interleaved_all.append(buckets[src].pop(0))

    return interleaved_all


def run_coordinated_batch(
    network_sample_size: int = 1000,
    host_sample_size: int = 1000,
    iam_sample_size: int = 1000,
    random_state: int = 42,
    db_path: Union[str, Path] = "logs/audit.db",
) -> Dict[str, Any]:
    """Runs a coordinated multi-agent batch across Network, Host, and IAM telemetry feeds.

    Parameters
    ----------
    network_sample_size : int, optional
        Number of network events to sample (default: 1000).
    host_sample_size : int, optional
        Number of host events to generate and process (default: 1000).
    iam_sample_size : int, optional
        Number of IAM events to generate and process (default: 1000).
    random_state : int, optional
        Random seed for sampling and generator reproducibility (default: 42).
    db_path : str or Path, optional
        Path to SQLite audit log database.

    Returns
    -------
    Dict[str, Any]
        Summary dictionary containing:
        - 'total_processed': count per source and overall total
        - 'gate_decisions_by_source': breakdown of decisions per agent
        - 'gate_decisions_combined': breakdown of decisions across all agents
        - 'priority_queue_top_10': top 10 highest-risk items across all feeds
        - 'unified_feed': all processed events sorted with round-robin tie-breaking
        - 'iam_crosstab': cross-tabulation of IAM gate decisions vs ground-truth labels
    """
    # 1. Initialize audit database once
    init_db(db_path)

    # 2. Paths
    network_model_path = PROJECT_ROOT / "models" / "threat_detector.pkl"
    host_model_path = PROJECT_ROOT / "models" / "host_detector.pkl"
    iam_model_path = PROJECT_ROOT / "models" / "iam_detector.pkl"
    network_csv_path = PROJECT_ROOT / "data" / "raw" / "Tuesday-WorkingHours.pcap_ISCX.csv"

    # Verify models
    for m_path in [network_model_path, host_model_path, iam_model_path]:
        if not m_path.exists():
            raise FileNotFoundError(f"Required model not found: {m_path}")

    # Load models
    network_model = joblib.load(network_model_path)
    host_model = joblib.load(host_model_path)
    iam_model = joblib.load(iam_model_path)

    # 3. Load & Sample Feeds
    # 3a. Network events (stratified from raw PCAP dataset)
    if not network_csv_path.exists():
        raise FileNotFoundError(f"Network raw dataset not found: {network_csv_path}")
    raw_df = load_raw_data(network_csv_path)
    all_net_events = normalize_events(raw_df)
    net_labels = [e["label"] for e in all_net_events]
    sampled_net_events, _, _, _ = train_test_split(
        all_net_events,
        net_labels,
        train_size=network_sample_size,
        stratify=net_labels,
        random_state=random_state,
    )

    # 3b. Host events (synthetic endpoint generator)
    host_df = generate_host_events(
        n_events=host_sample_size,
        attack_ratio=0.03,
        random_state=random_state,
    )
    host_events = host_df.to_dict(orient="records")

    # 3c. IAM events (synthetic identity generator)
    iam_df = generate_iam_events(
        n_events=iam_sample_size,
        attack_ratio=0.03,
        random_state=random_state,
    )
    iam_events = iam_df.to_dict(orient="records")

    # 4. Execute Pipelines
    all_results: List[Dict[str, Any]] = []

    # Run Network
    for ev in sampled_net_events:
        res = run_agent_pipeline(
            source_agent="NETWORK",
            event=ev,
            model=network_model,
            detect_fn=network_detect,
            propose_fn=network_propose_action,
            db_path=db_path,
        )
        all_results.append(res)

    # Run Host
    for ev in host_events:
        res = run_agent_pipeline(
            source_agent="HOST",
            event=ev,
            model=host_model,
            detect_fn=host_detect,
            propose_fn=host_propose_action,
            db_path=db_path,
        )
        all_results.append(res)

    # Run IAM
    for ev in iam_events:
        res = run_agent_pipeline(
            source_agent="IAM",
            event=ev,
            model=iam_model,
            detect_fn=iam_detect,
            propose_fn=iam_propose_action,
            db_path=db_path,
        )
        all_results.append(res)

    # 5. Build Unified Priority View (sort by risk_score desc, round-robin tie-breaking across sources)
    valid_results = [r for r in all_results if "error" not in r]
    sorted_feed = interleave_by_source_round_robin(valid_results)

    # 6. Aggregate Summary
    total_processed = {
        "NETWORK": len([r for r in valid_results if r["source_agent"] == "NETWORK"]),
        "HOST": len([r for r in valid_results if r["source_agent"] == "HOST"]),
        "IAM": len([r for r in valid_results if r["source_agent"] == "IAM"]),
        "TOTAL": len(valid_results),
    }

    gate_decisions_by_source: Dict[str, Dict[str, int]] = {}
    for src in ["NETWORK", "HOST", "IAM"]:
        src_items = [r for r in valid_results if r["source_agent"] == src]
        counts = Counter(r["gate_decision"] for r in src_items)
        gate_decisions_by_source[src] = {
            "AUTO_EXECUTE": counts.get("AUTO_EXECUTE", 0),
            "NEEDS_HUMAN_APPROVAL": counts.get("NEEDS_HUMAN_APPROVAL", 0),
        }

    combined_counts = Counter(r["gate_decision"] for r in valid_results)
    gate_decisions_combined = {
        "AUTO_EXECUTE": combined_counts.get("AUTO_EXECUTE", 0),
        "NEEDS_HUMAN_APPROVAL": combined_counts.get("NEEDS_HUMAN_APPROVAL", 0),
    }

    priority_queue_top_10 = [
        {
            "event_id": r["event_id"],
            "source_agent": r["source_agent"],
            "action": r.get("action", r.get("final_action")),
            "risk_score": r["risk_score"],
            "gate_decision": r["gate_decision"],
        }
        for r in sorted_feed[:10]
    ]

    # 7. Host & IAM Ground-Truth Breakdown (cross-tabulate gate_decision vs true_label)
    host_items = [r for r in valid_results if r["source_agent"] == "HOST"]
    host_true_labels = [r.get("true_label", "UNKNOWN") for r in host_items]
    host_gate_decisions = [r.get("gate_decision", "UNKNOWN") for r in host_items]

    host_crosstab_df = pd.crosstab(
        index=pd.Series(host_true_labels, name="true_label"),
        columns=pd.Series(host_gate_decisions, name="gate_decision"),
        margins=True,
        margins_name="Total",
    )

    iam_items = [r for r in valid_results if r["source_agent"] == "IAM"]
    iam_true_labels = [r.get("true_label", "UNKNOWN") for r in iam_items]
    iam_gate_decisions = [r.get("gate_decision", "UNKNOWN") for r in iam_items]

    iam_crosstab_df = pd.crosstab(
        index=pd.Series(iam_true_labels, name="true_label"),
        columns=pd.Series(iam_gate_decisions, name="gate_decision"),
        margins=True,
        margins_name="Total",
    )

    return {
        "total_processed": total_processed,
        "gate_decisions_by_source": gate_decisions_by_source,
        "gate_decisions_combined": gate_decisions_combined,
        "priority_queue_top_10": priority_queue_top_10,
        "unified_feed": sorted_feed,
        "host_crosstab": host_crosstab_df,
        "iam_crosstab": iam_crosstab_df,
    }


if __name__ == "__main__":
    print("=" * 75)
    print("COORDINATED MULTI-AGENT PIPELINE EXECUTION (Network, Host, IAM)")
    print("=" * 75)
    start_wall_time = time.time()

    summary = run_coordinated_batch(
        network_sample_size=1000,
        host_sample_size=1000,
        iam_sample_size=1000,
        random_state=42,
    )

    elapsed_sec = time.time() - start_wall_time

    # 1. Print Processing Counts
    print("\n" + "=" * 75)
    print("EXECUTION SUMMARY: TOTAL PROCESSED PER SOURCE")
    print("=" * 75)
    for src, count in summary["total_processed"].items():
        print(f"  {src:<15}: {count:>6,}")

    # 2. Print Gate Decision Breakdown
    print("\n" + "=" * 75)
    print("GATE DECISION BREAKDOWN BY SOURCE AGENT")
    print("=" * 75)
    for src, decisions in summary["gate_decisions_by_source"].items():
        print(f"[{src}]")
        for dec, cnt in decisions.items():
            print(f"    {dec:<22}: {cnt:>5,}")

    print("\n[COMBINED GATE DECISIONS]")
    tot = summary["total_processed"]["TOTAL"]
    for dec, cnt in summary["gate_decisions_combined"].items():
        pct = (cnt / tot * 100.0) if tot > 0 else 0.0
        print(f"    {dec:<22}: {cnt:>5,} ({pct:>5.1f}%)")

    # 3. Print Host Escalations Ground-Truth Cross-Tabulation
    print("\n" + "=" * 75)
    print("HOST ESCALATIONS: GROUND-TRUTH CROSS-TABULATION (Gate Decision vs True Label)")
    print("=" * 75)
    print(summary["host_crosstab"])

    # 4. Print IAM Escalations Ground-Truth Cross-Tabulation
    print("\n" + "=" * 75)
    print("IAM ESCALATIONS: GROUND-TRUTH CROSS-TABULATION (Gate Decision vs True Label)")
    print("=" * 75)
    print(summary["iam_crosstab"])

    # 5. Print Priority Queue Preview
    print("\n" + "=" * 75)
    print("UNIFIED PRIORITY QUEUE PREVIEW (TOP 10 HIGHEST-RISK INCIDENTS — ROUND-ROBIN)")
    print("=" * 75)
    header = f"{'Rank':<5} | {'Source':<8} | {'Event ID':<9} | {'Action':<20} | {'Risk':<6} | {'Gate Decision':<20}"
    print(header)
    print("-" * len(header))
    for rank, item in enumerate(summary["priority_queue_top_10"], start=1):
        print(
            f"{rank:<5} | "
            f"{item['source_agent']:<8} | "
            f"{str(item['event_id']):<9} | "
            f"{item['action']:<20} | "
            f"{item['risk_score']:<6.4f} | "
            f"{item['gate_decision']:<20}"
        )
    print("=" * 75)
    print(f"Total Wall-Clock Time: {elapsed_sec:.2f} seconds")
    print("=" * 75)
