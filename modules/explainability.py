"""Explainability Module.

Responsible for providing structured, model-grounded explainability data
(MITRE ATT&CK technique alignment and global model feature contributions)
consumable by the dashboard and immutable audit log.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# MITRE ATT&CK Matrix for Enterprise (Technique & Tactic Alignment)
# Independently verified against MITRE ATT&CK Enterprise v14/v15
MITRE_ATTCK_MAPPING: Dict[str, Optional[Dict[str, str]]] = {
    "FTP-Patator": {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": "Credential Access",
    },
    "SSH-Patator": {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": "Credential Access",
    },
    "BRUTE_FORCE_LOGIN": {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": "Credential Access",
    },
    "SUSPICIOUS_PROCESS_INJECTION": {
        "technique_id": "T1055",
        "technique_name": "Process Injection",
        "tactic": "Defense Evasion, Privilege Escalation",
    },
    "PRIVILEGE_ESCALATION_ATTEMPT": {
        "technique_id": "T1033",
        "technique_name": "System Owner/User Discovery",
        "tactic": "Discovery",
    },
    "PASS_THE_HASH": {
        "technique_id": "T1550.002",
        "technique_name": "Pass the Hash",
        "tactic": "Defense Evasion, Lateral Movement",
    },
    "BENIGN": None,
}


def get_attck_mapping(label: Optional[str]) -> Optional[Dict[str, str]]:
    """Retrieves MITRE ATT&CK technique information for a security event label.

    Parameters
    ----------
    label : Optional[str]
        Ground-truth, detection prediction, or signature label.

    Returns
    -------
    Optional[Dict[str, str]]
        Dictionary containing 'technique_id', 'technique_name', and 'tactic',
        or None if the label is unmapped, benign, or None.
    """
    if not label:
        return None
    return MITRE_ATTCK_MAPPING.get(label)


def format_attck_badge(technique_id: str, label: Optional[str] = None) -> str:
    """Formats a human-readable ATT&CK badge string.

    Example: 'MITRE ATT&CK: T1110 - Brute Force (Credential Access)'
    """
    # Find matching technique metadata across mapping
    for k, v in MITRE_ATTCK_MAPPING.items():
        if v and v.get("technique_id") == technique_id:
            return f"MITRE ATT&CK: {v['technique_id']} - {v['technique_name']} ({v['tactic']})"
    if label and label in MITRE_ATTCK_MAPPING and MITRE_ATTCK_MAPPING[label]:
        v = MITRE_ATTCK_MAPPING[label]
        return f"MITRE ATT&CK: {v['technique_id']} - {v['technique_name']} ({v['tactic']})"
    return f"MITRE ATT&CK: {technique_id}"


def explain_feature_contribution(
    model: Any,
    feature_columns: List[str],
    event_features: Dict[str, Any],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Identifies the model's top globally influential features and correlates them with event values.

    METHODOLOGICAL LIMITATION & SCOPE CAVEAT:
    -----------------------------------------
    This function reports the trained RandomForestClassifier's global Gini feature
    importances (`model.feature_importances_`) paired with THIS event's observed
    feature values for contextual analyst triaging.

    It is NOT a per-instance local attribution method (such as Shapley additive
    explanations / SHAP, TreeSHAP, or LIME). It does NOT claim that these specific
    features drove THIS specific event's prediction more than others. Local feature
    interactions, split thresholds, and non-linear tree decision paths mean that an
    individual prediction may have been driven by features outside the global top-N.
    This serves strictly as a model-grounded baseline indicating which high-leverage
    signals were active in this event.

    Parameters
    ----------
    model : Any
        Trained scikit-learn model possessing a `feature_importances_` attribute.
    feature_columns : List[str]
        Ordered list of feature column names corresponding to model inputs.
    event_features : Dict[str, Any]
        Dictionary of event features (supports raw or one-hot encoded representations).
    top_n : int, default=5
        Number of top globally influential features to extract.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with 'feature', 'global_importance', and 'event_value'.
    """
    if not hasattr(model, "feature_importances_"):
        return []

    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1][:top_n]

    contributions = []
    # If event dictionary has a nested 'features' dict (e.g. network events from ingestion.py), check both
    feat_dict = event_features.get("features") if isinstance(event_features.get("features"), dict) else event_features

    for idx in sorted_indices:
        feat_name = feature_columns[idx]
        val = feat_dict.get(feat_name)
        if val is None:
            val = event_features.get(feat_name)

        # Handle one-hot encoded categorical features when raw event is provided
        if val is None:
            for prefix in ("parent_process", "process_name", "auth_type", "logon_type", "auth_result"):
                if feat_name.startswith(prefix + "_"):
                    target = feat_name[len(prefix) + 1 :]
                    actual = str(event_features.get(prefix, ""))
                    val = 1 if actual.lower() == target.lower() else 0
                    break

        if val is None:
            val = "N/A"
        elif isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
            val = round(float(val), 4)

        contributions.append({
            "feature": feat_name,
            "global_importance": round(float(importances[idx]), 4),
            "event_value": val,
        })

    return contributions


def build_explainability_report(
    detection_result: Dict[str, Any],
    event: Dict[str, Any],
    model: Any,
    feature_columns: List[str],
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Combines ATT&CK alignment and feature contributions into a unified explainability report.

    Parameters
    ----------
    detection_result : Dict[str, Any]
        Output dictionary from detection module containing 'event_id', 'prediction', 'confidence'.
    event : Dict[str, Any]
        Raw or normalized event dictionary.
    model : Any
        Trained machine learning classifier.
    feature_columns : List[str]
        Feature column names matching model input dimensionality.
    label : Optional[str], default=None
        Ground-truth or signature attack label. If None, resolves from event or detection_result.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'event_id', 'attck_mapping', and 'top_features'.
    """
    event_id = detection_result.get("event_id", event.get("event_id", "N/A"))
    effective_label = label or event.get("label") or detection_result.get("prediction")

    attck = get_attck_mapping(effective_label)
    top_features = explain_feature_contribution(
        model=model,
        feature_columns=feature_columns,
        event_features=event,
        top_n=5,
    )

    return {
        "event_id": event_id,
        "attck_mapping": attck,
        "top_features": top_features,
    }


if __name__ == "__main__":
    import joblib
    import pandas as pd
    from modules.dataset_loader import load_raw_data
    from modules.detection import detect as network_detect, prepare_training_data
    from modules.host_dataset_generator import load_synthetic_host_events
    from modules.host_detection import detect as host_detect
    from modules.iam_dataset_generator import load_synthetic_iam_events
    from modules.iam_detection import detect as iam_detect
    from modules.ingestion import normalize_events

    print("=" * 80)
    print("MODULE 16: STRUCTURED EXPLAINABILITY REPORTS (NETWORK, HOST, IAM)")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Network Domain Sample: FTP-Patator
    # --------------------------------------------------------------------------
    net_model_path = PROJECT_ROOT / "models" / "threat_detector.pkl"
    net_csv_path = PROJECT_ROOT / "data" / "raw" / "Tuesday-WorkingHours.pcap_ISCX.csv"

    print("\n--- [DOMAIN 1: NETWORK TELEMETRY (FTP-Patator)] ---")
    net_model = joblib.load(net_model_path)
    raw_df = load_raw_data(net_csv_path)
    net_events = normalize_events(raw_df)
    _, X_test_net, _, _ = prepare_training_data(net_events)
    net_feature_cols = list(net_model.feature_names_in_)

    sample_ftp = next(ev for ev in (net_events[idx] for idx in X_test_net.index) if ev.get("label") == "FTP-Patator")
    net_det = network_detect(net_model, sample_ftp)

    net_report = build_explainability_report(
        detection_result=net_det,
        event=sample_ftp,
        model=net_model,
        feature_columns=net_feature_cols,
        label=sample_ftp.get("label"),
    )

    print(f"Event ID      : {net_report['event_id']}")
    print(f"Attack Label  : {sample_ftp.get('label')}")
    print(f"ATT&CK Mapping: {net_report['attck_mapping']}")
    print("Top-5 Feature Contributions (Global Importance vs Event Value):")
    for rank, f in enumerate(net_report["top_features"], 1):
        print(f"  {rank}. {f['feature']:<28} | Global Importance: {f['global_importance']:.4f} | Event Value: {f['event_value']}")

    # --------------------------------------------------------------------------
    # 2. Host Domain Sample: SUSPICIOUS_PROCESS_INJECTION
    # --------------------------------------------------------------------------
    print("\n--- [DOMAIN 2: HOST TELEMETRY (SUSPICIOUS_PROCESS_INJECTION)] ---")
    host_model_path = PROJECT_ROOT / "models" / "host_detector.pkl"
    host_csv_path = PROJECT_ROOT / "data" / "raw" / "synthetic_host_events.csv"

    host_model = joblib.load(host_model_path)
    df_host = load_synthetic_host_events(host_csv_path)
    host_feature_cols = list(host_model.feature_names_in_)

    sample_host_rows = df_host[df_host["event_id"] == 332]
    sample_host = sample_host_rows.iloc[0].to_dict() if not sample_host_rows.empty else df_host[df_host["label"] == "SUSPICIOUS_PROCESS_INJECTION"].iloc[0].to_dict()
    host_det = host_detect(host_model, host_feature_cols, sample_host)

    host_report = build_explainability_report(
        detection_result=host_det,
        event=sample_host,
        model=host_model,
        feature_columns=host_feature_cols,
        label=sample_host.get("label"),
    )

    print(f"Event ID      : {host_report['event_id']}")
    print(f"Process Tree  : {sample_host.get('parent_process')} -> {sample_host.get('process_name')}")
    print(f"Attack Label  : {sample_host.get('label')}")
    print(f"ATT&CK Mapping: {host_report['attck_mapping']}")
    print("Top-5 Feature Contributions (Global Importance vs Event Value):")
    for rank, f in enumerate(host_report["top_features"], 1):
        print(f"  {rank}. {f['feature']:<28} | Global Importance: {f['global_importance']:.4f} | Event Value: {f['event_value']}")

    # --------------------------------------------------------------------------
    # 3. IAM Domain Sample: PASS_THE_HASH (Event 396)
    # --------------------------------------------------------------------------
    print("\n--- [DOMAIN 3: IAM TELEMETRY (PASS_THE_HASH)] ---")
    iam_model_path = PROJECT_ROOT / "models" / "iam_detector.pkl"
    iam_csv_path = PROJECT_ROOT / "data" / "raw" / "synthetic_iam_events.csv"

    iam_model = joblib.load(iam_model_path)
    df_iam = load_synthetic_iam_events(iam_csv_path)
    iam_feature_cols = list(iam_model.feature_names_in_)

    sample_iam_rows = df_iam[df_iam["event_id"] == 396]
    sample_iam = sample_iam_rows.iloc[0].to_dict() if not sample_iam_rows.empty else df_iam[df_iam["label"] == "PASS_THE_HASH"].iloc[0].to_dict()
    iam_det = iam_detect(iam_model, iam_feature_cols, sample_iam)

    iam_report = build_explainability_report(
        detection_result=iam_det,
        event=sample_iam,
        model=iam_model,
        feature_columns=iam_feature_cols,
        label=sample_iam.get("label"),
    )

    print(f"Event ID      : {iam_report['event_id']}")
    print(f"Auth Context  : auth_type={sample_iam.get('auth_type')}, logon_type={sample_iam.get('logon_type')}, auth_result={sample_iam.get('auth_result')}")
    print(f"Attack Label  : {sample_iam.get('label')}")
    print(f"ATT&CK Mapping: {iam_report['attck_mapping']}")
    print("Top-5 Feature Contributions (Global Importance vs Event Value):")
    for rank, f in enumerate(iam_report["top_features"], 1):
        print(f"  {rank}. {f['feature']:<28} | Global Importance: {f['global_importance']:.4f} | Event Value: {f['event_value']}")

    print("\n" + "=" * 80)
    print("ALL 3 EXPLAINABILITY REPORTS GENERATED SUCCESSFULLY.")
    print("=" * 80)
