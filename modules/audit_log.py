"""Audit Log Module.

Responsible for persisting immutable audit events, risk determinations, system
decisions, and pipeline execution trails to an SQLite audit database.
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Union



PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_db_path(db_path: Union[str, Path]) -> Path:
    """Resolves relative database paths relative to the project root directory."""
    path = Path(db_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def init_db(db_path: Union[str, Path] = "logs/audit.db") -> None:
    """Creates the audit database and table if they do not exist.

    Parameters
    ----------
    db_path : str or Path, optional
        Filepath for the SQLite database (default: 'logs/audit.db').
    """
    resolved_path = _resolve_db_path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(resolved_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                true_label TEXT,
                detection_prediction TEXT,
                detection_confidence REAL,
                proposed_action TEXT,
                risk_score REAL,
                risk_level TEXT,
                gate_decision TEXT,
                final_action TEXT,
                gate_reason TEXT
            )
            """
        )
        conn.commit()


def log_decision(
    event: Dict[str, Any],
    detection_result: Dict[str, Any],
    action_proposal: Dict[str, Any],
    risk_result: Dict[str, Any],
    gate_result: Dict[str, Any],
    db_path: Union[str, Path] = "logs/audit.db",
) -> int:
    """Inserts a complete end-to-end decision trail into the audit database.

    Parameters
    ----------
    event : Dict[str, Any]
        Original normalized event dictionary (contains 'event_id' and 'label').
    detection_result : Dict[str, Any]
        Output dictionary from detection.detect().
    action_proposal : Dict[str, Any]
        Output dictionary from network_agent.propose_action().
    risk_result : Dict[str, Any]
        Output dictionary from risk_engine.compute_risk_score().
    gate_result : Dict[str, Any]
        Output dictionary from safety_gate.evaluate_gate().
    db_path : str or Path, optional
        Target SQLite database path.

    Returns
    -------
    int
        The auto-generated primary key ID of the inserted row.
    """
    resolved_path = _resolve_db_path(db_path)
    now_utc = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(resolved_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO audit_log (
                event_id,
                timestamp,
                true_label,
                detection_prediction,
                detection_confidence,
                proposed_action,
                risk_score,
                risk_level,
                gate_decision,
                final_action,
                gate_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"],
                now_utc,
                event.get("label"),
                detection_result.get("prediction"),
                detection_result.get("confidence"),
                action_proposal.get("action"),
                risk_result.get("risk_score"),
                risk_result.get("risk_level"),
                gate_result.get("gate_decision"),
                gate_result.get("final_action"),
                gate_result.get("gate_reason"),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_recent_decisions(
    limit: int = 50,
    db_path: Union[str, Path] = "logs/audit.db",
) -> List[Dict[str, Any]]:
    """Retrieves the most recent audit records from the database, newest first.

    Parameters
    ----------
    limit : int, optional
        Maximum number of recent records to return (default: 50).
    db_path : str or Path, optional
        Path to the SQLite database.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries corresponding to recent audit rows.
    """
    resolved_path = _resolve_db_path(db_path)
    if not resolved_path.exists():
        return []

    with sqlite3.connect(resolved_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                id,
                event_id,
                timestamp,
                true_label,
                detection_prediction,
                detection_confidence,
                proposed_action,
                risk_score,
                risk_level,
                gate_decision,
                final_action,
                gate_reason
            FROM audit_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


if __name__ == "__main__":
    import joblib
    from modules.dataset_loader import load_raw_data
    from modules.detection import detect, prepare_training_data
    from modules.ingestion import normalize_events
    from modules.network_agent import propose_action
    from modules.risk_engine import compute_risk_score
    from modules.safety_gate import evaluate_gate

    model_path = PROJECT_ROOT / "models" / "threat_detector.pkl"
    csv_path = PROJECT_ROOT / "data" / "raw" / "Tuesday-WorkingHours.pcap_ISCX.csv"

    if not model_path.exists():
        print(f"Error: Trained model not found at {model_path}")
        sys.exit(1)

    if not csv_path.exists():
        print(f"Error: Dataset not found at {csv_path}")
        sys.exit(1)

    print("Initializing audit log database...")
    init_db()

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

    print("\n" + "=" * 70)
    print("RUNNING 6-STAGE PIPELINE AND LOGGING DECISIONS:")
    print("=" * 70)

    for category, sample_ev in sample_targets:
        if sample_ev is not None:
            det_result = detect(model, sample_ev)
            action_proposal = propose_action(det_result, sample_ev)
            risk_result = compute_risk_score(action_proposal)
            gate_result = evaluate_gate(risk_result)

            row_id = log_decision(
                event=sample_ev,
                detection_result=det_result,
                action_proposal=action_proposal,
                risk_result=risk_result,
                gate_result=gate_result,
            )
            print(f"Logged {category} (event_id={sample_ev['event_id']}) -> Audit Log Row ID: {row_id}")
        else:
            print(f"Warning: Test sample for {category} could not be found.")

    print("\n" + "=" * 70)
    print("READING RECENT DECISIONS FROM AUDIT LOG (get_recent_decisions):")
    print("=" * 70)
    recent_rows = get_recent_decisions(limit=10)
    for row in recent_rows:
        print(f"\nRow ID: {row['id']}")
        for key, value in row.items():
            if key != "id":
                print(f"  {key:<22}: {value}")
