"""Audit Log Module.

Responsible for persisting immutable audit events, risk determinations, system
decisions, and pipeline execution trails to an SQLite audit database.

human_decision tracks reviewer approval/rejection for audit and evaluation purposes;
it does not trigger any live remediation action, since this prototype has no connection
to real infrastructure.
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Union



PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_db_path(db_path: Union[str, Path]) -> Path:
    """Resolves relative database paths relative to the project root directory."""
    if db_path is None:
        raise ValueError("Database path cannot be None")
    path = Path(db_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def get_connection(
    db_path: Union[str, Path] = "logs/audit.db",
    timeout: float = 30.0,
) -> sqlite3.Connection:
    """Creates a connection to the audit database with WAL mode and busy timeout configured."""
    resolved_path = _resolve_db_path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved_path, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db(db_path: Union[str, Path] = "logs/audit.db") -> str:
    """Creates the audit database and table if they do not exist.

    Ensures the schema is up-to-date (including 'source_agent' and 'llm_explanation'),
    and configures SQLite Write-Ahead Logging (WAL) for safe multi-process concurrency.

    Parameters
    ----------
    db_path : str or Path, optional
        Filepath for the SQLite database (default: 'logs/audit.db').

    Returns
    -------
    str
        Active SQLite journal mode (e.g. 'wal').
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        journal_mode = cursor.fetchone()[0]
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
                gate_reason TEXT,
                source_agent TEXT,
                llm_explanation TEXT,
                mitre_technique_id TEXT,
                human_decision TEXT,
                human_decision_timestamp TEXT,
                campaign_id TEXT,
                campaign_stage TEXT
            )
            """
        )
        # Migrate existing tables: add columns if not already present
        cursor.execute("PRAGMA table_info(audit_log)")
        columns = [row[1] for row in cursor.fetchall()]
        if "source_agent" not in columns:
            cursor.execute("ALTER TABLE audit_log ADD COLUMN source_agent TEXT")
        if "llm_explanation" not in columns:
            cursor.execute("ALTER TABLE audit_log ADD COLUMN llm_explanation TEXT")
        if "mitre_technique_id" not in columns:
            cursor.execute("ALTER TABLE audit_log ADD COLUMN mitre_technique_id TEXT")
        if "human_decision" not in columns:
            cursor.execute("ALTER TABLE audit_log ADD COLUMN human_decision TEXT")
        if "human_decision_timestamp" not in columns:
            cursor.execute("ALTER TABLE audit_log ADD COLUMN human_decision_timestamp TEXT")
        if "campaign_id" not in columns:
            cursor.execute("ALTER TABLE audit_log ADD COLUMN campaign_id TEXT")
        if "campaign_stage" not in columns:
            cursor.execute("ALTER TABLE audit_log ADD COLUMN campaign_stage TEXT")

        conn.commit()
        return str(journal_mode)


def log_decision(
    event: Dict[str, Any],
    detection_result: Dict[str, Any],
    action_proposal: Dict[str, Any],
    risk_result: Dict[str, Any],
    gate_result: Dict[str, Any],
    source_agent: str,
    llm_explanation: Optional[str] = None,
    mitre_technique_id: Optional[str] = None,
    db_path: Union[str, Path] = "logs/audit.db",
    campaign_id: Optional[str] = None,
    campaign_stage: Optional[str] = None,
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
    source_agent : str
        Source agent tag indicating telemetry domain ("NETWORK", "HOST", or "IAM").
    llm_explanation : str, optional
        Human-readable incident reasoning generated by LLM explanation layer.
    mitre_technique_id : str, optional
        MITRE ATT&CK technique identifier (e.g. 'T1110', 'T1055', 'T1550.002').
    db_path : str or Path, optional
        Target SQLite database path.
    campaign_id : str, optional
        Identifier of the cyber range campaign (e.g. 'INTRUSION-01', 'NORMALDAY-01').
    campaign_stage : str, optional
        Campaign stage description or identifier.

    Returns
    -------
    int
        The auto-generated primary key ID of the inserted row.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    gate_decision = gate_result.get("gate_decision")

    # Workflow rule: escalated decisions start in PENDING state; auto-executed actions require no human tracking
    human_decision = "PENDING" if gate_decision == "NEEDS_HUMAN_APPROVAL" else None
    human_decision_timestamp = None

    with get_connection(db_path) as conn:
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
                gate_reason,
                source_agent,
                llm_explanation,
                mitre_technique_id,
                human_decision,
                human_decision_timestamp,
                campaign_id,
                campaign_stage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                gate_decision,
                gate_result.get("final_action"),
                gate_result.get("gate_reason"),
                source_agent,
                llm_explanation,
                mitre_technique_id,
                human_decision,
                human_decision_timestamp,
                campaign_id,
                campaign_stage,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def record_human_decision(
    row_id: int,
    decision: str,
    db_path: Union[str, Path] = "logs/audit.db",
) -> bool:
    """Records a human analyst's authorization or rejection of an escalated security decision.

    Parameters
    ----------
    row_id : int
        The primary key ID of the audit log record.
    decision : str
        The human decision; must be strictly "APPROVED" or "REJECTED".
    db_path : str or Path, optional
        Target SQLite database path.

    Returns
    -------
    bool
        True if the record was in 'PENDING' status and successfully updated;
        False if the record was already decided (idempotent no-op) or not found.

    Raises
    ------
    ValueError
        If decision is not strictly 'APPROVED' or 'REJECTED'.
    """
    if decision not in ("APPROVED", "REJECTED"):
        raise ValueError(
            f"Invalid human decision '{decision}'. Decision must be exactly 'APPROVED' or 'REJECTED'."
        )

    resolved_path = _resolve_db_path(db_path)
    if not resolved_path.exists():
        print(f"Warning: Audit log database not found at {resolved_path}")
        return False

    with get_connection(resolved_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, human_decision FROM audit_log WHERE id = ?", (row_id,))
        row = cursor.fetchone()

        if row is None:
            print(f"Warning: Audit log row with ID {row_id} not found.")
            return False

        current_decision = row["human_decision"]
        if current_decision != "PENDING":
            print(
                f"Warning: Audit log row {row_id} already has decision '{current_decision}'. "
                f"Cannot overwrite with '{decision}' (idempotency guard enforced)."
            )
            return False

        now_utc = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            """
            UPDATE audit_log
            SET human_decision = ?, human_decision_timestamp = ?
            WHERE id = ?
            """,
            (decision, now_utc, row_id),
        )
        conn.commit()
        return True


def get_pending_approvals(
    limit: int = 50,
    db_path: Union[str, Path] = "logs/audit.db",
) -> List[Dict[str, Any]]:
    """Retrieves decisions pending human authorization, newest first.

    Parameters
    ----------
    limit : int, optional
        Maximum number of pending records to return (default: 50).
    db_path : str or Path, optional
        Target SQLite database path.

    Returns
    -------
    List[Dict[str, Any]]
        List of audit log rows awaiting human review.
    """
    resolved_path = _resolve_db_path(db_path)
    if not resolved_path.exists():
        return []

    with get_connection(resolved_path) as conn:
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
                gate_reason,
                source_agent,
                llm_explanation,
                mitre_technique_id,
                human_decision,
                human_decision_timestamp,
                campaign_id,
                campaign_stage
            FROM audit_log
            WHERE human_decision = 'PENDING'
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


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

    with get_connection(resolved_path) as conn:
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
                gate_reason,
                source_agent,
                llm_explanation,
                mitre_technique_id,
                human_decision,
                human_decision_timestamp,
                campaign_id,
                campaign_stage
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
    jmode = init_db()
    print(f"WAL Confirmation: PRAGMA journal_mode = {jmode.upper()}")

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
                source_agent="NETWORK",
            )
            print(f"Logged {category} (event_id={sample_ev['event_id']}) -> Audit Log Row ID: {row_id}")
        else:
            print(f"Warning: Test sample for {category} could not be found.")

    print("\n" + "=" * 70)
    print("READING RECENT DECISIONS FROM AUDIT LOG (get_recent_decisions):")
    print("=" * 70)
    recent_rows = get_recent_decisions(limit=5)
    for row in recent_rows:
        print(f"\nRow ID: {row['id']}")
        for key, value in row.items():
            if key != "id":
                print(f"  {key:<26}: {value}")

    print("\n" + "=" * 70)
    print("HUMAN APPROVAL WORKFLOW & IDEMPOTENCY TEST:")
    print("=" * 70)
    # 1. Log a decision with NEEDS_HUMAN_APPROVAL
    test_event = {"event_id": 99991, "label": "SSH-Patator"}
    test_det = {"prediction": "THREAT", "confidence": 0.99}
    test_act = {"action": "ISOLATE_FLOW"}
    test_risk = {"risk_score": 0.99, "risk_level": "HIGH"}
    test_gate = {
        "gate_decision": "NEEDS_HUMAN_APPROVAL",
        "final_action": "ISOLATE_FLOW",
        "gate_reason": "High risk requires human authorization.",
    }
    test_row_id = log_decision(
        event=test_event,
        detection_result=test_det,
        action_proposal=test_act,
        risk_result=test_risk,
        gate_result=test_gate,
        source_agent="NETWORK",
        mitre_technique_id="T1110",
    )
    print(f"Logged escalated test event -> Row ID: {test_row_id}")

    # Verify initially PENDING
    pending_list = get_pending_approvals(limit=10)
    matching_pending = [r for r in pending_list if r["id"] == test_row_id]
    assert len(matching_pending) == 1, f"Expected row {test_row_id} in pending approvals!"
    assert matching_pending[0]["human_decision"] == "PENDING"
    print(f"Confirmed: Row {test_row_id} is logged with human_decision='PENDING'.")

    # 2. Record approval -> should return True
    success_approve = record_human_decision(test_row_id, "APPROVED")
    assert success_approve is True, "Expected record_human_decision() to return True!"
    print("Confirmed: Approval recorded successfully (returned True).")

    # Verify row now shows APPROVED
    recent_rows_after = get_recent_decisions(limit=5)
    approved_row = next(r for r in recent_rows_after if r["id"] == test_row_id)
    assert approved_row["human_decision"] == "APPROVED"
    assert approved_row["human_decision_timestamp"] is not None
    print(f"Confirmed: Row {test_row_id} updated to 'APPROVED' at {approved_row['human_decision_timestamp']}.")

    # 3. Idempotency test: attempt to REJECT the SAME row -> should return False
    print("Testing idempotency guard: attempting to overwrite APPROVED with REJECTED...")
    attempt_overwrite = record_human_decision(test_row_id, "REJECTED")
    assert attempt_overwrite is False, "Expected idempotency guard to reject overwrite and return False!"
    recent_rows_final = get_recent_decisions(limit=5)
    final_row = next(r for r in recent_rows_final if r["id"] == test_row_id)
    assert final_row["human_decision"] == "APPROVED", "Row decision was overwritten despite idempotency guard!"
    print("Confirmed: Idempotency guard rejected overwrite (returned False), decision remains 'APPROVED'.")
    print("\n" + "=" * 70)
    print("ALL AUDIT LOG & HUMAN APPROVAL TESTS PASSED SUCCESSFULLY.")
    print("=" * 70)
