"""Streamlit Dashboard for Cyber Defense Pipeline (Prototype Dashboard).

Provides operational monitoring, summary metrics, decision log inspection,
and safety-gate audit transparency.
"""

import platform
try:
    platform.system = lambda: "Windows"
except Exception:
    pass

from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.audit_log import (
    get_connection,
    get_pending_approvals,
    get_recent_decisions,
    record_human_decision,
)
from modules.explainability import format_attck_badge

DB_PATH = PROJECT_ROOT / "logs" / "audit.db"

# 1. Page Configuration
st.set_page_config(
    page_title="Cyber Defense Pipeline — Prototype Dashboard",
    page_icon="🛡️",
    layout="wide",
)


def get_full_audit_metrics(db_path: Path) -> Dict[str, int]:
    """Computes summary metrics across the entire audit database."""
    if not db_path.exists():
        return {
            "total_events": 0,
            "needs_approval": 0,
            "auto_execute": 0,
            "high_risk": 0,
        }

    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(SUM(CASE WHEN gate_decision = 'NEEDS_HUMAN_APPROVAL' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN gate_decision = 'AUTO_EXECUTE' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END), 0)
                FROM audit_log
                """
            )
            row = cursor.fetchone()
            if row:
                return {
                    "total_events": int(row[0]),
                    "needs_approval": int(row[1]),
                    "auto_execute": int(row[2]),
                    "high_risk": int(row[3]),
                }
    except sqlite3.OperationalError as exc:
        st.error(f"❌ Database error: Could not read audit database ({exc})")

    return {
        "total_events": 0,
        "needs_approval": 0,
        "auto_execute": 0,
        "high_risk": 0,
    }


def highlight_needs_approval(row: pd.Series) -> List[str]:
    """Applies a distinct visual highlight to rows requiring human approval."""
    if row.get("gate_decision") == "NEEDS_HUMAN_APPROVAL":
        return ["background-color: rgba(255, 75, 75, 0.22); font-weight: 500"] * len(row)
    return [""] * len(row)


def main() -> None:
    st.title("🛡️ Cyber Defense Pipeline — Prototype Dashboard")
    st.caption("Real-time decision auditing, threat detection metrics, and safety-gate enforcement.")

    # 5. Sidebar with Refresh control
    with st.sidebar:
        st.header("Pipeline Controls")
        st.write(f"Database: `{DB_PATH.relative_to(PROJECT_ROOT)}`")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    # 7. Check if database exists and has data
    if not DB_PATH.exists():
        st.info("ℹ️ No decisions logged yet — run the pipeline first to generate audit logs (`logs/audit.db`).")
        return

    metrics = get_full_audit_metrics(DB_PATH)
    if metrics["total_events"] == 0:
        st.info("ℹ️ No decisions logged yet — run the pipeline first to populate the audit log.")
        return

    pending_approvals = get_pending_approvals(limit=100, db_path=DB_PATH)
    pending_count = len(pending_approvals)

    # 2. Summary row metrics across the entire audit database (5 metrics)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Events Processed", f"{metrics['total_events']:,}")
    with col2:
        st.metric(
            "Pending Human Review",
            f"{pending_count:,}",
            delta="Action Required" if pending_count > 0 else None,
            delta_color="inverse",
        )
    with col3:
        st.metric(
            "Needs Human Approval",
            f"{metrics['needs_approval']:,}",
            delta="Blocked Action" if metrics["needs_approval"] > 0 else None,
            delta_color="inverse",
        )
    with col4:
        st.metric("Auto Executed", f"{metrics['auto_execute']:,}")
    with col5:
        st.metric(
            "High Risk Events",
            f"{metrics['high_risk']:,}",
            delta="High Risk" if metrics["high_risk"] > 0 else None,
            delta_color="inverse",
        )

    st.markdown("---")

    # Actionable Pending Human Review Queue
    st.subheader("Pending Human Review")
    if not pending_approvals:
        st.success("✅ No pending reviews — all escalated decisions have been reviewed or auto-executed.")
    else:
        pending_table_data = [
            {
                "event_id": r["event_id"],
                "source_agent": r.get("source_agent", "NETWORK"),
                "proposed_action": r.get("proposed_action"),
                "risk_score": round(r.get("risk_score", 0.0), 4),
                "gate_reason": r.get("gate_reason"),
                "timestamp": r.get("timestamp"),
            }
            for r in pending_approvals
        ]
        st.dataframe(pd.DataFrame(pending_table_data), use_container_width=True, hide_index=True)

        st.markdown("##### ⚡ Authorize or Reject Pending Actions")
        for r in pending_approvals:
            row_id = r["id"]
            ev_id = r["event_id"]
            action = r.get("proposed_action")
            agent = r.get("source_agent", "NETWORK")
            risk = r.get("risk_score", 0.0)

            c1, c2, c3, c4 = st.columns([3, 4, 1.5, 1.5])
            with c1:
                st.write(f"**Event #{ev_id}** (`{agent}`)")
            with c2:
                st.write(f"Action: `{action}` &nbsp;|&nbsp; Risk: `{risk:.2f}`")
            with c3:
                if st.button("✅ Approve", key=f"approve_{row_id}", use_container_width=True, type="primary"):
                    if record_human_decision(row_id=row_id, decision="APPROVED", db_path=DB_PATH):
                        st.rerun()
            with c4:
                if st.button("❌ Reject", key=f"reject_{row_id}", use_container_width=True):
                    if record_human_decision(row_id=row_id, decision="REJECTED", db_path=DB_PATH):
                        st.rerun()

    st.markdown("---")

    # 3. Main table of recent decisions
    st.subheader("Recent Audit Trail (Latest 100 Decisions)")
    recent_records = get_recent_decisions(limit=100, db_path=DB_PATH)

    if not recent_records:
        st.info("No audit records found.")
        return

    # Prepare DataFrame with required columns including source_agent
    ordered_columns = [
        "timestamp",
        "source_agent",
        "true_label",
        "detection_prediction",
        "detection_confidence",
        "proposed_action",
        "risk_score",
        "risk_level",
        "gate_decision",
        "human_decision",
        "final_action",
    ]

    df = pd.DataFrame(recent_records)

    # Ensure all required display columns exist
    display_cols = [c for c in ordered_columns if c in df.columns]
    display_df = df[display_cols].copy()

    # 4. Highlight rows where gate_decision == "NEEDS_HUMAN_APPROVAL"
    styled_df = display_df.style.apply(highlight_needs_approval, axis=1)

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )

    # 6. Expandable section: Decision Detail
    st.markdown("### Decision Inspection")
    with st.expander("🔍 Decision Detail", expanded=True):
        event_options = [r["event_id"] for r in recent_records]
        selected_event_id = st.selectbox(
            "Select an Event ID to inspect complete audit trail & reasoning:",
            options=event_options,
            format_func=lambda eid: f"Event #{eid} — [{next((r.get('source_agent', 'NETWORK') for r in recent_records if r['event_id'] == eid), 'NETWORK')}] {next((r['gate_decision'] + ' (' + r['final_action'] + ')' for r in recent_records if r['event_id'] == eid), '')}",
        )

        selected_record = next((r for r in recent_records if r["event_id"] == selected_event_id), None)
        if selected_record:
            st.markdown(f"#### Audit Record for Event #{selected_event_id}")

            mitre_id = selected_record.get("mitre_technique_id")
            if mitre_id:
                badge_text = format_attck_badge(mitre_id, selected_record.get("true_label"))
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
                        border-left: 4px solid #f59e0b;
                        border-radius: 6px;
                        padding: 10px 16px;
                        margin-bottom: 16px;
                        display: flex;
                        align-items: center;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    ">
                        <span style="font-size: 1.2em; margin-right: 10px;">🛡️</span>
                        <div>
                            <span style="color: #94a3b8; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Adversary Technique Classification</span><br/>
                            <span style="color: #f8fafc; font-family: 'Segoe UI', monospace; font-size: 1.05em; font-weight: 700;">{badge_text}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Display Human Decision badge if present
            human_dec = selected_record.get("human_decision")
            human_ts = selected_record.get("human_decision_timestamp")
            if human_dec:
                if human_dec == "PENDING":
                    h_badge = "⏳ Pending Review"
                    h_color = "#f59e0b"
                    h_bg = "rgba(245, 158, 11, 0.15)"
                elif human_dec == "APPROVED":
                    h_badge = "✅ Approved by analyst (logged — no live system integration in this prototype)"
                    h_color = "#10b981"
                    h_bg = "rgba(16, 185, 129, 0.15)"
                elif human_dec == "REJECTED":
                    h_badge = "❌ Rejected by analyst (logged — proposed action will not be executed)"
                    h_color = "#ef4444"
                    h_bg = "rgba(239, 68, 68, 0.15)"
                else:
                    h_badge = f"Human Decision: {human_dec}"
                    h_color = "#94a3b8"
                    h_bg = "rgba(148, 163, 184, 0.15)"

                ts_info = f" &nbsp;•&nbsp; <span style='color: #94a3b8; font-weight: normal; font-size: 0.9em;'>Timestamp (UTC): {human_ts}</span>" if human_ts else ""
                st.markdown(
                    f"""
                    <div style="
                        background-color: {h_bg};
                        border-left: 4px solid {h_color};
                        border-radius: 6px;
                        padding: 8px 14px;
                        margin-bottom: 16px;
                        font-weight: 600;
                        color: {h_color};
                        font-size: 0.95em;
                    ">
                        {h_badge}{ts_info}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            with meta_col1:
                st.write(f"**Timestamp (UTC):** `{selected_record.get('timestamp')}`")
                st.write(f"**Source Domain:** `{selected_record.get('source_agent', 'NETWORK')}`")
            with meta_col2:
                st.write(f"**True Ground-Truth:** `{selected_record.get('true_label')}`")
                st.write(f"**Proposed Action:** `{selected_record.get('proposed_action')}`")
            with meta_col3:
                conf = selected_record.get("detection_confidence", 0.0)
                st.write(f"**ML Prediction:** `{selected_record.get('detection_prediction')}` (Confidence: `{conf:.4f}`)")
                r_score = selected_record.get("risk_score", 0.0)
                st.write(f"**Risk Score:** `{r_score:.4f}` ({selected_record.get('risk_level')})")
            with meta_col4:
                st.write(f"**Gate Decision:** `{selected_record.get('gate_decision')}`")
                st.write(f"**Final Action:** `{selected_record.get('final_action')}`")
                if human_dec:
                    st.write(f"**Human Decision:** `{human_dec}`")

            st.markdown("**Full Gate Reasoning:**")
            decision = selected_record.get("gate_decision")
            reason_text = selected_record.get("gate_reason", "No reason provided")

            if decision == "NEEDS_HUMAN_APPROVAL":
                st.warning(f"⚠️ **Pending Human Approval**: {reason_text}")
            else:
                st.success(f"✅ **Auto-Execution Permitted**: {reason_text}")

            llm_exp = selected_record.get("llm_explanation")
            if llm_exp:
                st.markdown("---")
                st.markdown("#### 🧠 AI Incident Explanation")
                st.info(llm_exp)


if __name__ == "__main__":
    main()
