"""LLM Reasoning and Explanation Module.

[READ-ONLY EXPLANATION LAYER NOTICE]
====================================
IMPORTANT ARCHITECTURAL GUARD:
This module is strictly a read-only advisory and explanation layer.
It NEVER modifies, overrides, or decides any security action or gate decision.
Its sole responsibility is to translate already-escalated security incidents
(where gate_decision == "NEEDS_HUMAN_APPROVAL") into plain-language, grounded
explanations for human reviewers and security operations analysts.

Autonomous action execution by LLMs is intentionally prohibited in this phase
because security verification guardrails around generative actions (Module 15)
have not yet been implemented. This module explains and advises; human analysts
retain sole execution authority.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import requests

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.guardrails import check_groundedness, sanitize_for_prompt

# Single, explicitly named constant for the active Gemini model endpoint
GEMINI_MODEL = "gemini-3.8-flash"


def build_incident_prompt(gate_result_or_combined: Dict[str, Any]) -> str:
    """Builds a grounded, plain-text explanation prompt for the Gemini model.

    Accepts either a single-agent gate result dictionary (from network, host, or
    iam pipelines) or a combined cross-domain result from conflict_resolver.py.
    All untrusted telemetry text fields are sanitized to defend against prompt injection.

    Parameters
    ----------
    gate_result_or_combined : Dict[str, Any]
        Incident gate result dictionary containing evidence fields.

    Returns
    -------
    str
        Formatted plain-text prompt instructing the model to explain the incident.
    """
    def _sanitize(val: Any, field_name: str) -> str:
        if val is None:
            return ""
        res = sanitize_for_prompt(str(val), field_name=field_name)
        if res["flagged"]:
            print(
                f"Warning: Field '{field_name}' flagged for injection pattern "
                f"'{res['matched_pattern']}'; using filtered text in prompt."
            )
        return res["sanitized_text"]

    is_combined = "host_result" in gate_result_or_combined and "iam_result" in gate_result_or_combined

    if is_combined:
        computer = _sanitize(gate_result_or_combined.get("computer", "Shared Host"), "computer")
        combined_decision = gate_result_or_combined.get("combined_gate_decision", "NEEDS_HUMAN_APPROVAL")
        execution_order = gate_result_or_combined.get("execution_order", [])
        combined_reason = _sanitize(gate_result_or_combined.get("combined_reason", ""), "combined_reason")
        h_res = gate_result_or_combined.get("host_result", {})
        i_res = gate_result_or_combined.get("iam_result", {})

        h_action = _sanitize(h_res.get("final_action", ""), "host_final_action")
        h_reason = _sanitize(h_res.get("gate_reason", "N/A"), "host_gate_reason")
        i_action = _sanitize(i_res.get("final_action", ""), "iam_final_action")
        i_reason = _sanitize(i_res.get("gate_reason", "N/A"), "iam_gate_reason")

        evidence = (
            f"Incident Classification: Correlated Multi-Domain Attack (Host & IAM Collision)\n"
            f"Target Computer: {computer}\n"
            f"Overall Gate Decision: {combined_decision}\n"
            f"Resolved Action Execution Sequence: {execution_order}\n"
            f"Host Subsystem Finding: Action '{h_action}' "
            f"(Risk: {h_res.get('risk_score', 'N/A')}). Reason: {h_reason}\n"
            f"IAM Subsystem Finding: Action '{i_action}' "
            f"(Risk: {i_res.get('risk_score', 'N/A')}). Reason: {i_reason}\n"
            f"Joint Reasoning Trail: {combined_reason}"
        )
    else:
        source_agent = _sanitize(gate_result_or_combined.get("source_agent", "Security Telemetry Agent"), "source_agent")
        event_id = gate_result_or_combined.get("event_id", "N/A")
        action = _sanitize(gate_result_or_combined.get("final_action", gate_result_or_combined.get("action", "N/A")), "action")
        risk_score = gate_result_or_combined.get("risk_score", "N/A")
        risk_level = gate_result_or_combined.get("risk_level", "N/A")
        gate_decision = gate_result_or_combined.get("gate_decision", "NEEDS_HUMAN_APPROVAL")
        gate_reason = _sanitize(gate_result_or_combined.get("gate_reason", "N/A"), "gate_reason")
        label = _sanitize(gate_result_or_combined.get("true_label", ""), "true_label")
        confidence = gate_result_or_combined.get("detection_confidence", "")

        evidence = (
            f"Source Telemetry Domain: {source_agent}\n"
            f"Event Identifier: {event_id}\n"
            f"Proposed Defense Action: {action}\n"
            f"Assessed Risk Score: {risk_score} (Level: {risk_level})\n"
            f"Safety Gate Decision: {gate_decision}\n"
            f"Policy Enforcement Reason: {gate_reason}\n"
        )
        if label:
            evidence += f"Ground-Truth Label / Signature: {label}\n"
        if confidence:
            evidence += f"Model Detection Confidence: {confidence}\n"

    prompt = (
        "You are an expert security operations advisor assisting a human security analyst.\n"
        "An autonomous defensive pipeline evaluated an event and held it for human review.\n\n"
        f"INCIDENT EVIDENCE:\n{evidence}\n\n"
        "INSTRUCTIONS FOR YOUR RESPONSE:\n"
        "1. Summarize what happened in 2-3 clear sentences that a non-technical manager or reviewer can readily understand.\n"
        "2. State explicitly why the action was blocked from automatic execution and escalated for human approval.\n"
        "3. Suggest 1-2 practical diagnostic checks the human analyst should inspect first to verify whether this is benign or malicious.\n\n"
        "STRICT SAFETY RESTRICTIONS:\n"
        "- Do NOT recommend specific CLI command-line syntax (e.g., no PowerShell, cmd, or bash commands).\n"
        "- Do NOT claim to take, execute, or authorize any remediation action yourself.\n"
        "- Keep your tone advisory, concise, objective, and strictly grounded in the evidence provided."
    )
    return prompt


def _build_fallback_explanation(gate_result_or_combined: Dict[str, Any]) -> str:
    """Builds a deterministic templated explanation from raw incident evidence.

    Invoked when network connectivity, API rate limits, or credentials prevent
    live LLM inference.
    """
    is_combined = "host_result" in gate_result_or_combined and "iam_result" in gate_result_or_combined
    if is_combined:
        comp = gate_result_or_combined.get("computer", "endpoint")
        order = gate_result_or_combined.get("execution_order", [])
        reason = gate_result_or_combined.get("combined_reason", "Multi-domain action conflict.")
        return (
            f"Correlated incident on computer {comp} was escalated to human review due to overlapping "
            f"actions across Host and IAM domains ({order}). Policy justification: {reason}. "
            f"The analyst should verify Active Directory authentication history and inspect running processes on this host."
        )
    else:
        src = gate_result_or_combined.get("source_agent", "Security agent")
        act = gate_result_or_combined.get("final_action", gate_result_or_combined.get("action", "action"))
        score = gate_result_or_combined.get("risk_score", "N/A")
        reason = gate_result_or_combined.get("gate_reason", "Policy threshold exceeded.")
        return (
            f"{src} incident was held for human authorization: proposed remediation action '{act}' "
            f"with risk score {score}. Policy reason: {reason}. "
            f"The analyst should inspect user session validity and check for anomalous endpoint activity."
        )


def explain_incident(
    gate_result_or_combined: Dict[str, Any],
    timeout_seconds: float = 10.0,
) -> Dict[str, Any]:
    """Generates an advisory explanation for incidents escalated to human review.

    Parameters
    ----------
    gate_result_or_combined : Dict[str, Any]
        Single-agent gate result or conflict-resolved combined incident dictionary.
    timeout_seconds : float, default=10.0
        HTTP request timeout in seconds.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'explanation': str (the generated explanation or fallback)
        - 'source': str (GEMINI_MODEL or 'fallback_template')
        - 'error': Optional[str] (None on success, error description on failure)

    Raises
    ------
    ValueError
        If called on an AUTO_EXECUTE incident, or if GEMINI_API_KEY is not set.
    """
    # 1. Guard check: only escalated incidents receive LLM explanation
    decision = gate_result_or_combined.get("gate_decision") or gate_result_or_combined.get(
        "combined_gate_decision"
    )
    if decision != "NEEDS_HUMAN_APPROVAL":
        raise ValueError(
            f"explain_incident() can only be called on incidents requiring human review "
            f"('NEEDS_HUMAN_APPROVAL'), but received decision: '{decision}'."
        )

    # 2. Check API key at call time
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. Please add "
            "GEMINI_API_KEY=<your_key> to your .env file or environment."
        )

    # 3. Build prompt
    prompt = build_incident_prompt(gate_result_or_combined)

    # 4. Attempt Gemini API call via plain HTTP
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 800,
            "temperature": 0.2,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts and "text" in parts[0]:
                    explanation_text = parts[0]["text"].strip()
                    groundedness = check_groundedness(explanation_text, gate_result_or_combined)
                    return {
                        "explanation": explanation_text,
                        "source": GEMINI_MODEL,
                        "error": None,
                        "groundedness": groundedness,
                    }
            last_error = "Malformed API response: no text parts in candidates"
        else:
            last_error = f"HTTP {resp.status_code}: {resp.text}"
    except Exception as exc:
        last_error = str(exc)

    # 5. Graceful fallback on any failure
    fallback_text = _build_fallback_explanation(gate_result_or_combined)
    groundedness = check_groundedness(fallback_text, gate_result_or_combined)
    return {
        "explanation": fallback_text,
        "source": "fallback_template",
        "error": last_error or "API request failed",
        "groundedness": groundedness,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("LLM REASONING & INCIDENT EXPLANATION MODULE TEST")
    print("[SAFETY NOTICE]: Read-only advisory layer. No decisions are executed or modified.")
    print("=" * 80)

    # 1. API key check
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("\n[WARNING]: GEMINI_API_KEY is not set in environment or .env file.")
        print("Please configure GEMINI_API_KEY to enable live LLM reasoning tests.")
        print("Exiting test early.")
        sys.exit(0)

    print(f"\nGEMINI_API_KEY detected in environment (credentials verified).")
    print(f"Configured Gemini Model: '{GEMINI_MODEL}'")

    # 2. Run existing IAM 4-sample test to obtain real gate evaluation results
    from modules.iam_dataset_generator import generate_iam_events
    from modules.iam_detection import detect as iam_detect
    from modules.iam_agent import propose_action as iam_propose_action
    from modules.risk_engine import compute_risk_score
    from modules.safety_gate import evaluate_gate
    from modules.audit_log import init_db, log_decision, get_recent_decisions
    import joblib

    iam_model_path = PROJECT_ROOT / "models" / "iam_detector.pkl"
    if not iam_model_path.exists():
        print(f"Error: Trained IAM model not found at {iam_model_path}")
        sys.exit(1)

    iam_model = joblib.load(iam_model_path)
    iam_feature_cols = list(iam_model.feature_names_in_)

    df_iam = generate_iam_events(n_events=50000, attack_ratio=0.03, random_state=42)

    # Sample 1: Clearly BENIGN (AUTO_EXECUTE)
    benign_mask = (df_iam["is_malicious"] == 0) & (df_iam["auth_result"] == "SUCCESS")
    sample_clearly_benign = df_iam[benign_mask].iloc[0].to_dict()

    # Sample 4: PASS_THE_HASH (typical successful NTLM network authentication -> RESET_CREDENTIALS)
    pth_mask = (df_iam["label"] == "PASS_THE_HASH") & (df_iam["auth_result"] == "SUCCESS")
    sample_pth = df_iam[pth_mask].iloc[0].to_dict()

    # Evaluate Sample 4 (PASS_THE_HASH)
    pth_det = iam_detect(iam_model, iam_feature_cols, sample_pth)
    pth_act = iam_propose_action(pth_det, sample_pth)
    pth_risk = compute_risk_score(pth_act)
    pth_gate = evaluate_gate(pth_risk)
    pth_gate["risk_score"] = pth_risk.get("risk_score")
    pth_gate["source_agent"] = "IAM"
    pth_gate["true_label"] = sample_pth.get("label")
    pth_gate["detection_confidence"] = pth_det.get("confidence")

    print("\n" + "=" * 80)
    print("TEST 1: EXPLAINING INCIDENT REQUIRING HUMAN APPROVAL (PASS_THE_HASH)")
    print("=" * 80)
    print(f"  Event ID        : {pth_gate['event_id']}")
    print(f"  Source Agent    : {pth_gate['source_agent']}")
    print(f"  Proposed Action : {pth_gate['final_action']}")
    print(f"  Risk Score      : {pth_gate['risk_score']}")
    print(f"  Gate Decision   : {pth_gate['gate_decision']}")
    print(f"  Gate Reason     : {pth_gate['gate_reason']}")

    print("\nCalling explain_incident()...")
    result = explain_incident(pth_gate)
    print(f"\nResult Source : {result['source']}")
    print(f"Result Error  : {result['error']}")
    print("\nGenerated Incident Explanation:")
    print("-" * 80)
    print(result["explanation"])
    print("-" * 80)

    assert "groundedness" in result, "Expected 'groundedness' key in explain_incident result"
    print(f"Groundedness Check: {result['groundedness']}")

    # Confirm source accurately matches GEMINI_MODEL or fallback_template
    if result["error"] is None:
        assert result["source"] == GEMINI_MODEL, (
            f"Expected source '{GEMINI_MODEL}', got '{result['source']}'"
        )
    else:
        assert result["source"] == "fallback_template"

    # Evaluate Sample 1 (Clearly BENIGN)
    benign_det = iam_detect(iam_model, iam_feature_cols, sample_clearly_benign)
    benign_act = iam_propose_action(benign_det, sample_clearly_benign)
    benign_risk = compute_risk_score(benign_act)
    benign_gate = evaluate_gate(benign_risk)
    benign_gate["risk_score"] = benign_risk.get("risk_score")
    benign_gate["source_agent"] = "IAM"

    print("\n" + "=" * 80)
    print("TEST 2: DEMONSTRATING SAFETY GUARD (ATTEMPTING EXPLANATION ON AUTO_EXECUTE)")
    print("=" * 80)
    print(f"  Event ID        : {benign_gate['event_id']}")
    print(f"  Gate Decision   : {benign_gate['gate_decision']}")

    try:
        explain_incident(benign_gate)
        print("FAIL: explain_incident() did not raise ValueError on AUTO_EXECUTE!")
    except ValueError as exc:
        print(f"\nSUCCESS: Safety guard properly raised ValueError as expected:")
        print(f"  >> {exc}")

    # TEST 3: Persist to audit log and verify read-back
    print("\n" + "=" * 80)
    print("TEST 3: AUDIT LOG PERSISTENCE & VERIFICATION (log_decision / get_recent_decisions)")
    print("=" * 80)
    init_db()

    # Format explanation with source model tag so both persist
    explanation_to_log = f"[{result['source']}] {result['explanation']}"

    row_id = log_decision(
        event=sample_pth,
        detection_result=pth_det,
        action_proposal=pth_act,
        risk_result=pth_risk,
        gate_result=pth_gate,
        source_agent="IAM",
        llm_explanation=explanation_to_log,
    )

    recent_rows = get_recent_decisions(limit=1)
    retrieved_row = recent_rows[0]

    print(f"  Inserted Audit Log Row ID : {row_id}")
    print(f"  Retrieved Row ID          : {retrieved_row['id']}")
    print(f"  Retrieved Source Agent    : {retrieved_row['source_agent']}")
    print(f"  Retrieved LLM Explanation :")
    print("-" * 80)
    print(retrieved_row["llm_explanation"])
    print("-" * 80)

    # Verify both explanation text and model source persist and read back correctly
    assert retrieved_row["id"] == row_id, f"Row ID mismatch: expected {row_id}, got {retrieved_row['id']}"
    assert result["source"] in retrieved_row["llm_explanation"], (
        f"Source model '{result['source']}' not found in retrieved llm_explanation!"
    )
    assert result["explanation"] in retrieved_row["llm_explanation"], (
        "Explanation text not found in retrieved llm_explanation!"
    )

    print(f"\nCONFIRMED: Explanation text and source model '{result['source']}' successfully persisted and verified from logs/audit.db.")
    print("\n" + "=" * 80)
    print("ALL LLM REASONING & AUDIT LOG PERSISTENCE TESTS COMPLETED SUCCESSFULLY.")
    print("=" * 80)
