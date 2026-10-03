"""IAM Threat Detection Module.

[SYNTHETIC DATA SOURCE NOTICE]
==============================
WARNING: ALL DATA PROCESSED AND EVALUATED BY THIS MODULE IS SYNTHETIC.
This module performs machine learning feature preparation, binary threat classification
training, evaluation, model serialization, and single-event inference for simulated
Windows IAM authentication and logon telemetry. It does NOT process authentic enterprise
Active Directory, Okta, or Kerberos/NTLM authentication logs.

Note on Detection Confidence:
Model confidence returned by detect() is NOT a calibrated probability due to
training with class_weight='balanced' on an imbalanced dataset (~97/3 ratio) with
natural class overlap. It serves as a relative decision-weighting signal rather than
a true posterior probability.

Real network flow detection in this project is handled separately by modules/detection.py
using CICIDS2017 flow data.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.iam_dataset_generator import (
    generate_iam_events,
    load_synthetic_iam_events,
)


def prepare_training_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Builds the IAM feature matrix and target vector, then performs an 80/20 stratified split.

    Features are strictly limited to one-hot encodings of 'auth_type', 'logon_type',
    and 'auth_result'. High-cardinality identity columns ('source_computer',
    'target_computer', 'user') are deliberately excluded to prevent spurious overfitting
    on non-generalizable machine/user IDs, mirroring the exclusion of 'Destination Port'
    in the network flow detection model.

    Parameters
    ----------
    df : pandas.DataFrame
        Synthetic IAM telemetry DataFrame with columns including 'auth_type',
        'logon_type', 'auth_result', and 'is_malicious'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]
        (X_train, X_test, y_train, y_test, feature_columns)
    """
    if df.empty:
        raise ValueError("Cannot prepare training data from an empty DataFrame.")

    # One-hot encode auth_type, logon_type, and auth_result ONLY
    X = pd.get_dummies(df[["auth_type", "logon_type", "auth_result"]], dtype=int)
    feature_columns = X.columns.tolist()

    # Binary target: 0 for BENIGN, 1 for THREAT
    y = df["is_malicious"].astype(int)

    # 80/20 train/test split, stratified on y with fixed random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_columns


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_path: Path = PROJECT_ROOT / "models" / "iam_detector.pkl",
) -> RandomForestClassifier:
    """Trains a RandomForestClassifier on the IAM feature matrix and saves to disk.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector (0 for BENIGN, 1 for THREAT).
    model_path : Path, optional
        Destination path for serializing the trained model.

    Returns
    -------
    RandomForestClassifier
        The fitted scikit-learn model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print(f"Training RandomForestClassifier on {len(X_train):,} IAM samples...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Save trained model to disk
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved trained model to: {model_path}")

    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluates the trained IAM model against test data and prints a classification report.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained model to evaluate.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True test labels (0 for BENIGN, 1 for THREAT).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing precision, recall, f1_score, confusion_matrix,
        and classification_report dictionary.
    """
    y_pred = model.predict(X_test)

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "=" * 60)
    print("IAM MODEL EVALUATION REPORT (Test Set):")
    print("=" * 60)
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["BENIGN (0)", "THREAT (1)"],
            digits=4,
        )
    )
    print("Confusion Matrix:")
    print(cm)
    print("=" * 60)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }


def print_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
) -> None:
    """Prints the top N features by importance from the trained IAM model.

    Parameters
    ----------
    model : Any
        Trained model with `feature_importances_` attribute.
    feature_names : List[str]
        List of feature names corresponding to the columns in X_train.
    top_n : int, optional
        Number of top features to display (default: 10).
    """
    if not hasattr(model, "feature_importances_"):
        print("Model does not have feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    print("\n" + "=" * 60)
    print(f"TOP {top_n} FEATURE IMPORTANCES:")
    print("=" * 60)
    for rank, idx in enumerate(sorted_indices[:top_n], start=1):
        col_name = feature_names[idx]
        score = importances[idx]
        print(f"  {rank:2d}. {col_name:<35} : {score:.6f}")
    print("=" * 60)


def detect(
    model: Any,
    feature_columns: List[str],
    event: Dict[str, Any],
) -> Dict[str, Any]:
    """Performs threat detection on a single IAM event dictionary.

    Encodes the single event using the same feature_columns list produced during
    training (missing dummy columns filled with 0) to ensure strict alignment.

    Parameters
    ----------
    model : Any
        Trained sklearn classifier with predict_proba and classes_ attributes.
    feature_columns : List[str]
        Canonical list of one-hot encoded feature names produced during training.
    event : Dict[str, Any]
        A single IAM event dict matching the generator's schema (must include
        'auth_type', 'logon_type', and 'auth_result', and optionally 'event_id').

    Returns
    -------
    Dict[str, Any]
        Inference result with keys:
        - "event_id": event identifier from input (or None)
        - "prediction": "THREAT" or "BENIGN"
        - "confidence": probability of the predicted class (float)
    """
    auth_type = event.get("auth_type")
    logon_type = event.get("logon_type")
    auth_result = event.get("auth_result")

    # Construct single-row DataFrame and encode using identical feature columns
    row_df = pd.DataFrame(
        [
            {
                "auth_type": auth_type,
                "logon_type": logon_type,
                "auth_result": auth_result,
            }
        ]
    )
    row_encoded = pd.get_dummies(row_df, dtype=int).reindex(
        columns=feature_columns, fill_value=0
    )

    probs = model.predict_proba(row_encoded)[0]
    classes = list(model.classes_)
    best_idx = int(np.argmax(probs))
    predicted_class = classes[best_idx]
    confidence = float(probs[best_idx])

    prediction_str = "THREAT" if predicted_class == 1 else "BENIGN"

    return {
        "event_id": event.get("event_id"),
        "prediction": prediction_str,
        "confidence": confidence,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("[SYNTHETIC IAM DETECTION PIPELINE]")
    print("NOTE: Evaluating synthetic IAM authentication telemetry for prototype testing.")
    print("=" * 70)

    # 1. Generate or load synthetic IAM events dataset
    csv_path = PROJECT_ROOT / "data" / "raw" / "synthetic_iam_events.csv"
    if csv_path.exists():
        print(f"\nLoading synthetic IAM dataset from: {csv_path} ...")
        df = load_synthetic_iam_events(csv_path)
    else:
        print("\nGenerating synthetic IAM dataset (50,000 events, 3% attacks)...")
        df = generate_iam_events(n_events=50000, attack_ratio=0.03, random_state=42)
    print(f"Loaded {len(df):,} synthetic IAM events.")

    # 2. Prepare training data (one-hot encoding auth_type, logon_type, auth_result ONLY)
    print("\nPreparing training data (80/20 train/test split)...")
    X_train, X_test, y_train, y_test, feature_columns = prepare_training_data(df)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Feature columns count: {len(feature_columns)}")
    print(f"y_train threat distribution:\n{y_train.value_counts()}")
    print(f"y_test threat distribution:\n{y_test.value_counts()}")

    # 3. Train model
    print("\nTraining IAM detector model...")
    model = train_model(X_train, y_train)

    # 4. Evaluate model
    print("\nEvaluating IAM detector model...")
    metrics = evaluate_model(model, X_test, y_test)

    # Immediately print feature importance after evaluate_model
    print_feature_importance(model, feature_columns, top_n=10)

    # 5. Attack-specific breakdown on test set (caught vs missed)
    print("\n" + "=" * 60)
    print("ATTACK-SPECIFIC TEST SET BREAKDOWN (Caught vs. Missed):")
    print("=" * 60)
    y_test_pred = model.predict(X_test)
    test_labels = df.loc[X_test.index, "label"]

    breakdown_df = pd.DataFrame(
        {
            "True Label": test_labels,
            "Predicted": ["THREAT" if p == 1 else "BENIGN" for p in y_test_pred],
            "Detected": y_test_pred == 1,
        }
    )

    ct = pd.crosstab(
        breakdown_df["True Label"],
        breakdown_df["Predicted"],
        margins=True,
    )
    print("Cross-Tabulation (True Label vs. Predicted):")
    print(ct.to_string())
    print("\nPer-Attack Detection Counts:")
    for attack in ["BRUTE_FORCE_LOGIN", "PASS_THE_HASH"]:
        sub = breakdown_df[breakdown_df["True Label"] == attack]
        caught = int(sub["Detected"].sum())
        total = len(sub)
        missed = total - caught
        pct_caught = (caught / total) * 100 if total else 0.0
        pct_missed = (missed / total) * 100 if total else 0.0
        print(
            f"  {attack:<20} : Caught={caught:>3}/{total} ({pct_caught:5.1f}%) | "
            f"Missed={missed:>2}/{total} ({pct_missed:4.1f}%)"
        )
    print("=" * 60)

    # 6. Run detect() on 4 sample events representing distinct IAM categories
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE (detect) ON 4 TEST SAMPLES:")
    print("=" * 60)

    # Sample 1: Clearly BENIGN (SUCCESS / Interactive / Kerberos-typical row)
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

    # Sample 3: BRUTE_FORCE_LOGIN (typical network NTLM login attempt)
    sample_brute_force = df[df["label"] == "BRUTE_FORCE_LOGIN"].iloc[0].to_dict()

    # Sample 4: PASS_THE_HASH (typical successful NTLM network authentication)
    sample_pass_the_hash = df[
        (df["label"] == "PASS_THE_HASH") & (df["auth_result"] == "SUCCESS")
    ].iloc[0].to_dict()

    test_samples = [
        ("1. Clearly BENIGN (SUCCESS / Interactive / Kerberos)", sample_clearly_benign),
        ("2. BENIGN-but-FAILURE (Natural Ambiguous Overlap)", sample_benign_failure),
        ("3. BRUTE_FORCE_LOGIN (Network / NTLM / High Failure)", sample_brute_force),
        ("4. PASS_THE_HASH (Network / NTLM / Successful Hash Auth)", sample_pass_the_hash),
    ]

    for category, sample_ev in test_samples:
        result = detect(model, feature_columns, sample_ev)
        print(f"Sample Category : {category}")
        print(
            f"Event Attributes: auth_type={sample_ev.get('auth_type')}, "
            f"logon_type={sample_ev.get('logon_type')}, "
            f"auth_result={sample_ev.get('auth_result')}"
        )
        print(f"True Label      : {sample_ev.get('label')} (is_malicious={sample_ev.get('is_malicious')})")
        print(f"detect() Result : {result}\n")
