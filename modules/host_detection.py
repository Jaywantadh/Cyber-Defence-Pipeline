"""Host Threat Detection Module.

[SYNTHETIC DATA SOURCE NOTICE]
==============================
WARNING: ALL DATA PROCESSED AND EVALUATED BY THIS MODULE IS SYNTHETIC.
This module performs machine learning feature preparation, binary classification
training, evaluation, model serialization, and single-event inference for
simulated Windows host endpoint telemetry. It does NOT process authentic enterprise
telemetry.

Real network flow detection in this project is handled separately by
modules/detection.py using CICIDS2017 flow data.
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

from modules.host_dataset_generator import (
    ATTACK_PROCESS_PAIRS,
    generate_host_events,
    load_synthetic_host_events,
)


def prepare_training_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Builds the host feature matrix and target vector, then performs an 80/20 stratified split.

    Features are strictly limited to one-hot encodings of 'process_name' and
    'parent_process'. High-cardinality identity columns ('computer', 'user')
    are deliberately excluded to prevent spurious overfitting, mirroring the
    exclusion of 'Destination Port' in the network flow detection model.

    Parameters
    ----------
    df : pandas.DataFrame
        Synthetic host telemetry DataFrame with columns including 'process_name',
        'parent_process', and 'is_malicious'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]
        (X_train, X_test, y_train, y_test, feature_columns)
    """
    if df.empty:
        raise ValueError("Cannot prepare training data from an empty DataFrame.")

    # One-hot encode process_name and parent_process ONLY
    X = pd.get_dummies(df[["process_name", "parent_process"]], dtype=int)
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
    model_path: Path = PROJECT_ROOT / "models" / "host_detector.pkl",
) -> RandomForestClassifier:
    """Trains a RandomForestClassifier on host feature matrix and saves to disk.

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

    print(f"Training RandomForestClassifier on {len(X_train):,} host samples...")
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
    """Evaluates the trained host model against test data and prints a classification report.

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
    print("HOST MODEL EVALUATION REPORT (Test Set):")
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
    """Prints the top N features by importance from the trained model.

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
    """Performs threat detection on a single host event dictionary.

    Encodes the single event using the same feature_columns list produced during
    training (missing dummy columns filled with 0) to ensure strict alignment.

    Parameters
    ----------
    model : Any
        Trained sklearn classifier with predict_proba and classes_ attributes.
    feature_columns : List[str]
        Canonical list of one-hot encoded feature names produced during training.
    event : Dict[str, Any]
        A single host event dict matching the generator's schema (must include
        'process_name' and 'parent_process', and optionally 'event_id').

    Returns
    -------
    Dict[str, Any]
        Inference result with keys:
        - "event_id": event identifier from input (or None)
        - "prediction": "THREAT" or "BENIGN"
        - "confidence": probability of the predicted class (float)
    """
    process_name = event.get("process_name")
    parent_process = event.get("parent_process")

    # Construct single-row DataFrame and encode using identical feature columns
    row_df = pd.DataFrame([{"process_name": process_name, "parent_process": parent_process}])
    row_encoded = pd.get_dummies(row_df, dtype=int).reindex(columns=feature_columns, fill_value=0)

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
    print("[SYNTHETIC HOST DETECTION PIPELINE]")
    print("NOTE: Evaluating synthetic endpoint process telemetry for prototype testing.")
    print("=" * 70)

    # 1. Generate or load synthetic host events dataset
    csv_path = PROJECT_ROOT / "data" / "raw" / "synthetic_host_events.csv"
    if csv_path.exists():
        print(f"\nLoading synthetic host dataset from: {csv_path} ...")
        df = load_synthetic_host_events(csv_path)
    else:
        print("\nGenerating synthetic host dataset (50,000 events, 3% attacks)...")
        df = generate_host_events(n_events=50000, attack_ratio=0.03, random_state=42)
    print(f"Loaded {len(df):,} synthetic host events.")

    # 2. Prepare training data (one-hot encoding process_name & parent_process ONLY)
    print("\nPreparing training data (80/20 train/test split)...")
    X_train, X_test, y_train, y_test, feature_columns = prepare_training_data(df)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Feature columns count: {len(feature_columns)}")
    print(f"y_train threat distribution:\n{y_train.value_counts()}")
    print(f"y_test threat distribution:\n{y_test.value_counts()}")

    # 3. Train model
    print("\nTraining host detector model...")
    model = train_model(X_train, y_train)

    # 4. Evaluate model
    print("\nEvaluating host detector model...")
    metrics = evaluate_model(model, X_test, y_test)

    # Immediately print feature importance after evaluate_model
    print_feature_importance(model, feature_columns, top_n=10)

    # 5. Run detect() on 3 sample events from distinct categories
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE (detect) ON 3 TEST SAMPLES:")
    print("=" * 60)

    attack_pairs_set = set(ATTACK_PROCESS_PAIRS)

    # Sample 1: Clearly BENIGN (non-overlap pair, e.g. explorer.exe spawning desktop apps)
    benign_non_overlap_mask = (df["is_malicious"] == 0) & (
        ~df.apply(lambda r: (r["parent_process"], r["process_name"]) in attack_pairs_set, axis=1)
    )
    sample_clearly_benign = df[benign_non_overlap_mask].iloc[0].to_dict()

    # Sample 2: Ambiguous overlap pair (e.g. winword.exe spawning cmd.exe or powershell.exe)
    ambiguous_overlap_mask = df.apply(
        lambda r: (r["parent_process"], r["process_name"]) in attack_pairs_set, axis=1
    )
    sample_ambiguous = df[ambiguous_overlap_mask].iloc[0].to_dict()

    # Sample 3: Clearly malicious (non-overlap attack pair with varied parent, e.g. explorer spawning rundll32)
    malicious_non_overlap_mask = (df["is_malicious"] == 1) & (
        ~df.apply(lambda r: (r["parent_process"], r["process_name"]) in attack_pairs_set, axis=1)
    )
    sample_clearly_malicious = df[malicious_non_overlap_mask].iloc[0].to_dict()

    test_samples = [
        ("1. Clearly BENIGN (non-overlap pair)", sample_clearly_benign),
        ("2. Ambiguous Overlap Pair (shared between benign & attack)", sample_ambiguous),
        ("3. Clearly Malicious (non-overlap attack pair)", sample_clearly_malicious),
    ]

    for category, sample_ev in test_samples:
        result = detect(model, feature_columns, sample_ev)
        print(f"Sample Category : {category}")
        print(f"Process Tree    : {sample_ev.get('parent_process')} -> {sample_ev.get('process_name')}")
        print(f"True Label      : {sample_ev.get('label')} (is_malicious={sample_ev.get('is_malicious')})")
        print(f"detect() Result : {result}\n")
