"""Threat Detection Module.

Responsible for feature matrix preparation, binary threat classification training,
model evaluation, persistence, and single-event inference.
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

from modules.dataset_loader import load_raw_data
from modules.ingestion import normalize_events


def prepare_training_data(
    events: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Converts normalized events into feature matrix X and binarized label vector y,

    then performs an 80/20 stratified train/test split.

    Parameters
    ----------
    events : List[Dict[str, Any]]
        List of normalized event dictionaries from ingestion.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    if not events:
        raise ValueError("Cannot prepare training data from an empty event list.")

    # Canonical column ordering from the first event, excluding 'Destination Port'
    feature_names = [col for col in events[0]["features"].keys() if col != "Destination Port"]

    # Build feature DataFrame with consistent column ordering (77 features)
    X = pd.DataFrame([event["features"] for event in events])[feature_names]

    # Binarize labels: 'BENIGN' -> 0, anything else -> 1
    y = pd.Series(
        [0 if event.get("label") == "BENIGN" else 1 for event in events],
        name="target",
    )

    # 80/20 train/test split, stratified on y with fixed random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_path: Path = PROJECT_ROOT / "models" / "threat_detector.pkl",
) -> RandomForestClassifier:
    """Trains a RandomForest baseline classifier and persists it to disk.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
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

    print(f"Training RandomForestClassifier on {len(X_train):,} samples...")
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
    """Evaluates the trained model against test data and prints a classification report.

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
    print("MODEL EVALUATION REPORT (Test Set):")
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


def detect(model: Any, event: Dict[str, Any]) -> Dict[str, Any]:
    """Performs threat detection on a single normalized event dictionary.

    Parameters
    ----------
    model : Any
        Trained sklearn classifier with predict_proba and classes_ attributes.
    event : Dict[str, Any]
        A single normalized event dict with keys 'event_id', 'features', and 'label'.

    Returns
    -------
    Dict[str, Any]
        Inference result with keys 'event_id', 'prediction' ('THREAT' or 'BENIGN'),
        and 'confidence' (float).
    """
    feature_dict = event["features"]

    # Align features with the exact columns and ordering expected by the model (excluding Destination Port)
    if hasattr(model, "feature_names_in_"):
        expected_cols = [col for col in model.feature_names_in_ if col != "Destination Port"]
        row_values = [feature_dict[col] for col in expected_cols]
        row_df = pd.DataFrame([row_values], columns=expected_cols)
    else:
        filtered_dict = {k: v for k, v in feature_dict.items() if k != "Destination Port"}
        row_df = pd.DataFrame([filtered_dict])

    probs = model.predict_proba(row_df)[0]
    classes = list(model.classes_)
    best_idx = int(np.argmax(probs))
    predicted_class = classes[best_idx]
    confidence = float(probs[best_idx])

    prediction_str = "THREAT" if predicted_class == 1 else "BENIGN"

    return {
        "event_id": event["event_id"],
        "prediction": prediction_str,
        "confidence": confidence,
    }


def print_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
) -> None:
    """Prints the top N features by importance from the trained RandomForestClassifier,

    sorted descending, with their importance scores.

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


if __name__ == "__main__":
    csv_path = PROJECT_ROOT / "data" / "raw" / "Tuesday-WorkingHours.pcap_ISCX.csv"
    if not csv_path.exists():
        print(f"Error: Target file not found at {csv_path}")
        sys.exit(1)

    print(f"Loading raw data from: {csv_path} ...")
    raw_df = load_raw_data(csv_path)

    print("\nNormalizing events...")
    events = normalize_events(raw_df)
    print(f"Normalized {len(events):,} events.")

    print("\nPreparing training data (80/20 train/test split)...")
    X_train, X_test, y_train, y_test = prepare_training_data(events)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train threat distribution:\n{y_train.value_counts()}")
    print(f"y_test threat distribution:\n{y_test.value_counts()}")

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print_feature_importance(model, X_train.columns.tolist(), top_n=10)

    print("\n" + "=" * 60)
    print("RUNNING INFERENCE (detect) ON 3 TEST SET SAMPLES:")
    print("=" * 60)

    # Find one sample of BENIGN, FTP-Patator, and SSH-Patator from the test set
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

    for label_category, sample_ev in sample_targets:
        if sample_ev is not None:
            result = detect(model, sample_ev)
            print(f"Sample Category : {label_category}")
            print(f"True Label      : {sample_ev.get('label')}")
            print(f"detect() Result : {result}\n")
        else:
            print(f"Warning: Could not find a test sample for category {label_category}\n")
