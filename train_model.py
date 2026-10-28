"""Train and serialize the Week 4 prediction model using the Week 3 dataset."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Use the same built-in dataset and stratified split approach as Week 3.
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # The scaler is part of the saved pipeline, so inference uses the same scaling.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob_benign = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "benign_precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "benign_recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "benign_f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "malignant_recall": float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "roc_auc_benign_positive": float(roc_auc_score(y_test, y_prob_benign)),
    }

    # Keep feature names and class names with the artifact for safer inference.
    bundle = {
        "model": model,
        "feature_names": list(X.columns),
        "target_names": list(dataset.target_names),
        "target_mapping": {"0": "malignant", "1": "benign"},
        "metrics": metrics,
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "model_description": "StandardScaler + LogisticRegression(max_iter=3000)",
    }
    joblib.dump(bundle, ARTIFACTS / "model.joblib")

    metadata = {
        "dataset": "Breast Cancer Wisconsin (built into scikit-learn)",
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "target_names": list(dataset.target_names),
        "target_mapping": {"0": "malignant", "1": "benign"},
        "model_description": bundle["model_description"],
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "metrics": metrics,
        "scikit_learn_version": __import__("sklearn").__version__,
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Model saved to:", ARTIFACTS / "model.joblib")
    print("Metadata saved to:", ARTIFACTS / "metadata.json")
    print("Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
