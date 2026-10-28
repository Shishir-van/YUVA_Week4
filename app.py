"""FastAPI application for the Week 4 ML deployment capstone."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model artifact not found. Run `python train_model.py` from the project folder first."
    )

BUNDLE: dict[str, Any] = joblib.load(MODEL_PATH)
MODEL = BUNDLE["model"]
FEATURE_NAMES: list[str] = BUNDLE["feature_names"]
TARGET_NAMES: list[str] = BUNDLE["target_names"]
METRICS: dict[str, Any] = BUNDLE["metrics"]

app = FastAPI(
    title="Breast Cancer Classification API",
    description=(
        "Educational deployment demo using the scikit-learn Breast Cancer Wisconsin dataset. "
        "This model is not intended for clinical diagnosis or medical decision-making."
    ),
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    """A single observation containing all 30 named dataset features."""

    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(
        ..., description="Map each of the 30 dataset feature names to a numeric value."
    )


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_label: int
    probabilities: dict[str, float]
    model: str
    disclaimer: str


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "message": "Breast Cancer Classification API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["General"])
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": "true"}


@app.get("/model-info", tags=["General"])
def model_info() -> dict[str, Any]:
    return {
        "model": BUNDLE["model_description"],
        "dataset": "Breast Cancer Wisconsin (scikit-learn)",
        "number_of_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "classes": TARGET_NAMES,
        "test_metrics": METRICS,
        "disclaimer": "Educational example only; not for clinical use.",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    supplied = set(request.features.keys())
    expected = set(FEATURE_NAMES)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        detail: dict[str, Any] = {}
        if missing:
            detail["missing_features"] = missing
        if extra:
            detail["unknown_features"] = extra
        raise HTTPException(status_code=422, detail=detail)

    # Reorder fields to the exact training order, regardless of JSON key order.
    row = pd.DataFrame([[request.features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    predicted_label = int(MODEL.predict(row)[0])
    probabilities_array = MODEL.predict_proba(row)[0]
    probabilities = {
        TARGET_NAMES[index]: float(probability)
        for index, probability in enumerate(probabilities_array)
    }
    return PredictionResponse(
        predicted_class=TARGET_NAMES[predicted_label],
        predicted_label=predicted_label,
        probabilities=probabilities,
        model=BUNDLE["model_description"],
        disclaimer="Educational example only; this prediction must not guide medical decisions.",
    )
