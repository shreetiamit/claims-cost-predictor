"""
Claims Risk API: serves the high-cost model over HTTP.

    GET  /health   is the service up, and did the model load?
    POST /predict  score one member at the care-management threshold (0.05)

Run locally:
    uvicorn app.main:app --reload --port 8000
    open http://localhost:8000/docs

Test:
    pytest tests/test_api.py -v
"""

import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pathlib import Path

from app.schemas import MemberFeatures, PredictionResponse
from config import HIGH_COST_THRESHOLD_USD

MODEL_PATH = Path("model.joblib")
DECISION_THRESHOLD = 0.05  # care-management operating point from evaluate.py
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of on every request.

    If model.joblib is missing, refuse to start: a service that boots and then
    fails every request is worse than one that fails loudly at deploy time.
    """
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run `python train.py` to create it, then restart."
        )
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Claims Risk API",
    description="Predicts whether a member is likely to exceed the plan-paid threshold.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Liveness check for Render. Reports whether the model actually loaded,
    since "the process is up" and "the service can serve" are different claims."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: MemberFeatures):
    """Score one member.

    Returns 503 if the model isn't loaded, and 422 if the counts contradict
    each other: the request is valid JSON but describes an impossible member,
    so we say so rather than return a score.

    The row goes to the pipeline untouched. Its preprocessing is inside it, so
    any rescaling or reordering here would cause train/serve skew.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python train.py` and restart the service.",
        )

    # Rules that span several fields, which the schema can't check on its own.
    f = features
    problems = []
    if f.medical_claims + f.pharmacy_claims != f.total_claims:
        problems.append("medical_claims + pharmacy_claims must equal total_claims")
    if f.inpatient_claims > f.medical_claims:
        problems.append("inpatient_claims cannot exceed medical_claims")
    if f.specialty_drug_fills > f.pharmacy_claims:
        problems.append("specialty_drug_fills cannot exceed pharmacy_claims")
    if f.unique_diagnosis_codes > f.medical_claims:
        problems.append("unique_diagnosis_codes cannot exceed medical_claims")
    if f.has_specialty_drug != (f.specialty_drug_fills > 0):
        problems.append("has_specialty_drug must be true exactly when specialty_drug_fills > 0")
    if problems:
        raise HTTPException(status_code=422, detail=problems)

    row = pd.DataFrame([features.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])

    return PredictionResponse(
        is_high_cost=probability >= DECISION_THRESHOLD,
        probability=round(probability, 4),
        threshold_dollars=HIGH_COST_THRESHOLD_USD,
    )