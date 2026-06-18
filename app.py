import json
import logging
from pathlib import Path
from typing import List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("artifacts/model.pkl")
FEATURE_COLUMNS_PATH = Path("artifacts/feature_columns.json")
MAX_BATCH_SIZE = 100

metrics = {
    "n_predictions": 0,
    "n_errors": 0,
    "n_batch_requests": 0,
    "n_batch_inputs_total": 0,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

model = None
feature_columns = []

try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
except Exception as e:
    logger.error("Erreur au chargement des artefacts : %s", e)


class CustomerInput(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract: Literal["Month-to-month", "One year", "Two year"]


class BatchInput(BaseModel):
    inputs: List[CustomerInput]


app = FastAPI(title="Churn Prediction API", version="1.0")


def _run_inference(payload: CustomerInput) -> dict:
    df = pd.DataFrame([payload.model_dump()])
    df_encoded = pd.get_dummies(df, drop_first=False)
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(df_aligned)[0]
    confidence = model.predict_proba(df_aligned)[0].max()

    return {
        "prediction": int(prediction),
        "label": "churn" if prediction == 1 else "no_churn",
        "confidence": float(confidence),
    }


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/metrics")
def get_metrics():
    return metrics


@app.post("/predict")
def predict(payload: CustomerInput):
    if model is None:
        metrics["n_errors"] += 1
        raise HTTPException(status_code=500, detail="Modèle indisponible")

    try:
        result = _run_inference(payload)
        metrics["n_predictions"] += 1
        logger.info("prediction=%s", result["prediction"])
        return result
    except Exception as e:
        metrics["n_errors"] += 1
        logger.error("Erreur pendant la prédiction : %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne pendant la prédiction")


@app.post("/predict_batch")
def predict_batch(payload: BatchInput):
    n_inputs = len(payload.inputs)

    if n_inputs > MAX_BATCH_SIZE:
        logger.warning(
            "Batch rejeté : %s entrées reçues (max autorisé %s)", n_inputs, MAX_BATCH_SIZE
        )
        raise HTTPException(
            status_code=413,
            detail=f"Batch trop volumineux : {n_inputs} entrées (maximum {MAX_BATCH_SIZE})",
        )

    if model is None:
        metrics["n_errors"] += 1
        raise HTTPException(status_code=500, detail="Modèle indisponible")

    predictions = []
    for index, item in enumerate(payload.inputs):
        try:
            predictions.append(_run_inference(item))
        except Exception as e:
            metrics["n_errors"] += 1
            logger.error(
                "Erreur pendant la prédiction batch (entrée %s) : %s", index, e, exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail=f"Erreur interne sur l'entrée {index} du batch",
            )

    metrics["n_batch_requests"] += 1
    metrics["n_batch_inputs_total"] += n_inputs
    metrics["n_predictions"] += n_inputs
    logger.info("Batch traité : %s entrées", n_inputs)

    return {"predictions": predictions, "n_inputs": n_inputs}
