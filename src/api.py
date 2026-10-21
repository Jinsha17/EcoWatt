
import json
import joblib

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "artifacts" / "model.joblib"
METADATA_PATH = BASE / "artifacts" / "metadata.json"

app = FastAPI(title="EcoWatt API", version="1.0.0")



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://localhost:8080"
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)


def load_artifacts():
    """Load and check the saved model and its metadata."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Model or metadata file is missing")

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    required = ["consumption_kwh", "hour", "day_of_week"]
    if metadata.get("features") != required:
        raise ValueError("Model feature metadata is incompatible")

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict"):
        raise ValueError("Saved artifact is not a prediction model")

    return model, metadata


class PredictionInput(BaseModel):
    consumption_kwh: float = Field(ge=0, le=100)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    tariff_per_kwh: float = Field(default=8.0, gt=0, le=100)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        load_artifacts()
        return {"ready": True, "message": "Model artifacts are available"}
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=503,
            detail="Model is not ready or artifacts are incompatible"
        )


@app.get("/model-info")
def model_info():
    try:
        _, metadata = load_artifacts()
        return metadata
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        raise HTTPException(status_code=503, detail="Model information unavailable")


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        model, _ = load_artifacts()
        features = [[
            data.consumption_kwh,
            data.hour,
            data.day_of_week
        ]]
        prediction = float(model.predict(features)[0])

        # Avoid displaying a physically impossible negative usage.
        prediction = max(0.0, prediction)

        return {
            "predicted_consumption_kwh": round(prediction, 3),
            "estimated_cost_inr": round(
                prediction * data.tariff_per_kwh, 2
            ),
            "tariff_per_kwh": data.tariff_per_kwh,
            "unit": "kWh"
        }
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        raise HTTPException(status_code=503, detail="Model unavailable")
