
import json
import joblib
import pandas as pd

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "artifacts" / "model.joblib"
METADATA_PATH = BASE / "artifacts" / "metadata.json"


def test_saved_model_can_predict():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "consumption_kwh": 2.5,
        "hour": 18,
        "day_of_week": 2
    }])

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] >= 0


def test_metadata_contains_test_metrics():
    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    assert metadata["model_name"] == "Random Forest"
    assert "MAE" in metadata["test_metrics"]
    assert "RMSE" in metadata["test_metrics"]
    assert "R2" in metadata["test_metrics"]
