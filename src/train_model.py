
import json
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "hourly_consumption.csv"
ARTIFACTS = BASE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA, index_col="datetime", parse_dates=True)

# Features available when making a prediction
features = ["consumption_kwh", "hour", "day_of_week"]
target = "target_kwh"

X = df[features]
y = df[target]

# Chronological split: 70% train, 15% validation, 15% test
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

models = {
    "Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

# Train and compare using validation data
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results).sort_values("MAE")
print("\nValidation results (lower MAE/RMSE is better):")
print(results_df.to_string(index=False))

# Select the best model using validation MAE
best_name = results_df.iloc[0]["model"]
best_model = models[best_name]

# Refit selected model on training + validation data
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
best_model.fit(X_train_val, y_train_val)

# Evaluate the selected model once on held-out test data
test_pred = best_model.predict(X_test)

test_results = {
    "MAE": float(mean_absolute_error(y_test, test_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, test_pred))),
    "R2": float(r2_score(y_test, test_pred))
}

# Save selected model and metadata
joblib.dump(best_model, ARTIFACTS / "model.joblib")

metadata = {
    "project": "EcoWatt",
    "model_name": best_name,
    "features": features,
    "target": target,
    "target_unit": "kWh",
    "validation_metric": "MAE",
    "validation_MAE": float(results_df.iloc[0]["MAE"]),
    "test_metrics": test_results,
    "train_rows": len(X_train),
    "validation_rows": len(X_val),
    "test_rows": len(X_test)
}

with open(ARTIFACTS / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

results_df.to_csv(ARTIFACTS / "validation_results.csv", index=False)

print("\nSelected model:", best_name)
print("Held-out test results:", test_results)
print("Saved model to:", ARTIFACTS / "model.joblib")
print("Saved metadata to:", ARTIFACTS / "metadata.json")
