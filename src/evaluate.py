
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data_loader import COLUMNS
from src.preprocessing import add_temporal_features

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
MODEL_PATH = ROOT_DIR / "models" / "xgboost_tuned.joblib"

CONSTANT_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]


def load_test_data():
    test_path = DATA_DIR / "test_FD001.txt"
    rul_path = DATA_DIR / "RUL_FD001.txt"

    if not test_path.exists() or not rul_path.exists():
        raise FileNotFoundError(
            "Missing test_FD001.txt or RUL_FD001.txt in data/"
        )

    test = pd.read_csv(
        test_path, sep=r"\s+", header=None
    ).iloc[:, :26]
    test.columns = COLUMNS

    true_rul = pd.read_csv(
        rul_path, sep=r"\s+", header=None
    ).iloc[:, 0].to_numpy()

    if len(true_rul) != test["engine_id"].nunique():
        raise ValueError("Test engine count and RUL label count differ.")

    return test, true_rul


def nasa_score(predictions, actual):
    # Standard asymmetric NASA C-MAPSS score.
    error = predictions - actual
    scores = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1,
        np.exp(error / 10.0) - 1,
    )
    return scores.sum()


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Tuned model not found. Run python -m src.tune_xgboost first."
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_cols = saved["feature_cols"]

    test, true_rul = load_test_data()

    test = test.drop(columns=CONSTANT_SENSORS)
    test = add_temporal_features(test)

    # Select the final observed cycle for each test engine.
    test_last = (
        test.sort_values(["engine_id", "cycle"])
        .groupby("engine_id", sort=True)
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    X_test = test_last[feature_cols]

    # The model was trained on uncapped RUL.
    predictions = np.maximum(model.predict(X_test), 0)

    results = {
        "MAE_actual_RUL": mean_absolute_error(
            true_rul, predictions
        ),
        "RMSE_actual_RUL": np.sqrt(
            mean_squared_error(true_rul, predictions)
        ),
        "R2_actual_RUL": r2_score(true_rul, predictions),
        "NASA_score_actual_RUL": nasa_score(
            predictions, true_rul
        ),
    }

    print("\n========== TUNED MODEL: OFFICIAL TEST RESULTS ==========")
    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    predictions_df = pd.DataFrame({
        "engine_id": test_last["engine_id"],
        "last_observed_cycle": test_last["cycle"],
        "actual_RUL": true_rul,
        "predicted_RUL": predictions,
    })

    predictions_df.to_csv(
        REPORTS_DIR / "fd001_tuned_test_predictions.csv",
        index=False,
    )
    pd.DataFrame([results]).to_csv(
        REPORTS_DIR / "fd001_tuned_test_metrics.csv",
        index=False,
    )

    print("\nSample predictions:")
    print(predictions_df.head(10).to_string(index=False))
    print("\nSaved tuned test results in reports/.")


if __name__ == "__main__":
    main()
