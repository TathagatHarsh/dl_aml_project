
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import COLUMNS
from src.preprocessing import add_temporal_features

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
MODEL_PATH = ROOT_DIR / "models" / "xgboost_final_capped.joblib"

CONSTANT_SENSORS = [
    "sensor_1", "sensor_5", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]


def nasa_score(predictions, actual):
    error = predictions - actual
    score = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1,
        np.exp(error / 10.0) - 1,
    )
    return score.sum()


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    rul_cap = artifact["rul_cap"]

    test = pd.read_csv(
        DATA_DIR / "test_FD001.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, :26]
    test.columns = COLUMNS

    actual_rul = pd.read_csv(
        DATA_DIR / "RUL_FD001.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0].to_numpy()

    test = test.drop(columns=CONSTANT_SENSORS)
    test = add_temporal_features(test)

    final_rows = (
        test.sort_values(["engine_id", "cycle"])
        .groupby("engine_id", sort=True)
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    if len(final_rows) != len(actual_rul):
        raise ValueError("Number of test engines and RUL labels differ.")

    X_test = final_rows[feature_cols]
    predictions = np.clip(model.predict(X_test), 0, rul_cap)

    capped_actual = np.minimum(actual_rul, rul_cap)

    metrics = {
        "capped_target_MAE": mean_absolute_error(
            capped_actual, predictions
        ),
        "capped_target_RMSE": np.sqrt(
            mean_squared_error(capped_actual, predictions)
        ),
        "capped_target_R2": r2_score(capped_actual, predictions),
        "actual_RUL_MAE": mean_absolute_error(actual_rul, predictions),
        "actual_RUL_RMSE": np.sqrt(
            mean_squared_error(actual_rul, predictions)
        ),
        "actual_RUL_R2": r2_score(actual_rul, predictions),
        "NASA_score_actual_RUL": nasa_score(predictions, actual_rul),
    }

    print("\n========== FINAL PHASE 1 MODEL ==========")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    predictions_df = pd.DataFrame({
        "engine_id": final_rows["engine_id"],
        "last_observed_cycle": final_rows["cycle"],
        "actual_RUL": actual_rul,
        "actual_RUL_capped": capped_actual,
        "predicted_RUL_capped": predictions,
    })

    predictions_df.to_csv(
        REPORTS_DIR / "fd001_final_ml_predictions.csv",
        index=False,
    )
    pd.DataFrame([metrics]).to_csv(
        REPORTS_DIR / "fd001_final_ml_metrics.csv",
        index=False,
    )

    print("\nSample predictions:")
    print(predictions_df.head(10).to_string(index=False))
    print("\nResults saved in reports/.")


if __name__ == "__main__":
    main()
