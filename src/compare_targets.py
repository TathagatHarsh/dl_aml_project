
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.preprocessing import (
    prepare_training_data,
    add_temporal_features,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"

EXCLUDED = {"engine_id", "cycle", "RUL", "RUL_capped"}


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_training_data()
    df = add_temporal_features(df)

    feature_cols = [
        col for col in df.columns
        if col not in EXCLUDED
    ]
    X = df[feature_cols]
    groups = df["engine_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(X, df["RUL_capped"], groups)
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    results = []

    for target in ["RUL_capped", "RUL"]:
        y_train = df[target].iloc[train_idx]
        y_val = df[target].iloc[val_idx]

        model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        print(f"\nTraining with target: {target}")
        model.fit(X_train, y_train)

        predictions = np.maximum(
            model.predict(X_val), 0
        )

        if target == "RUL_capped":
            predictions = np.minimum(predictions, 125)

        # Compare both models against actual and capped RUL.
        for evaluation_target in ["RUL", "RUL_capped"]:
            actual = df[evaluation_target].iloc[val_idx]

            results.append({
                "Training target": target,
                "Evaluation target": evaluation_target,
                "MAE": mean_absolute_error(actual, predictions),
                "RMSE": np.sqrt(
                    mean_squared_error(actual, predictions)
                ),
                "R2": r2_score(actual, predictions),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        REPORTS_DIR / "target_comparison.csv",
        index=False,
    )

    print("\n========== TARGET COMPARISON ==========")
    print(results_df.to_string(index=False))
    print("\nSaved to reports/target_comparison.csv")


if __name__ == "__main__":
    main()

