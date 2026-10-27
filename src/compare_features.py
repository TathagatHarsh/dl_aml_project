
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

from src.preprocessing import (
    prepare_training_data,
    add_temporal_features,
)

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"

EXCLUDED = {"engine_id", "cycle", "RUL", "RUL_capped"}


def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    return {
        "MAE": mean_absolute_error(y_val, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_val, predictions)),
        "R2": r2_score(y_val, predictions),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    original = prepare_training_data()
    engineered = add_temporal_features(original)

    # Use the same engine-level split for both experiments.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(
            original,
            original["RUL_capped"],
            groups=original["engine_id"],
        )
    )

    y_train = original["RUL_capped"].iloc[train_idx]
    y_val = original["RUL_capped"].iloc[val_idx]

    feature_sets = {
        "Original": original.drop(columns=list(EXCLUDED)),
        "Temporal features": engineered.drop(columns=list(EXCLUDED)),
    }

    results = []

    for feature_name, X in feature_sets.items():
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "XGBoost": XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            ),
        }

        for model_name, model in models.items():
            print(f"Training {model_name} — {feature_name}")

            metrics = evaluate_model(
                model, X_train, y_train, X_val, y_val
            )

            results.append({
                "Feature set": feature_name,
                "Model": model_name,
                **metrics,
            })

    results_df = pd.DataFrame(results).sort_values("RMSE")

    output_path = REPORTS_DIR / "feature_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\n========== FEATURE COMPARISON ==========")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
