
from pathlib import Path

import joblib

from xgboost import XGBRegressor

from src.preprocessing import (
    prepare_training_data,
    add_temporal_features,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_training_data()
    df = add_temporal_features(df)

    excluded = {"engine_id", "cycle", "RUL", "RUL_capped"}
    feature_cols = [
        col for col in df.columns if col not in excluded
    ]

    X = df[feature_cols]
    y = df["RUL_capped"]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )

    print("Training final Phase 1 XGBoost model...")
    model.fit(X, y)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target": "RUL_capped",
        "rul_cap": 125,
        "temporal_features": True,
        "dataset": "NASA C-MAPSS FD001",
    }

    output_path = MODELS_DIR / "xgboost_final_capped.joblib"
    joblib.dump(artifact, output_path)

    print(f"Training rows: {len(X)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target: RUL_capped (maximum 125 cycles)")
    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()
