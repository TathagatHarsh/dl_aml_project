
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

from src.preprocessing import prepare_training_data

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"

TARGET = "RUL_capped"

# Keep identifiers and targets out of the feature set.
EXCLUDED_COLUMNS = {
    "engine_id",
    "cycle",
    "RUL",
    "RUL_capped",
}


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_training_data()

    feature_cols = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    X = df[feature_cols]
    y = df[TARGET]
    groups = df["engine_id"]

    # Split by engine to avoid engine-level leakage.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    print("Features:", len(feature_cols))
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print(
        "Training engines:",
        groups.iloc[train_idx].nunique(),
    )
    print(
        "Validation engines:",
        groups.iloc[val_idx].nunique(),
    )

    models = {
        "Dummy Baseline": DummyRegressor(
            strategy="mean"
        ),
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

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        mae = mean_absolute_error(
            y_val, predictions
        )
        rmse = np.sqrt(
            mean_squared_error(y_val, predictions)
        )
        r2 = r2_score(y_val, predictions)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        print(f"MAE:  {mae:.3f} cycles")
        print(f"RMSE: {rmse:.3f} cycles")
        print(f"R2:   {r2:.4f}")
    

    # Save XGBoost feature importance
    xgb_model = models["XGBoost"]

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": xgb_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importance_path = REPORTS_DIR / "xgboost_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    print("\n========== TOP 10 FEATURES ==========")
    print(importance_df.head(10).to_string(index=False))

    # Plot the top 10 features
    import matplotlib.pyplot as plt

    top_features = importance_df.head(10).sort_values("Importance")

    plt.figure(figsize=(9, 6))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.xlabel("XGBoost feature importance")
    plt.title("Top 10 Features for RUL Prediction")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "xgboost_feature_importance.png",
        dpi=150,
    )
    plt.close()

    print(f"\nFeature importance saved to: {importance_path}")


    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("RMSE")

    output_path = REPORTS_DIR / "ml_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
