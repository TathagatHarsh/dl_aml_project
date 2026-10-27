
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor

from src.preprocessing import (
    prepare_training_data,
    add_temporal_features,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"

EXCLUDED = {
    "engine_id",
    "cycle",
    "RUL",
    "RUL_capped",
}


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and construct temporal features.
    df = prepare_training_data()
    df = add_temporal_features(df)

    feature_cols = [
        col for col in df.columns
        if col not in EXCLUDED
    ]

    X = df[feature_cols]
    y = df["RUL"]  # Uncapped target
    groups = df["engine_id"]

    # Hold out engines for a final validation check.
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
    train_groups = groups.iloc[train_idx]

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Training engines:", train_groups.nunique())
    print("Validation engines:", groups.iloc[val_idx].nunique())
    print("Number of features:", len(feature_cols))

    # Grouped CV keeps each engine entirely within one fold.
    cv = GroupKFold(n_splits=4)

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )

    param_distributions = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [1, 3, 5, 10],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    print("\nStarting grouped cross-validation tuning...")
    search.fit(X_train, y_train, groups=train_groups)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    print("\n========== TUNING RESULTS ==========")
    print("Best CV MAE:", round(-search.best_score_, 4))
    print("Best parameters:")
    print(search.best_params_)

    print("\nHeld-out validation results:")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

    # Save all tuning trials.
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(
        REPORTS_DIR / "xgboost_tuning_results.csv",
        index=False,
    )

    # Save validation metrics.
    metrics = pd.DataFrame([{
        "Target": "RUL",
        "Best_CV_MAE": -search.best_score_,
        "Validation_MAE": mae,
        "Validation_RMSE": rmse,
        "Validation_R2": r2,
        "Best_Parameters": str(search.best_params_),
    }])

    metrics.to_csv(
        REPORTS_DIR / "xgboost_tuned_metrics.csv",
        index=False,
    )

    # Save the selected model and feature order.
    joblib.dump(
        {
            "model": best_model,
            "feature_cols": feature_cols,
            "target": "RUL",
            "temporal_features": True,
        },
        MODELS_DIR / "xgboost_tuned.joblib",
    )

    print("\nSaved tuning results to reports/")
    print("Saved model to models/xgboost_tuned.joblib")


if __name__ == "__main__":
    main()
