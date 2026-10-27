
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from src.preprocessing import (
    prepare_training_data,
    add_temporal_features,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"


def get_features(df):
    excluded = {"engine_id", "cycle", "RUL", "RUL_capped"}
    return [col for col in df.columns if col not in excluded]


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data and construct temporal features.
    df = prepare_training_data()
    df = add_temporal_features(df)

    groups = df["engine_id"]
    feature_cols = get_features(df)
    X = df[feature_cols]

    # Use four folds, keeping each engine entirely in one fold.
    cv = GroupKFold(n_splits=4)

    experiments = [
        {
            "name": "Random Forest - capped RUL",
            "model": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "target": "RUL_capped",
        },
        {
            "name": "XGBoost - capped RUL",
            "model": XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
                tree_method="hist",
            ),
            "target": "RUL_capped",
        },
        {
            "name": "XGBoost - uncapped RUL",
            "model": XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
                tree_method="hist",
            ),
            "target": "RUL",
        },
        {
            "name": "XGBoost tuned - uncapped RUL",
            "model": XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=7,
                reg_alpha=0,
                reg_lambda=1,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
                tree_method="hist",
            ),
            "target": "RUL",
        },
    ]

    rows = []

    for experiment in experiments:
        name = experiment["name"]
        model = experiment["model"]
        target = experiment["target"]
        y = df[target]

        print(f"\nEvaluating: {name}")

        scores = cross_validate(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
            return_train_score=False,
        )

        # Compute fold RMSE separately to report it alongside MAE and R².
        fold_rmses = []

        for train_idx, val_idx in cv.split(X, y, groups):
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = fold_model.predict(X.iloc[val_idx])
            fold_rmses.append(
                root_mean_squared_error(y.iloc[val_idx], pred)
            )

        mean_mae = -scores["test_mae"].mean()
        mean_rmse = np.mean(fold_rmses)
        mean_r2 = scores["test_r2"].mean()

        rows.append({
            "model": name,
            "target": target,
            "mean_CV_MAE": mean_mae,
            "std_fold_MAE": scores["test_mae"].std(),
            "mean_CV_RMSE": mean_rmse,
            "mean_CV_R2": mean_r2,
        })

        print(f"Mean CV MAE:  {mean_mae:.4f}")
        print(f"Mean CV RMSE: {mean_rmse:.4f}")
        print(f"Mean CV R²:   {mean_r2:.4f}")

    results = pd.DataFrame(rows).sort_values("mean_CV_MAE")
    output_path = REPORTS_DIR / "grouped_cv_model_comparison.csv"
    results.to_csv(output_path, index=False)

    print("\n========== GROUPED CROSS-VALIDATION COMPARISON ==========")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
