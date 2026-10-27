
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"

def main():
    file_path = REPORTS_DIR / "fd001_test_predictions.csv"
    df = pd.read_csv(file_path)

    # Positive error means overprediction.
    df["error"] = df["predicted_RUL"] - df["actual_RUL"]
    df["absolute_error"] = df["error"].abs()

    print("\n========== ERROR ANALYSIS ==========")
    print("Mean error:", round(df["error"].mean(), 3))
    print("Median absolute error:", round(df["absolute_error"].median(), 3))
    print("Maximum absolute error:", round(df["absolute_error"].max(), 3))

    print("\nWorst 10 predictions:")
    print(
        df.nlargest(10, "absolute_error")[
            [
                "engine_id",
                "actual_RUL",
                "predicted_RUL",
                "error",
                "absolute_error",
            ]
        ].to_string(index=False)
    )


    # Group errors by actual RUL range
    df["RUL_range"] = pd.cut(
        df["actual_RUL"],
        bins=[-1, 30, 60, 90, 125, float("inf")],
        labels=[
            "0–30",
            "31–60",
            "61–90",
            "91–125",
            "Above 125",
        ],
    )

    range_analysis = (
        df.groupby("RUL_range", observed=False)
        .agg(
            engine_count=("engine_id", "count"),
            MAE=("absolute_error", "mean"),
            Mean_Error=("error", "mean"),
        )
        .reset_index()
    )

    print("\n========== ERROR BY ACTUAL RUL RANGE ==========")
    print(range_analysis.to_string(index=False))

    range_analysis.to_csv(
        REPORTS_DIR / "error_by_rul_range.csv",
        index=False,
    )


    # Actual versus predicted RUL
    plt.figure(figsize=(7, 6))
    plt.scatter(df["actual_RUL"], df["predicted_RUL"])

    lower = min(df["actual_RUL"].min(), df["predicted_RUL"].min())
    upper = max(df["actual_RUL"].max(), df["predicted_RUL"].max())

    plt.plot([lower, upper], [lower, upper], "--")
    plt.xlabel("Actual RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title("Actual vs Predicted RUL — FD001 Test Set")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Prediction error distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["error"], bins=15, edgecolor="black")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Prediction error (predicted - actual)")
    plt.ylabel("Number of engines")
    plt.title("RUL Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "prediction_errors.png", dpi=150)
    plt.close()

    df.to_csv(REPORTS_DIR / "fd001_error_analysis.csv", index=False)

    print("\nSaved error analysis and plots in reports/.")

if __name__ == "__main__":
    main()
