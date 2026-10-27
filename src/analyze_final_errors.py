
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"


def main():
    path = REPORTS_DIR / "fd001_final_ml_predictions.csv"
    df = pd.read_csv(path)

    df["error"] = df["predicted_RUL_capped"] - df["actual_RUL"]
    df["absolute_error"] = df["error"].abs()

    print("\n========== FINAL MODEL ERROR ANALYSIS ==========")
    print(f"Mean absolute error: {df['absolute_error'].mean():.3f}")
    print(f"Mean signed error:   {df['error'].mean():.3f}")
    print(f"Median absolute error: {df['absolute_error'].median():.3f}")

    print("\nWorst 10 predictions:")
    print(
        df.nlargest(10, "absolute_error")[
            [
                "engine_id",
                "actual_RUL",
                "predicted_RUL_capped",
                "error",
                "absolute_error",
            ]
        ].to_string(index=False)
    )

    bins = [0, 30, 60, 90, 125, float("inf")]
    labels = ["0–30", "31–60", "61–90", "91–125", ">125"]

    df["actual_RUL_range"] = pd.cut(
        df["actual_RUL"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    grouped = df.groupby(
        "actual_RUL_range", observed=False
    ).agg(
        engine_count=("engine_id", "count"),
        MAE=("absolute_error", "mean"),
        mean_error=("error", "mean"),
    ).reset_index()

    print("\nErrors by actual RUL range:")
    print(grouped.to_string(index=False))

    grouped.to_csv(
        REPORTS_DIR / "fd001_final_error_by_rul.csv",
        index=False,
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(df["actual_RUL"], df["predicted_RUL_capped"])
    plt.plot([0, 125], [0, 125], linestyle="--")
    plt.xlabel("Actual RUL (cycles)")
    plt.ylabel("Predicted capped RUL (cycles)")
    plt.title("EngineMind AI: Actual vs Predicted RUL")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "fd001_final_actual_vs_predicted.png",
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["error"], bins=20)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Prediction error: predicted - actual")
    plt.ylabel("Number of engines")
    plt.title("EngineMind AI: Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "fd001_final_error_distribution.png",
        dpi=150,
    )
    plt.close()

    print("\nSaved error analysis and plots in reports/.")


if __name__ == "__main__":
    main()
