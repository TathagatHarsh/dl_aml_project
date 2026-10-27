
from pathlib import Path

import pandas as pd

from src.data_loader import load_train_data

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

RUL_CAP = 125

# These sensors have effectively zero variance in FD001.
CONSTANT_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]


def prepare_training_data() -> pd.DataFrame:
    df = load_train_data()

    # Calculate the final observed cycle for each engine.
    final_cycles = df.groupby("engine_id")["cycle"].max()

    # RUL = final cycle - current cycle.
    df["RUL"] = (
        df["engine_id"].map(final_cycles) - df["cycle"]
    )

    # Create a capped target for the first experiment.
    df["RUL_capped"] = df["RUL"].clip(upper=RUL_CAP)

    # Remove constant sensors from the feature set.
    df = df.drop(columns=CONSTANT_SENSORS)

    return df



def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(
        ["engine_id", "cycle"]
    ).copy()

    selected_sensors = [
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_9",
        "sensor_11",
        "sensor_12",
        "sensor_14",
    ]

    windows = [5, 10]

    for sensor in selected_sensors:
        grouped = df.groupby("engine_id")[sensor]

        for window in windows:
            rolling = grouped.rolling(
                window=window,
                min_periods=1,
            )

            df[f"{sensor}_mean_{window}"] = (
                rolling.mean()
                .reset_index(level=0, drop=True)
            )

            df[f"{sensor}_std_{window}"] = (
                grouped.rolling(
                    window=window,
                    min_periods=2,
                )
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0)
            )

    return df




if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_training_data()

    df_with_features = add_temporal_features(df)

    print("Original shape:", df.shape)
    print("Shape with temporal features:", df_with_features.shape)

    print(
        df_with_features[
            ["engine_id", "cycle", "sensor_11",
            "sensor_11_mean_5", "sensor_11_std_5"]
        ].head(10)
    )

    output_path = DATA_DIR / "train_FD001_prepared.csv"
    df.to_csv(output_path, index=False)

    print("Prepared dataset shape:", df.shape)
    print("Number of engines:", df["engine_id"].nunique())

    print("\nRUL statistics:")
    print(df["RUL"].describe())

    print("\nCapped RUL statistics:")
    print(df["RUL_capped"].describe())

    print("\nSample rows:")
    print(df[["engine_id", "cycle", "RUL", "RUL_capped"]].head(10))

    print(f"\nSaved prepared data to: {output_path}")
