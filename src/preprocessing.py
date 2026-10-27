"""Data Preprocessing and Feature Engineering Pipeline for RUL Prediction.

Module: src.preprocessing
Role: Member 1 (Data Preparation Pipeline)

Core Responsibilities:
    1. Ground-Truth RUL Computation:
       - Computes Remaining Useful Life (RUL = max_cycle - current_cycle) for run-to-failure engines.
       - Implements piece-wise linear degradation target capping (threshold: 125 cycles).
    2. Low-Variance Feature Pruning:
       - Removes flat/constant telemetry sensors that provide no discriminative degradation signal.
    3. Temporal Feature Engineering:
       - Extracts rolling statistical moments (moving average and moving standard deviation)
         across multi-scale operational windows (5 and 10 cycles) per engine.
"""

from pathlib import Path
import pandas as pd

from src.data_loader import load_train_data

# Project directory paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

# Piece-wise linear RUL ceiling (literature standard for C-MAPSS benchmark)
# In early operational cycles, health state is nominal and deterioration is negligible.
# Capping prevents the regressor from penalizing healthy-state baseline fluctuations.
RUL_CAP = 125

# Uninformative telemetry channels with near-zero / zero standard deviation in FD001
# (Fan inlet temperature, bypass duct pressure, etc. stay invariant under single sea-level operating mode)
CONSTANT_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]


def prepare_training_data() -> pd.DataFrame:
    """Prepare raw telemetry data with target RUL calculation and feature pruning.

    Steps:
        1. Ingest raw observations via `load_train_data`.
        2. Derive ground truth RUL per unit: RUL(t) = max_cycle(unit) - t.
        3. Create a capped RUL column (RUL_capped = min(RUL, 125)) for piece-wise modeling.
        4. Drop constant sensor columns to prevent noise and multicollinearity.

    Returns:
        pd.DataFrame: Cleaned dataframe with computed target columns and active sensors.
    """
    df = load_train_data()

    # Step 1: Calculate the final failure cycle observed for each engine unit
    final_cycles = df.groupby("engine_id")["cycle"].max()

    # Step 2: Compute true remaining operating cycles until failure
    df["RUL"] = (
        df["engine_id"].map(final_cycles) - df["cycle"]
    )

    # Step 3: Piece-wise linear transformation (clips upper bound to 125 cycles)
    df["RUL_capped"] = df["RUL"].clip(upper=RUL_CAP)

    # Step 4: Drop sensors with zero variance in single-condition FD001
    df = df.drop(columns=CONSTANT_SENSORS)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer multi-scale rolling temporal features grouped by engine unit.

    Computes moving window statistics (mean and standard deviation) over 5-cycle
    and 10-cycle horizons for the most degradation-sensitive sensors. Grouping by
    `engine_id` guarantees that temporal history does not leak across different units.

    Args:
        df (pd.DataFrame): Preprocessed dataframe containing engine_id, cycle, and sensors.

    Returns:
        pd.DataFrame: Augmented dataframe including lag-windowed temporal features.
    """
    # Enforce strict chronological order within each engine trajectory
    df = df.sort_values(
        ["engine_id", "cycle"]
    ).copy()

    # Most informative sensors exhibiting monotonic degradation trends
    selected_sensors = [
        "sensor_2",   # LPC outlet temperature
        "sensor_3",   # HPC outlet temperature
        "sensor_4",   # LPT outlet temperature
        "sensor_7",   # HPC outlet pressure
        "sensor_9",   # Core speed
        "sensor_11",  # Static HPC pressure
        "sensor_12",  # Ratio of fuel flow to Ps30
        "sensor_14",  # Corrected core speed
    ]

    windows = [5, 10]

    for sensor in selected_sensors:
        grouped = df.groupby("engine_id")[sensor]

        for window in windows:
            rolling = grouped.rolling(
                window=window,
                min_periods=1,
            )

            # Rolling mean captures smoothed trend of degradation progression
            df[f"{sensor}_mean_{window}"] = (
                rolling.mean()
                .reset_index(level=0, drop=True)
            )

            # Rolling std captures increasing measurement volatility / vibration before failure
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

    print("\n--- Executing Data Preparation & Feature Engineering Pipeline ---")
    df = prepare_training_data()

    df_with_features = add_temporal_features(df)

    print("Base preprocessed dataset shape:", df.shape)
    print("Dataset shape after temporal feature augmentation:", df_with_features.shape)

    print("\nSample engineered temporal features:")
    print(
        df_with_features[
            ["engine_id", "cycle", "sensor_11",
            "sensor_11_mean_5", "sensor_11_std_5"]
        ].head(10)
    )

    # Persist prepared dataset for downstream modeling (Members 2 & 3)
    output_path = DATA_DIR / "train_FD001_prepared.csv"
    df.to_csv(output_path, index=False)

    print(f"\nPrepared dataset successfully saved to: {output_path}")
    print("Total engine units processed:", df["engine_id"].nunique())

    print("\n--- Summary Statistics for Ground Truth Targets ---")
    print("\nUncapped RUL Summary:")
    print(df["RUL"].describe())

    print("\nCapped RUL (Piece-wise, 125-cycle threshold) Summary:")
    print(df["RUL_capped"].describe())
