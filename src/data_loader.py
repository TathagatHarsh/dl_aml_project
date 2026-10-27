"""NASA C-MAPSS Turbofan Engine Degradation Dataset Loader.

Module: src.data_loader
Role: Member 1 (Data Preparation Pipeline)

Dataset Description:
    The Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset
    provided by NASA simulates turbofan engine degradation over operational cycles.
    Engines begin with varying degrees of initial wear and run until system failure.

Column Definitions:
    - engine_id: Unique identifier for each simulated engine unit.
    - cycle: Time elapsed in operational flight cycles (time-series index).
    - setting_1 to setting_3: Operational flight condition parameters (Altitude,
      Mach number, Throttle Resolver Angle).
    - sensor_1 to sensor_21: Continuous telemetry measurements across engine
      subsystems (fan inlet/outlet, compressor stages, combustor, turbines, etc.).
"""

from pathlib import Path
import pandas as pd

# Define base project directory paths for reliable asset resolution
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Standard schema definition for NASA C-MAPSS 26-column format
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_train_data() -> pd.DataFrame:
    """Load and format raw training observations for FD001.

    Reads whitespace-delimited raw telemetry logs, validates existence,
    extracts the 26 canonical features (dropping trailing delimiter artifacts),
    and assigns standard descriptive column labels.

    Returns:
        pd.DataFrame: Structured run-to-failure telemetry dataset with columns
            ['engine_id', 'cycle', 'setting_1'..'setting_3', 'sensor_1'..'sensor_21'].

    Raises:
        FileNotFoundError: If the raw dataset file `train_FD001.txt` is missing.
    """
    file_path = DATA_DIR / "train_FD001.txt"

    # Verify input file existence prior to parsing
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at expected location: {file_path}"
        )

    # Parse whitespace-separated telemetry stream
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None
    )

    # Slice first 26 columns to remove potential trailing whitespace columns created by pandas
    df = df.iloc[:, :26]
    df.columns = COLUMNS

    return df


if __name__ == "__main__":
    # Ingestion pipeline validation and sanity checks
    print("\n--- Ingesting Training Telemetry (FD001) ---")
    data = load_train_data()

    print("\nDataset shape:", data.shape)
    print("\nFirst five rows:")
    print(data.head())

    # Data integrity metrics
    print("\nNumber of engines:", data["engine_id"].nunique())
    print("Missing values across all features:", data.isnull().sum().sum())
