
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_train_data() -> pd.DataFrame:
    file_path = DATA_DIR / "train_FD001.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None
    )

    df = df.iloc[:, :26]
    df.columns = COLUMNS

    return df


if __name__ == "__main__":
    data = load_train_data()

    print("\nDataset shape:", data.shape)
    print("\nFirst five rows:")
    print(data.head())

    print("\nNumber of engines:", data["engine_id"].nunique())
    print("\nMissing values:", data.isnull().sum().sum())
