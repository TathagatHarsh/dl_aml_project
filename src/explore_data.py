
from src.data_loader import load_train_data
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = load_train_data()

# Create reports directory if it doesn't exist
from pathlib import Path
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# 1. Dataset overview
print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("Number of engines:", df["engine_id"].nunique())
print("Missing values:", df.isnull().sum().sum())

# 2. Engine lifetime
engine_lifetimes = df.groupby("engine_id")["cycle"].max()

print("\n========== ENGINE LIFETIME ==========")
print(engine_lifetimes.describe())

plt.figure(figsize=(10, 5))
plt.hist(engine_lifetimes, bins=15, edgecolor="black")
plt.xlabel("Total operating cycles")
plt.ylabel("Number of engines")
plt.title("Distribution of Engine Lifetimes")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "engine_lifetimes.png", dpi=150)
plt.close()

# 3. Sensor variability
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
sensor_std = df[sensor_cols].std().sort_values()

print("\n========== SENSOR VARIABILITY ==========")
print(sensor_std.to_string())

plt.figure(figsize=(10, 6))
sensor_std.plot(kind="bar")
plt.ylabel("Standard deviation")
plt.title("Variability of Sensor Measurements")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "sensor_variability.png", dpi=150)
plt.close()

# 4. Sensor trends for engine 1
engine = df[df["engine_id"] == 1]

plt.figure(figsize=(12, 6))

for sensor in ["sensor_2", "sensor_3", "sensor_4"]:
    plt.plot(engine["cycle"], engine[sensor], label=sensor)

plt.xlabel("Operating cycle")
plt.ylabel("Sensor reading (dataset units)")
plt.title("Sensor Trends for Engine 1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "engine_1_sensor_trends.png", dpi=150)
plt.close()

# 5. Identify constant or nearly constant sensors
print("\n========== LOW-VARIABILITY SENSORS ==========")
print(sensor_std[sensor_std < 0.01])

print(f"\nPlots saved to: {REPORTS_DIR}")
