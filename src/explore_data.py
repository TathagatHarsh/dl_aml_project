"""Exploratory Data Analysis (EDA) and Telemetry Profiling.

Module: src.explore_data
Role: Member 1 (Data Preparation Pipeline)

Objectives:
    1. Validate data integrity (missingness, shape, cohort size).
    2. Analyze distribution of total engine operational lifespans.
    3. Profile sensor variability to empirically isolate invariant channels.
    4. Visualize run-to-failure sensor degradation trajectories for representative units.
    5. Generate diagnostic figures for the evaluation report.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_train_data

# Ensure report output directory exists for diagnostic plots
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Step 1: Ingestion & Integrity Verification
# -------------------------------------------------------------------------
df = load_train_data()

print("\n========== 1. DATASET OVERVIEW & INTEGRITY ==========")
print("Total observation records:", df.shape[0])
print("Total telemetry features:", df.shape[1])
print("Unique engine units:", df["engine_id"].nunique())
print("Null / missing values detected:", df.isnull().sum().sum())

# -------------------------------------------------------------------------
# Step 2: Engine Lifetime Distribution Analysis
# -------------------------------------------------------------------------
# Calculate the maximum operational cycle reached by each engine before failure
engine_lifetimes = df.groupby("engine_id")["cycle"].max()

print("\n========== 2. ENGINE LIFETIME DISTRIBUTION ==========")
print(engine_lifetimes.describe())

# Plot histogram of engine lifespans to understand lifetime variance
plt.figure(figsize=(10, 5))
plt.hist(engine_lifetimes, bins=15, edgecolor="black", color="#3b82f6", alpha=0.85)
plt.xlabel("Total Operating Cycles Until Failure")
plt.ylabel("Number of Engines")
plt.title("Distribution of Engine Lifetimes (FD001)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "engine_lifetimes.png", dpi=150)
plt.close()

# -------------------------------------------------------------------------
# Step 3: Sensor Variability Profiling
# -------------------------------------------------------------------------
# Evaluate standard deviation across all 21 telemetry channels
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
sensor_std = df[sensor_cols].std().sort_values()

print("\n========== 3. SENSOR VARIABILITY (STD DEV) ==========")
print(sensor_std.to_string())

# Bar chart visualizing sensor measurement dispersion
plt.figure(figsize=(10, 6))
sensor_std.plot(kind="bar", color="#0284c7", edgecolor="black")
plt.ylabel("Standard Deviation")
plt.xlabel("Sensor Channel")
plt.title("Variability of Sensor Telemetry (FD001)")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "sensor_variability.png", dpi=150)
plt.close()

# -------------------------------------------------------------------------
# Step 4: Run-to-Failure Sensor Trajectory (Engine #1)
# -------------------------------------------------------------------------
# Track individual sensor drift over time for a single unit from healthy to failed state
engine_1 = df[df["engine_id"] == 1]

plt.figure(figsize=(12, 6))
for sensor, label in [
    ("sensor_2", "Sensor 2 (LPC Outlet Temp)"),
    ("sensor_3", "Sensor 3 (HPC Outlet Temp)"),
    ("sensor_4", "Sensor 4 (LPT Outlet Temp)"),
]:
    plt.plot(engine_1["cycle"], engine_1[sensor], label=label, linewidth=1.8)

plt.xlabel("Operating Cycle")
plt.ylabel("Sensor Reading (Standardized Physical Units)")
plt.title("Run-to-Failure Sensor Degradation Trends for Engine #1")
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "engine_1_sensor_trends.png", dpi=150)
plt.close()

# -------------------------------------------------------------------------
# Step 5: Identification of Invariant / Low-Variability Sensors
# -------------------------------------------------------------------------
# Confirm sensor channels showing virtually zero standard deviation (< 0.01)
# These channels will be formally pruned during preprocessing to reduce noise.
low_var_sensors = sensor_std[sensor_std < 0.01]
print("\n========== 5. INVARIANT SENSORS (STD < 0.01) ==========")
print(low_var_sensors)

print(f"\nDiagnostic figures successfully generated in: {REPORTS_DIR}")
