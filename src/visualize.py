import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
m1 = m2 = 1.0
l1 = l2 = 1.0
g = 9.81

# Paths
DATA_PATH = Path("../data/processed/double_pendulum_processed.csv")
PLOT_DIR = Path("../plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Check for required columns
required_cols = ["theta1", "theta2", "omega1", "omega2"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

# Compute velocities
v1_sq = (l1 * df["omega1"])**2

v2x = l1 * df["omega1"] * np.cos(df["theta1"]) + l2 * df["omega2"] * np.cos(df["theta2"])
v2y = l1 * df["omega1"] * np.sin(df["theta1"]) + l2 * df["omega2"] * np.sin(df["theta2"])
v2_sq = v2x**2 + v2y**2

# Energy computations
df["KE"] = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
df["PE"] = - (m1 + m2) * g * l1 * np.cos(df["theta1"]) - m2 * g * l2 * np.cos(df["theta2"])
df["TME"] = df["KE"] + df["PE"]

# Energy plot
plt.figure(figsize=(12, 6))
plt.plot(df["time"], df["TME"], label="Total Energy", color="black")
plt.plot(df["time"], df["KE"], label="Kinetic Energy", color="orange")
plt.plot(df["time"], df["PE"], label="Potential Energy", color="skyblue")
plt.title("Energy Conservation in Double Pendulum System")
plt.xlabel("Time (s)")
plt.ylabel("Energy (Joules)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "energy_plot.png", dpi=300)
plt.show()

print(f"✅ Energy plot saved to {PLOT_DIR / 'energy_plot.png'}")
