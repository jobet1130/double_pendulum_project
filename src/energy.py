import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

DATA_PATH = Path("../data/processed/double_pendulum_processed.csv")
PLOT_DIR = Path("../plots")
PLOT_FILENAME = f"energy_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
PLOT_PATH = PLOT_DIR / PLOT_FILENAME

# Required columns
REQUIRED_COLUMNS = {"time", "KE", "PE", "TME"}

def load_data(path: Path) -> pd.DataFrame:
    """Load CSV and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"❌ File not found: {path}")
    
    df = pd.read_csv(path)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")
    
    logging.info(f"✅ Loaded data from {path}")
    return df

def plot_energy(df: pd.DataFrame, save_path: Path) -> None:
    """Plot KE, PE, TME, save to file, and display plot."""
    plt.figure(figsize=(12, 6))
    
    plt.plot(df["time"], df["TME"], label="Total Energy", color="black", linewidth=1.8)
    plt.plot(df["time"], df["KE"], label="Kinetic Energy", color="orange", linestyle="--", alpha=0.8)
    plt.plot(df["time"], df["PE"], label="Potential Energy", color="skyblue", linestyle="-.", alpha=0.8)
    
    plt.title("Energy Conservation in Double Pendulum", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Energy (Joules)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)

    plt.show()

    plt.close()
    
    logging.info(f"📈 Energy plot saved to: {save_path}")

def main():
    try:
        df = load_data(DATA_PATH)
        plot_energy(df, PLOT_PATH)
    except Exception as e:
        logging.error(str(e))

if __name__ == "__main__":
    main()
