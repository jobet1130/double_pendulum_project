import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


def load_processed_data(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = ["theta1", "theta2", "omega1", "omega2"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")
    return df[required_cols].values, df["time"].values


def compute_lyapunov(data, time, delta=1e-8, max_iter=5000):
    ref = data[0].copy()
    perturbed = ref + delta

    d0 = euclidean(ref, perturbed)
    d_list = []
    time_list = []

    for i in range(1, min(max_iter, len(data))):
        d = euclidean(data[i], data[i] + (perturbed - ref))
        d_list.append(np.log(d / d0))
        time_list.append(time[i])

    return time_list, d_list


def plot_lyapunov(time_list, divergence, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(time_list, divergence, label="log(d(t)/d0)", color="crimson")
    plt.title("Lyapunov Exponent Estimation")
    plt.xlabel("Time (s)")
    plt.ylabel("Log Divergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Lyapunov plot saved to: {save_path}")

    plt.show()


def estimate_lyapunov_slope(time_list, divergence):
    from sklearn.linear_model import LinearRegression

    X = np.array(time_list).reshape(-1, 1)
    y = np.array(divergence)

    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0]


def main():
    csv_path = Path("../data/processed/double_pendulum_processed.csv")
    data, time = load_processed_data(csv_path)

    time_list, divergence = compute_lyapunov(data, time)
    plot_lyapunov(time_list, divergence, save_path="../plots/lyapunov_divergence.png")

    λ = estimate_lyapunov_slope(time_list[:1000], divergence[:1000])
    print(f"📈 Estimated Largest Lyapunov Exponent: λ ≈ {λ:.4f} 1/s")


if __name__ == "__main__":
    main()
