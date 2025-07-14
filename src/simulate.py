import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pathlib import Path


def double_pendulum_ode(t, y, m1, m2, l1, l2, g):
    θ1, ω1, θ2, ω2 = y
    Δ = θ2 - θ1

    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(Δ)**2
    den2 = (l2 / l1) * den1

    dydt = np.zeros_like(y)
    dydt[0] = ω1
    dydt[1] = (
        m2 * l1 * ω1**2 * np.sin(Δ) * np.cos(Δ)
        + m2 * g * np.sin(θ2) * np.cos(Δ)
        + m2 * l2 * ω2**2 * np.sin(Δ)
        - (m1 + m2) * g * np.sin(θ1)
    ) / den1
    dydt[2] = ω2
    dydt[3] = (
        -m2 * l2 * ω2**2 * np.sin(Δ) * np.cos(Δ)
        + (m1 + m2) * g * np.sin(θ1) * np.cos(Δ)
        - (m1 + m2) * l1 * ω1**2 * np.sin(Δ)
        - (m1 + m2) * g * np.sin(θ2)
    ) / den2
    return dydt


def solve_double_pendulum(t_span, y0, t_eval, params):
    m1, m2, l1, l2, g = params
    return solve_ivp(
        double_pendulum_ode,
        t_span,
        y0,
        t_eval=t_eval,
        args=(m1, m2, l1, l2, g),
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
    )


def compute_cartesian(θ1, θ2, l1, l2):
    x1 = l1 * np.sin(θ1)
    y1 = -l1 * np.cos(θ1)
    x2 = x1 + l2 * np.sin(θ2)
    y2 = y1 - l2 * np.cos(θ2)
    return x1, y1, x2, y2


def compute_energy(θ1, θ2, ω1, ω2, m1, m2, l1, l2, g):
    KE = 0.5 * m1 * (l1 * ω1)**2 + \
         0.5 * m2 * ((l1 * ω1)**2 + (l2 * ω2)**2 + 2 * l1 * l2 * ω1 * ω2 * np.cos(θ1 - θ2))
    PE = - (m1 + m2) * g * l1 * np.cos(θ1) - m2 * g * l2 * np.cos(θ2)
    TME = KE + PE
    return KE, PE, TME


def save_simulation_data(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Data saved to: {output_path}")


def main():
    m1, m2 = 1.0, 1.0
    l1, l2 = 1.0, 1.0
    g = 9.81
    y0 = [np.pi / 2, 0, np.pi / 2, 0]

    t_span = (0, 20)
    n_steps = 50_000
    t_eval = np.linspace(*t_span, n_steps)

    sol = solve_double_pendulum(t_span, y0, t_eval, (m1, m2, l1, l2, g))
    θ1, ω1, θ2, ω2 = sol.y

    x1, y1, x2, y2 = compute_cartesian(θ1, θ2, l1, l2)
    KE, PE, TME = compute_energy(θ1, θ2, ω1, ω2, m1, m2, l1, l2, g)

    df = pd.DataFrame({
        "time": t_eval,
        "theta1": θ1,
        "theta2": θ2,
        "omega1": ω1,
        "omega2": ω2,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "KE": KE,
        "PE": PE,
        "TME": TME
    })

    DATA_PATH = Path("../data/processed/double_pendulum_processed.csv")
    save_simulation_data(df, DATA_PATH)


if __name__ == "__main__":
    main()
