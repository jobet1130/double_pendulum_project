import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.animation import FuncAnimation
import io

st.set_page_config(page_title="Double Pendulum Simulator", layout="wide")

# -----------------------------------------------
# Physics and Math Functions
# -----------------------------------------------
def double_pendulum_ode(t, y, m1, m2, l1, l2, g):
    θ1, ω1, θ2, ω2 = y
    Δ = θ2 - θ1
    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(Δ) ** 2
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

def solve(t_span, y0, t_eval, params):
    return solve_ivp(
        double_pendulum_ode,
        t_span,
        y0,
        t_eval=t_eval,
        args=params,
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

def compute_energies(θ1, θ2, ω1, ω2, m1, m2, l1, l2, g):
    KE1 = 0.5 * m1 * (l1 * ω1)**2
    KE2 = 0.5 * m2 * (
        (l1 * ω1)**2 + (l2 * ω2)**2 + 2 * l1 * l2 * ω1 * ω2 * np.cos(θ1 - θ2)
    )
    KE = KE1 + KE2
    PE = -m1 * g * l1 * np.cos(θ1) - m2 * g * (l1 * np.cos(θ1) + l2 * np.cos(θ2))
    TME = KE + PE
    return KE, PE, TME

def compute_divergence(sol1, sol2):
    θ1_1, ω1_1, θ2_1, ω2_1 = sol1.y
    θ1_2, ω1_2, θ2_2, ω2_2 = sol2.y
    dist = np.sqrt((θ1_1 - θ1_2)**2 + (θ2_1 - θ2_2)**2 + (ω1_1 - ω1_2)**2 + (ω2_1 - ω2_2)**2)
    return dist

def compute_energy_error(TME):
    return np.abs((TME - TME[0]) / TME[0])  # relative energy error

# -----------------------------------------------
# Visualization Functions
# -----------------------------------------------
def plot_selected(option, t, θ1, θ2, ω1, ω2, x1, y1, x2, y2, KE, PE, TME, divergence=None, energy_error=None):
    fig, ax = plt.subplots(figsize=(10, 4))

    if option == "Angle vs Time":
        ax.plot(t, θ1, label="θ₁", color="blue")
        ax.plot(t, θ2, label="θ₂", color="green")
        ax.set_title("Angle vs Time")
        ax.set_ylabel("Angle (rad)")

    elif option == "Omega vs Time":
        ax.plot(t, ω1, label="ω₁", color="orange")
        ax.plot(t, ω2, label="ω₂", color="red")
        ax.set_title("Angular Velocity vs Time")
        ax.set_ylabel("Angular Velocity (rad/s)")

    elif option == "Phase Space θ₁":
        ax.plot(θ1, ω1, color="purple")
        ax.set_title("Phase Space: θ₁ vs ω₁")
        ax.set_xlabel("θ₁")
        ax.set_ylabel("ω₁")

    elif option == "Phase Space θ₂":
        ax.plot(θ2, ω2, color="teal")
        ax.set_title("Phase Space: θ₂ vs ω₂")
        ax.set_xlabel("θ₂")
        ax.set_ylabel("ω₂")

    elif option == "Trajectory":
        ax.plot(x2, y2, color="magenta", alpha=0.8)
        ax.set_title("Trajectory of Second Bob")
        ax.set_xlabel("x₂ (m)")
        ax.set_ylabel("y₂ (m)")

    elif option == "Energy":
        ax.plot(t, TME, label="Total Energy", color="black", linewidth=2)
        ax.plot(t, KE, label="Kinetic", linestyle="--", color="orange")
        ax.plot(t, PE, label="Potential", linestyle=":", color="skyblue")
        ax.set_title("Energy Over Time")
        ax.set_ylabel("Energy (J)")

    elif option == "Divergence" and divergence is not None:
        ax.plot(t, divergence, label="Divergence", color="crimson")
        ax.set_yscale("log")
        ax.set_title("Divergence Over Time (Log Scale)")
        ax.set_ylabel("Distance in State Space")

    elif option == "Energy Error" and energy_error is not None:
        ax.plot(t, energy_error, label="Energy Error", color="darkred")
        ax.set_title("Relative Energy Error")
        ax.set_ylabel("Relative Error")

    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return fig

def animate_pendulum(t, x1, y1, x2, y2, interval=20):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)

    line, = ax.plot([], [], 'o-', lw=2, color='blue')
    trace, = ax.plot([], [], '-', lw=1, color='magenta', alpha=0.5)
    path_x, path_y = [], []

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        return line, trace

    def update(i):
        path_x.append(x2[i])
        path_y.append(y2[i])
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        trace.set_data(path_x, path_y)
        return line, trace

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=interval)
    return fig

# -----------------------------------------------
# Streamlit UI
# -----------------------------------------------
st.title("🌀 Double Pendulum Simulator")

col1, col2 = st.columns(2)
with col1:
    steps = st.slider("Number of Steps", 1000, 100_000, 50000, step=1000)
    t_end = st.slider("Simulation Time (s)", 5.0, 60.0, 20.0, step=1.0)
with col2:
    m1 = st.number_input("Mass 1 (kg)", 0.1, 10.0, 1.0)
    m2 = st.number_input("Mass 2 (kg)", 0.1, 10.0, 1.0)
    l1 = st.number_input("Length 1 (m)", 0.1, 5.0, 1.0)
    l2 = st.number_input("Length 2 (m)", 0.1, 5.0, 1.0)

plot_option = st.selectbox("📊 Select Visualization", [
    "Angle vs Time",
    "Omega vs Time",
    "Energy",
    "Phase Space θ₁",
    "Phase Space θ₂",
    "Trajectory",
    "Divergence",
    "Energy Error",
    "Animate Pendulum"
])

if st.button("▶️ Run Simulation"):
    g = 9.81
    y0 = [np.pi / 2, 0, np.pi / 2, 0]
    y0_perturbed = [np.pi / 2 + 1e-5, 0, np.pi / 2, 0]
    t_span = (0, t_end)
    t_eval = np.linspace(*t_span, steps)

    with st.spinner("Running simulation..."):
        sol = solve(t_span, y0, t_eval, (m1, m2, l1, l2, g))
        sol2 = solve(t_span, y0_perturbed, t_eval, (m1, m2, l1, l2, g))

        θ1, ω1, θ2, ω2 = sol.y
        x1, y1, x2, y2 = compute_cartesian(θ1, θ2, l1, l2)
        KE, PE, TME = compute_energies(θ1, θ2, ω1, ω2, m1, m2, l1, l2, g)
        divergence = compute_divergence(sol, sol2)
        energy_error = compute_energy_error(TME)

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
            "TME": TME,
            "divergence": divergence,
            "energy_error": energy_error
        })

        st.success("✅ Simulation complete!")
        st.dataframe(df.head(10), use_container_width=True)

        if plot_option == "Animate Pendulum":
            anim_fig = animate_pendulum(t_eval[::50], x1[::50], y1[::50], x2[::50], y2[::50])
            st.pyplot(anim_fig)
        else:
            fig = plot_selected(plot_option, t_eval, θ1, θ2, ω1, ω2, x1, y1, x2, y2, KE, PE, TME, divergence, energy_error)
            st.pyplot(fig)

            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png")
            st.download_button("🖼️ Download Plot", img_buf.getvalue(), "pendulum_plot.png", "image/png")

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("📥 Download CSV", csv_buf.getvalue(), "double_pendulum.csv", "text/csv")
