
# 🌀 Double Pendulum Chaos Simulation

A computational project that explores the fascinating world of **chaos theory** through the behavior of a **double pendulum** — a classic example of a **deterministic system with unpredictable motion**.

This repository demonstrates the **simulation**, **analysis**, and **visualization** of the double pendulum's chaotic dynamics using **Python**, grounded in **physical modeling** and **numerical methods**.

---

## 🎯 Objectives

- Model a **double pendulum system** using **Lagrangian mechanics**  
- Numerically solve its **equations of motion** using `scipy.solve_ivp`  
- Generate and export a **5,000-record dataset**  
- Visualize **time evolution**, **phase space**, and **spatial trajectories**  
- Measure **chaotic behavior** through **divergence analysis**  
- Validate **conservation of mechanical energy**  
- Create **animations** of the pendulum's dynamic motion  

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/jobet1130/double_pendulum_project.git
cd double_pendulum_project
```

### 2. Set Up a Virtual Environment

```bash
python -m venv .venv
source .venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Run the Main Simulation

```bash
python main.py
```

Or launch the notebook:

```bash
jupyter notebook notebooks/01_Generate_Double_Pendulum_Data.ipynb
```

---

## 📒 Notebook Guide

| Notebook                                | Description                                                                 |
|-----------------------------------------|-----------------------------------------------------------------------------|
| **01_Generate_Double_Pendulum_Data**     | Simulates 5,000 time steps and saves results to CSV                         |
| **02_Visualize_Angles_and_Motion**       | Plots time series of angles (θ) and angular velocities (ω)                  |
| **03_Plot_Phase_Space_and_Trajectories** | Displays phase portraits and spatial motion of the pendulum                 |
| **04_Analyze_Chaos_and_Divergence**      | Compares similar initial conditions to demonstrate chaotic divergence       |
| **05_Compute_Energy_Conservation**       | Calculates and validates kinetic, potential, and total energy               |
| **06_Create_Animated_Simulation**        | Generates an MP4 animation of pendulum motion over time                     |

---

## 🧮 Mathematical Model

The double pendulum is modeled using **Lagrangian mechanics**, producing a system of **coupled second-order nonlinear differential equations**.

These are solved numerically using a **4th/5th-order Runge-Kutta method (RK45)** from `scipy.integrate.solve_ivp`.

Simulated parameters:

- **Angular positions**: θ₁, θ₂  
- **Angular velocities**: ω₁, ω₂  
- **Masses**: m₁, m₂  
- **Rod lengths**: L₁, L₂  
- **Gravitational constant**: g  

This system exhibits **extreme sensitivity to initial conditions**, a hallmark of **deterministic chaos**.

---

## 📊 Data & Outputs

- `data/raw/` — **Unmodified simulation results** (5,000 records)  
- `data/processed/` — **Enhanced datasets** with energy fields, downsampling, etc.  
- `plots/` — **Time series**, **phase space**, **divergence**, and **trajectory plots**  
- `results/` — Logs, metrics, and statistical outputs  
- `double_pendulum_animation.mp4` — **Animation** of the system’s motion  

---

## 📦 Dependencies

Install all dependencies via:

```bash
pip install -r requirements.txt
```

### Core Python Packages:

```text
numpy
pandas
scipy
matplotlib
tqdm
nolds    # Optional: for Lyapunov exponent analysis
```

---

## 📚 Topics Explored

- **Nonlinear dynamics** and **deterministic chaos**  
- **Lagrangian mechanics** for physical modeling  
- **Numerical integration** using **Runge-Kutta methods**  
- **Time series analysis** and **phase space visualization**  
- **Energy conservation** validation in mechanical systems  
- **Scientific animation** using Matplotlib  

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 👤 Author

**Jobet P. Casquejo**  
[🐙 GitHub](https://github.com/jobet1130) • [🔗 LinkedIn](https://www.linkedin.com/in/jobet-casquejo)

---

## ⭐ Support

If you found this project helpful, please consider **starring ⭐ the repository** to support future work!
