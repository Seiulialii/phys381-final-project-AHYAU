#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# Constants
g = 9.81
rho = 1.225
r = 0.105
A = np.pi * r**2
m = 0.27
v0 = 30 * 0.44704  # Convert mph to m/s

# Coefficients
Cd = 0.47
Cl_values = {'float': 0.0, 'topspin': 0.2, 'jump': 0.1}
initial_y = 2.2

# Launch angles
angles_deg = np.arange(10, 61, 10)

# Parabolic fit model
def parabola_model(t, a, b, c):
    return a * t**2 + b * t + c

# Drag + Magnus force with variability
def nonlinear_drag_forces(t, y, Cd, Cl, spin_dir, Cd_variability=0.05):
    vx, vy, x, y_pos = y
    v = np.sqrt(vx**2 + vy**2)
    Cd_random = Cd * (1 + np.random.normal(0, Cd_variability))
    Fd = 0.5 * rho * Cd_random * A * v**2
    Fl = 0.5 * rho * Cl * A * v**2
    ax = -Fd * vx / (m * v) + spin_dir * Fl * vy / (m * v)
    ay = -g - Fd * vy / (m * v) - spin_dir * Fl * vx / (m * v)
    return [ax, ay, vx, vy]

# Simulation function
def simulate_nonlinear_serve_with_height(angle_deg, Cd, Cl, spin_dir, variability=0.05):
    angle_rad = np.radians(angle_deg)
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)
    sol = solve_ivp(nonlinear_drag_forces, [0, 5], [vx0, vy0, 0, initial_y],
                    args=(Cd, Cl, spin_dir, variability), dense_output=True, max_step=0.01)
    t_vals = np.linspace(0, 5, 500)
    y_vals = sol.sol(t_vals)
    return t_vals, y_vals

# Run simulations
nonlinear_results_height = {}
for serve_type, Cl in Cl_values.items():
    serve_data = []
    for angle in angles_deg:
        t, vals = simulate_nonlinear_serve_with_height(angle, Cd, Cl, spin_dir=1)
        x, y = vals[2], vals[3]
        mask = y >= 0
        x, y, t = x[mask], y[mask], t[mask]
        apex = max(y)
        range_ = x[-1]
        flight_time = t[-1]
        popt, _ = curve_fit(parabola_model, t, y)
        serve_data.append({
            'angle': angle,
            't': t,
            'x': x,
            'y': y,
            'apex': apex,
            'range': range_,
            'flight_time': flight_time,
            'fit_coeffs': popt
        })
    nonlinear_results_height[serve_type] = serve_data

# Plot serve trajectories on court
court_length = 18
court_half = court_length / 2
court_width = 9
net_position = court_half

for serve_type, data in nonlinear_results_height.items():
    plt.figure(figsize=(14, 6))
    plt.axvline(net_position, color='black', linestyle='--', label='Net (9m)')
    plt.axhline(0, color='gray')
    plt.fill_between([0, court_length], 0, court_width, color='beige', alpha=0.2, label='Court Area')
    for entry in data:
        plt.plot(entry['x'], entry['y'], label=f"{entry['angle']}°")
    plt.xlim(0, 20)
    plt.ylim(0, 6)
    plt.xlabel("Distance from Server (m)")
    plt.ylabel("Height (m)")
    plt.title(f"{serve_type.capitalize()} Serve Trajectories on Full Volleyball Court")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[ ]:




