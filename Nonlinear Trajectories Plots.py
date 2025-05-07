#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Re-run the full simulation after kernel reset

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import pandas as pd

# Constants
g = 9.81  # gravity (m/s^2)
rho = 1.225  # air density (kg/m^3)
r = 0.105  # volleyball radius (m)
A = np.pi * r**2  # cross-sectional area (m^2)
m = 0.27  # mass of volleyball (kg)
v0_mph = 30
v0 = v0_mph * 0.44704  # convert mph to m/s
initial_y = 2.2  # realistic initial height in meters

# Drag and lift coefficients
Cd = 0.47  # drag coefficient for a sphere
Cl_values = {'float': 0.0, 'topspin': 0.2, 'jump': 0.1}  # approximate lift coefficients

# Nonlinear drag and Magnus effect force model with variability
def nonlinear_drag_forces(t, y, Cd, Cl, spin_dir, Cd_variability=0.05):
    vx, vy, x, y_pos = y
    v = np.sqrt(vx**2 + vy**2)
    Cd_random = Cd * (1 + np.random.normal(0, Cd_variability))  # apply variability
    Fd = 0.5 * rho * Cd_random * A * v**2  # quadratic drag
    Fl = 0.5 * rho * Cl * A * v**2
    ax = -Fd * vx / (m * v) + spin_dir * Fl * vy / (m * v)
    ay = -g - Fd * vy / (m * v) - spin_dir * Fl * vx / (m * v)
    return [ax, ay, vx, vy]

# Parabolic model for curve fitting
def parabola_model(t, a, b, c):
    return a * t**2 + b * t + c

# Simulate trajectory
def simulate_nonlinear_serve(angle_deg, Cd, Cl, spin_dir, variability=0.05):
    angle_rad = np.radians(angle_deg)
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)
    sol = solve_ivp(nonlinear_drag_forces, [0, 5], [vx0, vy0, 0, initial_y],
                    args=(Cd, Cl, spin_dir, variability), dense_output=True, max_step=0.01)
    t_vals = np.linspace(0, 5, 500)
    y_vals = sol.sol(t_vals)
    return t_vals, y_vals

# Launch angles
angles_deg = np.arange(10, 61, 10)

# Run simulation
nonlinear_results = {}

for serve_type, Cl in Cl_values.items():
    serve_data = []
    for angle in angles_deg:
        t, vals = simulate_nonlinear_serve(angle, Cd, Cl, spin_dir=1)
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
    nonlinear_results[serve_type] = serve_data

# Plotting
for serve_type, data in nonlinear_results.items():
    plt.figure(figsize=(10, 6))
    for entry in data:
        plt.plot(entry['x'], entry['y'], label=f"{entry['angle']}°")
        t_fit = np.linspace(0, entry['flight_time'], len(entry['x']))
        y_fit = parabola_model(t_fit, *entry['fit_coeffs'])
        plt.plot(entry['x'], y_fit, linestyle='--', alpha=0.5)
    plt.title(f"Nonlinear Trajectories (Start Height 2.2m) - {serve_type.capitalize()} Serve")
    plt.xlabel("Range (m)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Compile and display results
df_results = pd.DataFrame([
    {
        'Serve Type': serve,
        'Launch Angle (deg)': d['angle'],
        'Apex (m)': d['apex'],
        'Range (m)': d['range'],
        'Flight Time (s)': d['flight_time'],
        'Fit Coeffs (a,b,c)': d['fit_coeffs']
    }
    for serve, entries in nonlinear_results.items() for d in entries
])

import ace_tools as tools; tools.display_dataframe_to_user(name="Serve Simulation with Realistic Height", dataframe=df_results)


# In[ ]:




