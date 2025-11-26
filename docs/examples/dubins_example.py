"""
Dubins Path Planning Example
=============================

This example demonstrates the Dubins path planner for fixed-wing aircraft.
"""

import numpy as np

from cyecca.planning import DubinsPathType, derive_dubins, plot_dubins_path

# Create planner and evaluator functions
plan_fn, eval_fn = derive_dubins()

# Define start and goal configurations
p0 = np.array([0.0, 0.0])  # Start position (x, y)
psi0 = 0.0  # Start heading (radians)

p1 = np.array([10.0, 5.0])  # Goal position (x, y)
psi1 = np.pi / 2  # Goal heading (radians)

R = 2.0  # Turn radius

# Plan the path
cost, path_type, a1, d, a2, tp0, tp1, c0, c1 = plan_fn(p0, psi0, p1, psi1, R)

print(f"Path Type: {DubinsPathType.name(path_type)}")
print(f"Total Cost: {float(cost):.4f}")
print(f"Arc 1 Angle: {float(a1):.4f} rad")
print(f"Straight Distance: {float(d):.4f}")
print(f"Arc 2 Angle: {float(a2):.4f} rad")

# Evaluate path at multiple points
n_points = 100
s_vals = np.linspace(0, 1, n_points)

print(f"\nEvaluating path at {n_points} points...")
for i, s in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
    x, y, psi = eval_fn(s, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)
    print(f"s={s:.2f}: x={float(x):.4f}, y={float(y):.4f}, psi={float(psi):.4f} rad")

# Visualize the path (optional - requires matplotlib)
try:
    import matplotlib.pyplot as plt

    ax, path_data = plot_dubins_path(p0, psi0, p1, psi1, R, plan_fn, eval_fn)
    plt.title(f"Dubins Path: {path_data['type']}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.savefig("dubins_example.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to dubins_example.png")
except ImportError:
    print("\nMatplotlib not available - skipping visualization")
