#!/usr/bin/env python3
"""
Example: Mass-Spring-Damper System

This demonstrates how to:
1. Build a model using the IR
2. Compile it with the CasADi backend
3. Simulate the system
4. Linearize at an operating point
"""

import numpy as np
import matplotlib.pyplot as plt

from cyecca.ir import Model, Variable, VariableType, Expr, Equation, der
from cyecca.backends.casadi import CasadiBackend


def main():
    # Create the model
    model = Model(name="MassSpringDamper", description="Simple mass-spring-damper")

    # Add variables
    model.add_variable(
        Variable(
            name="x",
            var_type=VariableType.STATE,
            start=1.0,
            description="position",
            unit="m",
        )
    )
    model.add_variable(
        Variable(
            name="v",
            var_type=VariableType.STATE,
            start=0.0,
            description="velocity",
            unit="m/s",
        )
    )
    model.add_variable(
        Variable(name="F", var_type=VariableType.INPUT, description="applied force", unit="N")
    )
    model.add_variable(
        Variable(
            name="m", var_type=VariableType.PARAMETER, value=1.0, description="mass", unit="kg"
        )
    )
    model.add_variable(
        Variable(
            name="c",
            var_type=VariableType.PARAMETER,
            value=0.5,
            description="damping",
            unit="N*s/m",
        )
    )
    model.add_variable(
        Variable(
            name="k",
            var_type=VariableType.PARAMETER,
            value=2.0,
            description="spring constant",
            unit="N/m",
        )
    )

    # Build expressions
    x = Expr.var_ref("x")
    v = Expr.var_ref("v")
    F = Expr.var_ref("F")
    m = Expr.var_ref("m")
    c = Expr.var_ref("c")
    k = Expr.var_ref("k")

    # Add equations
    # der(x) = v
    model.add_equation(Equation.simple(der(x), v))

    # der(v) = (F - c*v - k*x) / m
    damping_force = Expr.mul(c, v)
    spring_force = Expr.mul(k, x)
    net_force = Expr.sub(F, Expr.add(damping_force, spring_force))
    accel = Expr.div(net_force, m)
    model.add_equation(Equation.simple(der(v), accel))

    print(model)
    print()

    # Validate
    errors = model.validate()
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return

    print("Model is valid!")
    print()

    # Compile with CasADi backend
    print("Compiling with CasADi backend...")
    backend = CasadiBackend(model)
    backend.compile()
    print("Compilation successful!")
    print()

    # Simulate (free oscillation, no input)
    print("Simulating for 20 seconds...")
    t, sol = backend.simulate(t_final=20.0, dt=0.01)
    print(f"Simulation complete! {len(t)} time steps")
    print()

    # Linearize at equilibrium (x=0, v=0, F=0)
    print("Linearizing at equilibrium...")
    A, B, C, D = backend.linearize(x0={"x": 0.0, "v": 0.0}, u0={"F": 0.0})

    print("State-space matrices:")
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"C = \n{C}")
    print(f"D = \n{D}")
    print()

    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvals(A)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"System is stable: {np.all(np.real(eigenvalues) < 0)}")
    print()

    # Natural frequency and damping ratio
    # For second-order system: eigenvalues = -zeta*omega_n ± j*omega_n*sqrt(1-zeta^2)
    omega_n = np.abs(eigenvalues[0])
    zeta = -np.real(eigenvalues[0]) / omega_n
    print(f"Natural frequency: {omega_n:.3f} rad/s")
    print(f"Damping ratio: {zeta:.3f}")
    print()

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t, sol["x"], label="Position (m)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")
    ax1.set_title("Mass-Spring-Damper: Position vs Time")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t, sol["v"], label="Velocity (m/s)", color="orange")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_title("Mass-Spring-Damper: Velocity vs Time")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("mass_spring_damper.png", dpi=150)
    print("Plot saved to mass_spring_damper.png")

    # Phase portrait
    plt.figure(figsize=(8, 6))
    plt.plot(sol["x"], sol["v"])
    plt.xlabel("Position (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Phase Portrait")
    plt.grid(True)
    plt.savefig("phase_portrait.png", dpi=150)
    print("Phase portrait saved to phase_portrait.png")


if __name__ == "__main__":
    main()
