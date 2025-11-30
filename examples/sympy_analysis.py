"""
Example: Symbolic analysis with SymPy backend.

This demonstrates the unique capabilities of the SymPy backend:
- Symbolic Jacobian computation
- LaTeX export for documentation
- Expression simplification
- Taylor series expansion
- Analytical linearization
"""

from cyecca.ir import Model, Variable, Equation, Expr, VariableType, der
from cyecca.backends.sympy import SympyBackend
import sympy as sp


def pendulum_model():
    """Create a simple pendulum model."""
    model = Model(name="SimplePendulum", description="Nonlinear pendulum dynamics")

    # Variables
    model.add_variable(
        Variable(
            name="theta",
            var_type=VariableType.STATE,
            start=0.1,
            description="angle from vertical",
            unit="rad",
        )
    )
    model.add_variable(
        Variable(
            name="omega",
            var_type=VariableType.STATE,
            start=0.0,
            description="angular velocity",
            unit="rad/s",
        )
    )
    model.add_variable(
        Variable(
            name="g",
            var_type=VariableType.PARAMETER,
            value=9.81,
            description="gravity",
            unit="m/s^2",
        )
    )
    model.add_variable(
        Variable(
            name="L", var_type=VariableType.PARAMETER, value=1.0, description="length", unit="m"
        )
    )

    # Equations
    theta = Expr.var_ref("theta")
    omega = Expr.var_ref("omega")
    g = Expr.var_ref("g")
    L = Expr.var_ref("L")

    # d(theta)/dt = omega
    model.add_equation(Equation.simple(der(theta), omega))

    # d(omega)/dt = -(g/L) * sin(theta)
    sin_theta = Expr.sin(theta)
    rhs = Expr.mul(Expr.neg(Expr.div(g, L)), sin_theta)
    model.add_equation(Equation.simple(der(omega), rhs))

    return model


def main():
    """Demonstrate SymPy backend capabilities."""
    print("=" * 80)
    print("SymPy Backend - Symbolic Analysis Example")
    print("=" * 80)
    print()

    # Create model
    model = pendulum_model()
    print(f"Model: {model.name}")
    print(f"  States: {model.n_states}")
    print(f"  Parameters: {model.n_parameters}")
    print()

    # Compile to SymPy
    backend = SympyBackend(model)
    backend.compile()
    print("✓ Compiled to SymPy")
    print()

    # 1. Show symbolic equations
    print("─" * 80)
    print("1. SYMBOLIC EQUATIONS")
    print("─" * 80)
    print()

    print("State variables:")
    for var in model.states:
        print(f"  {var.name}: {backend.symbols[var.name]}")
    print()

    print("Derivative equations:")
    for var in model.states:
        der_expr = backend.derivatives[var.name]
        print(f"  d({var.name})/dt = {der_expr}")
    print()

    # 2. LaTeX export
    print("─" * 80)
    print("2. LaTeX EXPORT (for documentation)")
    print("─" * 80)
    print()

    for var in model.states:
        latex_eq = backend.to_latex(var.name, simplified=False)
        print(f"  {latex_eq}")
    print()

    # 3. Symbolic Jacobian
    print("─" * 80)
    print("3. SYMBOLIC JACOBIAN MATRIX (A = ∂f/∂x)")
    print("─" * 80)
    print()

    A_sym = backend.get_jacobian_state(simplified=False)
    print("A matrix:")
    sp.pprint(A_sym)
    print()

    print("LaTeX form:")
    print(f"  A = {sp.latex(A_sym)}")
    print()

    # 4. Linearization at equilibrium
    print("─" * 80)
    print("4. LINEARIZATION AT EQUILIBRIUM (θ=0, ω=0)")
    print("─" * 80)
    print()

    A, B, C, D = backend.linearize(x0={"theta": 0.0, "omega": 0.0})
    print("Numerical A matrix:")
    print(A)
    print()

    # Eigenvalue analysis
    eigenvalues = sp.Matrix(A).eigenvals()
    print("Eigenvalues:")
    for eigval, mult in eigenvalues.items():
        print(f"  λ = {eigval} (multiplicity {mult})")
    print()

    # Stability
    import numpy as np

    eigs = np.linalg.eigvals(A)
    if all(np.real(eigs) < 0):
        print("System is STABLE at equilibrium")
    elif all(np.real(eigs) <= 0):
        print("System is MARGINALLY STABLE at equilibrium")
    else:
        print("System is UNSTABLE at equilibrium")
    print()

    # 5. Simplification
    print("─" * 80)
    print("5. SYMBOLIC SIMPLIFICATION")
    print("─" * 80)
    print()

    for var in model.states:
        original = backend.derivatives[var.name]
        simplified = backend.simplify(var.name)
        print(f"d({var.name})/dt:")
        print(f"  Original:   {original}")
        print(f"  Simplified: {simplified}")
        print()

    # 6. Taylor series expansion
    print("─" * 80)
    print("6. TAYLOR SERIES EXPANSION (around θ=0)")
    print("─" * 80)
    print()

    # Expand d(omega)/dt around theta=0
    taylor = backend.taylor_series("omega", around={"theta": 0.0}, order=3)
    print("d(omega)/dt ≈ ", end="")
    sp.pprint(taylor)
    print()

    print("This shows that for small angles, sin(θ) ≈ θ, giving:")
    print("  d(omega)/dt ≈ -(g/L) * theta")
    print("  which is the linear approximation used in small-angle pendulum theory")
    print()

    # 7. Substitution
    print("─" * 80)
    print("7. SYMBOLIC SUBSTITUTION")
    print("─" * 80)
    print()

    # Substitute specific values
    result = backend.substitute("omega", {"theta": sp.pi / 6, "g": 9.81, "L": 1.0})
    print(f"At θ = π/6 rad (30°), with g=9.81, L=1.0:")
    print(f"  d(omega)/dt = {result}")
    print(f"  d(omega)/dt ≈ {float(result):.4f} rad/s²")
    print()

    # 8. Phase portrait symbolic analysis
    print("─" * 80)
    print("8. SYMBOLIC PHASE PORTRAIT ANALYSIS")
    print("─" * 80)
    print()

    # Find equilibrium points (where derivatives are zero)
    print("Equilibrium points (d(theta)/dt = 0, d(omega)/dt = 0):")
    print("  From d(theta)/dt = omega = 0:")
    print("    ⟹ omega = 0")
    print("  From d(omega)/dt = -(g/L)*sin(theta) = 0:")
    print("    ⟹ sin(theta) = 0")
    print("    ⟹ theta = n*π  (n = 0, ±1, ±2, ...)")
    print()
    print("Equilibrium points:")
    print("  (theta, omega) = (0, 0)      ← Stable (hanging down)")
    print("  (theta, omega) = (±π, 0)     ← Unstable (inverted)")
    print()

    # 9. Analytical Jacobian at arbitrary point
    print("─" * 80)
    print("9. ANALYTICAL JACOBIAN (at arbitrary operating point)")
    print("─" * 80)
    print()

    # Show how A depends on the operating point
    theta_sym = backend.symbols["theta"]
    g_sym = backend.symbols["g"]
    L_sym = backend.symbols["L"]

    print("At arbitrary point (θ₀, ω₀):")
    print()
    print("A(θ₀) = [[0,        1      ],")
    print(f"         [-(g/L)cos(θ₀), 0      ]]")
    print()

    # Evaluate at theta = pi/4
    A_at_pi4 = backend.get_jacobian_state().subs({theta_sym: sp.pi / 4, g_sym: 9.81, L_sym: 1.0})
    print("At θ = π/4 (45°):")
    print(A_at_pi4)
    print()

    print("=" * 80)
    print("Summary:")
    print("=" * 80)
    print()
    print("The SymPy backend enables:")
    print("  ✓ Symbolic equation manipulation")
    print("  ✓ Analytical Jacobian computation")
    print("  ✓ LaTeX export for papers/documentation")
    print("  ✓ Taylor series for approximations")
    print("  ✓ Symbolic simplification")
    print("  ✓ Analytical stability analysis")
    print()
    print("Use SymPy backend for:")
    print("  - Understanding system dynamics symbolically")
    print("  - Deriving linear approximations")
    print("  - Generating documentation")
    print("  - Teaching and learning")
    print()
    print("Use CasADi backend for:")
    print("  - Fast numerical simulation")
    print("  - Optimization")
    print("  - Embedded code generation")
    print()


if __name__ == "__main__":
    main()
