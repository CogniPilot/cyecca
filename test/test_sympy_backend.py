"""
Tests for the SymPy backend.
"""

import pytest

pytest.importorskip("sympy")  # Skip if sympy not installed

import numpy as np
import sympy as sp
from cyecca.ir import Model, Variable, Equation, Expr, VariableType, der
from cyecca.backends.sympy import SympyBackend


def test_simple_ode():
    """Test compilation of a simple ODE: dx/dt = -k*x"""
    model = Model(name="SimpleDecay")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=0.5))

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")
    rhs = Expr.mul(Expr.neg(k), x)
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), rhs))

    # Compile to SymPy
    backend = SympyBackend(model)
    backend.compile()

    # Check that symbols were created
    assert "x" in backend.symbols
    assert "k" in backend.symbols

    # Check that derivative expression was created
    assert "x" in backend.derivatives
    der_x = backend.derivatives["x"]

    # Should be: -k*x
    x_sym = backend.symbols["x"]
    k_sym = backend.symbols["k"]
    expected = -k_sym * x_sym
    assert sp.simplify(der_x - expected) == 0


def test_mass_spring_damper():
    """Test mass-spring-damper system."""
    model = Model(name="MSD")

    # Variables
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="v", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="F", var_type=VariableType.INPUT))
    model.add_variable(Variable(name="m", var_type=VariableType.PARAMETER, value=1.0))
    model.add_variable(Variable(name="c", var_type=VariableType.PARAMETER, value=0.1))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    # Equations
    x = Expr.var_ref("x")
    v = Expr.var_ref("v")
    F = Expr.var_ref("F")
    m = Expr.var_ref("m")
    c = Expr.var_ref("c")
    k = Expr.var_ref("k")

    model.add_equation(Equation.simple(der(Expr.var_ref("x")), v))

    damping_force = Expr.mul(c, v)
    spring_force = Expr.mul(k, x)
    net_force = Expr.sub(F, Expr.add(damping_force, spring_force))
    acceleration = Expr.div(net_force, m)
    model.add_equation(Equation.simple(der(Expr.var_ref("v")), acceleration))

    # Compile
    backend = SympyBackend(model)
    backend.compile()

    # Check derivatives
    assert "x" in backend.derivatives
    assert "v" in backend.derivatives

    # der(x) should be v
    assert backend.derivatives["x"] == backend.symbols["v"]


def test_jacobian_computation():
    """Test symbolic Jacobian computation."""
    model = Model(name="TwoState")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="y", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="a", var_type=VariableType.PARAMETER, value=1.0))

    x = Expr.var_ref("x")
    y = Expr.var_ref("y")
    a = Expr.var_ref("a")

    # dx/dt = -a*x
    # dy/dt = x - y
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(a), x)))
    model.add_equation(Equation.simple(der(Expr.var_ref("y")), Expr.sub(x, y)))

    backend = SympyBackend(model)
    backend.compile()

    # Get Jacobian
    A = backend.get_jacobian_state()

    # Should be:
    # [[-a,  0],
    #  [ 1, -1]]
    assert A.shape == (2, 2)

    x_sym = backend.symbols["x"]
    y_sym = backend.symbols["y"]
    a_sym = backend.symbols["a"]

    assert A[0, 0] == -a_sym
    assert A[0, 1] == 0
    assert A[1, 0] == 1
    assert A[1, 1] == -1


def test_linearization():
    """Test numerical linearization at an operating point."""
    model = Model(name="Pendulum")
    model.add_variable(Variable(name="theta", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="omega", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="g", var_type=VariableType.PARAMETER, value=9.81))
    model.add_variable(Variable(name="L", var_type=VariableType.PARAMETER, value=1.0))

    theta = Expr.var_ref("theta")
    omega = Expr.var_ref("omega")
    g = Expr.var_ref("g")
    L = Expr.var_ref("L")

    # d(theta)/dt = omega
    model.add_equation(Equation.simple(der(Expr.var_ref("theta")), omega))

    # d(omega)/dt = -g/L * sin(theta)
    sin_theta = Expr.sin(theta)
    rhs = Expr.mul(Expr.neg(Expr.div(g, L)), sin_theta)
    model.add_equation(Equation.simple(der(Expr.var_ref("omega")), rhs))

    backend = SympyBackend(model)
    backend.compile()

    # Linearize at equilibrium (theta=0, omega=0)
    A, B, C, D = backend.linearize(x0={"theta": 0.0, "omega": 0.0})

    # At theta=0, sin(theta) ≈ theta, so:
    # A should be approximately:
    # [[0,      1],
    #  [-g/L,   0]]
    assert A.shape == (2, 2)
    assert np.isclose(A[0, 0], 0.0)
    assert np.isclose(A[0, 1], 1.0)
    assert np.isclose(A[1, 0], -9.81 / 1.0)
    assert np.isclose(A[1, 1], 0.0)


def test_latex_export():
    """Test LaTeX export."""
    model = Model(name="Simple")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(k), x)))

    backend = SympyBackend(model)
    backend.compile()

    latex_str = backend.to_latex("x", simplified=False)

    # Should contain \dot{x} and k, x symbols
    assert r"\dot{x}" in latex_str
    assert "k" in latex_str or "x" in latex_str


def test_simplification():
    """Test symbolic simplification."""
    model = Model(name="Complex")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE))
    model.add_variable(Variable(name="a", var_type=VariableType.PARAMETER, value=2.0))
    model.add_variable(Variable(name="b", var_type=VariableType.PARAMETER, value=3.0))

    x = Expr.var_ref("x")
    a = Expr.var_ref("a")
    b = Expr.var_ref("b")

    # Create a complex expression: (a*x + b*x)
    term1 = Expr.mul(a, x)
    term2 = Expr.mul(b, x)
    rhs = Expr.add(term1, term2)

    model.add_equation(Equation.simple(der(Expr.var_ref("x")), rhs))

    backend = SympyBackend(model)
    backend.compile()

    # Get simplified version
    simplified = backend.simplify("x")

    # Should simplify to (a + b)*x
    x_sym = backend.symbols["x"]
    a_sym = backend.symbols["a"]
    b_sym = backend.symbols["b"]
    expected = (a_sym + b_sym) * x_sym

    assert sp.simplify(simplified - expected) == 0


def test_substitution():
    """Test symbolic substitution."""
    model = Model(name="Sub")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE))
    model.add_variable(Variable(name="y", var_type=VariableType.STATE))

    x = Expr.var_ref("x")
    y = Expr.var_ref("y")

    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.add(x, y)))

    backend = SympyBackend(model)
    backend.compile()

    # Substitute x=2, y=3
    result = backend.substitute("x", {"x": 2.0, "y": 3.0})

    # Should give 2 + 3 = 5
    assert float(result) == 5.0


def test_trig_functions():
    """Test trigonometric functions."""
    model = Model(name="Trig")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE))

    x = Expr.var_ref("x")

    # Create expression with trig functions
    sin_x = Expr.sin(x)
    cos_x = Expr.cos(x)
    rhs = Expr.add(sin_x, cos_x)

    model.add_equation(Equation.simple(der(Expr.var_ref("x")), rhs))

    backend = SympyBackend(model)
    backend.compile()

    # Check that it compiled
    assert "x" in backend.derivatives

    # Derivative should involve sin and cos
    der_x = backend.derivatives["x"]
    x_sym = backend.symbols["x"]
    expected = sp.sin(x_sym) + sp.cos(x_sym)

    assert sp.simplify(der_x - expected) == 0


def test_simulation():
    """Test that simulation works (using lambdified functions)."""
    model = Model(name="SimpleODE")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(k), x)))

    backend = SympyBackend(model)
    backend.compile()

    # Simulate
    t, sol = backend.simulate(t_final=5.0, dt=0.1)

    # Check solution
    assert len(t) > 0
    assert "x" in sol
    assert len(sol["x"]) == len(t)

    # Should decay exponentially: x(t) = exp(-k*t)
    # At t=0, x should be close to 1.0
    assert np.isclose(sol["x"][0], 1.0, atol=0.1)

    # At t=5, x should be close to exp(-5) ≈ 0.0067
    assert sol["x"][-1] < 0.1  # Should have decayed significantly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
