"""
Tests for the SymPy backend.
"""

import pytest

pytest.importorskip("sympy")  # Skip if sympy not installed

import numpy as np
import sympy as sp
from cyecca.ir import Model, Variable, Equation, Expr, VariableType, der, IfExpr
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


def test_array_variable():
    """Test array state variables (vectors)."""
    model = Model(name="ArrayTest")

    # Create a 3-element state vector
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, shape=[3], start=0.0))
    model.add_variable(
        Variable(name="v", var_type=VariableType.PARAMETER, shape=[3], value=[1.0, 2.0, 3.0])
    )

    # Each element evolves at a different rate: der(x[i]) = v[i]
    for i in range(1, 4):  # Modelica 1-based indexing
        x_i = Expr.component_ref(("x", [Expr.literal(i)]))
        v_i = Expr.component_ref(("v", [Expr.literal(i)]))
        model.add_equation(Equation.simple(der(x_i), v_i))

    backend = SympyBackend(model)
    backend.compile()

    # Check that element-wise derivatives were created
    assert ("x", 0) in backend.derivatives
    assert ("x", 1) in backend.derivatives
    assert ("x", 2) in backend.derivatives

    # Check derivative expressions reference correct parameter symbols
    assert backend.derivatives[("x", 0)] == backend.symbols[("v", 0)]
    assert backend.derivatives[("x", 1)] == backend.symbols[("v", 1)]
    assert backend.derivatives[("x", 2)] == backend.symbols[("v", 2)]


def test_array_indexing_expression():
    """Test array indexing in expressions."""
    model = Model(name="ArrayIndexTest")

    # Create scalar state
    model.add_variable(Variable(name="y", var_type=VariableType.STATE, start=0.0))
    # Create array parameter
    model.add_variable(
        Variable(name="k", var_type=VariableType.PARAMETER, shape=[2], value=[1.0, 2.0])
    )

    # der(y) = k[1] + k[2]
    k1 = Expr.component_ref(("k", [Expr.literal(1)]))
    k2 = Expr.component_ref(("k", [Expr.literal(2)]))
    rhs = Expr.add(k1, k2)
    model.add_equation(Equation.simple(der(Expr.var_ref("y")), rhs))

    backend = SympyBackend(model)
    backend.compile()

    # Check derivative was created
    assert "y" in backend.derivatives

    # Check expression uses correct symbols
    k1_sym = backend.symbols[("k", 0)]
    k2_sym = backend.symbols[("k", 1)]
    expected = k1_sym + k2_sym
    assert sp.simplify(backend.derivatives["y"] - expected) == 0


def test_hierarchical_component_ref():
    """Test hierarchical component references like vehicle.engine.temp."""
    model = Model(name="HierarchicalTest")

    # Create flattened variables with dot notation
    model.add_variable(
        Variable(name="vehicle.engine.temp", var_type=VariableType.STATE, start=20.0)
    )
    model.add_variable(Variable(name="T_ambient", var_type=VariableType.PARAMETER, value=15.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=0.1))

    # der(vehicle.engine.temp) = k * (T_ambient - vehicle.engine.temp)
    temp = Expr.component_ref("vehicle", "engine", "temp")
    ambient = Expr.var_ref("T_ambient")
    k = Expr.var_ref("k")
    rhs = Expr.mul(k, Expr.sub(ambient, temp))
    model.add_equation(Equation.simple(der(temp), rhs))

    backend = SympyBackend(model)
    backend.compile()

    # Check that the variable was recognized
    assert "vehicle.engine.temp" in backend.symbols
    assert "vehicle.engine.temp" in backend.derivatives

    # Check symbolic derivative is correct
    temp_sym = backend.symbols["vehicle.engine.temp"]
    amb_sym = backend.symbols["T_ambient"]
    k_sym = backend.symbols["k"]
    expected = k_sym * (amb_sym - temp_sym)
    assert sp.simplify(backend.derivatives["vehicle.engine.temp"] - expected) == 0


def test_hierarchical_ref_latex():
    """Test LaTeX export with hierarchical refs."""
    model = Model(name="HierarchicalLatex")

    model.add_variable(Variable(name="body.position", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="body.velocity", var_type=VariableType.STATE, start=1.0))

    pos = Expr.component_ref("body", "position")
    vel = Expr.component_ref("body", "velocity")
    model.add_equation(Equation.simple(der(pos), vel))
    model.add_equation(Equation.simple(der(vel), Expr.neg(pos)))

    backend = SympyBackend(model)
    backend.compile()

    # Both should have derivatives
    assert "body.position" in backend.derivatives
    assert "body.velocity" in backend.derivatives


def test_output_variable_linearization():
    """Test linearization with output variables (C, D matrices)."""
    model = Model(name="OutputTest")

    # States
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="v", var_type=VariableType.STATE, start=0.0))

    # Input
    model.add_variable(Variable(name="u", var_type=VariableType.INPUT))

    # Output
    model.add_variable(Variable(name="y", var_type=VariableType.OUTPUT))

    # Parameters
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    # Equations: der(x) = v, der(v) = -k*x + u, y = x
    x = Expr.var_ref("x")
    v = Expr.var_ref("v")
    u = Expr.var_ref("u")
    k = Expr.var_ref("k")

    model.add_equation(Equation.simple(der(Expr.var_ref("x")), v))
    model.add_equation(
        Equation.simple(der(Expr.var_ref("v")), Expr.add(Expr.mul(Expr.neg(k), x), u))
    )
    model.add_equation(Equation.simple(Expr.var_ref("y"), x))

    backend = SympyBackend(model)
    backend.compile()

    A, B, C, D = backend.linearize()

    # A should be [[0, 1], [-k, 0]] = [[0, 1], [-1, 0]]
    assert A.shape == (2, 2)
    assert np.isclose(A[0, 0], 0.0)
    assert np.isclose(A[0, 1], 1.0)
    assert np.isclose(A[1, 0], -1.0)
    assert np.isclose(A[1, 1], 0.0)

    # B should be [[0], [1]]
    assert B.shape == (2, 1)
    assert np.isclose(B[0, 0], 0.0)
    assert np.isclose(B[1, 0], 1.0)

    # C should be [[1, 0]] (y = x)
    assert C.shape == (1, 2)
    assert np.isclose(C[0, 0], 1.0)
    assert np.isclose(C[0, 1], 0.0)

    # D should be [[0]] (no direct feedthrough)
    assert D.shape == (1, 1)
    assert np.isclose(D[0, 0], 0.0)


def test_array_state_simulation():
    """Test simulation with array state variables."""
    model = Model(name="ArraySimTest")

    # 3-element state vector with initial values (using value for arrays, start is scalar)
    model.add_variable(
        Variable(name="x", var_type=VariableType.STATE, shape=[3], value=[1.0, 2.0, 3.0])
    )
    # Decay rates
    model.add_variable(
        Variable(name="k", var_type=VariableType.PARAMETER, shape=[3], value=[0.1, 0.2, 0.3])
    )

    # Each element decays: der(x[i]) = -k[i] * x[i]
    for i in range(1, 4):  # 1-based Modelica indexing
        x_i = Expr.component_ref(("x", [Expr.literal(i)]))
        k_i = Expr.component_ref(("k", [Expr.literal(i)]))
        model.add_equation(Equation.simple(der(x_i), Expr.mul(Expr.neg(k_i), x_i)))

    backend = SympyBackend(model)
    backend.compile()

    # Simulate for 5 seconds
    t, sol = backend.simulate(t_final=5.0, dt=0.1)

    # Check that x is a 2D array (3 elements x timesteps)
    assert "x" in sol
    assert sol["x"].shape[0] == 3

    # Check initial conditions
    assert np.isclose(sol["x"][0, 0], 1.0, atol=0.1)
    assert np.isclose(sol["x"][1, 0], 2.0, atol=0.1)
    assert np.isclose(sol["x"][2, 0], 3.0, atol=0.1)

    # Check that faster decay rate leads to smaller final values
    # x[2] has the highest decay rate (0.3), x[0] has the lowest (0.1)
    final_ratios = sol["x"][:, -1] / sol["x"][:, 0]
    assert final_ratios[2] < final_ratios[1] < final_ratios[0]


def test_logical_operators():
    """Test logical operators (and, or) in expressions."""
    model = Model(name="LogicalTest")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="a", var_type=VariableType.PARAMETER, value=1.0))
    model.add_variable(Variable(name="b", var_type=VariableType.PARAMETER, value=2.0))

    # Use if expression with logical condition
    a = Expr.var_ref("a")
    b = Expr.var_ref("b")

    # condition: (a > 0) and (b > 1)
    cond1 = Expr.binary_op(">", a, Expr.literal(0.0))
    cond2 = Expr.binary_op(">", b, Expr.literal(1.0))
    combined_cond = Expr.binary_op("and", cond1, cond2)

    # der(x) = if (a > 0 and b > 1) then 1 else -1
    rhs = IfExpr(
        condition=combined_cond, true_expr=Expr.literal(1.0), false_expr=Expr.literal(-1.0)
    )
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), rhs))

    backend = SympyBackend(model)
    backend.compile()

    # Check that it compiled (the expression involves Piecewise with And)
    assert "x" in backend.derivatives


def test_hyperbolic_functions():
    """Test hyperbolic functions (sinh, cosh, tanh)."""
    model = Model(name="HyperbolicTest")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.0))

    x = Expr.var_ref("x")

    # Create expression with hyperbolic functions
    sinh_x = Expr.call("sinh", x)
    cosh_x = Expr.call("cosh", x)
    tanh_x = Expr.call("tanh", x)

    # der(x) = sinh(x) + cosh(x) - tanh(x)
    rhs = Expr.sub(Expr.add(sinh_x, cosh_x), tanh_x)
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), rhs))

    backend = SympyBackend(model)
    backend.compile()

    # Check that it compiled
    assert "x" in backend.derivatives

    # Verify symbolic expression
    x_sym = backend.symbols["x"]
    expected = sp.sinh(x_sym) + sp.cosh(x_sym) - sp.tanh(x_sym)
    assert sp.simplify(backend.derivatives["x"] - expected) == 0


def test_controllability_matrix():
    """Test controllability matrix computation."""
    model = Model(name="ControllabilityTest")

    model.add_variable(Variable(name="x1", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="x2", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="u", var_type=VariableType.INPUT))

    # Simple controllable system: der(x1) = x2, der(x2) = u
    x2 = Expr.var_ref("x2")
    u = Expr.var_ref("u")

    model.add_equation(Equation.simple(der(Expr.var_ref("x1")), x2))
    model.add_equation(Equation.simple(der(Expr.var_ref("x2")), u))

    backend = SympyBackend(model)
    backend.compile()

    # Get controllability matrix
    Co = backend.controllability_matrix()

    # Should be [B, AB] = [[0, 1], [1, 0]]
    assert Co.shape == (2, 2)

    # Check rank (should be 2 for controllable system)
    assert Co.rank() == 2


def test_observability_matrix():
    """Test observability matrix computation."""
    model = Model(name="ObservabilityTest")

    model.add_variable(Variable(name="x1", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="x2", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="y", var_type=VariableType.OUTPUT))

    # Simple observable system: der(x1) = x2, der(x2) = 0, y = x1
    x2 = Expr.var_ref("x2")
    x1 = Expr.var_ref("x1")

    model.add_equation(Equation.simple(der(Expr.var_ref("x1")), x2))
    model.add_equation(Equation.simple(der(Expr.var_ref("x2")), Expr.literal(0.0)))
    model.add_equation(Equation.simple(Expr.var_ref("y"), x1))

    backend = SympyBackend(model)
    backend.compile()

    # Get observability matrix
    Ob = backend.observability_matrix()

    # Should be [C; CA] = [[1, 0], [0, 1]]
    assert Ob.shape == (2, 2)

    # Check rank (should be 2 for observable system)
    assert Ob.rank() == 2


def test_state_space_symbolic():
    """Test getting full symbolic state-space representation."""
    model = Model(name="StateSpaceTest")

    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="u", var_type=VariableType.INPUT))
    model.add_variable(Variable(name="y", var_type=VariableType.OUTPUT))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    x = Expr.var_ref("x")
    u = Expr.var_ref("u")
    k = Expr.var_ref("k")

    # der(x) = -k*x + u, y = x
    model.add_equation(
        Equation.simple(der(Expr.var_ref("x")), Expr.add(Expr.mul(Expr.neg(k), x), u))
    )
    model.add_equation(Equation.simple(Expr.var_ref("y"), x))

    backend = SympyBackend(model)
    backend.compile()

    ss = backend.get_state_space_symbolic()

    # Check all components exist
    assert "A" in ss
    assert "B" in ss
    assert "C" in ss
    assert "D" in ss
    assert "f" in ss
    assert "y" in ss

    # Check shapes
    assert ss["A"].shape == (1, 1)
    assert ss["B"].shape == (1, 1)
    assert ss["C"].shape == (1, 1)
    assert ss["D"].shape == (1, 1)


def test_get_all_expressions():
    """Test getting all symbolic expressions."""
    model = Model(name="AllExprsTest")

    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="y", var_type=VariableType.OUTPUT))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")

    # der(x) = -k*x, y = 2*x
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(k), x)))
    model.add_equation(Equation.simple(Expr.var_ref("y"), Expr.mul(Expr.literal(2.0), x)))

    backend = SympyBackend(model)
    backend.compile()

    exprs = backend.get_all_expressions()

    # Should have der(x) and y
    assert "der(x)" in exprs
    assert "y" in exprs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
