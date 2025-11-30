"""
Tests for the IR data structures.
"""

import pytest
from cyecca.ir import (
    Model,
    Variable,
    VariableType,
    Expr,
    Equation,
    EquationType,
    ComponentRef,
    ComponentRefPart,
    VarRef,
    der,
    pre,
    edge,
)


def test_variable_creation():
    """Test creating variables."""
    x = Variable(name="x", var_type=VariableType.STATE, start=1.0)
    assert x.name == "x"
    assert x.is_state
    assert x.start == 1.0
    assert x.is_scalar


def test_variable_types():
    """Test variable type predicates."""
    state = Variable(name="x", var_type=VariableType.STATE)
    param = Variable(name="m", var_type=VariableType.PARAMETER, value=1.0)
    input_var = Variable(name="F", var_type=VariableType.INPUT)

    assert state.is_state
    assert not state.is_parameter
    assert param.is_parameter
    assert not param.is_state
    assert input_var.is_input


def test_expression_builders():
    """Test expression building helpers."""
    x = Expr.var_ref("x")
    v = Expr.var_ref("v")

    # Binary operations
    sum_expr = Expr.add(x, v)
    assert isinstance(sum_expr, Expr)

    # Function calls
    sin_expr = Expr.sin(x)
    assert isinstance(sin_expr, Expr)


def test_der_pre_edge_operators():
    """Test Modelica-specific operators."""
    x = Expr.var_ref("x")

    # Test der()
    dx = der(x)
    assert dx.func == "der"
    assert len(dx.args) == 1

    # Test pre()
    pre_x = pre(x)
    assert pre_x.func == "pre"

    # Test edge()
    edge_x = edge(x)
    assert edge_x.func == "edge"


def test_equation_creation():
    """Test creating equations with new SIMPLE type."""
    x = Expr.var_ref("x")
    v = Expr.var_ref("v")

    # Derivative equation: der(x) = v
    der_eq = Equation.simple(der(x), v)
    assert der_eq.eq_type == EquationType.SIMPLE
    assert der_eq.lhs.func == "der"

    # Algebraic equation: y = x
    y = Expr.var_ref("y")
    alg_eq = Equation.simple(y, x)
    assert alg_eq.eq_type == EquationType.SIMPLE
    assert alg_eq.lhs.simple_name == "y"


def test_model_creation():
    """Test creating a model."""
    model = Model(name="TestModel")

    # Add variables
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="v", var_type=VariableType.STATE, start=0.0))

    assert model.n_states == 2
    assert model.has_variable("x")
    assert model.has_variable("v")


def test_model_variable_access():
    """Test accessing variables from model."""
    model = Model(name="Test")

    state = Variable(name="x", var_type=VariableType.STATE)
    param = Variable(name="m", var_type=VariableType.PARAMETER)
    input_var = Variable(name="F", var_type=VariableType.INPUT)

    model.add_variable(state)
    model.add_variable(param)
    model.add_variable(input_var)

    assert len(model.states) == 1
    assert len(model.parameters) == 1
    assert len(model.inputs) == 1

    assert model.get_variable("x") == state


def test_model_validation():
    """Test model validation."""
    model = Model(name="Invalid")

    # Add state without derivative equation
    model.add_variable(Variable(name="x", var_type=VariableType.STATE))

    errors = model.validate()
    assert len(errors) > 0  # Should have validation error


def test_simple_model():
    """Test building a simple complete model with new syntax."""
    model = Model(name="SimpleODE")

    # dx/dt = -k*x, x(0) = 1
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=0.5))

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")
    rhs = Expr.mul(Expr.neg(k), x)

    # New style: der(x) = rhs
    model.add_equation(Equation.simple(der(x), rhs))

    # Should validate successfully
    errors = model.validate()
    assert len(errors) == 0
