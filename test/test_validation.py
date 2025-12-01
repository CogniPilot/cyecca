"""
Tests for model validation.
"""

import pytest
from cyecca.ir import (
    Model,
    Variable,
    VariableType,
    Expr,
    Equation,
    der,
    validate_model,
    ValidationResult,
    ValidationSeverity,
    ValidationCategory,
)


def test_valid_simple_model():
    """Test that a valid simple model passes validation."""
    model = Model(name="ValidModel")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=0.5))

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(k), x)))

    result = validate_model(model)

    # Should have no errors
    assert result.is_valid
    assert not result.has_errors


def test_undefined_variable_error():
    """Test detection of undefined variable references."""
    model = Model(name="UndefinedVar")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))

    x = Expr.var_ref("x")
    y = Expr.var_ref("y")  # y is not defined!
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.add(x, y)))

    result = validate_model(model)

    assert result.has_errors
    errors = result.errors
    assert len(errors) >= 1

    # Check that we found the undefined variable
    undefined_errors = [e for e in errors if e.category == ValidationCategory.UNDEFINED_VARIABLE]
    assert len(undefined_errors) >= 1
    assert any("y" in str(e) for e in undefined_errors)


def test_array_bounds_error():
    """Test detection of array index out of bounds."""
    model = Model(name="ArrayBounds")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, shape=[3], start=0.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    # Access x[5] which is out of bounds (x has shape [3])
    x_5 = Expr.component_ref(("x", [Expr.literal(5)]))
    model.add_equation(Equation.simple(der(x_5), Expr.var_ref("k")))

    result = validate_model(model)

    # Should have array bounds error
    bounds_errors = [e for e in result.errors if e.category == ValidationCategory.ARRAY_BOUNDS]
    assert len(bounds_errors) >= 1
    assert any("5" in str(e) and "x" in str(e) for e in bounds_errors)


def test_array_index_below_one():
    """Test detection of array index below 1 (Modelica is 1-indexed)."""
    model = Model(name="ArrayIndexZero")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, shape=[3], start=0.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    # Access x[0] which is invalid in Modelica (1-indexed)
    x_0 = Expr.component_ref(("x", [Expr.literal(0)]))
    model.add_equation(Equation.simple(der(x_0), Expr.var_ref("k")))

    result = validate_model(model)

    bounds_errors = [e for e in result.errors if e.category == ValidationCategory.ARRAY_BOUNDS]
    assert len(bounds_errors) >= 1


def test_scalar_indexing_error():
    """Test detection of indexing a scalar variable."""
    model = Model(name="ScalarIndex")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.0))  # Scalar!
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    # Try to access x[1] but x is a scalar
    x_1 = Expr.component_ref(("x", [Expr.literal(1)]))
    model.add_equation(Equation.simple(der(x_1), Expr.var_ref("k")))

    result = validate_model(model)

    bounds_errors = [e for e in result.errors if e.category == ValidationCategory.ARRAY_BOUNDS]
    assert len(bounds_errors) >= 1
    assert any("scalar" in str(e).lower() for e in bounds_errors)


def test_missing_state_start_warning():
    """Test warning for state without start value."""
    model = Model(name="NoStart")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE))  # No start value
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(k), x)))

    result = validate_model(model)

    # Should have a warning about missing start value
    assert result.has_warnings
    missing_warnings = [
        w for w in result.warnings if w.category == ValidationCategory.MISSING_VALUE
    ]
    assert len(missing_warnings) >= 1


def test_missing_parameter_value_warning():
    """Test warning for parameter without value."""
    model = Model(name="NoParamValue")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER))  # No value!

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(k), x)))

    result = validate_model(model)

    missing_warnings = [
        w for w in result.warnings if w.category == ValidationCategory.MISSING_VALUE
    ]
    assert len(missing_warnings) >= 1
    assert any("k" in str(w) for w in missing_warnings)


def test_constant_without_value_error():
    """Test error for constant without value."""
    model = Model(name="NoConstValue")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="c", var_type=VariableType.CONSTANT))  # No value!

    x = Expr.var_ref("x")
    c = Expr.var_ref("c")
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(c), x)))

    result = validate_model(model)

    # Constant without value should be an error
    missing_errors = [e for e in result.errors if e.category == ValidationCategory.MISSING_VALUE]
    assert len(missing_errors) >= 1


def test_missing_derivative_equation():
    """Test detection of state without derivative equation."""
    model = Model(name="NoDerEq")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="y", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    # Only x has a derivative equation, y does not
    x = Expr.var_ref("x")
    k = Expr.var_ref("k")
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(k), x)))

    result = validate_model(model)

    der_errors = [e for e in result.errors if e.category == ValidationCategory.DERIVATIVE]
    assert len(der_errors) >= 1
    assert any("y" in str(e) for e in der_errors)


def test_der_on_undefined_variable():
    """Test detection of der() on undefined variable."""
    model = Model(name="DerUndefined")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    # der(z) where z is not defined
    k = Expr.var_ref("k")
    model.add_equation(Equation.simple(der(Expr.var_ref("z")), k))

    result = validate_model(model)

    # Should have undefined variable error
    undefined_errors = [
        e for e in result.errors if e.category == ValidationCategory.UNDEFINED_VARIABLE
    ]
    assert len(undefined_errors) >= 1


def test_equation_balance_under_determined():
    """Test detection of under-determined system."""
    model = Model(name="UnderDetermined")
    # Two states but only one equation
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="y", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(k), x)))
    # Missing: der(y) = ...

    result = validate_model(model)

    # Should have balance error or derivative error
    assert result.has_errors


def test_hierarchical_ref_validation():
    """Test validation with hierarchical component references."""
    model = Model(name="HierarchicalRef")
    model.add_variable(Variable(name="body.position", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="body.velocity", var_type=VariableType.PARAMETER, value=1.0))

    # Use hierarchical reference
    pos = Expr.component_ref("body", "position")
    vel = Expr.component_ref("body", "velocity")
    model.add_equation(Equation.simple(der(pos), vel))

    result = validate_model(model)

    # Should be valid
    assert result.is_valid


def test_hierarchical_ref_undefined():
    """Test detection of undefined hierarchical reference."""
    model = Model(name="HierarchicalUndefined")
    model.add_variable(Variable(name="body.position", var_type=VariableType.STATE, start=0.0))

    # Reference undefined body.velocity
    pos = Expr.component_ref("body", "position")
    vel = Expr.component_ref("body", "velocity")  # Not defined!
    model.add_equation(Equation.simple(der(pos), vel))

    result = validate_model(model)

    undefined_errors = [
        e for e in result.errors if e.category == ValidationCategory.UNDEFINED_VARIABLE
    ]
    assert len(undefined_errors) >= 1


def test_when_equation_validation():
    """Test validation of when-equations."""
    model = Model(name="WhenEq")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="v", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="e", var_type=VariableType.PARAMETER, value=0.9))

    x = Expr.var_ref("x")
    v = Expr.var_ref("v")
    e = Expr.var_ref("e")

    model.add_equation(Equation.simple(der(Expr.var_ref("x")), v))
    model.add_equation(Equation.simple(der(Expr.var_ref("v")), Expr.literal(-9.81)))

    # When equation with reference to undefined variable
    condition = Expr.binary_op("<", x, Expr.literal(0.0))
    pre_v = Expr.call("pre", v)
    reset = Expr.mul(Expr.neg(e), Expr.var_ref("undefined_var"))  # Undefined!
    when_eq = Equation.when(condition, [Equation.simple(v, reset)])
    model.add_equation(when_eq)

    result = validate_model(model)

    undefined_errors = [
        e for e in result.errors if e.category == ValidationCategory.UNDEFINED_VARIABLE
    ]
    assert len(undefined_errors) >= 1


def test_validation_result_summary():
    """Test ValidationResult summary formatting."""
    result = ValidationResult()
    result.add_error(ValidationCategory.UNDEFINED_VARIABLE, "Test error", location="equation 1")
    result.add_warning(ValidationCategory.MISSING_VALUE, "Test warning")
    result.add_info(ValidationCategory.STRUCTURAL, "Test info")

    summary = result.summary()

    assert "INVALID" in summary
    assert "Errors: 1" in summary
    assert "Warnings: 1" in summary
    assert "Test error" in summary


def test_model_validate_method():
    """Test that Model.validate() method works correctly."""
    model = Model(name="ValidateMethod")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="y", var_type=VariableType.PARAMETER))  # Missing value

    x = Expr.var_ref("x")
    y = Expr.var_ref("y")
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(y), x)))

    # Basic validate() returns list of error strings
    errors = model.validate()
    assert isinstance(errors, list)

    # validate_detailed() returns ValidationResult
    result = model.validate_detailed()
    assert isinstance(result, ValidationResult)


def test_selective_validation():
    """Test that validation checks can be selectively disabled."""
    model = Model(name="Selective")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE))  # No start

    x = Expr.var_ref("x")
    y = Expr.var_ref("y")  # Undefined
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.add(x, y)))

    # With all checks
    full_result = validate_model(model)
    assert full_result.has_errors  # Undefined var

    # Disable undefined var check
    partial_result = validate_model(model, check_undefined_vars=False)
    undefined_errors = [
        e for e in partial_result.errors if e.category == ValidationCategory.UNDEFINED_VARIABLE
    ]
    assert len(undefined_errors) == 0


def test_valid_array_access():
    """Test that valid array access passes validation."""
    model = Model(name="ValidArray")
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, shape=[3], start=0.0))
    model.add_variable(
        Variable(name="v", var_type=VariableType.PARAMETER, shape=[3], value=[1.0, 2.0, 3.0])
    )

    # Valid 1-based indexing
    for i in range(1, 4):
        x_i = Expr.component_ref(("x", [Expr.literal(i)]))
        v_i = Expr.component_ref(("v", [Expr.literal(i)]))
        model.add_equation(Equation.simple(der(x_i), v_i))

    result = validate_model(model)

    # Should have no bounds errors
    bounds_errors = [e for e in result.errors if e.category == ValidationCategory.ARRAY_BOUNDS]
    assert len(bounds_errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
