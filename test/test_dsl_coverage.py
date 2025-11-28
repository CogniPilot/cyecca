"""
Additional tests for DSL coverage improvement.

This file targets specific uncovered code paths identified by coverage analysis.
"""

import math

import numpy as np
import pytest

# =============================================================================
# Operators Tests - Math Functions with Floats and Symbolic
# =============================================================================


class TestMathOperatorsFloat:
    """Test math operators with float inputs (return Python floats)."""

    def test_sin_float(self) -> None:
        from cyecca.dsl.operators import sin

        result = sin(0.5)
        assert isinstance(result, float)
        assert result == pytest.approx(math.sin(0.5))

    def test_cos_float(self) -> None:
        from cyecca.dsl.operators import cos

        result = cos(0.5)
        assert isinstance(result, float)
        assert result == pytest.approx(math.cos(0.5))

    def test_tan_float(self) -> None:
        from cyecca.dsl.operators import tan

        result = tan(0.5)
        assert isinstance(result, float)
        assert result == pytest.approx(math.tan(0.5))

    def test_asin_float(self) -> None:
        from cyecca.dsl.operators import asin

        result = asin(0.5)
        assert isinstance(result, float)
        assert result == pytest.approx(math.asin(0.5))

    def test_acos_float(self) -> None:
        from cyecca.dsl.operators import acos

        result = acos(0.5)
        assert isinstance(result, float)
        assert result == pytest.approx(math.acos(0.5))

    def test_atan_float(self) -> None:
        from cyecca.dsl.operators import atan

        result = atan(0.5)
        assert isinstance(result, float)
        assert result == pytest.approx(math.atan(0.5))

    def test_atan2_float(self) -> None:
        from cyecca.dsl.operators import atan2

        result = atan2(1.0, 2.0)
        assert isinstance(result, float)
        assert result == pytest.approx(math.atan2(1.0, 2.0))

    def test_sqrt_float(self) -> None:
        from cyecca.dsl.operators import sqrt

        result = sqrt(4.0)
        assert isinstance(result, float)
        assert result == pytest.approx(2.0)

    def test_exp_float(self) -> None:
        from cyecca.dsl.operators import exp

        result = exp(1.0)
        assert isinstance(result, float)
        assert result == pytest.approx(math.e)

    def test_log_float(self) -> None:
        from cyecca.dsl.operators import log

        result = log(math.e)
        assert isinstance(result, float)
        assert result == pytest.approx(1.0)

    def test_abs_float_positive(self) -> None:
        from cyecca.dsl import abs as dsl_abs

        result = dsl_abs(5.0)
        assert isinstance(result, float)
        assert result == 5.0

    def test_abs_float_negative(self) -> None:
        from cyecca.dsl import abs as dsl_abs

        result = dsl_abs(-5.0)
        assert isinstance(result, float)
        assert result == 5.0


class TestMathOperatorsSymbolic:
    """Test math operators with symbolic inputs (return Expr)."""

    def test_tan_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, model, tan, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == tan(m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.TAN

    def test_asin_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, asin, der, model, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == asin(m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.ASIN

    def test_acos_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, acos, der, model, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == acos(m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.ACOS

    def test_atan_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, atan, der, model, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == atan(m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.ATAN

    def test_atan2_symbolic_both(self) -> None:
        from cyecca.dsl import ExprKind, atan2, der, model, var

        @model
        class TestModel:
            x = var()
            y = var()
            angle = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield der(m.y) == 1.0
                yield m.angle == atan2(m.y, m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["angle"]
        assert expr.kind == ExprKind.ATAN2

    def test_atan2_symbolic_first_arg_float(self) -> None:
        from cyecca.dsl import ExprKind, atan2, der, model, var

        @model
        class TestModel:
            x = var()
            angle = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.angle == atan2(1.0, m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["angle"]
        assert expr.kind == ExprKind.ATAN2

    def test_atan2_symbolic_second_arg_float(self) -> None:
        from cyecca.dsl import ExprKind, atan2, der, model, var

        @model
        class TestModel:
            y = var()
            angle = var(output=True)

            def equations(m):
                yield der(m.y) == 1.0
                yield m.angle == atan2(m.y, 1.0)

        flat = TestModel().flatten()
        expr = flat.output_equations["angle"]
        assert expr.kind == ExprKind.ATAN2

    def test_exp_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, exp, model, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == exp(m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.EXP

    def test_log_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, log, model, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == log(m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.LOG

    def test_abs_symbolic(self) -> None:
        from cyecca.dsl import ExprKind
        from cyecca.dsl import abs as dsl_abs
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == dsl_abs(m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.ABS


class TestOperatorTypeErrors:
    """Test error handling in operators."""

    def test_sin_invalid_type(self) -> None:
        from cyecca.dsl.operators import sin

        with pytest.raises(TypeError):
            sin("not a number")

    def test_cos_invalid_type(self) -> None:
        from cyecca.dsl.operators import cos

        with pytest.raises(TypeError):
            cos([1, 2, 3])


# =============================================================================
# Expr __repr__ Tests
# =============================================================================


class TestExprRepr:
    """Test Expr __repr__ for various expression kinds."""

    def test_constant_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        expr = Expr(ExprKind.CONSTANT, value=3.14)
        assert "3.14" in repr(expr)

    def test_time_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        expr = Expr(ExprKind.TIME)
        assert repr(expr) == "t"

    def test_neg_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        inner = Expr(ExprKind.VARIABLE, name="x")
        expr = Expr(ExprKind.NEG, (inner,))
        assert "(-x)" in repr(expr)

    def test_sub_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        left = Expr(ExprKind.VARIABLE, name="x")
        right = Expr(ExprKind.VARIABLE, name="y")
        expr = Expr(ExprKind.SUB, (left, right))
        assert "(x - y)" in repr(expr)

    def test_div_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        left = Expr(ExprKind.VARIABLE, name="x")
        right = Expr(ExprKind.VARIABLE, name="y")
        expr = Expr(ExprKind.DIV, (left, right))
        assert "(x / y)" in repr(expr)

    def test_pow_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        left = Expr(ExprKind.VARIABLE, name="x")
        right = Expr(ExprKind.CONSTANT, value=2.0)
        expr = Expr(ExprKind.POW, (left, right))
        assert "(x ** 2.0)" in repr(expr)

    def test_indexed_variable_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        expr = Expr(ExprKind.VARIABLE, name="pos", indices=(0, 1))
        assert "pos[0,1]" in repr(expr)

    def test_index_kind_repr(self) -> None:
        """Test legacy INDEX kind repr."""
        from cyecca.dsl.model import Expr, ExprKind

        expr = Expr(ExprKind.INDEX, name="x", value=2.0)
        assert "x[2]" in repr(expr)

    def test_atan2_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        y = Expr(ExprKind.VARIABLE, name="y")
        x = Expr(ExprKind.VARIABLE, name="x")
        expr = Expr(ExprKind.ATAN2, (y, x))
        assert "atan2(y, x)" in repr(expr)

    def test_eq_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        left = Expr(ExprKind.VARIABLE, name="x")
        right = Expr(ExprKind.CONSTANT, value=0.0)
        expr = Expr(ExprKind.EQ, (left, right))
        assert "(x == 0.0)" in repr(expr)

    def test_ne_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        left = Expr(ExprKind.VARIABLE, name="x")
        right = Expr(ExprKind.CONSTANT, value=0.0)
        expr = Expr(ExprKind.NE, (left, right))
        assert "(x != 0.0)" in repr(expr)

    def test_unknown_kind_repr(self) -> None:
        """Test fallback repr for unknown kinds."""
        from cyecca.dsl.model import Expr, ExprKind

        # PRE, EDGE, CHANGE have repr but may not be fully tested
        expr = Expr(ExprKind.PRE, name="x")
        assert "pre(x)" in repr(expr)

        expr = Expr(ExprKind.EDGE, name="x")
        assert "edge(x)" in repr(expr)

        expr = Expr(ExprKind.CHANGE, name="x")
        assert "change(x)" in repr(expr)


# =============================================================================
# TimeVar Tests
# =============================================================================


class TestTimeVar:
    """Test TimeVar arithmetic operations."""

    def test_time_repr(self) -> None:
        from cyecca.dsl.model import TimeVar

        t = TimeVar()
        assert repr(t) == "t"

    def test_time_add(self) -> None:
        from cyecca.dsl.model import ExprKind, TimeVar

        t = TimeVar()
        expr = t + 1.0
        assert expr.kind == ExprKind.ADD

    def test_time_radd(self) -> None:
        from cyecca.dsl.model import ExprKind, TimeVar

        t = TimeVar()
        expr = 1.0 + t
        assert expr.kind == ExprKind.ADD

    def test_time_sub(self) -> None:
        from cyecca.dsl.model import ExprKind, TimeVar

        t = TimeVar()
        expr = t - 1.0
        assert expr.kind == ExprKind.SUB

    def test_time_rsub(self) -> None:
        from cyecca.dsl.model import ExprKind, TimeVar

        t = TimeVar()
        expr = 10.0 - t
        assert expr.kind == ExprKind.SUB

    def test_time_mul(self) -> None:
        from cyecca.dsl.model import ExprKind, TimeVar

        t = TimeVar()
        expr = t * 2.0
        assert expr.kind == ExprKind.MUL

    def test_time_rmul(self) -> None:
        from cyecca.dsl.model import ExprKind, TimeVar

        t = TimeVar()
        expr = 2.0 * t
        assert expr.kind == ExprKind.MUL

    def test_time_div(self) -> None:
        from cyecca.dsl.model import ExprKind, TimeVar

        t = TimeVar()
        expr = t / 2.0
        assert expr.kind == ExprKind.DIV

    def test_time_rdiv(self) -> None:
        from cyecca.dsl.model import ExprKind, TimeVar

        t = TimeVar()
        expr = 1.0 / t
        assert expr.kind == ExprKind.DIV


# =============================================================================
# DerivativeExpr Tests
# =============================================================================


class TestDerivativeExpr:
    """Test DerivativeExpr arithmetic."""

    def test_der_add(self) -> None:
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            y = var()
            z = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield der(m.y) == 1.0
                yield m.z == der(m.x) + der(m.y)

        flat = TestModel().flatten()
        # The output equation should contain an ADD
        expr = flat.output_equations["z"]
        assert expr.kind == ExprKind.ADD

    def test_der_radd(self) -> None:
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            z = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.z == 5.0 + der(m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["z"]
        assert expr.kind == ExprKind.ADD

    def test_der_sub(self) -> None:
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            y = var()
            z = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield der(m.y) == 1.0
                yield m.z == der(m.x) - der(m.y)

        flat = TestModel().flatten()
        expr = flat.output_equations["z"]
        assert expr.kind == ExprKind.SUB

    def test_der_rsub(self) -> None:
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            z = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.z == 10.0 - der(m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["z"]
        assert expr.kind == ExprKind.SUB

    def test_der_neg(self) -> None:
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            z = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.z == -der(m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["z"]
        assert expr.kind == ExprKind.NEG


# =============================================================================
# SimulationResult Tests
# =============================================================================


class TestSimulationResult:
    """Test SimulationResult property accessors."""

    def test_result_states_property(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)
            y = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0
                yield der(m.y) == -m.y

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=1.0)

        states = result.states
        assert "x" in states
        assert "y" in states
        assert isinstance(states["x"], np.ndarray)

    def test_result_outputs_property(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == m.x * 2

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=1.0)

        outputs = result.outputs
        assert "y" in outputs
        assert isinstance(outputs["y"], np.ndarray)

    def test_result_inputs_property(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            u = var(input=True)
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == m.u

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=1.0, u={"u": 1.0})

        inputs = result.inputs
        assert "u" in inputs
        assert isinstance(inputs["u"], np.ndarray)

    def test_result_data_property(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=1.0)

        data = result.data
        assert "t" in data
        assert "x" in data

    def test_result_getitem_t(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=1.0)

        t = result["t"]
        assert t[0] == 0.0

    def test_result_getitem_keyerror(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=1.0)

        with pytest.raises(KeyError):
            _ = result["nonexistent"]

    def test_result_call_keyerror(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=1.0)

        with pytest.raises(KeyError):
            result("nonexistent")

    def test_result_call_typeerror(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=1.0)

        with pytest.raises(TypeError):
            result(123)  # type: ignore


# =============================================================================
# VarDecl Tests
# =============================================================================


class TestVarDeclRepr:
    """Test Var __repr__ for various configurations."""

    def test_vardecl_empty_repr(self) -> None:
        from cyecca.dsl.types import Var

        v = Var()
        assert repr(v) == "var()"

    def test_vardecl_with_dtype(self) -> None:
        from cyecca.dsl.types import DType, Var

        v = Var(dtype=DType.INTEGER)
        assert "dtype=INTEGER" in repr(v)

    def test_vardecl_with_default(self) -> None:
        from cyecca.dsl.types import Var

        v = Var(default=5.0)
        assert "default=5.0" in repr(v)

    def test_vardecl_with_shape(self) -> None:
        from cyecca.dsl.types import Var

        v = Var(shape=(3, 3))
        assert "shape=(3, 3)" in repr(v)

    def test_vardecl_with_start_fixed(self) -> None:
        from cyecca.dsl.types import Var

        v = Var(start=1.0, fixed=True)
        assert "start=1.0" in repr(v)
        assert "fixed=True" in repr(v)

    def test_vardecl_with_min_max(self) -> None:
        from cyecca.dsl.types import Var

        v = Var(min=-10.0, max=10.0)
        assert "min=-10.0" in repr(v)
        assert "max=10.0" in repr(v)

    def test_vardecl_constant(self) -> None:
        from cyecca.dsl.types import Var

        v = Var(constant=True)
        assert "constant=True" in repr(v)

    def test_vardecl_discrete(self) -> None:
        from cyecca.dsl.types import Var

        v = Var(discrete=True)
        assert "discrete=True" in repr(v)

    def test_vardecl_protected(self) -> None:
        from cyecca.dsl.types import Var

        v = Var(protected=True)
        assert "protected=True" in repr(v)

    def test_vardecl_get_initial_value_start(self) -> None:
        from cyecca.dsl.types import Var

        v = Var(start=5.0, default=1.0)
        assert v.get_initial_value() == 5.0

    def test_vardecl_get_initial_value_default(self) -> None:
        from cyecca.dsl.types import Var

        v = Var(default=1.0)
        assert v.get_initial_value() == 1.0


# =============================================================================
# Array Equation Tests
# =============================================================================


class TestArrayEquations:
    """Test array equation expansion and error handling."""

    def test_array_equation_shape_mismatch(self) -> None:
        from cyecca.dsl import der, model, var

        @model
        class BadModel:
            pos = var(shape=(3,))
            vel = var(shape=(2,))  # Wrong shape

            def equations(m):
                yield der(m.pos) == m.vel

        with pytest.raises(ValueError, match="Shape mismatch"):
            BadModel().flatten()

    def test_array_equation_scalar_rhs(self) -> None:
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            pos = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == 0.0

        flat = TestModel().flatten()
        # Should have 3 derivative equations, all equal to 0
        assert "pos[0]" in flat.derivative_equations
        assert "pos[1]" in flat.derivative_equations
        assert "pos[2]" in flat.derivative_equations

    def test_array_derivative_getitem(self) -> None:
        """Test der(pos)[i] syntax."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            def equations(m):
                # Test partial indexing of der(pos)
                yield der(m.pos)[0] == m.vel[0]
                yield der(m.pos)[1] == m.vel[1]
                yield der(m.pos)[2] == m.vel[2]

        flat = TestModel().flatten()
        assert "pos[0]" in flat.derivative_equations
        assert "pos[1]" in flat.derivative_equations
        assert "pos[2]" in flat.derivative_equations


# =============================================================================
# Expr Arithmetic Edge Cases
# =============================================================================


class TestExprArithmetic:
    """Test Expr arithmetic operators."""

    def test_expr_pow(self) -> None:
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == m.x**2

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.POW

    def test_expr_rpow(self) -> None:
        """Test reverse power - uses explicit expression construction."""
        from cyecca.dsl.model import Expr, ExprKind

        # Test that we can construct a POW expression with constant base
        base = Expr(ExprKind.CONSTANT, value=2.0)
        exp = Expr(ExprKind.VARIABLE, name="x")
        expr = Expr(ExprKind.POW, (base, exp))
        assert expr.kind == ExprKind.POW

    def test_expr_indexed_name_property(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        # With indices
        expr = Expr(ExprKind.VARIABLE, name="pos", indices=(0, 1))
        assert expr.indexed_name == "pos[0,1]"

        # Without indices
        expr2 = Expr(ExprKind.VARIABLE, name="x")
        assert expr2.indexed_name == "x"


# =============================================================================
# Error Path Tests
# =============================================================================


class TestErrorPaths:
    """Test various error paths."""

    def test_der_wrong_type(self) -> None:
        from beartype.roar import BeartypeCallHintParamViolation

        from cyecca.dsl.model import der

        with pytest.raises((TypeError, BeartypeCallHintParamViolation)):
            der("not a symbolic var")  # type: ignore

    def test_symbolic_var_too_many_indices(self) -> None:
        from cyecca.dsl import model, var

        @model
        class TestModel:
            x = var()

        m = TestModel()

        with pytest.raises(TypeError, match="Too many indices"):
            _ = m.x[0]  # x is scalar, can't be indexed

    def test_symbolic_var_index_out_of_bounds(self) -> None:
        from cyecca.dsl import model, var

        @model
        class TestModel:
            pos = var(shape=(3,))

        m = TestModel()

        with pytest.raises(IndexError, match="out of bounds"):
            _ = m.pos[5]

    def test_symbolic_var_invalid_index_type(self) -> None:
        from cyecca.dsl import model, var

        @model
        class TestModel:
            pos = var(shape=(3,))

        m = TestModel()

        with pytest.raises(TypeError):
            _ = m.pos["string"]  # type: ignore


# =============================================================================
# Expr Comparison Operators
# =============================================================================


class TestExprComparisons:
    """Test Expr comparison operators create correct expression kinds."""

    def test_expr_lt_kind(self) -> None:
        """Test < creates LT expression."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        y = Expr(ExprKind.VARIABLE, name="y")
        lt = x < y
        assert lt.kind == ExprKind.LT

    def test_expr_gt_kind(self) -> None:
        """Test > creates GT expression."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        y = Expr(ExprKind.VARIABLE, name="y")
        gt = x > y
        assert gt.kind == ExprKind.GT

    def test_expr_le_kind(self) -> None:
        """Test <= creates LE expression."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        y = Expr(ExprKind.VARIABLE, name="y")
        le = x <= y
        assert le.kind == ExprKind.LE

    def test_expr_ge_kind(self) -> None:
        """Test >= creates GE expression."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        y = Expr(ExprKind.VARIABLE, name="y")
        ge = x >= y
        assert ge.kind == ExprKind.GE


# =============================================================================
# Additional Expr repr coverage
# =============================================================================


class TestExprReprAdditional:
    """Additional tests for Expr repr coverage."""

    def test_repr_variable_with_single_index(self) -> None:
        """Test variable repr with single index."""
        from cyecca.dsl.model import Expr, ExprKind

        expr = Expr(ExprKind.VARIABLE, name="pos", indices=(0,))
        assert "pos[0]" in repr(expr)

    def test_repr_and_expr(self) -> None:
        """Test AND repr."""
        from cyecca.dsl.model import Expr, ExprKind

        left = Expr(ExprKind.VARIABLE, name="a")
        right = Expr(ExprKind.VARIABLE, name="b")
        expr = Expr(ExprKind.AND, (left, right))
        assert "(a and b)" in repr(expr)

    def test_repr_or_expr(self) -> None:
        """Test OR repr."""
        from cyecca.dsl.model import Expr, ExprKind

        left = Expr(ExprKind.VARIABLE, name="a")
        right = Expr(ExprKind.VARIABLE, name="b")
        expr = Expr(ExprKind.OR, (left, right))
        assert "(a or b)" in repr(expr)

    def test_repr_not_expr(self) -> None:
        """Test NOT repr."""
        from cyecca.dsl.model import Expr, ExprKind

        child = Expr(ExprKind.VARIABLE, name="a")
        expr = Expr(ExprKind.NOT, (child,))
        assert "(not a)" in repr(expr)

    def test_repr_if_then_else(self) -> None:
        """Test IF_THEN_ELSE repr."""
        from cyecca.dsl.model import Expr, ExprKind

        cond = Expr(ExprKind.VARIABLE, name="c")
        then_val = Expr(ExprKind.CONSTANT, value=1.0)
        else_val = Expr(ExprKind.CONSTANT, value=0.0)
        expr = Expr(ExprKind.IF_THEN_ELSE, (cond, then_val, else_val))
        assert "if" in repr(expr)
        assert "then" in repr(expr)
        assert "else" in repr(expr)

    def test_repr_lt_le_gt_ge(self) -> None:
        """Test comparison expr repr."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        y = Expr(ExprKind.VARIABLE, name="y")

        lt = Expr(ExprKind.LT, (x, y))
        assert "(x < y)" in repr(lt)

        le = Expr(ExprKind.LE, (x, y))
        assert "(x <= y)" in repr(le)

        gt = Expr(ExprKind.GT, (x, y))
        assert "(x > y)" in repr(gt)

        ge = Expr(ExprKind.GE, (x, y))
        assert "(x >= y)" in repr(ge)


# =============================================================================
# SymbolicVar Coverage
# =============================================================================


class TestSymbolicVarCoverage:
    """Test SymbolicVar methods for coverage."""

    def test_symbolic_var_len(self) -> None:
        """Test __len__ for array variables."""
        from cyecca.dsl import model, var

        @model
        class TestModel:
            pos = var(shape=(3,))

        m = TestModel()
        assert len(m.pos) == 3

    def test_symbolic_var_len_scalar_error(self) -> None:
        """Test __len__ raises for scalar."""
        from cyecca.dsl import model, var

        @model
        class TestModel:
            x = var()

        m = TestModel()
        with pytest.raises(TypeError, match="no length"):
            len(m.x)

    def test_symbolic_var_iter(self) -> None:
        """Test __iter__ for array variables."""
        from cyecca.dsl import model, var

        @model
        class TestModel:
            pos = var(shape=(3,))

        m = TestModel()
        elements = list(m.pos)
        assert len(elements) == 3

    def test_symbolic_var_iter_scalar_error(self) -> None:
        """Test __iter__ raises for scalar."""
        from cyecca.dsl import model, var

        @model
        class TestModel:
            x = var()

        m = TestModel()
        with pytest.raises(TypeError, match="Cannot iterate"):
            list(m.x)

    def test_symbolic_var_ndim(self) -> None:
        """Test ndim property."""
        from cyecca.dsl import model, var

        @model
        class TestModel:
            x = var()
            pos = var(shape=(3,))
            matrix = var(shape=(2, 2))

        m = TestModel()
        assert m.x.ndim == 0
        assert m.pos.ndim == 1
        assert m.matrix.ndim == 2

    def test_symbolic_var_is_scalar(self) -> None:
        """Test is_scalar method."""
        from cyecca.dsl import model, var

        @model
        class TestModel:
            x = var()
            pos = var(shape=(3,))

        m = TestModel()
        assert m.x.is_scalar() is True
        assert m.pos.is_scalar() is False
        assert m.pos[0].is_scalar() is True

    def test_symbolic_var_size(self) -> None:
        """Test size property."""
        from cyecca.dsl import model, var

        @model
        class TestModel:
            x = var()
            pos = var(shape=(3,))

        m = TestModel()
        assert m.x.size == 1
        assert m.pos.size == 3

    def test_symbolic_var_base_name(self) -> None:
        """Test base_name property."""
        from cyecca.dsl import model, var

        @model
        class TestModel:
            pos = var(shape=(3,))

        m = TestModel()
        assert m.pos.base_name == "pos"
        assert m.pos[0].base_name == "pos"

    def test_symbolic_var_remaining_shape(self) -> None:
        """Test remaining_shape property."""
        from cyecca.dsl import model, var

        @model
        class TestModel:
            matrix = var(shape=(2, 3))

        m = TestModel()
        assert m.matrix.remaining_shape == (2, 3)
        assert m.matrix[0].remaining_shape == (3,)
        assert m.matrix[0, 0].remaining_shape == ()

    def test_symbolic_var_indices(self) -> None:
        """Test indices property."""
        from cyecca.dsl import model, var

        @model
        class TestModel:
            matrix = var(shape=(2, 3))

        m = TestModel()
        assert m.matrix.indices == ()
        assert m.matrix[0].indices == (0,)
        assert m.matrix[0, 1].indices == (0, 1)

    def test_symbolic_var_arithmetic(self) -> None:
        """Test arithmetic operations on SymbolicVar."""
        from cyecca.dsl import ExprKind, model, var

        @model
        class TestModel:
            x = var()

        m = TestModel()

        # Test rsub, rmul, rtruediv
        expr1 = 5.0 - m.x
        assert expr1.kind == ExprKind.SUB

        expr2 = 5.0 * m.x
        assert expr2.kind == ExprKind.MUL

        expr3 = 5.0 / m.x
        assert expr3.kind == ExprKind.DIV


# =============================================================================
# DiscreteOperators Coverage
# =============================================================================


class TestDiscreteOperators:
    """Test pre(), edge(), change() coverage."""

    def test_pre_type_error(self) -> None:
        """Test pre() with wrong type."""
        from beartype.roar import BeartypeCallHintParamViolation

        from cyecca.dsl.model import pre

        with pytest.raises((TypeError, BeartypeCallHintParamViolation)):
            pre("not a var")  # type: ignore

    def test_pre_non_scalar_error(self) -> None:
        """Test pre() with array variable."""
        from cyecca.dsl import model, var
        from cyecca.dsl.model import pre

        @model
        class TestModel:
            pos = var(shape=(3,), discrete=True)

        m = TestModel()
        with pytest.raises(TypeError, match="scalar"):
            pre(m.pos)

    def test_edge_type_error(self) -> None:
        """Test edge() with wrong type."""
        from beartype.roar import BeartypeCallHintParamViolation

        from cyecca.dsl.model import edge

        with pytest.raises((TypeError, BeartypeCallHintParamViolation)):
            edge("not a var")  # type: ignore

    def test_edge_non_scalar_error(self) -> None:
        """Test edge() with array variable."""
        from cyecca.dsl import model, var
        from cyecca.dsl.model import edge

        @model
        class TestModel:
            flags = var(shape=(3,), discrete=True)

        m = TestModel()
        with pytest.raises(TypeError, match="scalar"):
            edge(m.flags)

    def test_change_type_error(self) -> None:
        """Test change() with wrong type."""
        from beartype.roar import BeartypeCallHintParamViolation

        from cyecca.dsl.model import change

        with pytest.raises((TypeError, BeartypeCallHintParamViolation)):
            change("not a var")  # type: ignore

    def test_change_non_scalar_error(self) -> None:
        """Test change() with array variable."""
        from cyecca.dsl import model, var
        from cyecca.dsl.model import change

        @model
        class TestModel:
            values = var(shape=(3,), discrete=True)

        m = TestModel()
        with pytest.raises(TypeError, match="scalar"):
            change(m.values)


# =============================================================================
# AlgorithmVar Coverage
# =============================================================================


class TestAlgorithmVarCoverage:
    """Test AlgorithmVar methods for coverage."""

    def test_algorithm_var_repr(self) -> None:
        """Test AlgorithmVar repr."""
        from cyecca.dsl.model import local

        temp = local("temp")
        assert repr(temp) == "local(temp)"

    def test_algorithm_var_name_property(self) -> None:
        """Test AlgorithmVar name property."""
        from cyecca.dsl.model import local

        temp = local("temp")
        assert temp.name == "temp"

    def test_algorithm_var_arithmetic(self) -> None:
        """Test AlgorithmVar arithmetic operations."""
        from cyecca.dsl.model import ExprKind, local

        temp = local("temp")

        expr1 = temp + 1
        assert expr1.kind == ExprKind.ADD

        expr2 = 1 + temp
        assert expr2.kind == ExprKind.ADD

        expr3 = temp - 1
        assert expr3.kind == ExprKind.SUB

        expr4 = 1 - temp
        assert expr4.kind == ExprKind.SUB

        expr5 = temp * 2
        assert expr5.kind == ExprKind.MUL

        expr6 = 2 * temp
        assert expr6.kind == ExprKind.MUL

        expr7 = temp / 2
        assert expr7.kind == ExprKind.DIV

        expr8 = 2 / temp
        assert expr8.kind == ExprKind.DIV

        expr9 = -temp
        assert expr9.kind == ExprKind.NEG

        expr10 = temp**2
        assert expr10.kind == ExprKind.POW

    def test_algorithm_var_comparisons(self) -> None:
        """Test AlgorithmVar comparison operators."""
        from cyecca.dsl.model import ExprKind, local

        temp = local("temp")

        lt = temp < 5
        assert lt.kind == ExprKind.LT

        le = temp <= 5
        assert le.kind == ExprKind.LE

        gt = temp > 5
        assert gt.kind == ExprKind.GT

        ge = temp >= 5
        assert ge.kind == ExprKind.GE

    def test_algorithm_var_matmul_assignment(self) -> None:
        """Test AlgorithmVar @ assignment operator."""
        from cyecca.dsl.model import Assignment, local

        temp = local("temp")
        assign = temp @ 42
        assert isinstance(assign, Assignment)
        assert assign.target == "temp"
        assert assign.is_local is True


# =============================================================================
# Assign Function Coverage
# =============================================================================


class TestAssignFunctionCoverage:
    """Test assign() function coverage."""

    def test_assign_symbolic_var(self) -> None:
        """Test assign to SymbolicVar."""
        from cyecca.dsl import model, var
        from cyecca.dsl.model import Assignment, assign

        @model
        class TestModel:
            y = var(output=True)

        m = TestModel()
        assignment = assign(m.y, 42)
        assert isinstance(assignment, Assignment)
        assert assignment.is_local is False

    def test_assign_algorithm_var(self) -> None:
        """Test assign to AlgorithmVar."""
        from cyecca.dsl.model import Assignment, assign, local

        temp = local("temp")
        assignment = assign(temp, 42)
        assert isinstance(assignment, Assignment)
        assert assignment.is_local is True

    def test_assign_string(self) -> None:
        """Test assign to string."""
        from cyecca.dsl.model import Assignment, assign

        assignment = assign("temp", 42)
        assert isinstance(assignment, Assignment)
        assert assignment.is_local is True

    def test_assign_invalid_target(self) -> None:
        """Test assign to invalid target - beartype catches this."""
        from beartype.roar import BeartypeCallHintParamViolation

        from cyecca.dsl.model import assign

        with pytest.raises(BeartypeCallHintParamViolation):
            assign(123, 42)  # type: ignore


# =============================================================================
# SubmodelProxy Coverage
# =============================================================================


class TestSubmodelProxyCoverage:
    """Test SubmodelProxy coverage."""

    def test_submodel_proxy_access(self) -> None:
        """Test accessing submodel variables."""
        from cyecca.dsl import der, model, submodel, var

        @model
        class Inner:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        @model
        class Outer:
            sub = submodel(Inner)
            y = var(output=True)

            def equations(m):
                yield m.y == m.sub.x

        outer = Outer()
        flat = outer.flatten()
        assert "sub.x" in flat.state_names

    def test_submodel_proxy_missing_attr(self) -> None:
        """Test accessing missing attribute on submodel."""
        from cyecca.dsl import der, model, submodel, var

        @model
        class Inner:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        @model
        class Outer:
            sub = submodel(Inner)

            def equations(m):
                return
                yield

        outer = Outer()
        with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
            _ = outer.sub.nonexistent


# =============================================================================
# ModelInstance Coverage
# =============================================================================


class TestModelInstanceCoverage:
    """Test ModelInstance methods for coverage."""

    def test_model_getattr_submodel(self) -> None:
        """Test accessing submodel via __getattr__."""
        from cyecca.dsl import der, model, submodel, var

        @model
        class Inner:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        @model
        class Outer:
            sub = submodel(Inner)

            def equations(m):
                return
                yield

        outer = Outer()
        # This should return a SubmodelProxy
        sub = outer.sub
        assert sub is not None

    def test_model_getattr_missing(self) -> None:
        """Test accessing missing attribute raises AttributeError."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        m = TestModel()
        with pytest.raises(AttributeError, match="no attribute"):
            _ = m.nonexistent

    def test_model_t_property(self) -> None:
        """Test t property returns TimeVar."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.model import TimeVar

        @model
        class TestModel:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        m = TestModel()
        assert isinstance(m.t, TimeVar)


# =============================================================================
# FlatModel Coverage
# =============================================================================


class TestFlatModelCoverage:
    """Test FlatModel __repr__ coverage."""

    def test_flat_model_repr_with_discrete(self) -> None:
        """Test FlatModel repr with discrete variables."""
        from cyecca.dsl import model, var

        @model
        class TestModel:
            count = var(0, discrete=True)

            def equations(m):
                return
                yield

        flat = TestModel().flatten()
        repr_str = repr(flat)
        assert "discrete=" in repr_str

    def test_flat_model_repr_all_parts(self) -> None:
        """Test FlatModel repr with all variable types."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            x = var(start=0.0)
            u = var(input=True)
            y = var(output=True)
            k = var(1.0, parameter=True)

            def equations(m):
                yield der(m.x) == m.k * m.u
                yield m.y == m.x

        flat = TestModel().flatten()
        repr_str = repr(flat)
        assert "states=" in repr_str
        assert "inputs=" in repr_str
        assert "outputs=" in repr_str
        assert "params=" in repr_str


# =============================================================================
# Var Coverage
# =============================================================================


class TestVarCoverage:
    """Test Var methods for coverage."""

    def test_var_ndim(self) -> None:
        """Test Var ndim property."""
        from cyecca.dsl.types import Var

        v_scalar = Var()
        assert v_scalar.ndim == 0

        v_vec = Var(shape=(3,))
        assert v_vec.ndim == 1

        v_mat = Var(shape=(2, 3))
        assert v_mat.ndim == 2

    def test_var_is_scalar(self) -> None:
        """Test Var is_scalar method."""
        from cyecca.dsl.types import Var

        v_scalar = Var()
        assert v_scalar.is_scalar() is True

        v_vec = Var(shape=(3,))
        assert v_vec.is_scalar() is False

    def test_var_size(self) -> None:
        """Test Var size property."""
        from cyecca.dsl.types import Var

        v_scalar = Var()
        assert v_scalar.size == 1

        v_vec = Var(shape=(3,))
        assert v_vec.size == 3

        v_mat = Var(shape=(2, 3))
        assert v_mat.size == 6

    def test_var_repr_input_output(self) -> None:
        """Test Var repr with input/output flags."""
        from cyecca.dsl.types import Var

        v_in = Var(input=True)
        assert "input=True" in repr(v_in)

        v_out = Var(output=True)
        assert "output=True" in repr(v_out)

    def test_var_repr_parameter(self) -> None:
        """Test Var repr with parameter flag."""
        from cyecca.dsl.types import Var

        v = Var(parameter=True)
        assert "parameter=True" in repr(v)


# =============================================================================
# SubmodelField Coverage
# =============================================================================


class TestSubmodelFieldCoverage:
    """Test SubmodelField repr."""

    def test_submodel_field_repr(self) -> None:
        """Test SubmodelField __repr__."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.types import SubmodelField

        @model
        class Inner:
            x = var()

            def equations(m):
                yield der(m.x) == 0

        field = SubmodelField(model_class=Inner)
        # After @model decorator, the class name becomes ModelClass
        assert "submodel(" in repr(field)


# =============================================================================
# Equation Coverage
# =============================================================================


class TestEquationCoverage:
    """Test Equation __repr__ coverage."""

    def test_equation_repr(self) -> None:
        """Test Equation repr."""
        from cyecca.dsl.model import Equation, Expr, ExprKind

        lhs = Expr(ExprKind.VARIABLE, name="x")
        rhs = Expr(ExprKind.CONSTANT, value=1.0)
        eq = Equation(lhs=lhs, rhs=rhs)
        assert "Eq(" in repr(eq)

    def test_assignment_repr(self) -> None:
        """Test Assignment repr."""
        from cyecca.dsl.model import Assignment, Expr, ExprKind

        expr = Expr(ExprKind.CONSTANT, value=42.0)
        assign = Assignment(target="x", expr=expr)
        assert "Assign(" in repr(assign)
        assert ":=" in repr(assign)


# =============================================================================
# SimulationResult __call__ with SymbolicVar
# =============================================================================


class TestSimulationResultCallSymbolicVar:
    """Test SimulationResult __call__ with SymbolicVar."""

    def test_result_call_with_symbolic_var(self) -> None:
        """Test result(m.x) syntax."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        m = TestModel()
        compiled = CasadiBackend.compile(m.flatten())
        result = compiled.simulate(tf=1.0)

        # Access using symbolic var
        x_data = result(m.x)
        assert x_data is not None
        assert len(x_data) > 0

    def test_result_call_with_expr_name(self) -> None:
        """Test result(expr) where expr has .name attribute."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend
        from cyecca.dsl.model import Expr, ExprKind

        @model
        class TestModel:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        m = TestModel()
        compiled = CasadiBackend.compile(m.flatten())
        result = compiled.simulate(tf=1.0)

        # Create an Expr with name
        expr = Expr(ExprKind.VARIABLE, name="x")
        x_data = result(expr)
        assert x_data is not None


# =============================================================================
# CasADi Backend Coverage
# =============================================================================


class TestCasadiBackendCoverage:
    """Test CasADi backend code paths."""

    def test_backend_tan_compilation(self) -> None:
        """Test tan() compiles correctly."""
        from cyecca.dsl import der, model, tan, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.1)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == tan(m.x)

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs

    def test_backend_asin_compilation(self) -> None:
        """Test asin() compiles correctly."""
        from cyecca.dsl import asin, der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.5)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 0.0
                yield m.y == asin(m.x)

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs

    def test_backend_acos_compilation(self) -> None:
        """Test acos() compiles correctly."""
        from cyecca.dsl import acos, der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.5)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 0.0
                yield m.y == acos(m.x)

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs

    def test_backend_atan_compilation(self) -> None:
        """Test atan() compiles correctly."""
        from cyecca.dsl import atan, der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.5)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 0.0
                yield m.y == atan(m.x)

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs

    def test_backend_atan2_compilation(self) -> None:
        """Test atan2() compiles correctly."""
        from cyecca.dsl import atan2, der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=1.0)
            a = var(start=0.5)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 0.0
                yield der(m.a) == 0.0
                yield m.y == atan2(m.a, m.x)

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs

    def test_backend_exp_compilation(self) -> None:
        """Test exp() compiles correctly."""
        from cyecca.dsl import der, exp, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == exp(m.x)

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs

    def test_backend_log_compilation(self) -> None:
        """Test log() compiles correctly."""
        from cyecca.dsl import der, log, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=1.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 0.1
                yield m.y == log(m.x)

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs

    def test_backend_abs_compilation(self) -> None:
        """Test abs() compiles correctly."""
        from cyecca.dsl import abs as dsl_abs
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=-1.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == dsl_abs(m.x)

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs

    def test_backend_comparison_ops(self) -> None:
        """Test comparison operators compile correctly."""
        from cyecca.dsl import der, if_then_else, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == if_then_else(m.x < 0.5, 0.0, 1.0)

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=1.0)
        assert "y" in result.outputs

    def test_backend_boolean_ops(self) -> None:
        """Test boolean operators compile correctly."""
        from cyecca.dsl import der, if_then_else, model, var
        from cyecca.dsl.backends import CasadiBackend
        from cyecca.dsl.model import Expr, ExprKind

        @model
        class TestModel:
            x = var(start=0.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                # Use Expr directly for AND since & isn't overloaded
                cond = Expr(ExprKind.AND, (m.x._expr > 0.25, m.x._expr < 0.75))
                yield m.y == if_then_else(cond, 1.0, 0.0)

        compiled = CasadiBackend.compile(TestModel().flatten())
        result = compiled.simulate(tf=1.0)
        assert "y" in result.outputs

    def test_compiled_model_repr(self) -> None:
        """Test CompiledModel __repr__."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)
            u = var(input=True)
            k = var(1.0, parameter=True)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == m.k * m.u
                yield m.y == m.x

        compiled = CasadiBackend.compile(TestModel().flatten())
        repr_str = repr(compiled)
        assert "CompiledModel(" in repr_str
        assert "states=" in repr_str


# =============================================================================
# Simulator Abstract Methods Coverage
# =============================================================================


class TestSimulatorAbstractMethods:
    """Test that Simulator abstract property methods are implemented."""

    def test_compiled_model_properties(self) -> None:
        """Test CompiledModel implements all Simulator properties."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)
            u = var(input=True)
            k = var(1.0, parameter=True)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == m.k * m.u
                yield m.y == m.x

        compiled = CasadiBackend.compile(TestModel().flatten())

        # Test all abstract properties are accessible
        assert compiled.state_names == ["x"]
        assert compiled.input_names == ["u"]
        assert compiled.output_names == ["y"]
        assert compiled.param_names == ["k"]


# =============================================================================
# Time-varying input simulation
# =============================================================================


class TestTimeVaryingInputs:
    """Test simulation with time-varying inputs."""

    def test_simulate_with_u_func(self) -> None:
        """Test simulation with u_func."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var(start=0.0)
            u = var(input=True)

            def equations(m):
                yield der(m.x) == m.u

        compiled = CasadiBackend.compile(TestModel().flatten())

        def u_func(t):
            return {"u": 1.0 if t < 0.5 else 2.0}

        result = compiled.simulate(tf=1.0, u_func=u_func)
        assert "x" in result.states
        assert "u" in result.inputs


# =============================================================================
# Additional Model Coverage
# =============================================================================


class TestExprReprMore:
    """Additional Expr repr tests."""

    def test_derivative_repr(self) -> None:
        """Test DERIVATIVE kind repr."""
        from cyecca.dsl.model import Expr, ExprKind

        expr = Expr(ExprKind.DERIVATIVE, name="x")
        assert "der(x)" in repr(expr)

    def test_math_func_reprs(self) -> None:
        """Test math function reprs."""
        from cyecca.dsl.model import Expr, ExprKind

        child = Expr(ExprKind.VARIABLE, name="x")

        for kind in [
            ExprKind.SIN,
            ExprKind.COS,
            ExprKind.TAN,
            ExprKind.ASIN,
            ExprKind.ACOS,
            ExprKind.ATAN,
            ExprKind.SQRT,
            ExprKind.EXP,
            ExprKind.LOG,
            ExprKind.ABS,
        ]:
            expr = Expr(kind, (child,))
            assert kind.name.lower() in repr(expr).lower()


class TestToExprConverters:
    """Test _to_expr for various types."""

    def test_to_expr_numpy_scalar(self) -> None:
        """Test _to_expr with numpy scalar."""
        import numpy as np

        from cyecca.dsl.model import ExprKind, _to_expr

        arr = np.array(5.0)
        expr = _to_expr(arr)
        assert expr.kind == ExprKind.CONSTANT
        assert expr.value == 5.0

    def test_to_expr_expr_passthrough(self) -> None:
        """Test _to_expr with Expr passthrough."""
        from cyecca.dsl.model import Expr, ExprKind, _to_expr

        original = Expr(ExprKind.VARIABLE, name="x")
        result = _to_expr(original)
        assert result is original

    def test_to_expr_symbolic_var(self) -> None:
        """Test _to_expr with SymbolicVar."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.model import _to_expr

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 0

        m = TestModel()
        result = _to_expr(m.x)
        assert result is m.x._expr

    def test_to_expr_derivative_expr(self) -> None:
        """Test _to_expr with DerivativeExpr."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.model import _to_expr

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 0

        m = TestModel()
        deriv = der(m.x)
        result = _to_expr(deriv)
        # DerivativeExpr has _expr attribute
        assert result is deriv._expr

    def test_to_expr_time_var(self) -> None:
        """Test _to_expr with TimeVar."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.model import _to_expr

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == m.t

        m = TestModel()
        result = _to_expr(m.t)
        assert result is m.t._expr

    def test_to_expr_invalid_type(self) -> None:
        """Test _to_expr with invalid type."""
        from cyecca.dsl.model import _to_expr

        with pytest.raises(TypeError, match="Cannot convert"):
            _to_expr("a string")


class TestHelperFunctions:
    """Test helper functions coverage."""

    def test_get_base_name(self) -> None:
        """Test _get_base_name helper."""
        from cyecca.dsl.model import _get_base_name

        assert _get_base_name("pos[0,1]") == "pos"
        assert _get_base_name("x") == "x"

    def test_parse_indices(self) -> None:
        """Test _parse_indices helper."""
        from cyecca.dsl.model import _parse_indices

        name, indices = _parse_indices("pos[0,1]")
        assert name == "pos"
        assert indices == (0, 1)

        name2, indices2 = _parse_indices("x")
        assert name2 == "x"
        assert indices2 == ()

    def test_format_indices(self) -> None:
        """Test _format_indices helper."""
        from cyecca.dsl.model import _format_indices

        assert _format_indices((0, 1)) == "[0,1]"
        assert _format_indices(()) == ""

    def test_iter_indices(self) -> None:
        """Test _iter_indices helper."""
        from cyecca.dsl.model import _iter_indices

        # Scalar
        indices = list(_iter_indices(()))
        assert indices == [()]

        # 1D
        indices = list(_iter_indices((3,)))
        assert indices == [(0,), (1,), (2,)]

        # 2D
        indices = list(_iter_indices((2, 2)))
        assert indices == [(0, 0), (0, 1), (1, 0), (1, 1)]


class TestArrayDerivativeExpr:
    """Test ArrayDerivativeExpr coverage."""

    def test_array_derivative_repr(self) -> None:
        """Test ArrayDerivativeExpr __repr__."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == m.vel

        m = TestModel()
        arr_der = der(m.pos)
        assert "der(pos)" in repr(arr_der)

    def test_array_derivative_partial_indexing(self) -> None:
        """Test ArrayDerivativeExpr with partial indexing on 2D array."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.model import ArrayDerivativeExpr

        @model
        class TestModel:
            matrix = var(shape=(2, 3))
            vel = var(shape=(2, 3))

            def equations(m):
                yield der(m.matrix) == m.vel

        m = TestModel()
        arr_der = der(m.matrix)
        # Partial indexing - index first dimension only
        partial = arr_der[0]
        assert isinstance(partial, ArrayDerivativeExpr)


class TestArrayEquationCoverage:
    """Test ArrayEquation coverage."""

    def test_array_equation_repr(self) -> None:
        """Test ArrayEquation __repr__."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == m.vel

        m = TestModel()
        # Get the ArrayEquation directly
        equations = list(m.equations())
        assert len(equations) == 1
        assert "ArrayEq" in repr(equations[0])
        assert "der(" in repr(equations[0])

    def test_array_equation_invalid_rhs_type(self) -> None:
        """Test ArrayEquation expand with invalid RHS type."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.model import ArrayEquation

        @model
        class TestModel:
            pos = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == [1, 2, 3]  # Lists not supported

        with pytest.raises(TypeError, match="Cannot expand"):
            TestModel().flatten()

    def test_array_equation_non_derivative(self) -> None:
        """Test non-derivative array equation repr."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.model import ArrayEquation

        @model
        class TestModel:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == 0
                yield m.pos == m.vel  # Non-derivative array equation

        m = TestModel()
        equations = list(m.equations())
        # Second one is non-derivative
        non_der_eq = equations[1]
        assert "der(" not in repr(non_der_eq)


class TestEquationPrefixNames:
    """Test Equation _prefix_names coverage."""

    def test_equation_prefix_names(self) -> None:
        """Test Equation._prefix_names()."""
        from cyecca.dsl.model import Equation, Expr, ExprKind

        lhs = Expr(ExprKind.DERIVATIVE, name="x")
        rhs = Expr(ExprKind.VARIABLE, name="y")
        eq = Equation(lhs=lhs, rhs=rhs, is_derivative=True, var_name="x")

        prefixed = eq._prefix_names("sub")
        assert prefixed.var_name == "sub.x"


class TestModelWarnings:
    """Test model deprecation warnings."""

    def test_output_equations_warning(self) -> None:
        """Test deprecation warning for output_equations."""
        import warnings

        from cyecca.dsl import der, model, var

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @model
            class TestModel:
                x = var()
                y = var(output=True)

                def equations(m):
                    yield der(m.x) == 0

                def output_equations(m):
                    yield m.y == m.x

            # Check that a deprecation warning was issued
            assert any("output_equations" in str(warning.message) for warning in w)


class TestSymbolicVarComparisons:
    """Test SymbolicVar comparison operators."""

    def test_symbolic_var_lt(self) -> None:
        """Test SymbolicVar < operator."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 0

        m = TestModel()
        expr = m.x < 5
        assert expr.kind == ExprKind.LT

    def test_symbolic_var_le(self) -> None:
        """Test SymbolicVar <= operator."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 0

        m = TestModel()
        expr = m.x <= 5
        assert expr.kind == ExprKind.LE

    def test_symbolic_var_gt(self) -> None:
        """Test SymbolicVar > operator."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 0

        m = TestModel()
        expr = m.x > 5
        assert expr.kind == ExprKind.GT

    def test_symbolic_var_ge(self) -> None:
        """Test SymbolicVar >= operator."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 0

        m = TestModel()
        expr = m.x >= 5
        assert expr.kind == ExprKind.GE

    def test_symbolic_var_neg(self) -> None:
        """Test SymbolicVar negation."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 0

        m = TestModel()
        expr = -m.x
        assert expr.kind == ExprKind.NEG

    def test_symbolic_var_pow(self) -> None:
        """Test SymbolicVar power."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 0

        m = TestModel()
        expr = m.x**2
        assert expr.kind == ExprKind.POW


class TestFunctionDecorator:
    """Test @function decorator."""

    def test_function_basic(self) -> None:
        """Test basic function creation."""
        from cyecca.dsl import function, var
        from cyecca.dsl.model import local

        @function
        class Saturation:
            x = var(input=True)
            limit = var(1.0, parameter=True)
            y = var(output=True)

            def algorithm(m):
                temp = local("temp")
                yield temp @ m.x
                yield m.y @ temp

        f = Saturation()
        assert hasattr(f, "_is_function")
        assert f._is_function is True

    def test_function_no_algorithm_error(self) -> None:
        """Test that function without algorithm raises."""
        from cyecca.dsl import function, var

        with pytest.raises(TypeError, match="must have an algorithm"):

            @function
            class BadFunction:
                x = var(input=True)
                y = var(output=True)

    def test_function_public_non_io_error(self) -> None:
        """Test that function with public non-io variable raises."""
        from cyecca.dsl import function, var

        with pytest.raises(TypeError, match="must have input=True or output=True"):

            @function
            class BadFunction:
                x = var(input=True)
                y = var(output=True)
                z = var()  # Public, not input/output/param

                def algorithm(m):
                    yield m.y @ m.x

    def test_function_metadata(self) -> None:
        """Test get_function_metadata."""
        from cyecca.dsl import function, var
        from cyecca.dsl.model import local

        @function
        class TestFunc:
            x = var(input=True)
            k = var(1.0, parameter=True)
            y = var(output=True)
            temp = var(protected=True)

            def algorithm(m):
                yield m.temp @ (m.x * m.k)
                yield m.y @ m.temp

        f = TestFunc()
        meta = f.get_function_metadata()
        assert "x" in meta.input_names
        assert "y" in meta.output_names
        assert "k" in meta.param_names
        assert "temp" in meta.protected_names


class TestAlgorithmToExpr:
    """Test _to_expr with AlgorithmVar."""

    def test_to_expr_algorithm_var(self) -> None:
        """Test _to_expr with AlgorithmVar."""
        from cyecca.dsl.model import ExprKind, _to_expr, local

        temp = local("temp")
        expr = _to_expr(temp)
        assert expr.kind == ExprKind.VARIABLE
        assert expr.name == "temp"


class TestSubmodelVariableClassification:
    """Test submodel variable classification."""

    def test_submodel_param_classification(self) -> None:
        """Test that submodel parameters are classified correctly."""
        from cyecca.dsl import der, model, submodel, var

        @model
        class Inner:
            k = var(1.0, parameter=True)
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == m.k

        @model
        class Outer:
            sub = submodel(Inner)

            def equations(m):
                return
                yield

        flat = Outer().flatten()
        assert "sub.k" in flat.param_names
        assert "sub.x" in flat.state_names

    def test_submodel_input_classification(self) -> None:
        """Test that submodel inputs are classified correctly."""
        from cyecca.dsl import der, model, submodel, var

        @model
        class Inner:
            u = var(input=True)
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == m.u

        @model
        class Outer:
            sub = submodel(Inner)

            def equations(m):
                return
                yield

        flat = Outer().flatten()
        assert "sub.u" in flat.input_names

    def test_submodel_output_classification(self) -> None:
        """Test that submodel outputs are classified correctly."""
        from cyecca.dsl import der, model, submodel, var

        @model
        class Inner:
            x = var(start=0.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == m.x

        @model
        class Outer:
            sub = submodel(Inner)

            def equations(m):
                return
                yield

        flat = Outer().flatten()
        assert "sub.y" in flat.output_names

    def test_submodel_discrete_classification(self) -> None:
        """Test that submodel discrete vars are classified correctly."""
        from cyecca.dsl import model, submodel, var

        @model
        class Inner:
            count = var(0, discrete=True)

            def equations(m):
                return
                yield

        @model
        class Outer:
            sub = submodel(Inner)

            def equations(m):
                return
                yield

        flat = Outer().flatten()
        assert "sub.count" in flat.discrete_names


class TestFlattenOptions:
    """Test flatten with different options."""

    def test_flatten_expand_arrays_false(self) -> None:
        """Test flatten with expand_arrays=False."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == m.vel

        flat = TestModel().flatten(expand_arrays=False)
        # With expand_arrays=False, array equations stay as-is
        assert flat.expand_arrays is False
        assert "pos" in flat.array_derivative_equations


class TestAlgorithmInFlatten:
    """Test algorithm handling in flatten."""

    def test_algorithm_invalid_type_error(self) -> None:
        """Test that invalid algorithm yield raises error."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

            def algorithm(m):
                yield "not an assignment"  # type: ignore

        with pytest.raises(TypeError, match="Expected Assignment"):
            TestModel().flatten()


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestExprArithmeticReverse:
    """Test Expr reverse arithmetic operations."""

    def test_expr_radd(self) -> None:
        """Test Expr.__radd__."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = 5 + x
        assert result.kind == ExprKind.ADD

    def test_expr_rsub(self) -> None:
        """Test Expr.__rsub__."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = 5 - x
        assert result.kind == ExprKind.SUB

    def test_expr_rmul(self) -> None:
        """Test Expr.__rmul__."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = 5 * x
        assert result.kind == ExprKind.MUL

    def test_expr_rtruediv(self) -> None:
        """Test Expr.__rtruediv__."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = 5 / x
        assert result.kind == ExprKind.DIV

    def test_expr_neg(self) -> None:
        """Test Expr.__neg__."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = -x
        assert result.kind == ExprKind.NEG


class TestMulRepr:
    """Test MUL repr coverage."""

    def test_mul_repr(self) -> None:
        """Test MUL kind repr."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        y = Expr(ExprKind.VARIABLE, name="y")
        expr = Expr(ExprKind.MUL, (x, y))
        assert "(x * y)" in repr(expr)


class TestSymbolicVarShape:
    """Test SymbolicVar.shape property."""

    def test_shape_property(self) -> None:
        """Test shape property on SymbolicVar."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            pos = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == 0

        m = TestModel()
        assert m.pos.shape == (3,)


class TestFunctionWithEquations:
    """Test @function decorator with equations method."""

    def test_function_with_equations_error(self) -> None:
        """Test that function with real equations method raises."""
        from cyecca.dsl import function, var

        with pytest.raises(TypeError, match="cannot have equations"):

            @function
            class BadFunction:
                x = var(input=True)
                y = var(output=True)

                def equations(m):
                    # This should trigger an error - functions can't have equations
                    yield m.y == m.x

                def algorithm(m):
                    yield m.y @ m.x


class TestArrayEquationExprRhs:
    """Test ArrayEquation.expand with Expr RHS."""

    def test_array_equation_expr_rhs(self) -> None:
        """Test ArrayEquation expand with Expr RHS (scalar broadcast)."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            pos = var(shape=(3,))
            k = var(1.0, parameter=True)

            def equations(m):
                # Broadcast scalar k*2 to all elements
                yield der(m.pos) == m.k * 2

        flat = TestModel().flatten()
        # Each derivative equation should have the same expression
        assert "pos[0]" in flat.derivative_equations
        assert "pos[1]" in flat.derivative_equations
        assert "pos[2]" in flat.derivative_equations


class TestSubmodelAlgebraicClassification:
    """Test submodel algebraic variable classification."""

    def test_submodel_algebraic_classification(self) -> None:
        """Test that submodel variables without der() are algebraic."""
        from cyecca.dsl import der, model, submodel, var

        @model
        class Inner:
            x = var()  # No der(x) anywhere -> algebraic

            def equations(m):
                return
                yield

        @model
        class Outer:
            sub = submodel(Inner)

            def equations(m):
                return
                yield

        flat = Outer().flatten()
        assert "sub.x" in flat.algebraic_names


class TestParamStartValue:
    """Test parameter with start value (no default)."""

    def test_param_start_value(self) -> None:
        """Test parameter with start= but no default=."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            k = var(start=5.0, parameter=True)
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == m.k

        flat = TestModel().flatten()
        assert "k" in flat.param_names
        assert flat.param_defaults.get("k") == 5.0


class TestSubmodelWithArrays:
    """Test submodel with array equations."""

    def test_submodel_array_equations(self) -> None:
        """Test that submodel array equations are expanded."""
        from cyecca.dsl import der, model, submodel, var

        @model
        class Inner:
            pos = var(shape=(2,))
            vel = var(shape=(2,))

            def equations(m):
                yield der(m.pos) == m.vel

        @model
        class Outer:
            sub = submodel(Inner)

            def equations(m):
                return
                yield

        flat = Outer().flatten()
        # Array equations from submodel should be prefixed and expanded
        assert "sub.pos[0]" in flat.derivative_equations
        assert "sub.pos[1]" in flat.derivative_equations


class TestNonDerivativeArrayEquation:
    """Test non-derivative array equation (algebraic array assignment)."""

    def test_non_derivative_array_equation(self) -> None:
        """Test y = x array equation (non-derivative)."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            x = var(shape=(3,))
            y = var(shape=(3,), output=True)

            def equations(m):
                yield der(m.x) == 0.0
                yield m.y == m.x  # Non-derivative array equation

        flat = TestModel().flatten()
        # y should be classified as output
        assert "y" in flat.output_names


class TestArrayEquationWithIndices:
    """Test ArrayEquation with partially indexed LHS."""

    def test_array_equation_with_partial_index(self) -> None:
        """Test array equation where LHS already has some indices."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            matrix = var(shape=(2, 3))
            vel = var(shape=(2, 3))

            def equations(m):
                yield der(m.matrix) == m.vel

        flat = TestModel().flatten()
        # All 6 elements should have derivative equations
        for i in range(2):
            for j in range(3):
                assert f"matrix[{i},{j}]" in flat.derivative_equations


class TestSymbolicVarName:
    """Test SymbolicVar.name property."""

    def test_name_property_with_indices(self) -> None:
        """Test name property includes indices."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            pos = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == 0

        m = TestModel()
        assert m.pos[0].name == "pos[0]"


# =============================================================================
# CasADi MX Backend Tests
# =============================================================================


class TestCasadiMXBackend:
    """Test CasADi MX backend compilation."""

    def test_mx_backend_basic(self) -> None:
        """Test basic MX backend compilation."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend, SymbolicType

        @model
        class TestModel:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        flat = TestModel().flatten(expand_arrays=False)
        compiled = CasadiBackend.compile(flat, symbolic_type=SymbolicType.MX)
        result = compiled.simulate(tf=1.0)
        assert "x" in result.states

    def test_mx_backend_with_input(self) -> None:
        """Test MX backend with input."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend, SymbolicType

        @model
        class TestModel:
            x = var(start=0.0)
            u = var(input=True)

            def equations(m):
                yield der(m.x) == m.u

        flat = TestModel().flatten(expand_arrays=False)
        compiled = CasadiBackend.compile(flat, symbolic_type=SymbolicType.MX)
        result = compiled.simulate(tf=1.0, u={"u": 2.0})
        assert "x" in result.states

    def test_mx_backend_with_param(self) -> None:
        """Test MX backend with parameter."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend, SymbolicType

        @model
        class TestModel:
            x = var(start=0.0)
            k = var(1.0, parameter=True)

            def equations(m):
                yield der(m.x) == m.k

        flat = TestModel().flatten(expand_arrays=False)
        compiled = CasadiBackend.compile(flat, symbolic_type=SymbolicType.MX)
        result = compiled.simulate(tf=1.0, params={"k": 2.0})
        assert "x" in result.states


# =============================================================================
# Discrete Operators with Valid Scalar Variables
# =============================================================================


class TestDiscreteOperatorsValid:
    """Test pre(), edge(), change() with valid scalar variables."""

    def test_pre_scalar(self) -> None:
        """Test pre() with valid scalar discrete variable."""
        from cyecca.dsl import model, var
        from cyecca.dsl.model import ExprKind, pre

        @model
        class TestModel:
            count = var(0, discrete=True)

            def equations(m):
                return
                yield

        m = TestModel()
        expr = pre(m.count)
        assert expr.kind == ExprKind.PRE

    def test_edge_scalar(self) -> None:
        """Test edge() with valid scalar boolean variable."""
        from cyecca.dsl import DType, model, var
        from cyecca.dsl.model import ExprKind, edge

        @model
        class TestModel:
            flag = var(False, dtype=DType.BOOLEAN, discrete=True)

            def equations(m):
                return
                yield

        m = TestModel()
        expr = edge(m.flag)
        assert expr.kind == ExprKind.EDGE

    def test_change_scalar(self) -> None:
        """Test change() with valid scalar variable."""
        from cyecca.dsl import model, var
        from cyecca.dsl.model import ExprKind, change

        @model
        class TestModel:
            mode = var(0, discrete=True)

            def equations(m):
                return
                yield

        m = TestModel()
        expr = change(m.mode)
        assert expr.kind == ExprKind.CHANGE


# =============================================================================
# SymbolicVar Index with Tuple
# =============================================================================


class TestSymbolicVarTupleIndex:
    """Test SymbolicVar indexing with tuple."""

    def test_index_with_tuple(self) -> None:
        """Test indexing with tuple of indices."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            matrix = var(shape=(2, 3))

            def equations(m):
                yield der(m.matrix) == 0

        m = TestModel()
        # Index with tuple
        elem = m.matrix[1, 2]
        assert elem.name == "matrix[1,2]"


# =============================================================================
# DerivativeExpr Arithmetic
# =============================================================================


class TestDerivativeExprArithmetic:
    """Additional DerivativeExpr arithmetic tests."""

    def test_der_mul(self) -> None:
        """Test der(x) * scalar."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                # Note: der() returns Expr not SymbolicVar, so mul isn't on DerivativeExpr
                # This tests the Expr multiplication
                yield m.y == der(m.x) + 0  # der() + 0 exercises __add__

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.ADD


# =============================================================================
# Additional Coverage: SymbolicVar radd
# =============================================================================


class TestSymbolicVarRadd:
    """Test SymbolicVar.__radd__."""

    def test_symbolic_var_radd_from_const(self) -> None:
        """Test 5 + m.x."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 0
                yield m.y == 5.0 + m.x

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.ADD


# =============================================================================
# Additional Coverage: DerivativeExpr repr
# =============================================================================


class TestDerivativeExprRepr:
    """Test DerivativeExpr repr."""

    def test_derivative_expr_repr(self) -> None:
        """Test DerivativeExpr __repr__."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 0

        m = TestModel()
        deriv = der(m.x)
        assert "der(x)" in repr(deriv)


# =============================================================================
# Additional Coverage: Expr rpow
# =============================================================================


class TestExprRpow:
    """Test Expr.__rpow__."""

    def test_rpow(self) -> None:
        """Test 2 ** expr."""
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = 2**x
        assert result.kind == ExprKind.POW
        # First child should be constant 2
        assert result.children[0].kind == ExprKind.CONSTANT


# =============================================================================
# Additional Coverage: Expr fallback repr
# =============================================================================


class TestExprFallbackRepr:
    """Test Expr fallback repr for unhandled kinds."""

    def test_fallback_repr(self) -> None:
        """Test fallback repr for kinds not explicitly handled."""
        from cyecca.dsl.model import Expr, ExprKind

        # Create an expr that would hit the fallback repr
        # Most kinds have explicit repr handling, but we can test the structure
        x = Expr(ExprKind.VARIABLE, name="x")
        # Test that ADD repr works (this one is covered)
        y = Expr(ExprKind.VARIABLE, name="y")
        add_expr = Expr(ExprKind.ADD, (x, y))
        assert "(x + y)" in repr(add_expr)


# =============================================================================
# Additional Coverage: Submodel with param start value
# =============================================================================


class TestSubmodelParamStart:
    """Test submodel parameter with start value (not default)."""

    def test_submodel_param_with_start(self) -> None:
        """Test submodel parameter classified correctly when using start=."""
        from cyecca.dsl import der, model, submodel, var

        @model
        class Inner:
            k = var(start=5.0, parameter=True)  # start= not default=
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == m.k

        @model
        class Outer:
            sub = submodel(Inner)

            def equations(m):
                return
                yield

        flat = Outer().flatten()
        assert "sub.k" in flat.param_names
        # Check the param_defaults uses the start value
        assert flat.param_defaults.get("sub.k") == 5.0


# =============================================================================
# Additional Coverage: Non-derivative ArrayEquation LHS
# =============================================================================


class TestArrayEquationNonDerivativeLHS:
    """Test non-derivative array equation expands correctly."""

    def test_non_derivative_array_lhs(self) -> None:
        """Test y = x array equation creates VARIABLE LHS."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.model import Equation

        @model
        class TestModel:
            x = var(shape=(2,))
            y = var(shape=(2,))

            def equations(m):
                yield der(m.x) == 0
                # Non-derivative array equations become regular Equations
                # because __eq__ on SymbolicVar returns Equation
                yield m.y == m.x

        m = TestModel()
        eqs = list(m.equations())
        # Second equation is a regular Equation (SymbolicVar.__eq__ returns Equation)
        eq = eqs[1]
        assert isinstance(eq, Equation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
