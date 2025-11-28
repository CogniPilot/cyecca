"""
Tests for cyecca.dsl.model module.

Covers: Expr, ExprKind, SymbolicVar, TimeVar, DerivativeExpr, ArrayDerivativeExpr,
        Equation, Assignment, model/function/block decorators, submodel, der, flatten,
        local, assign, pre, edge, change
"""

import numpy as np
import pytest

# =============================================================================
# Expr Tests
# =============================================================================


class TestExpr:
    """Test Expr class."""

    def test_constant_expr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        expr = Expr(ExprKind.CONSTANT, value=3.14)
        assert expr.kind == ExprKind.CONSTANT
        assert expr.value == 3.14
        assert "3.14" in repr(expr)

    def test_variable_expr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        expr = Expr(ExprKind.VARIABLE, name="x")
        assert expr.kind == ExprKind.VARIABLE
        assert expr.name == "x"
        assert "x" in repr(expr)

    def test_time_expr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        expr = Expr(ExprKind.TIME)
        assert repr(expr) == "t"

    def test_indexed_variable_expr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        expr = Expr(ExprKind.VARIABLE, name="pos", indices=(0, 1))
        assert expr.indexed_name == "pos[0,1]"
        assert "pos[0,1]" in repr(expr)


class TestExprArithmetic:
    """Test Expr arithmetic operators."""

    def test_add(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        y = Expr(ExprKind.VARIABLE, name="y")
        result = x + y
        assert result.kind == ExprKind.ADD
        assert "(x + y)" in repr(result)

    def test_sub(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        y = Expr(ExprKind.VARIABLE, name="y")
        result = x - y
        assert result.kind == ExprKind.SUB
        assert "(x - y)" in repr(result)

    def test_mul(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        y = Expr(ExprKind.VARIABLE, name="y")
        result = x * y
        assert result.kind == ExprKind.MUL
        assert "(x * y)" in repr(result)

    def test_div(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        y = Expr(ExprKind.VARIABLE, name="y")
        result = x / y
        assert result.kind == ExprKind.DIV
        assert "(x / y)" in repr(result)

    def test_pow(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = x**2
        assert result.kind == ExprKind.POW
        assert "x **" in repr(result)

    def test_neg(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = -x
        assert result.kind == ExprKind.NEG
        assert "(-x)" in repr(result)

    def test_radd(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = 5 + x
        assert result.kind == ExprKind.ADD

    def test_rsub(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = 5 - x
        assert result.kind == ExprKind.SUB

    def test_rmul(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = 5 * x
        assert result.kind == ExprKind.MUL

    def test_rtruediv(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = 5 / x
        assert result.kind == ExprKind.DIV

    def test_rpow(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = 2**x
        assert result.kind == ExprKind.POW


class TestExprComparisons:
    """Test Expr comparison operators."""

    def test_lt(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = x < 5
        assert result.kind == ExprKind.LT
        assert "x <" in repr(result)

    def test_le(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = x <= 5
        assert result.kind == ExprKind.LE
        assert "x <=" in repr(result)

    def test_gt(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = x > 5
        assert result.kind == ExprKind.GT
        assert "x >" in repr(result) and ">=" not in repr(result)

    def test_ge(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        result = x >= 5
        assert result.kind == ExprKind.GE
        assert "x >=" in repr(result)

    def test_eq_ne(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        eq = Expr(ExprKind.EQ, (x, Expr(ExprKind.CONSTANT, value=0)))
        ne = Expr(ExprKind.NE, (x, Expr(ExprKind.CONSTANT, value=0)))
        assert "(x == 0)" in repr(eq)
        assert "(x != 0)" in repr(ne)


class TestExprReprMathFunctions:
    """Test Expr repr for math functions."""

    def test_math_func_reprs(self) -> None:
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

    def test_atan2_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        y = Expr(ExprKind.VARIABLE, name="y")
        x = Expr(ExprKind.VARIABLE, name="x")
        expr = Expr(ExprKind.ATAN2, (y, x))
        assert "atan2(y, x)" in repr(expr)

    def test_derivative_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        expr = Expr(ExprKind.DERIVATIVE, name="x")
        assert "der(x)" in repr(expr)


class TestExprReprBoolean:
    """Test Expr repr for boolean expressions."""

    def test_and_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        a = Expr(ExprKind.VARIABLE, name="a")
        b = Expr(ExprKind.VARIABLE, name="b")
        expr = Expr(ExprKind.AND, (a, b))
        assert "(a and b)" in repr(expr)

    def test_or_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        a = Expr(ExprKind.VARIABLE, name="a")
        b = Expr(ExprKind.VARIABLE, name="b")
        expr = Expr(ExprKind.OR, (a, b))
        assert "(a or b)" in repr(expr)

    def test_not_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        a = Expr(ExprKind.VARIABLE, name="a")
        expr = Expr(ExprKind.NOT, (a,))
        assert "(not a)" in repr(expr)

    def test_if_then_else_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        cond = Expr(ExprKind.VARIABLE, name="c")
        then_val = Expr(ExprKind.CONSTANT, value=1.0)
        else_val = Expr(ExprKind.CONSTANT, value=0.0)
        expr = Expr(ExprKind.IF_THEN_ELSE, (cond, then_val, else_val))
        assert "if" in repr(expr)
        assert "then" in repr(expr)
        assert "else" in repr(expr)


class TestExprReprDiscrete:
    """Test Expr repr for discrete operators."""

    def test_pre_edge_change_repr(self) -> None:
        from cyecca.dsl.model import Expr, ExprKind

        assert "pre(x)" in repr(Expr(ExprKind.PRE, name="x"))
        assert "edge(x)" in repr(Expr(ExprKind.EDGE, name="x"))
        assert "change(x)" in repr(Expr(ExprKind.CHANGE, name="x"))


# =============================================================================
# TimeVar Tests
# =============================================================================


class TestTimeVar:
    """Test TimeVar class."""

    def test_repr(self) -> None:
        from cyecca.dsl.model import TimeVar

        t = TimeVar()
        assert repr(t) == "t"

    def test_arithmetic(self) -> None:
        from cyecca.dsl.model import ExprKind, TimeVar

        t = TimeVar()
        assert (t + 1).kind == ExprKind.ADD
        assert (1 + t).kind == ExprKind.ADD
        assert (t - 1).kind == ExprKind.SUB
        assert (1 - t).kind == ExprKind.SUB
        assert (t * 2).kind == ExprKind.MUL
        assert (2 * t).kind == ExprKind.MUL
        assert (t / 2).kind == ExprKind.DIV
        assert (1 / t).kind == ExprKind.DIV


# =============================================================================
# SymbolicVar Tests
# =============================================================================


class TestSymbolicVar:
    """Test SymbolicVar class."""

    def test_basic_properties(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            x = var()

            @equations
            def _(m):
                der(m.x) == 0

        m = M()
        assert m.x.name == "x"
        assert m.x.base_name == "x"
        assert m.x.is_scalar() is True
        assert m.x.ndim == 0
        assert m.x.size == 1
        assert m.x.shape == ()
        assert m.x.indices == ()

    def test_array_properties(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            pos = var(shape=(3,))

            @equations
            def _(m):
                der(m.pos) == 0

        m = M()
        assert m.pos.is_scalar() is False
        assert m.pos.ndim == 1
        assert m.pos.size == 3
        assert m.pos.shape == (3,)
        assert len(m.pos) == 3

    def test_indexing(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            pos = var(shape=(3,))
            matrix = var(shape=(2, 3))

            @equations
            def _(m):
                der(m.pos) == 0
                der(m.matrix) == 0

        m = M()
        # 1D indexing
        assert m.pos[0].name == "pos[0]"
        assert m.pos[0].is_scalar() is True
        assert m.pos[0].indices == (0,)

        # 2D indexing
        assert m.matrix[0].remaining_shape == (3,)
        assert m.matrix[0, 1].name == "matrix[0,1]"
        assert m.matrix[0][1].is_scalar() is True

    def test_iteration(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            pos = var(shape=(3,))

            @equations
            def _(m):
                der(m.pos) == 0

        m = M()
        elements = list(m.pos)
        assert len(elements) == 3
        assert elements[0].indices == (0,)

    def test_index_errors(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            x = var()
            pos = var(shape=(3,))

            @equations
            def _(m):
                der(m.x) == 0
                der(m.pos) == 0

        m = M()
        with pytest.raises(TypeError, match="Too many indices"):
            _ = m.x[0]
        with pytest.raises(IndexError, match="out of bounds"):
            _ = m.pos[5]
        with pytest.raises(TypeError):
            _ = m.pos["str"]  # type: ignore

    def test_arithmetic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, model, var

        @model
        class M:
            x = var()

            @equations
            def _(m):
                der(m.x) == 0

        m = M()
        assert (m.x + 1).kind == ExprKind.ADD
        assert (1 + m.x).kind == ExprKind.ADD
        assert (m.x - 1).kind == ExprKind.SUB
        assert (1 - m.x).kind == ExprKind.SUB
        assert (m.x * 2).kind == ExprKind.MUL
        assert (2 * m.x).kind == ExprKind.MUL
        assert (m.x / 2).kind == ExprKind.DIV
        assert (2 / m.x).kind == ExprKind.DIV
        assert (m.x**2).kind == ExprKind.POW
        assert (-m.x).kind == ExprKind.NEG

    def test_comparisons(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, model, var

        @model
        class M:
            x = var()

            @equations
            def _(m):
                der(m.x) == 0

        m = M()
        assert (m.x < 5).kind == ExprKind.LT
        assert (m.x <= 5).kind == ExprKind.LE
        assert (m.x > 5).kind == ExprKind.GT
        assert (m.x >= 5).kind == ExprKind.GE

    def test_len_iter_errors_scalar(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            x = var()

            @equations
            def _(m):
                der(m.x) == 0

        m = M()
        with pytest.raises(TypeError, match="no length"):
            len(m.x)
        with pytest.raises(TypeError, match="Cannot iterate"):
            list(m.x)


# =============================================================================
# DerivativeExpr Tests
# =============================================================================


class TestDerivativeExpr:
    """Test DerivativeExpr class."""

    def test_der_repr(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            x = var()

            @equations
            def _(m):
                der(m.x) == 0

        m = M()
        deriv = der(m.x)
        assert "der(x)" in repr(deriv)

    def test_der_arithmetic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, model, var

        @model
        class M:
            x = var()
            y = var()
            z = var(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                der(m.y) == 1.0
                m.z == der(m.x) + der(m.y)

        flat = M().flatten()
        assert flat.output_equations["z"].kind == ExprKind.ADD


class TestArrayDerivativeExpr:
    """Test ArrayDerivativeExpr class."""

    def test_array_der_repr(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            @equations
            def _(m):
                der(m.pos) == m.vel

        m = M()
        arr_der = der(m.pos)
        assert "der(pos)" in repr(arr_der)

    def test_array_der_indexing(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            @equations
            def _(m):
                der(m.pos)[0] == m.vel[0]
                der(m.pos)[1] == m.vel[1]
                der(m.pos)[2] == m.vel[2]

        flat = M().flatten()
        assert "pos[0]" in flat.derivative_equations


# =============================================================================
# Equation and Assignment Tests
# =============================================================================


class TestEquation:
    """Test Equation class."""

    def test_equation_repr(self) -> None:
        from cyecca.dsl.model import Equation, Expr, ExprKind

        lhs = Expr(ExprKind.VARIABLE, name="x")
        rhs = Expr(ExprKind.CONSTANT, value=1.0)
        eq = Equation(lhs=lhs, rhs=rhs)
        assert "Eq(" in repr(eq)


class TestAssignment:
    """Test Assignment class."""

    def test_assignment_repr(self) -> None:
        from cyecca.dsl.model import Assignment, Expr, ExprKind

        expr = Expr(ExprKind.CONSTANT, value=42.0)
        assign = Assignment(target="x", expr=expr)
        assert "Assign(" in repr(assign)
        assert ":=" in repr(assign)


# =============================================================================
# AlgorithmVar Tests
# =============================================================================


class TestAlgorithmVar:
    """Test AlgorithmVar (local) class."""

    def test_local_repr(self) -> None:
        from cyecca.dsl.model import local

        temp = local("temp")
        assert repr(temp) == "local(temp)"
        assert temp.name == "temp"

    def test_local_arithmetic(self) -> None:
        from cyecca.dsl.model import ExprKind, local

        temp = local("temp")
        assert (temp + 1).kind == ExprKind.ADD
        assert (1 + temp).kind == ExprKind.ADD
        assert (temp - 1).kind == ExprKind.SUB
        assert (1 - temp).kind == ExprKind.SUB
        assert (temp * 2).kind == ExprKind.MUL
        assert (2 * temp).kind == ExprKind.MUL
        assert (temp / 2).kind == ExprKind.DIV
        assert (2 / temp).kind == ExprKind.DIV
        assert (-temp).kind == ExprKind.NEG
        assert (temp**2).kind == ExprKind.POW

    def test_local_comparisons(self) -> None:
        from cyecca.dsl.model import ExprKind, local

        temp = local("temp")
        assert (temp < 5).kind == ExprKind.LT
        assert (temp <= 5).kind == ExprKind.LE
        assert (temp > 5).kind == ExprKind.GT
        assert (temp >= 5).kind == ExprKind.GE

    def test_local_matmul_assignment(self) -> None:
        from cyecca.dsl.model import Assignment, local

        temp = local("temp")
        assign = temp @ 42
        assert isinstance(assign, Assignment)
        assert assign.target == "temp"
        assert assign.is_local is True


# =============================================================================
# assign Function Tests
# =============================================================================


class TestAssignFunction:
    """Test assign() function."""

    def test_assign_symbolic_var(self) -> None:
        from cyecca.dsl import model, var
        from cyecca.dsl.model import Assignment, assign

        @model
        class M:
            y = var(output=True)

        m = M()
        assignment = assign(m.y, 42)
        assert isinstance(assignment, Assignment)
        assert assignment.is_local is False

    def test_assign_algorithm_var(self) -> None:
        from cyecca.dsl.model import Assignment, assign, local

        temp = local("temp")
        assignment = assign(temp, 42)
        assert isinstance(assignment, Assignment)
        assert assignment.is_local is True

    def test_assign_string(self) -> None:
        from cyecca.dsl.model import Assignment, assign

        assignment = assign("temp", 42)
        assert isinstance(assignment, Assignment)
        assert assignment.is_local is True


# =============================================================================
# Discrete Operators Tests
# =============================================================================


class TestDiscreteOperators:
    """Test pre(), edge(), change() operators."""

    def test_pre_scalar(self) -> None:
        from cyecca.dsl import equations, model, var
        from cyecca.dsl.model import ExprKind, pre

        @model
        class M:
            count = var(0, discrete=True)

            @equations
            def _(m):
                pass

        m = M()
        expr = pre(m.count)
        assert expr.kind == ExprKind.PRE

    def test_edge_scalar(self) -> None:
        from cyecca.dsl import DType, equations, model, var
        from cyecca.dsl.model import ExprKind, edge

        @model
        class M:
            flag = var(False, dtype=DType.BOOLEAN, discrete=True)

            @equations
            def _(m):
                pass

        m = M()
        expr = edge(m.flag)
        assert expr.kind == ExprKind.EDGE

    def test_change_scalar(self) -> None:
        from cyecca.dsl import equations, model, var
        from cyecca.dsl.model import ExprKind, change

        @model
        class M:
            mode = var(0, discrete=True)

            @equations
            def _(m):
                pass

        m = M()
        expr = change(m.mode)
        assert expr.kind == ExprKind.CHANGE

    def test_discrete_non_scalar_error(self) -> None:
        from cyecca.dsl import equations, model, var
        from cyecca.dsl.model import change, edge, pre

        @model
        class M:
            arr = var(shape=(3,), discrete=True)

            @equations
            def _(m):
                pass

        m = M()
        with pytest.raises(TypeError, match="scalar"):
            pre(m.arr)
        with pytest.raises(TypeError, match="scalar"):
            edge(m.arr)
        with pytest.raises(TypeError, match="scalar"):
            change(m.arr)


# =============================================================================
# Model Decorator Tests
# =============================================================================


class TestModelDecorator:
    """Test @model decorator."""

    def test_basic_model(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        m = M()
        flat = m.flatten()
        assert flat.name == "M"
        assert "x" in flat.state_names

    def test_model_t_property(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.model import TimeVar

        @model
        class M:
            x = var()

            @equations
            def _(m):
                der(m.x) == m.t

        m = M()
        assert isinstance(m.t, TimeVar)

    def test_model_getattr_missing(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            x = var()

            @equations
            def _(m):
                der(m.x) == 0

        m = M()
        with pytest.raises(AttributeError, match="no attribute"):
            _ = m.nonexistent


# =============================================================================
# Submodel Tests
# =============================================================================


class TestSubmodel:
    """Test submodel composition."""

    def test_submodel_access(self) -> None:
        from cyecca.dsl import der, equations, model, submodel, var

        @model
        class Inner:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        @model
        class Outer:
            sub = submodel(Inner)

            @equations
            def _(m):
                pass

        outer = Outer()
        flat = outer.flatten()
        assert "sub.x" in flat.state_names

    def test_submodel_missing_attr(self) -> None:
        from cyecca.dsl import der, equations, model, submodel, var

        @model
        class Inner:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        @model
        class Outer:
            sub = submodel(Inner)

            @equations
            def _(m):
                pass

        outer = Outer()
        with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
            _ = outer.sub.nonexistent

    def test_submodel_variable_classification(self) -> None:
        from cyecca.dsl import der, equations, model, submodel, var

        @model
        class Inner:
            k = var(1.0, parameter=True)
            u = var(input=True)
            y = var(output=True)
            x = var(start=0.0)
            count = var(0, discrete=True)

            @equations
            def _(m):
                der(m.x) == m.k * m.u
                m.y == m.x

        @model
        class Outer:
            sub = submodel(Inner)

            @equations
            def _(m):
                pass

        flat = Outer().flatten()
        assert "sub.k" in flat.param_names
        assert "sub.u" in flat.input_names
        assert "sub.y" in flat.output_names
        assert "sub.x" in flat.state_names
        assert "sub.count" in flat.discrete_names


# =============================================================================
# Initial Equations Tests
# =============================================================================


class TestInitialEquations:
    """Test initial_equations method (Modelica Spec Section 8.6)."""

    def test_basic_initial_equations(self) -> None:
        """Test basic initial equations definition."""
        from cyecca.dsl import der, equations, model, var

        @model
        class Spring:
            x = var()
            v = var()

            @equations
            def _(m):
                der(m.x) == m.v
                der(m.v) == -10.0 * m.x

            def initial_equations(m):
                yield m.x == 1.0
                yield m.v == 0.0

        flat = Spring().flatten()
        assert len(flat.initial_equations) == 2

        # Check lhs values
        lhs_names = [str(eq.lhs) for eq in flat.initial_equations]
        assert "x" in lhs_names
        assert "v" in lhs_names

    def test_initial_equations_with_expressions(self) -> None:
        """Test initial equations with RHS expressions."""
        from cyecca.dsl import der, equations, model, sin, var

        @model
        class Pendulum:
            theta = var()
            omega = var()

            @equations
            def _(m):
                der(m.theta) == m.omega
                der(m.omega) == -9.81 * sin(m.theta)

            def initial_equations(m):
                yield m.theta == 0.5  # 0.5 radians
                yield m.omega == 0.0  # At rest

        flat = Pendulum().flatten()
        assert len(flat.initial_equations) == 2

        # Check specific values
        for eq in flat.initial_equations:
            if str(eq.lhs) == "theta":
                assert str(eq.rhs) == "0.5"
            elif str(eq.lhs) == "omega":
                assert str(eq.rhs) == "0.0"

    def test_initial_equations_with_submodels(self) -> None:
        """Test initial equations are properly prefixed in submodels."""
        from cyecca.dsl import der, equations, model, submodel, var

        @model
        class Spring:
            x = var()
            v = var()

            @equations
            def _(m):
                der(m.x) == m.v
                der(m.v) == -10.0 * m.x

            def initial_equations(m):
                yield m.x == 1.0
                yield m.v == 0.0

        @model
        class TwoSprings:
            spring1 = submodel(Spring)
            spring2 = submodel(Spring)

            @equations
            def _(m):
                pass

        flat = TwoSprings().flatten()
        assert len(flat.initial_equations) == 4

        # Check prefixes
        lhs_names = [str(eq.lhs) for eq in flat.initial_equations]
        assert "spring1.x" in lhs_names
        assert "spring1.v" in lhs_names
        assert "spring2.x" in lhs_names
        assert "spring2.v" in lhs_names

    def test_no_initial_equations(self) -> None:
        """Test model without initial_equations method."""
        from cyecca.dsl import der, equations, model, var

        @model
        class Simple:
            x = var()

            @equations
            def _(m):
                der(m.x) == 1.0

        flat = Simple().flatten()
        assert len(flat.initial_equations) == 0

    def test_empty_initial_equations(self) -> None:
        """Test model with empty initial_equations method."""
        from cyecca.dsl import der, equations, model, var

        @model
        class Simple:
            x = var()

            @equations
            def _(m):
                der(m.x) == 1.0

            def initial_equations(m):
                return
                yield

        flat = Simple().flatten()
        assert len(flat.initial_equations) == 0

    def test_initial_equations_structure(self) -> None:
        """Test that initial equations have proper Equation structure."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.model import Equation, Expr

        @model
        class Model:
            x = var()

            @equations
            def _(m):
                der(m.x) == 1.0

            def initial_equations(m):
                yield m.x == 5.0

        flat = Model().flatten()
        assert len(flat.initial_equations) == 1

        eq = flat.initial_equations[0]
        assert isinstance(eq, Equation)
        assert isinstance(eq.lhs, Expr)
        # rhs could be Expr or float/int
        assert eq.is_derivative is False  # initial equations are not derivative equations


# =============================================================================
# Function and Block Decorator Tests
# =============================================================================


class TestFunctionDecorator:
    """Test @function decorator."""

    def test_basic_function(self) -> None:
        from cyecca.dsl import function, var
        from cyecca.dsl.model import local

        @function
        class Saturate:
            x = var(input=True)
            y = var(output=True)

            def algorithm(m):
                yield m.y @ m.x

        f = Saturate()
        assert hasattr(f, "_is_function")
        assert f._is_function is True

    def test_function_metadata(self) -> None:
        from cyecca.dsl import function, var

        @function
        class Func:
            x = var(input=True)
            k = var(1.0, parameter=True)
            y = var(output=True)
            temp = var(protected=True)

            def algorithm(m):
                yield m.temp @ (m.x * m.k)
                yield m.y @ m.temp

        meta = Func().get_function_metadata()
        assert "x" in meta.input_names
        assert "y" in meta.output_names
        assert "k" in meta.param_names
        assert "temp" in meta.protected_names

    def test_function_requires_algorithm(self) -> None:
        from cyecca.dsl import function, var

        with pytest.raises(TypeError, match="must have an algorithm"):

            @function
            class BadFunc:
                x = var(input=True)
                y = var(output=True)

    def test_function_requires_io(self) -> None:
        from cyecca.dsl import function, var

        with pytest.raises(TypeError, match="must have input=True or output=True"):

            @function
            class BadFunc:
                x = var()  # Missing input/output
                y = var(output=True)

                def algorithm(m):
                    yield m.y @ m.x


class TestBlockDecorator:
    """Test @block decorator."""

    def test_valid_block(self) -> None:
        from cyecca.dsl import block, der, equations, var

        @block
        class Integrator:
            u = var(input=True)
            y = var(output=True)
            x = var(protected=True)

            @equations
            def _(m):
                der(m.x) == m.u
                m.y == m.x

        flat = Integrator().flatten()
        assert "u" in flat.input_names
        assert "y" in flat.output_names

    def test_block_rejects_public_without_causality(self) -> None:
        from cyecca.dsl import block, der, equations, var

        with pytest.raises(TypeError, match="violates Modelica block constraints"):

            @block
            class BadBlock:
                x = var()  # Public without input/output
                u = var(input=True)
                y = var(output=True)

                @equations
                def _(m):
                    der(m.x) == m.u
                    m.y == m.x


# =============================================================================
# Flatten Tests
# =============================================================================


class TestFlatten:
    """Test model flattening."""

    def test_flatten_states_from_der(self) -> None:
        from cyecca.dsl import VarKind, der, equations, model, var

        @model
        class M:
            x = var(start=0.0)
            v = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == m.v
                der(m.v) == -9.81

        flat = M().flatten()
        assert "x" in flat.state_names
        assert "v" in flat.state_names
        assert flat.state_vars["x"].kind == VarKind.STATE

    def test_flatten_algebraic_without_der(self) -> None:
        from cyecca.dsl import VarKind, der, equations, model, var

        @model
        class M:
            x = var()
            y = var()

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == m.x * 2

        flat = M().flatten()
        assert "y" in flat.algebraic_names
        assert flat.algebraic_vars["y"].kind == VarKind.ALGEBRAIC

    def test_flatten_array_equations_expand(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            @equations
            def _(m):
                der(m.pos) == m.vel

        flat = M().flatten()
        assert len(flat.derivative_equations) == 3
        assert "pos[0]" in flat.derivative_equations
        assert "pos[1]" in flat.derivative_equations
        assert "pos[2]" in flat.derivative_equations

    def test_flatten_array_equation_shape_mismatch(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            pos = var(shape=(3,))
            vel = var(shape=(2,))

            @equations
            def _(m):
                der(m.pos) == m.vel

        with pytest.raises(ValueError, match="Shape mismatch"):
            M().flatten()

    def test_flatten_expand_arrays_false(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            @equations
            def _(m):
                der(m.pos) == m.vel

        flat = M().flatten(expand_arrays=False)
        assert flat.expand_arrays is False
        assert "pos" in flat.array_derivative_equations

    def test_flat_model_repr(self) -> None:
        from cyecca.dsl import der, equations, model, var

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)
            y = var(output=True)
            k = var(1.0, parameter=True)
            count = var(0, discrete=True)

            @equations
            def _(m):
                der(m.x) == m.k * m.u
                m.y == m.x

        repr_str = repr(M().flatten())
        assert "states=" in repr_str
        assert "inputs=" in repr_str
        assert "outputs=" in repr_str
        assert "params=" in repr_str
        assert "discrete=" in repr_str


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_base_name(self) -> None:
        from cyecca.dsl.model import _get_base_name

        assert _get_base_name("pos[0,1]") == "pos"
        assert _get_base_name("x") == "x"

    def test_parse_indices(self) -> None:
        from cyecca.dsl.model import _parse_indices

        name, indices = _parse_indices("pos[0,1]")
        assert name == "pos"
        assert indices == (0, 1)

        name2, indices2 = _parse_indices("x")
        assert name2 == "x"
        assert indices2 == ()

    def test_format_indices(self) -> None:
        from cyecca.dsl.model import _format_indices

        assert _format_indices((0, 1)) == "[0,1]"
        assert _format_indices(()) == ""

    def test_iter_indices(self) -> None:
        from cyecca.dsl.model import _iter_indices

        assert list(_iter_indices(())) == [()]
        assert list(_iter_indices((3,))) == [(0,), (1,), (2,)]
        assert list(_iter_indices((2, 2))) == [(0, 0), (0, 1), (1, 0), (1, 1)]


class TestToExpr:
    """Test _to_expr conversion."""

    def test_to_expr_types(self) -> None:
        import numpy as np

        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.model import Expr, ExprKind, _to_expr

        # float
        expr = _to_expr(5.0)
        assert expr.kind == ExprKind.CONSTANT

        # numpy scalar
        arr = np.array(5.0)
        expr = _to_expr(arr)
        assert expr.kind == ExprKind.CONSTANT

        # Expr passthrough
        original = Expr(ExprKind.VARIABLE, name="x")
        assert _to_expr(original) is original

        # SymbolicVar
        @model
        class M:
            x = var()

            @equations
            def _(m):
                der(m.x) == 0

        m = M()
        result = _to_expr(m.x)
        assert result is m.x._expr

        # TimeVar
        result = _to_expr(m.t)
        assert result is m.t._expr

    def test_to_expr_invalid(self) -> None:
        from cyecca.dsl.model import _to_expr

        with pytest.raises(TypeError, match="Cannot convert"):
            _to_expr("a string")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
