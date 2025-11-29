"""
Tests for cyecca.dsl.operators module.

Covers: sin, cos, tan, asin, acos, atan, atan2, sqrt, exp, log, log10, abs,
        sign, floor, ceil, sinh, cosh, tanh, min, max
"""

import math

import pytest


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

    def test_log10_float(self) -> None:
        from cyecca.dsl.operators import log10

        result = log10(100.0)
        assert isinstance(result, float)
        assert result == pytest.approx(2.0)

    def test_abs_float(self) -> None:
        from cyecca.dsl import abs as dsl_abs

        assert dsl_abs(5.0) == 5.0
        assert dsl_abs(-5.0) == 5.0

    def test_sign_float(self) -> None:
        from cyecca.dsl.operators import sign

        assert sign(5.0) == 1.0
        assert sign(-5.0) == -1.0
        assert sign(0.0) == 0.0

    def test_floor_float(self) -> None:
        from cyecca.dsl.operators import floor

        assert floor(2.7) == 2.0
        assert floor(-2.7) == -3.0

    def test_ceil_float(self) -> None:
        from cyecca.dsl.operators import ceil

        assert ceil(2.3) == 3.0
        assert ceil(-2.3) == -2.0

    def test_sinh_float(self) -> None:
        from cyecca.dsl.operators import sinh

        result = sinh(1.0)
        assert isinstance(result, float)
        assert result == pytest.approx(math.sinh(1.0))

    def test_cosh_float(self) -> None:
        from cyecca.dsl.operators import cosh

        result = cosh(1.0)
        assert isinstance(result, float)
        assert result == pytest.approx(math.cosh(1.0))

    def test_tanh_float(self) -> None:
        from cyecca.dsl.operators import tanh

        result = tanh(1.0)
        assert isinstance(result, float)
        assert result == pytest.approx(math.tanh(1.0))

    def test_min_float(self) -> None:
        from cyecca.dsl.operators import min as dsl_min

        assert dsl_min(3.0, 5.0) == 3.0
        assert dsl_min(5.0, 3.0) == 3.0
        assert dsl_min(-1.0, 1.0) == -1.0

    def test_max_float(self) -> None:
        from cyecca.dsl.operators import max as dsl_max

        assert dsl_max(3.0, 5.0) == 5.0
        assert dsl_max(5.0, 3.0) == 5.0
        assert dsl_max(-1.0, 1.0) == 1.0


class TestMathOperatorsSymbolic:
    """Test math operators with symbolic inputs (return Expr)."""

    def test_sin_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, model, sin, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == sin(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.SIN

    def test_cos_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, cos, der, equations, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == cos(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.COS

    def test_tan_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, model, tan, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == tan(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.TAN

    def test_asin_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, asin, der, equations, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == asin(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.ASIN

    def test_acos_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, acos, der, equations, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == acos(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.ACOS

    def test_atan_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, atan, der, equations, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == atan(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.ATAN

    def test_atan2_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, atan2, der, equations, model, Real, var

        @model
        class M:
            x = Real()
            y = Real()
            angle = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                der(m.y) == 1.0
                m.angle == atan2(m.y, m.x)

        assert M().flatten().output_equations["angle"].kind == ExprKind.ATAN2

    def test_sqrt_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, model, sqrt, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == sqrt(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.SQRT

    def test_exp_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, exp, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == exp(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.EXP

    def test_log_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, log, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == log(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.LOG

    def test_log10_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, log10, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == log10(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.LOG10

    def test_abs_symbolic(self) -> None:
        from cyecca.dsl import ExprKind
        from cyecca.dsl import abs as dsl_abs
        from cyecca.dsl import der, equations, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == dsl_abs(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.ABS

    def test_sign_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, model, sign, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == sign(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.SIGN

    def test_floor_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, floor, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == floor(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.FLOOR

    def test_ceil_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, ceil, der, equations, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == ceil(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.CEIL

    def test_sinh_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, model, sinh, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == sinh(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.SINH

    def test_cosh_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, cosh, der, equations, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == cosh(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.COSH

    def test_tanh_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, model, tanh, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == tanh(m.x)

        assert M().flatten().output_equations["y"].kind == ExprKind.TANH

    def test_min_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations
        from cyecca.dsl import min as dsl_min
        from cyecca.dsl import model, Real, var

        @model
        class M:
            x = Real()
            y = Real()
            z = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                der(m.y) == 1.0
                m.z == dsl_min(m.x, m.y)

        assert M().flatten().output_equations["z"].kind == ExprKind.MIN

    def test_max_symbolic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations
        from cyecca.dsl import max as dsl_max
        from cyecca.dsl import model, Real, var

        @model
        class M:
            x = Real()
            y = Real()
            z = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                der(m.y) == 1.0
                m.z == dsl_max(m.x, m.y)

        assert M().flatten().output_equations["z"].kind == ExprKind.MAX


class TestBooleanOperators:
    """Test boolean operators (and_, or_, not_)."""

    def test_and_operator(self) -> None:
        from cyecca.dsl import ExprKind, and_, der, equations, model, Real, var

        @model
        class M:
            a = Real()
            b = Real()
            both = Real(output=True)

            @equations
            def _(m):
                der(m.a) == 1.0
                der(m.b) == 1.0
                m.both == and_(m.a > 0, m.b > 0)

        assert M().flatten().output_equations["both"].kind == ExprKind.AND

    def test_or_operator(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, model, or_, Real, var

        @model
        class M:
            a = Real()
            b = Real()
            either = Real(output=True)

            @equations
            def _(m):
                der(m.a) == 1.0
                der(m.b) == 1.0
                m.either == or_(m.a > 0, m.b > 0)

        assert M().flatten().output_equations["either"].kind == ExprKind.OR

    def test_not_operator(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, model, not_, Real, var

        @model
        class M:
            a = Real()
            result = Real(output=True)

            @equations
            def _(m):
                der(m.a) == 1.0
                m.result == not_(m.a > 0)

        assert M().flatten().output_equations["result"].kind == ExprKind.NOT


class TestConditionalOperators:
    """Test conditional operators (if_then_else)."""

    def test_if_then_else_basic(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, if_then_else, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == if_then_else(m.x > 0, 1.0, -1.0)

        assert M().flatten().output_equations["y"].kind == ExprKind.IF_THEN_ELSE

    def test_if_then_else_nested(self) -> None:
        from cyecca.dsl import ExprKind, der, equations, if_then_else, model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == if_then_else(m.x < 0, -1.0, if_then_else(m.x < 1, 0.0, 1.0))

        expr = M().flatten().output_equations["y"]
        assert expr.kind == ExprKind.IF_THEN_ELSE
        assert expr.children[2].kind == ExprKind.IF_THEN_ELSE


class TestOperatorErrors:
    """Test error handling in operators."""

    def test_sin_invalid_type(self) -> None:
        from cyecca.dsl.operators import sin

        with pytest.raises(TypeError):
            sin("not a number")

    def test_cos_invalid_type(self) -> None:
        from cyecca.dsl.operators import cos

        with pytest.raises(TypeError):
            cos([1, 2, 3])


class TestMixedFloatSymbolic:
    """Test operators with mixed float and symbolic inputs."""

    def test_atan2_float_symbolic(self) -> None:
        """Test atan2 where y is float and x is symbolic."""
        from cyecca.dsl import ExprKind, der, equations, model, Real, var
        from cyecca.dsl.operators import atan2

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == atan2(1.0, m.x)  # float y, symbolic x

        flat = M().flatten()
        assert flat.output_equations["y"].kind == ExprKind.ATAN2

    def test_atan2_symbolic_float(self) -> None:
        """Test atan2 where y is symbolic and x is float."""
        from cyecca.dsl import ExprKind, der, equations, model, Real, var
        from cyecca.dsl.operators import atan2

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == atan2(m.x, 1.0)  # symbolic y, float x

        flat = M().flatten()
        assert flat.output_equations["y"].kind == ExprKind.ATAN2

    def test_min_float_symbolic(self) -> None:
        """Test min where first arg is float and second is symbolic."""
        from cyecca.dsl import ExprKind, der, equations
        from cyecca.dsl import min as dsl_min
        from cyecca.dsl import model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == dsl_min(5.0, m.x)  # float, symbolic

        flat = M().flatten()
        assert flat.output_equations["y"].kind == ExprKind.MIN

    def test_min_symbolic_float(self) -> None:
        """Test min where first arg is symbolic and second is float."""
        from cyecca.dsl import ExprKind, der, equations
        from cyecca.dsl import min as dsl_min
        from cyecca.dsl import model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == dsl_min(m.x, 5.0)  # symbolic, float

        flat = M().flatten()
        assert flat.output_equations["y"].kind == ExprKind.MIN

    def test_max_float_symbolic(self) -> None:
        """Test max where first arg is float and second is symbolic."""
        from cyecca.dsl import ExprKind, der, equations
        from cyecca.dsl import max as dsl_max
        from cyecca.dsl import model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == dsl_max(5.0, m.x)  # float, symbolic

        flat = M().flatten()
        assert flat.output_equations["y"].kind == ExprKind.MAX

    def test_max_symbolic_float(self) -> None:
        """Test max where first arg is symbolic and second is float."""
        from cyecca.dsl import ExprKind, der, equations
        from cyecca.dsl import max as dsl_max
        from cyecca.dsl import model, Real, var

        @model
        class M:
            x = Real()
            y = Real(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == dsl_max(m.x, 5.0)  # symbolic, float

        flat = M().flatten()
        assert flat.output_equations["y"].kind == ExprKind.MAX


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
