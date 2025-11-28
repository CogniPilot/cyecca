"""
Tests for recently added Cyecca DSL features.

Covers:
- Relational operators (<, <=, >, >=)
- Boolean operators (and_, or_, not_)
- Conditional expressions (if_then_else)
- Algorithm sections with local variables and @ operator
- @function decorator
- @block decorator constraints

================================================================================
PROTOTYPE MODE - API IS IN FLUX
================================================================================
"""

import numpy as np
import pytest

# =============================================================================
# Relational Operators Tests
# =============================================================================


class TestRelationalOperators:
    """Test relational operators (<, <=, >, >=) on Expr and SymbolicVar."""

    def test_less_than_operator(self) -> None:
        """Test < operator creates LT expression."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            threshold = var(10.0, parameter=True)
            below = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.below == (m.x < m.threshold)

        flat = TestModel().flatten()
        assert "below" in flat.output_equations
        expr = flat.output_equations["below"]
        assert expr.kind == ExprKind.LT

    def test_less_than_or_equal_operator(self) -> None:
        """Test <= operator creates LE expression."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            threshold = var(10.0, parameter=True)
            at_or_below = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.at_or_below == (m.x <= m.threshold)

        flat = TestModel().flatten()
        expr = flat.output_equations["at_or_below"]
        assert expr.kind == ExprKind.LE

    def test_greater_than_operator(self) -> None:
        """Test > operator creates GT expression."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            threshold = var(10.0, parameter=True)
            above = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.above == (m.x > m.threshold)

        flat = TestModel().flatten()
        expr = flat.output_equations["above"]
        assert expr.kind == ExprKind.GT

    def test_greater_than_or_equal_operator(self) -> None:
        """Test >= operator creates GE expression."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            threshold = var(10.0, parameter=True)
            at_or_above = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.at_or_above == (m.x >= m.threshold)

        flat = TestModel().flatten()
        expr = flat.output_equations["at_or_above"]
        assert expr.kind == ExprKind.GE

    def test_relational_with_constants(self) -> None:
        """Test relational operators with numeric constants."""
        from cyecca.dsl import ExprKind, der, model, var

        @model
        class TestModel:
            x = var()
            positive = var(output=True)
            large = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.positive == (m.x > 0)
                yield m.large == (m.x >= 100.0)

        flat = TestModel().flatten()
        assert flat.output_equations["positive"].kind == ExprKind.GT
        assert flat.output_equations["large"].kind == ExprKind.GE

    def test_chained_relational(self) -> None:
        """Test combining relational operators with boolean operators."""
        from cyecca.dsl import ExprKind, and_, der, model, var

        @model
        class TestModel:
            x = var()
            in_range = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.in_range == and_(m.x > 0, m.x < 100)

        flat = TestModel().flatten()
        expr = flat.output_equations["in_range"]
        assert expr.kind == ExprKind.AND


# =============================================================================
# Boolean Operators Tests
# =============================================================================


class TestBooleanOperators:
    """Test boolean operators (and_, or_, not_)."""

    def test_and_operator(self) -> None:
        """Test and_() creates AND expression."""
        from cyecca.dsl import ExprKind, and_, der, model, var

        @model
        class TestModel:
            a = var()
            b = var()
            both = var(output=True)

            def equations(m):
                yield der(m.a) == 1.0
                yield der(m.b) == 1.0
                yield m.both == and_(m.a > 0, m.b > 0)

        flat = TestModel().flatten()
        expr = flat.output_equations["both"]
        assert expr.kind == ExprKind.AND
        assert len(expr.children) == 2

    def test_or_operator(self) -> None:
        """Test or_() creates OR expression."""
        from cyecca.dsl import ExprKind, der, model, or_, var

        @model
        class TestModel:
            a = var()
            b = var()
            either = var(output=True)

            def equations(m):
                yield der(m.a) == 1.0
                yield der(m.b) == 1.0
                yield m.either == or_(m.a > 0, m.b > 0)

        flat = TestModel().flatten()
        expr = flat.output_equations["either"]
        assert expr.kind == ExprKind.OR
        assert len(expr.children) == 2

    def test_not_operator(self) -> None:
        """Test not_() creates NOT expression."""
        from cyecca.dsl import ExprKind, der, model, not_, var

        @model
        class TestModel:
            a = var()
            not_positive = var(output=True)

            def equations(m):
                yield der(m.a) == 1.0
                yield m.not_positive == not_(m.a > 0)

        flat = TestModel().flatten()
        expr = flat.output_equations["not_positive"]
        assert expr.kind == ExprKind.NOT
        assert len(expr.children) == 1

    def test_nested_boolean_operators(self) -> None:
        """Test nested boolean expressions."""
        from cyecca.dsl import ExprKind, and_, der, model, not_, or_, var

        @model
        class TestModel:
            a = var()
            b = var()
            c = var()
            complex_logic = var(output=True)

            def equations(m):
                yield der(m.a) == 1.0
                yield der(m.b) == 1.0
                yield der(m.c) == 1.0
                # (a > 0 AND b > 0) OR (NOT c > 0)
                yield m.complex_logic == or_(and_(m.a > 0, m.b > 0), not_(m.c > 0))

        flat = TestModel().flatten()
        expr = flat.output_equations["complex_logic"]
        assert expr.kind == ExprKind.OR

    def test_boolean_with_constants(self) -> None:
        """Test boolean operators with constant values."""
        from cyecca.dsl import ExprKind, and_, der, model, or_, var

        @model
        class TestModel:
            x = var()
            result = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                # x > 0 AND x < 100
                yield m.result == and_(m.x > 0, m.x < 100)

        flat = TestModel().flatten()
        assert "result" in flat.output_equations


# =============================================================================
# If-Then-Else Tests
# =============================================================================


class TestIfThenElse:
    """Test if_then_else() conditional expression."""

    def test_basic_if_then_else(self) -> None:
        """Test basic if_then_else creates IF_THEN_ELSE expression."""
        from cyecca.dsl import ExprKind, der, if_then_else, model, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == if_then_else(m.x > 0, 1.0, -1.0)

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.IF_THEN_ELSE
        assert len(expr.children) == 3

    def test_if_then_else_with_expressions(self) -> None:
        """Test if_then_else with expression branches."""
        from cyecca.dsl import ExprKind, der, if_then_else, model, var

        @model
        class TestModel:
            x = var()
            a = var(2.0, parameter=True)
            b = var(3.0, parameter=True)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == if_then_else(m.x > 0, m.a * m.x, m.b * m.x)

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.IF_THEN_ELSE

    def test_nested_if_then_else_saturation(self) -> None:
        """Test nested if_then_else for saturation function."""
        from cyecca.dsl import ExprKind, der, if_then_else, model, var

        @model
        class Saturation:
            u = var(input=True)
            y = var(output=True)
            limit = var(10.0, parameter=True)

            def equations(m):
                yield m.y == if_then_else(m.u > m.limit, m.limit, if_then_else(m.u < -m.limit, -m.limit, m.u))

        flat = Saturation().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.IF_THEN_ELSE
        # Check nested structure
        else_branch = expr.children[2]
        assert else_branch.kind == ExprKind.IF_THEN_ELSE

    def test_if_then_else_in_derivative(self) -> None:
        """Test if_then_else in derivative equation."""
        from cyecca.dsl import ExprKind, der, if_then_else, model, var

        @model
        class SwitchedSystem:
            x = var()
            mode = var(input=True)

            def equations(m):
                yield der(m.x) == if_then_else(m.mode > 0, 1.0, -1.0)

        flat = SwitchedSystem().flatten()
        expr = flat.derivative_equations["x"]
        assert expr.kind == ExprKind.IF_THEN_ELSE

    def test_if_then_else_with_boolean_condition(self) -> None:
        """Test if_then_else with complex boolean condition."""
        from cyecca.dsl import ExprKind, and_, der, if_then_else, model, var

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                # If x in range (0, 100) then x else 0
                yield m.y == if_then_else(and_(m.x > 0, m.x < 100), m.x, 0.0)

        flat = TestModel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.IF_THEN_ELSE
        assert expr.children[0].kind == ExprKind.AND


# =============================================================================
# Algorithm Section Tests
# =============================================================================


class TestAlgorithmSection:
    """Test algorithm sections with local variables and @ operator."""

    def test_basic_algorithm_assignment(self) -> None:
        """Test basic algorithm with @ assignment operator."""
        from cyecca.dsl import Assignment, model, var

        @model
        class TestModel:
            u = var(input=True)
            y = var(output=True)

            def algorithm(m):
                yield m.y @ (m.u * 2)

        flat = TestModel().flatten()
        assert len(flat.algorithm_assignments) == 1
        assign = flat.algorithm_assignments[0]
        assert isinstance(assign, Assignment)
        assert assign.target == "y"
        assert assign.is_local == False

    def test_algorithm_with_local_variable(self) -> None:
        """Test algorithm with local() variable."""
        from cyecca.dsl import Assignment, local, model, var

        @model
        class TestModel:
            u = var(input=True)
            y = var(output=True)

            def algorithm(m):
                temp = local("temp")
                yield temp @ (m.u * 2)
                yield m.y @ (temp + 1)

        flat = TestModel().flatten()
        assert len(flat.algorithm_assignments) == 2
        assert "temp" in flat.algorithm_locals

        # First assignment: temp := u * 2
        assert flat.algorithm_assignments[0].target == "temp"
        assert flat.algorithm_assignments[0].is_local == True

        # Second assignment: y := temp + 1
        assert flat.algorithm_assignments[1].target == "y"
        assert flat.algorithm_assignments[1].is_local == False

    def test_algorithm_with_if_then_else(self) -> None:
        """Test algorithm with conditional logic using if_then_else."""
        from cyecca.dsl import if_then_else, local, model, var

        @model
        class Saturation:
            u = var(input=True)
            y = var(output=True)
            limit = var(5.0, parameter=True)

            def algorithm(m):
                temp = local("temp")
                yield temp @ (m.u * 2)
                yield m.y @ if_then_else(temp > m.limit, m.limit, if_then_else(temp < -m.limit, -m.limit, temp))

        flat = Saturation().flatten()
        assert len(flat.algorithm_assignments) == 2
        assert "temp" in flat.algorithm_locals

    def test_algorithm_multiple_locals(self) -> None:
        """Test algorithm with multiple local variables."""
        from cyecca.dsl import local, model, var

        @model
        class TestModel:
            a = var(input=True)
            b = var(input=True)
            y = var(output=True)

            def algorithm(m):
                sum_val = local("sum")
                diff_val = local("diff")
                yield sum_val @ (m.a + m.b)
                yield diff_val @ (m.a - m.b)
                yield m.y @ (sum_val * diff_val)

        flat = TestModel().flatten()
        assert len(flat.algorithm_assignments) == 3
        assert "sum" in flat.algorithm_locals
        assert "diff" in flat.algorithm_locals

    def test_assign_function(self) -> None:
        """Test explicit assign() function."""
        from cyecca.dsl import Assignment, assign, model, var

        @model
        class TestModel:
            u = var(input=True)
            y = var(output=True)

            def algorithm(m):
                yield assign(m.y, m.u * 3)

        flat = TestModel().flatten()
        assert len(flat.algorithm_assignments) == 1
        assert flat.algorithm_assignments[0].target == "y"

    def test_algorithm_local_arithmetic(self) -> None:
        """Test arithmetic with local variables."""
        from cyecca.dsl import ExprKind, local, model, var

        @model
        class TestModel:
            x = var(input=True)
            y = var(output=True)

            def algorithm(m):
                a = local("a")
                b = local("b")
                yield a @ (m.x + 1)
                yield b @ (a * 2)
                yield m.y @ (b - 3)

        flat = TestModel().flatten()
        assert len(flat.algorithm_assignments) == 3

        # Check expression structure
        b_expr = flat.algorithm_assignments[1].expr
        assert b_expr.kind == ExprKind.MUL


# =============================================================================
# @function Decorator Tests
# =============================================================================


class TestFunctionDecorator:
    """Test @function decorator for Modelica-style functions."""

    def test_basic_function(self) -> None:
        """Test basic function definition."""
        from cyecca.dsl import function, if_then_else, var

        @function
        class Saturate:
            x = var(input=True)
            lo = var(input=True)
            hi = var(input=True)
            y = var(output=True)

            def algorithm(f):
                yield f.y @ if_then_else(f.x < f.lo, f.lo, if_then_else(f.x > f.hi, f.hi, f.x))

        sat = Saturate()
        flat = sat.flatten()

        assert "x" in flat.input_names
        assert "lo" in flat.input_names
        assert "hi" in flat.input_names
        assert "y" in flat.output_names
        assert len(flat.algorithm_assignments) == 1

    def test_function_with_protected(self) -> None:
        """Test function with protected intermediate variable."""
        from cyecca.dsl import function, sqrt, var

        @function
        class Quadratic:
            a = var(input=True)
            b = var(input=True)
            c = var(input=True)
            x1 = var(output=True)
            x2 = var(output=True)
            d = var(protected=True)

            def algorithm(f):
                yield f.d @ sqrt(f.b**2 - 4 * f.a * f.c)
                yield f.x1 @ ((-f.b + f.d) / (2 * f.a))
                yield f.x2 @ ((-f.b - f.d) / (2 * f.a))

        quad = Quadratic()
        flat = quad.flatten()

        assert "a" in flat.input_names
        assert "b" in flat.input_names
        assert "c" in flat.input_names
        assert "x1" in flat.output_names
        assert "x2" in flat.output_names
        assert len(flat.algorithm_assignments) == 3

    def test_function_get_metadata(self) -> None:
        """Test get_function_metadata() method."""
        from cyecca.dsl import function, var

        @function
        class SimpleFunc:
            x = var(input=True)
            y = var(output=True)

            def algorithm(f):
                yield f.y @ (f.x * 2)

        func = SimpleFunc()
        meta = func.get_function_metadata()

        assert meta.name == "SimpleFunc"
        assert "x" in meta.input_names
        assert "y" in meta.output_names

    def test_function_requires_input(self) -> None:
        """Test that function requires at least one input."""
        from cyecca.dsl import function, var

        with pytest.raises(TypeError, match="must have input=True or output=True"):

            @function
            class BadFunc:
                x = var()  # Missing input/output
                y = var(output=True)

                def algorithm(f):
                    yield f.y @ f.x

    def test_function_requires_output(self) -> None:
        """Test that function requires at least one output."""
        from cyecca.dsl import function, var

        with pytest.raises(TypeError, match="must have input=True or output=True"):

            @function
            class BadFunc:
                x = var(input=True)
                y = var()  # Missing output

                def algorithm(f):
                    yield f.y @ f.x

    def test_function_requires_algorithm(self) -> None:
        """Test that function requires algorithm() method."""
        from cyecca.dsl import function, var

        with pytest.raises(TypeError, match="must have an algorithm"):

            @function
            class BadFunc:
                x = var(input=True)
                y = var(output=True)
                # Missing algorithm() method

    def test_function_is_model_subclass(self) -> None:
        """Test that @function creates a model-like class."""
        from cyecca.dsl import function, var

        @function
        class TestFunc:
            x = var(input=True)
            y = var(output=True)

            def algorithm(f):
                yield f.y @ (f.x + 1)

        func = TestFunc()

        # Should have flatten() method like a model
        flat = func.flatten()
        assert flat is not None
        assert hasattr(func, "_is_function")
        assert func._is_function == True


# =============================================================================
# @block Decorator Tests
# =============================================================================


class TestBlockDecorator:
    """Test @block decorator constraints."""

    def test_valid_block(self) -> None:
        """Test valid block with input/output on all public variables."""
        from cyecca.dsl import block, der, var

        @block
        class Integrator:
            u = var(input=True)
            y = var(output=True)
            x = var(protected=True)  # Internal state

            def equations(m):
                yield der(m.x) == m.u
                yield m.y == m.x

        integ = Integrator()
        flat = integ.flatten()

        assert "u" in flat.input_names
        assert "y" in flat.output_names

    def test_block_with_parameters(self) -> None:
        """Test block allows parameters without input/output."""
        from cyecca.dsl import block, der, var

        @block
        class Gain:
            K = var(1.0, parameter=True)  # Parameters are OK
            u = var(input=True)
            y = var(output=True)

            def equations(m):
                yield m.y == m.K * m.u

        gain = Gain()
        flat = gain.flatten()

        assert "K" in flat.param_names
        assert "u" in flat.input_names
        assert "y" in flat.output_names

    def test_block_rejects_public_without_causality(self) -> None:
        """Test that block rejects public non-parameter without input/output."""
        from cyecca.dsl import block, der, var

        with pytest.raises(TypeError, match="violates Modelica block constraints"):

            @block
            class BadBlock:
                x = var()  # Public non-parameter without input/output!
                u = var(input=True)
                y = var(output=True)

                def equations(m):
                    yield der(m.x) == m.u
                    yield m.y == m.x

    def test_block_allows_protected_state(self) -> None:
        """Test that block allows protected internal state."""
        from cyecca.dsl import block, der, var

        @block
        class PIDController:
            setpoint = var(input=True)
            measurement = var(input=True)
            output = var(output=True)
            Kp = var(1.0, parameter=True)
            Ki = var(0.1, parameter=True)
            integral = var(start=0.0, protected=True)  # Protected OK

            def equations(m):
                error = m.setpoint - m.measurement
                yield der(m.integral) == error
                yield m.output == m.Kp * error + m.Ki * m.integral

        pid = PIDController()
        flat = pid.flatten()

        # Protected integral should be a state (der used)
        assert "integral" in flat.state_names


# =============================================================================
# CasADi Backend Integration Tests
# =============================================================================


class TestCasadiBackendNewFeatures:
    """Test CasADi backend compilation of new features."""

    def test_compile_relational_operators(self) -> None:
        """Test CasADi compilation of relational operators."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var()
            positive = var(output=True)
            large = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.positive == (m.x > 0)
                yield m.large == (m.x >= 10)

        compiled = CasadiBackend.compile(TestModel().flatten())
        assert compiled is not None
        assert "positive" in compiled.output_names
        assert "large" in compiled.output_names

    def test_compile_boolean_operators(self) -> None:
        """Test CasADi compilation of boolean operators."""
        from cyecca.dsl import and_, der, model, not_, or_, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var()
            in_range = var(output=True)
            out_of_range = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.in_range == and_(m.x > 0, m.x < 100)
                yield m.out_of_range == not_(and_(m.x > 0, m.x < 100))

        compiled = CasadiBackend.compile(TestModel().flatten())
        assert compiled is not None

    def test_compile_if_then_else(self) -> None:
        """Test CasADi compilation of if_then_else."""
        from cyecca.dsl import der, if_then_else, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == if_then_else(m.x > 0, m.x, -m.x)

        compiled = CasadiBackend.compile(TestModel().flatten())
        assert compiled is not None

    def test_simulate_saturation(self) -> None:
        """Test simulation with saturation using if_then_else."""
        from cyecca.dsl import der, if_then_else, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class SaturatedRamp:
            x = var()
            y = var(output=True)
            limit = var(5.0, parameter=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == if_then_else(m.x > m.limit, m.limit, if_then_else(m.x < -m.limit, -m.limit, m.x))

        compiled = CasadiBackend.compile(SaturatedRamp().flatten())
        result = compiled.simulate(t0=0.0, tf=10.0, dt=0.1, x0={"x": -10.0})

        # x starts at -10, ramps up at 1/s
        # y should be -5 (saturated) initially, then follow x, then +5
        assert result is not None
        y_values = result._data["y"]

        # At t=0, x=-10, y should be -5 (lower saturation)
        assert y_values[0] == pytest.approx(-5.0, abs=0.1)

        # At t=10, x=0, y should be 0
        mid_idx = len(result.t) // 2

        # At end, x=0, should be within bounds
        assert abs(y_values[-1]) <= 5.0


# =============================================================================
# Expression Tree Representation Tests
# =============================================================================


class TestExpressionRepresentation:
    """Test Expr __repr__ for new expression kinds."""

    def test_relational_repr(self) -> None:
        """Test repr of relational expressions."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 1.0

        m = TestModel()

        # Create expressions
        lt_expr = m.x._expr < 5
        le_expr = m.x._expr <= 5
        gt_expr = m.x._expr > 5
        ge_expr = m.x._expr >= 5

        assert "< 5" in str(lt_expr)
        assert "<= 5" in str(le_expr)
        assert "> 5" in str(gt_expr)
        assert ">= 5" in str(ge_expr)

    def test_boolean_repr(self) -> None:
        """Test repr of boolean expressions."""
        from cyecca.dsl import and_, der, model, not_, or_, var

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 1.0

        m = TestModel()

        and_expr = and_(m.x > 0, m.x < 10)
        or_expr = or_(m.x < 0, m.x > 10)
        not_expr = not_(m.x > 0)

        assert "and" in str(and_expr)
        assert "or" in str(or_expr)
        assert "not" in str(not_expr)

    def test_if_then_else_repr(self) -> None:
        """Test repr of if_then_else expression."""
        from cyecca.dsl import der, if_then_else, model, var

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 1.0

        m = TestModel()

        ite_expr = if_then_else(m.x > 0, 1.0, -1.0)
        repr_str = str(ite_expr)

        assert "if" in repr_str
        assert "then" in repr_str
        assert "else" in repr_str


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_algorithm_empty(self) -> None:
        """Test model with empty algorithm section."""
        from cyecca.dsl import der, model, var

        @model
        class TestModel:
            x = var()

            def equations(m):
                yield der(m.x) == 1.0

            def algorithm(m):
                return
                yield  # Make it a generator

        flat = TestModel().flatten()
        assert len(flat.algorithm_assignments) == 0

    def test_local_variable_reuse(self) -> None:
        """Test that local variables can be assigned multiple times."""
        from cyecca.dsl import local, model, var

        @model
        class TestModel:
            u = var(input=True)
            y = var(output=True)

            def algorithm(m):
                temp = local("temp")
                yield temp @ (m.u * 2)
                yield temp @ (temp + 1)  # Reassign
                yield m.y @ temp

        flat = TestModel().flatten()
        assert len(flat.algorithm_assignments) == 3
        # temp appears once in locals list (not duplicated)
        assert flat.algorithm_locals.count("temp") == 1

    def test_nested_if_then_else_deep(self) -> None:
        """Test deeply nested if_then_else."""
        from cyecca.dsl import ExprKind, der, if_then_else, model, var

        @model
        class MultiLevel:
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                # 4 levels: maps x to 0,1,2,3 based on value
                yield m.y == if_then_else(m.x < 0, 0.0, if_then_else(m.x < 1, 1.0, if_then_else(m.x < 2, 2.0, 3.0)))

        flat = MultiLevel().flatten()
        expr = flat.output_equations["y"]
        assert expr.kind == ExprKind.IF_THEN_ELSE

    def test_combined_algorithm_and_equations(self) -> None:
        """Test model with both equations and algorithm sections."""
        from cyecca.dsl import der, local, model, var

        @model
        class Combined:
            u = var(input=True)
            x = var()
            y = var(output=True)

            def equations(m):
                yield der(m.x) == m.u

            def algorithm(m):
                temp = local("temp")
                yield temp @ (m.x * 2)
                yield m.y @ (temp + 1)

        flat = Combined().flatten()

        # Should have both
        assert len(flat.derivative_equations) == 1
        assert len(flat.algorithm_assignments) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
