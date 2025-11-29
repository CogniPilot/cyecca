"""
Tests for if-equations (conditional equations) in the Cyecca DSL.

Covers:
- if_eq() context manager
- elseif_eq() for additional conditions
- else_eq() for default branches
- IfEquation dataclass
- If-equation expansion to conditional expressions
- CasADi backend simulation with if-equations

If-equations are defined inside @equations methods using the if_eq() context
manager. They select which equations are active based on a condition.

Reference: Modelica Language Spec 3.7-dev, Section 8.3.4 - If-Equations
"""

import numpy as np
import pytest


class TestIfEquationBasics:
    """Test basic if-equation syntax and data structures."""

    def test_if_eq_creates_if_equation(self) -> None:
        """Test that if_eq() context manager creates IfEquation."""
        from cyecca.dsl import IfEquation, else_eq, equations, if_eq, model, var

        @model
        class M:
            use_linear = var(1.0, parameter=True)  # 1.0 = True
            x = var(start=1.0)
            y = var()

            @equations
            def _(m):
                with if_eq(m.use_linear > 0.5):
                    m.y == m.x
                with else_eq():
                    m.y == m.x**2

        instance = M()
        flat = instance.flatten()

        # If-equations should be expanded into regular equations
        # Check that we have an equation for y
        y_eqs = [eq for eq in flat.equations if "y" in str(eq.lhs)]
        assert len(y_eqs) >= 1

    def test_if_eq_with_elseif(self) -> None:
        """Test if_eq with elseif branches."""
        from cyecca.dsl import else_eq, elseif_eq, equations, if_eq, model, var

        @model
        class M:
            mode = var(1.0, parameter=True)
            x = var(start=1.0)
            y = var()

            @equations
            def _(m):
                with if_eq(m.mode > 1.5):
                    m.y == 2 * m.x
                with elseif_eq(m.mode > 0.5):
                    m.y == m.x
                with else_eq():
                    m.y == 0.0

        instance = M()
        flat = instance.flatten()

        # Should have equation for y
        y_eqs = [eq for eq in flat.equations if "y" in str(eq.lhs)]
        assert len(y_eqs) >= 1

    def test_if_eq_single_branch(self) -> None:
        """Test if_eq with only a then branch (no else)."""
        from cyecca.dsl import equations, if_eq, model, var

        @model
        class M:
            flag = var(1.0, parameter=True)
            x = var(start=1.0)
            y = var()

            @equations
            def _(m):
                with if_eq(m.flag > 0.5):
                    m.y == m.x
                # No else branch - equations are conditional

        instance = M()
        flat = instance.flatten()

        # Single branch if-equation should create equation
        y_eqs = [eq for eq in flat.equations if "y" in str(eq.lhs)]
        assert len(y_eqs) >= 1


class TestIfEquationExpansion:
    """Test expansion of if-equations to conditional expressions."""

    def test_if_else_expands_to_if_then_else(self) -> None:
        """Test that if/else expands to if_then_else expression."""
        from cyecca.dsl import ExprKind, else_eq, equations, if_eq, model, var

        @model
        class M:
            cond = var(1.0, parameter=True)
            x = var(start=1.0)
            y = var()

            @equations
            def _(m):
                with if_eq(m.cond > 0.5):
                    m.y == m.x + 1
                with else_eq():
                    m.y == m.x - 1

        instance = M()
        flat = instance.flatten()

        # Find the y equation
        y_eqs = [eq for eq in flat.equations if "y" in str(eq.lhs)]
        assert len(y_eqs) == 1

        # The RHS should be an IF_THEN_ELSE expression
        rhs = y_eqs[0].rhs
        assert rhs.kind == ExprKind.IF_THEN_ELSE

    def test_multiple_elseif_creates_nested_conditionals(self) -> None:
        """Test that multiple elseif creates nested if_then_else."""
        from cyecca.dsl import ExprKind, else_eq, elseif_eq, equations, if_eq, model, var

        @model
        class M:
            mode = var(1.0, parameter=True)
            x = var(start=1.0)
            y = var()

            @equations
            def _(m):
                with if_eq(m.mode > 2.5):
                    m.y == 3 * m.x
                with elseif_eq(m.mode > 1.5):
                    m.y == 2 * m.x
                with elseif_eq(m.mode > 0.5):
                    m.y == m.x
                with else_eq():
                    m.y == 0.0

        instance = M()
        flat = instance.flatten()

        y_eqs = [eq for eq in flat.equations if "y" in str(eq.lhs)]
        assert len(y_eqs) == 1

        # RHS should be nested IF_THEN_ELSE
        rhs = y_eqs[0].rhs
        assert rhs.kind == ExprKind.IF_THEN_ELSE
        # The else part should also be IF_THEN_ELSE (nested)
        assert rhs.children[2].kind == ExprKind.IF_THEN_ELSE


class TestIfEquationSimulation:
    """Test if-equations in CasADi backend simulation."""

    def test_if_eq_simulation_selects_correct_branch(self) -> None:
        """Test that simulation uses correct branch based on condition."""
        from cyecca.dsl import der, else_eq, equations, if_eq, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class LinearOrQuadratic:
            use_linear = var(1.0, parameter=True)  # 1.0 = True
            x = var(start=0.0)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0  # x increases linearly

                with if_eq(m.use_linear > 0.5):
                    m.y == m.x  # Linear: y = x
                with else_eq():
                    m.y == m.x**2  # Quadratic: y = x^2

        # Test with linear mode (default use_linear=1.0)
        linear_model = LinearOrQuadratic()
        flat_linear = linear_model.flatten()
        compiled_linear = CasadiBackend.compile(flat_linear)
        result_linear = compiled_linear.simulate(tf=2.0)

        # At t=2, x=2, so y should be 2 (linear)
        final_y_linear = result_linear("y")[-1]
        assert abs(final_y_linear - 2.0) < 0.1

        # Test with quadratic mode using parameter override in simulate
        result_quad = compiled_linear.simulate(tf=2.0, params={"use_linear": 0.0})

        # At t=2, x=2, so y should be 4 (quadratic)
        final_y_quad = result_quad("y")[-1]
        assert abs(final_y_quad - 4.0) < 0.2

    def test_if_eq_with_state_variable_condition(self) -> None:
        """Test if-equation with state-dependent condition."""
        from cyecca.dsl import der, else_eq, equations, if_eq, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class SaturatedGrowth:
            x = var(start=0.1)
            rate = var(output=True)

            @equations
            def _(m):
                # Rate changes based on state
                with if_eq(m.x < 1.0):
                    m.rate == 1.0  # Fast growth when small
                with else_eq():
                    m.rate == 0.1  # Slow growth when large

                der(m.x) == m.rate

        instance = SaturatedGrowth()
        flat = instance.flatten()
        compiled = CasadiBackend.compile(flat)
        result = compiled.simulate(tf=3.0)

        # Initially x < 1, so rate = 1.0
        # After x crosses 1, rate drops to 0.1
        # Get x trajectory from states
        final_x = result("x")[-1]
        # x should be > 1 but not grow much after crossing
        assert final_x > 1.0
        assert final_x < 2.0  # Growth slowed after crossing


class TestIfEquationValidation:
    """Test validation of if-equation constraints."""

    def test_elseif_requires_preceding_if(self) -> None:
        """Test that elseif_eq without preceding if_eq raises error."""
        from cyecca.dsl import elseif_eq, equations, model, var

        @model
        class M:
            x = var()
            y = var()

            @equations
            def _(m):
                # elseif without if should fail
                with elseif_eq(m.x > 0):
                    m.y == m.x

        instance = M()
        with pytest.raises(RuntimeError, match="must immediately follow"):
            instance.flatten()

    def test_else_requires_preceding_if(self) -> None:
        """Test that else_eq without preceding if_eq raises error."""
        from cyecca.dsl import else_eq, equations, model, var

        @model
        class M:
            x = var()
            y = var()

            @equations
            def _(m):
                # else without if should fail
                with else_eq():
                    m.y == m.x

        instance = M()
        with pytest.raises(RuntimeError, match="must immediately follow"):
            instance.flatten()


class TestIfEquationBranch:
    """Test IfEquationBranch dataclass."""

    def test_if_equation_branch_creation(self) -> None:
        """Test creating IfEquationBranch directly."""
        from cyecca.dsl import Equation, Expr, ExprKind, IfEquationBranch

        condition = Expr(
            ExprKind.GT,
            children=[
                Expr(ExprKind.VARIABLE, name="x"),
                Expr(ExprKind.CONSTANT, value=0.0),
            ],
        )
        eq = Equation(
            lhs=Expr(ExprKind.VARIABLE, name="y"),
            rhs=Expr(ExprKind.VARIABLE, name="x"),
        )
        branch = IfEquationBranch(condition=condition, body=[eq])

        assert branch.condition is not None
        assert len(branch.body) == 1
        assert "IfBranch" in repr(branch)

    def test_else_branch_has_no_condition(self) -> None:
        """Test that else branch has condition=None."""
        from cyecca.dsl import Equation, Expr, ExprKind, IfEquationBranch

        eq = Equation(
            lhs=Expr(ExprKind.VARIABLE, name="y"),
            rhs=Expr(ExprKind.CONSTANT, value=0.0),
        )
        else_branch = IfEquationBranch(condition=None, body=[eq])

        assert else_branch.condition is None
        assert "ElseBranch" in repr(else_branch)


class TestIfEquationDataclass:
    """Test IfEquation dataclass."""

    def test_if_equation_creation(self) -> None:
        """Test creating IfEquation directly."""
        from cyecca.dsl import Equation, Expr, ExprKind, IfEquation, IfEquationBranch

        cond = Expr(
            ExprKind.GT,
            children=[
                Expr(ExprKind.VARIABLE, name="x"),
                Expr(ExprKind.CONSTANT, value=0.0),
            ],
        )
        then_eq = Equation(
            lhs=Expr(ExprKind.VARIABLE, name="y"),
            rhs=Expr(ExprKind.VARIABLE, name="x"),
        )
        else_eq = Equation(
            lhs=Expr(ExprKind.VARIABLE, name="y"),
            rhs=Expr(ExprKind.CONSTANT, value=0.0),
        )

        if_eq = IfEquation(
            branches=[
                IfEquationBranch(condition=cond, body=[then_eq]),
                IfEquationBranch(condition=None, body=[else_eq]),  # else
            ]
        )

        assert len(if_eq.branches) == 2
        assert "IfEquation" in repr(if_eq)

    def test_if_equation_expand(self) -> None:
        """Test IfEquation.expand() creates conditional equations."""
        from cyecca.dsl import Equation, Expr, ExprKind, IfEquation, IfEquationBranch

        cond = Expr(
            ExprKind.GT,
            children=[
                Expr(ExprKind.VARIABLE, name="x"),
                Expr(ExprKind.CONSTANT, value=0.0),
            ],
        )
        then_eq = Equation(
            lhs=Expr(ExprKind.VARIABLE, name="y"),
            rhs=Expr(ExprKind.VARIABLE, name="x"),
        )
        else_eq = Equation(
            lhs=Expr(ExprKind.VARIABLE, name="y"),
            rhs=Expr(ExprKind.CONSTANT, value=0.0),
        )

        if_eq = IfEquation(
            branches=[
                IfEquationBranch(condition=cond, body=[then_eq]),
                IfEquationBranch(condition=None, body=[else_eq]),
            ]
        )

        expanded = if_eq.expand()

        assert len(expanded) == 1
        eq = expanded[0]
        # RHS should be IF_THEN_ELSE
        assert eq.rhs.kind == ExprKind.IF_THEN_ELSE

    def test_if_equation_prefix_names(self) -> None:
        """Test IfEquation._prefix_names for submodel flattening."""
        from cyecca.dsl import Equation, Expr, ExprKind, IfEquation, IfEquationBranch

        cond = Expr(
            ExprKind.GT,
            children=[
                Expr(ExprKind.VARIABLE, name="x"),
                Expr(ExprKind.CONSTANT, value=0.0),
            ],
        )
        eq = Equation(
            lhs=Expr(ExprKind.VARIABLE, name="y"),
            rhs=Expr(ExprKind.VARIABLE, name="x"),
        )
        if_eq = IfEquation(
            branches=[
                IfEquationBranch(condition=cond, body=[eq]),
            ]
        )

        prefixed = if_eq._prefix_names("sub")

        # The condition and equations should have prefixed names
        assert prefixed.branches[0].condition is not None
