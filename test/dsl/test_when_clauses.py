"""
Tests for when-clauses (hybrid systems support) in the Cyecca DSL.

Covers:
- when() context manager
- reinit() function
- WhenClause dataclass
- Event detection in CasADi backend simulation

When-clauses are defined inside @equations methods using the when() context
manager. The reinit() function auto-registers when inside a when() block.
"""

import numpy as np
import pytest


class TestWhenClauseBasics:
    """Test basic when-clause syntax and data structures."""

    def test_when_context_creates_when_clause(self) -> None:
        """Test that when() context manager creates WhenClause."""
        from cyecca.dsl import model, var, der, when, reinit, pre, WhenClause, equations

        @model
        class M:
            h = var(start=1.0)
            v = var(start=0.0)
            e = var(0.8, parameter=True)

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81
                
                # When-clause using new @equations syntax
                with when(m.h < 0):
                    reinit(m.v, -m.e * pre(m.v))

        instance = M()
        flat = instance.flatten()
        
        assert len(flat.when_clauses) == 1
        wc = flat.when_clauses[0]
        assert isinstance(wc, WhenClause)
        assert len(wc.body) == 1

    def test_reinit_requires_symbolic_var(self) -> None:
        """Test that reinit() requires a SymbolicVar."""
        from cyecca.dsl import reinit
        from beartype.roar import BeartypeCallHintParamViolation

        with pytest.raises(BeartypeCallHintParamViolation):
            reinit("not_a_var", 1.0)

    def test_reinit_requires_scalar_var(self) -> None:
        """Test that reinit() currently requires scalar variables."""
        from cyecca.dsl import model, var, reinit

        @model
        class M:
            pos = var(shape=(3,))

        instance = M()
        with pytest.raises(TypeError, match="scalar variables"):
            reinit(instance.pos, 0.0)

    def test_flat_model_contains_when_clauses(self) -> None:
        """Test that FlatModel correctly collects when_clauses."""
        from cyecca.dsl import model, var, der, when, reinit, pre, equations

        @model
        class Counter:
            trigger = var(start=0.0)
            count = var(start=0.0)

            @equations
            def _(m):
                der(m.trigger) == 1.0
                der(m.count) == 0.0
                
                # When-clause in @equations method
                with when(m.trigger > 0.5):
                    reinit(m.count, pre(m.count) + 1)

        flat = Counter().flatten()
        assert len(flat.when_clauses) == 1


class TestBouncingBall:
    """Test the classic bouncing ball hybrid system."""

    def test_bouncing_ball_model_definition(self) -> None:
        """Test that bouncing ball model can be defined."""
        from cyecca.dsl import model, var, der, when, reinit, pre, equations

        @model
        class BouncingBall:
            """Bouncing ball with restitution coefficient."""
            h = var(start=1.0, unit="m")
            v = var(start=0.0, unit="m/s")
            e = var(0.8, parameter=True)  # Restitution coefficient

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81
                
                # When-clause for bounce
                with when(m.h < 0):
                    reinit(m.v, -m.e * pre(m.v))

        ball = BouncingBall()
        flat = ball.flatten()
        
        assert "h" in flat.state_names
        assert "v" in flat.state_names
        assert "e" in flat.param_names
        assert len(flat.when_clauses) == 1

    def test_bouncing_ball_compiles(self) -> None:
        """Test that bouncing ball model compiles with CasADi backend."""
        from cyecca.dsl import model, var, der, when, reinit, pre, equations
        from cyecca.dsl.backends import CasadiBackend

        @model
        class BouncingBall:
            h = var(start=1.0)
            v = var(start=0.0)
            e = var(0.8, parameter=True)

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81
                
                with when(m.h < 0):
                    reinit(m.v, -m.e * pre(m.v))

        compiled = CasadiBackend.compile(BouncingBall().flatten())
        
        assert compiled.has_events
        assert len(compiled.when_clause_funcs) == 1

    def test_bouncing_ball_simulation(self) -> None:
        """Test bouncing ball simulation with event detection."""
        from cyecca.dsl import model, var, der, when, reinit, pre, equations
        from cyecca.dsl.backends import CasadiBackend

        @model
        class BouncingBall:
            h = var(start=1.0)
            v = var(start=0.0)
            e = var(0.8, parameter=True)

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81
                
                with when(m.h < 0):
                    reinit(m.v, -m.e * pre(m.v))

        ball = BouncingBall()
        compiled = CasadiBackend.compile(ball.flatten())
        result = compiled.simulate(tf=3.0, dt=0.001)

        # Check that simulation ran
        assert len(result.t) > 0
        
        # Height should be mostly positive (ball bounces)
        h = result(ball.h)
        
        # Ball should have bounced at least once
        # After first bounce, velocity should have changed sign
        v = result(ball.v)
        
        # Find sign changes in velocity (bounces)
        sign_changes = np.diff(np.sign(v))
        n_bounces = np.sum(sign_changes > 0)  # Positive to negative (hitting ground)
        
        assert n_bounces >= 1, "Ball should bounce at least once"

    def test_bouncing_ball_energy_loss(self) -> None:
        """Test that bouncing ball loses energy at each bounce."""
        from cyecca.dsl import model, var, der, when, reinit, pre, equations
        from cyecca.dsl.backends import CasadiBackend

        @model
        class BouncingBall:
            h = var(start=1.0)
            v = var(start=0.0)
            e = var(0.5, parameter=True)  # 50% energy loss per bounce

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81
                
                with when(m.h < 0):
                    reinit(m.v, -m.e * pre(m.v))

        compiled = CasadiBackend.compile(BouncingBall().flatten())
        result = compiled.simulate(tf=5.0, dt=0.001)

        h = result("h")
        
        # Find local maxima (apex of each bounce)
        # A local max is where h[i] > h[i-1] and h[i] > h[i+1]
        apex_heights = []
        for i in range(1, len(h) - 1):
            if h[i] > h[i-1] and h[i] > h[i+1]:
                apex_heights.append(h[i])

        # Each apex should be lower than the previous (energy loss)
        if len(apex_heights) >= 2:
            for i in range(1, len(apex_heights)):
                assert apex_heights[i] < apex_heights[i-1] + 0.1, \
                    f"Apex {i} should be lower than apex {i-1}"


class TestMultipleWhenClauses:
    """Test models with multiple when-clauses."""

    def test_multiple_when_clauses(self) -> None:
        """Test model with multiple independent when-clauses."""
        from cyecca.dsl import model, var, der, when, reinit, pre, equations
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TwoEvents:
            x = var(start=0.0)
            y = var(start=0.0)
            counter = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0
                der(m.y) == 2.0
                der(m.counter) == 0.0
                
                # First when-clause: Reset x when it exceeds 1
                with when(m.x > 1.0):
                    reinit(m.x, 0.0)
                
                # Second when-clause: Increment counter when y exceeds 2
                with when(m.y > 2.0):
                    reinit(m.y, 0.0)
                    reinit(m.counter, pre(m.counter) + 1)

        flat = TwoEvents().flatten()
        assert len(flat.when_clauses) == 2

        compiled = CasadiBackend.compile(flat)
        assert len(compiled.when_clause_funcs) == 2


class TestWhenClauseExpressions:
    """Test various condition expressions in when-clauses."""

    def test_when_with_lt_condition(self) -> None:
        """Test when-clause with less-than condition."""
        from cyecca.dsl import model, var, der, when, reinit, pre, ExprKind, equations

        @model
        class M:
            x = var(start=1.0)

            @equations
            def _(m):
                der(m.x) == -1.0
                
                with when(m.x < 0):
                    reinit(m.x, 1.0)

        flat = M().flatten()
        wc = flat.when_clauses[0]
        assert wc.condition.kind == ExprKind.LT

    def test_when_with_gt_condition(self) -> None:
        """Test when-clause with greater-than condition."""
        from cyecca.dsl import model, var, der, when, reinit, pre, ExprKind, equations

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0
                
                with when(m.x > 1.0):
                    reinit(m.x, 0.0)

        flat = M().flatten()
        wc = flat.when_clauses[0]
        assert wc.condition.kind == ExprKind.GT

    def test_when_with_le_condition(self) -> None:
        """Test when-clause with less-than-or-equal condition."""
        from cyecca.dsl import model, var, der, when, reinit, pre, ExprKind, equations

        @model
        class M:
            x = var(start=1.0)

            @equations
            def _(m):
                der(m.x) == -1.0
                
                with when(m.x <= 0):
                    reinit(m.x, 1.0)

        flat = M().flatten()
        wc = flat.when_clauses[0]
        assert wc.condition.kind == ExprKind.LE


class TestPreOperator:
    """Test the pre() operator in when-clauses."""

    def test_pre_in_reinit_expression(self) -> None:
        """Test that pre() works in reinit expressions."""
        from cyecca.dsl import model, var, der, when, reinit, pre, equations
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Accumulator:
            """Accumulates value every time x crosses threshold."""
            x = var(start=0.0)
            total = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0
                der(m.total) == 0.0
                
                with when(m.x > 1.0):
                    # Add pre(x) to total, then reset x
                    reinit(m.total, pre(m.total) + pre(m.x))
                    reinit(m.x, 0.0)

        compiled = CasadiBackend.compile(Accumulator().flatten())
        result = compiled.simulate(tf=5.0, dt=0.01)

        # Total should accumulate the x value at each crossing
        total = result("total")
        assert total[-1] > 0, "Total should have accumulated values"


class TestSubmodelWhenClauses:
    """Test when-clauses in submodels."""

    def test_submodel_when_clauses_are_collected(self) -> None:
        """Test that when-clauses in submodels are collected with prefix."""
        from cyecca.dsl import model, var, der, when, reinit, pre, submodel, equations

        @model
        class Ball:
            h = var(start=1.0)
            v = var(start=0.0)
            e = var(0.8, parameter=True)

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81
                
                with when(m.h < 0):
                    reinit(m.v, -m.e * pre(m.v))

        @model
        class TwoBalls:
            ball1 = submodel(Ball)
            ball2 = submodel(Ball)

        flat = TwoBalls().flatten()
        
        # Should have 2 when-clauses (one from each ball)
        assert len(flat.when_clauses) == 2
