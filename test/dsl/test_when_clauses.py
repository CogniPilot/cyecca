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
        from cyecca.dsl import WhenClause, der, equations, model, pre, reinit, var, when

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
        from beartype.roar import BeartypeCallHintParamViolation

        from cyecca.dsl import reinit

        with pytest.raises(BeartypeCallHintParamViolation):
            reinit("not_a_var", 1.0)

    def test_reinit_requires_scalar_var(self) -> None:
        """Test that reinit() currently requires scalar variables."""
        from cyecca.dsl import model, reinit, var

        @model
        class M:
            pos = var(shape=(3,))

        instance = M()
        with pytest.raises(TypeError, match="scalar variables"):
            reinit(instance.pos, 0.0)

    def test_flat_model_contains_when_clauses(self) -> None:
        """Test that FlatModel correctly collects when_clauses."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when

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
        from cyecca.dsl import der, equations, model, pre, reinit, var, when

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
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
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
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
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
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
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
            if h[i] > h[i - 1] and h[i] > h[i + 1]:
                apex_heights.append(h[i])

        # Each apex should be lower than the previous (energy loss)
        if len(apex_heights) >= 2:
            for i in range(1, len(apex_heights)):
                assert apex_heights[i] < apex_heights[i - 1] + 0.1, f"Apex {i} should be lower than apex {i-1}"


class TestMultipleWhenClauses:
    """Test models with multiple when-clauses."""

    def test_multiple_when_clauses(self) -> None:
        """Test model with multiple independent when-clauses."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
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
        from cyecca.dsl import ExprKind, der, equations, model, pre, reinit, var, when

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
        from cyecca.dsl import ExprKind, der, equations, model, pre, reinit, var, when

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
        from cyecca.dsl import ExprKind, der, equations, model, pre, reinit, var, when

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
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
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
        from cyecca.dsl import der, equations, model, pre, reinit, submodel, var, when

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


class TestBouncingBallPhysics:
    """Comprehensive tests for bouncing ball physics simulation."""

    def test_bouncing_ball_first_bounce_time(self) -> None:
        """Test that first bounce occurs at the correct time.

        For h0=1.0, g=9.81: t = sqrt(2*h0/g) ≈ 0.4515 s
        """
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
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
        result = compiled.simulate(tf=1.0, dt=0.0001)

        h = result("h")
        t = result.t

        # Find first time when h is near zero (first bounce)
        # After initial drop, h should reach near 0
        first_bounce_idx = np.argmin(h[: len(h) // 2])  # Look in first half
        first_bounce_time = t[first_bounce_idx]

        expected_time = np.sqrt(2 * 1.0 / 9.81)  # ~0.4515 s
        assert (
            abs(first_bounce_time - expected_time) < 0.01
        ), f"First bounce at t={first_bounce_time:.4f}, expected {expected_time:.4f}"

    def test_bouncing_ball_velocity_reversal(self) -> None:
        """Test that velocity reverses with correct coefficient at bounce."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
        from cyecca.dsl.backends import CasadiBackend

        e_value = 0.7  # Restitution coefficient

        @model
        class BouncingBall:
            h = var(start=1.0)
            v = var(start=0.0)
            e = var(e_value, parameter=True)

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81

                with when(m.h < 0):
                    reinit(m.v, -m.e * pre(m.v))

        compiled = CasadiBackend.compile(BouncingBall().flatten())
        result = compiled.simulate(tf=2.0, dt=0.0001)

        v = result("v")

        # Velocity at first bounce: v = sqrt(2*g*h0) ≈ 4.43 m/s (downward, negative)
        expected_v_at_bounce = -np.sqrt(2 * 9.81 * 1.0)

        # Find velocity just before and after first bounce
        # Velocity goes from negative to positive
        for i in range(1, len(v)):
            if v[i - 1] < -3.0 and v[i] > 0:  # Large negative to positive
                v_before = v[i - 1]
                v_after = v[i]
                # Check restitution: v_after = -e * v_before
                ratio = -v_after / v_before
                assert abs(ratio - e_value) < 0.1, f"Velocity ratio {ratio:.3f}, expected {e_value}"
                break

    def test_bouncing_ball_multiple_bounces(self) -> None:
        """Test that ball bounces multiple times with decreasing height."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
        from cyecca.dsl.backends import CasadiBackend

        @model
        class BouncingBall:
            h = var(start=2.0)  # Start higher for more bounces
            v = var(start=0.0)
            e = var(0.8, parameter=True)

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
        apex_heights = []
        for i in range(1, len(h) - 1):
            if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > 0.01:
                apex_heights.append(h[i])

        # Should have multiple bounces
        assert len(apex_heights) >= 3, f"Expected at least 3 bounces, got {len(apex_heights)}"

        # Each apex should be lower than previous (energy loss)
        for i in range(1, len(apex_heights)):
            assert (
                apex_heights[i] < apex_heights[i - 1] * 1.01
            ), f"Apex {i} ({apex_heights[i]:.3f}) should be lower than apex {i-1} ({apex_heights[i-1]:.3f})"

    def test_bouncing_ball_energy_conservation_ratio(self) -> None:
        """Test that height ratio between bounces equals e^2."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
        from cyecca.dsl.backends import CasadiBackend

        e_value = 0.9  # High restitution for clearer bounces

        @model
        class BouncingBall:
            h = var(start=1.0)
            v = var(start=0.0)
            e = var(e_value, parameter=True)

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81

                with when(m.h < 0):
                    reinit(m.v, -m.e * pre(m.v))

        compiled = CasadiBackend.compile(BouncingBall().flatten())
        result = compiled.simulate(tf=5.0, dt=0.0001)

        h = result("h")

        # Find apex heights
        apex_heights = []
        for i in range(1, len(h) - 1):
            if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > 0.05:
                apex_heights.append(h[i])

        # Height ratio should be e^2 (energy goes as v^2, height ~ v^2/2g)
        expected_ratio = e_value**2

        if len(apex_heights) >= 2:
            actual_ratio = apex_heights[1] / apex_heights[0]
            assert (
                abs(actual_ratio - expected_ratio) < 0.05
            ), f"Height ratio {actual_ratio:.3f}, expected {expected_ratio:.3f}"

    def test_bouncing_ball_with_initial_velocity(self) -> None:
        """Test bouncing ball thrown upward."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
        from cyecca.dsl.backends import CasadiBackend

        @model
        class BouncingBall:
            h = var(start=0.5)  # Start at 0.5m
            v = var(start=5.0)  # Throw upward at 5 m/s
            e = var(0.8, parameter=True)

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81

                with when(m.h < 0):
                    reinit(m.v, -m.e * pre(m.v))

        compiled = CasadiBackend.compile(BouncingBall().flatten())
        result = compiled.simulate(tf=3.0, dt=0.001)

        h = result("h")

        # Ball should go up first then come down
        # Max height = h0 + v0^2/(2g) = 0.5 + 25/19.62 ≈ 1.77m
        max_h = np.max(h)
        expected_max = 0.5 + (5.0**2) / (2 * 9.81)

        assert abs(max_h - expected_max) < 0.1, f"Max height {max_h:.3f}, expected {expected_max:.3f}"

    def test_bouncing_ball_perfect_elastic(self) -> None:
        """Test bouncing ball with e=1.0 (perfect elastic collision)."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
        from cyecca.dsl.backends import CasadiBackend

        @model
        class BouncingBall:
            h = var(start=1.0)
            v = var(start=0.0)
            e = var(1.0, parameter=True)  # Perfect elastic

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81

                with when(m.h < 0):
                    reinit(m.v, -m.e * pre(m.v))

        compiled = CasadiBackend.compile(BouncingBall().flatten())
        result = compiled.simulate(tf=3.0, dt=0.001)

        h = result("h")

        # Find apex heights - should all be approximately equal
        apex_heights = []
        for i in range(1, len(h) - 1):
            if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > 0.5:
                apex_heights.append(h[i])

        if len(apex_heights) >= 2:
            # All apex heights should be close to initial height
            for apex in apex_heights:
                assert abs(apex - 1.0) < 0.15, f"Apex height {apex:.3f} should be ~1.0 for elastic collision"

    def test_bouncing_ball_zero_restitution(self) -> None:
        """Test bouncing ball with e=0 (completely inelastic)."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
        from cyecca.dsl.backends import CasadiBackend

        @model
        class BouncingBall:
            h = var(start=1.0)
            v = var(start=0.0)
            e = var(0.0, parameter=True)  # Completely inelastic

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81

                with when(m.h < 0):
                    reinit(m.v, -m.e * pre(m.v))

        compiled = CasadiBackend.compile(BouncingBall().flatten())
        result = compiled.simulate(tf=2.0, dt=0.001)

        h = result("h")
        v = result("v")

        # After first bounce, ball should stop (v=0)
        # Find index after first bounce
        first_bounce_idx = None
        for i in range(1, len(v)):
            if v[i - 1] < -1.0 and v[i] >= -0.5:  # Velocity changes from negative
                first_bounce_idx = i
                break

        if first_bounce_idx is not None:
            # After bounce, velocity should be near zero
            v_after = np.abs(v[first_bounce_idx : first_bounce_idx + 100])
            assert np.mean(v_after) < 0.5, "Ball should stop after inelastic collision"


class TestDiscreteVariables:
    """Test discrete variables with when-clauses.

    Discrete variables are piecewise constant and can only change at events
    via reinit() statements in when-clauses.
    """

    def test_discrete_variable_counter(self) -> None:
        """Test a counter that increments on each event."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
        from cyecca.dsl.backends import CasadiBackend

        @model
        class EventCounter:
            # Continuous variable that triggers events
            x = var(start=0.0)
            # Discrete counter - increments on each event
            count = var(0, discrete=True)
            # Threshold for events (crosses every 1.0 units of x)
            threshold = var(1.0, parameter=True)

            @equations
            def _(m):
                der(m.x) == 1.0  # x increases linearly

                # When x crosses threshold, reset x and increment count
                with when(m.x > m.threshold):
                    reinit(m.x, 0.0)
                    reinit(m.count, pre(m.count) + 1)

        flat = EventCounter().flatten()

        # Verify discrete variable is correctly classified
        assert "count" in flat.discrete_names
        assert "x" in flat.state_names

        # Compile and simulate
        compiled = CasadiBackend.compile(flat)
        assert compiled.has_discrete
        assert compiled.has_events

        result = compiled.simulate(tf=5.0, dt=0.01)

        # Check that count is in results
        assert "count" in result.available_names

        # After 5 seconds with events every 1 second, should have ~5 events
        count = result("count")
        assert count[-1] >= 4, f"Expected at least 4 events, got {count[-1]}"
        assert count[-1] <= 6, f"Expected at most 6 events, got {count[-1]}"

        # Count should only increase (discrete, monotonic)
        for i in range(1, len(count)):
            assert count[i] >= count[i - 1], "Counter should never decrease"

    def test_discrete_variable_in_result(self) -> None:
        """Test that discrete variable trajectory is available in result."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            t_trigger = var(start=0.0)
            mode = var(0, discrete=True)

            @equations
            def _(m):
                der(m.t_trigger) == 1.0

                with when(m.t_trigger > 2.0):
                    reinit(m.t_trigger, 0.0)
                    reinit(m.mode, pre(m.mode) + 1)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=7.0, dt=0.01)

        # Access discrete variable via different methods
        mode_via_call = result("mode")
        assert mode_via_call is not None

        # Check discrete property
        assert "mode" in result.discrete
        assert np.array_equal(result.discrete["mode"], mode_via_call)

    def test_discrete_without_events_stays_constant(self) -> None:
        """Test discrete variable without events stays at initial value."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            flag = var(42, discrete=True)  # Initial value 42

            @equations
            def _(m):
                der(m.x) == 1.0
                # No when-clauses - flag should never change

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=5.0, dt=0.1)

        # Without events, discrete stays constant
        flag = result("flag")
        assert np.all(flag == 42), "Discrete var should stay at initial value"

    def test_discrete_multiple_reinits_same_event(self) -> None:
        """Test multiple discrete variables updated in same event."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
        from cyecca.dsl.backends import CasadiBackend

        @model
        class MultiDiscrete:
            x = var(start=0.0)
            a = var(0, discrete=True)
            b = var(10, discrete=True)

            @equations
            def _(m):
                der(m.x) == 1.0

                with when(m.x > 1.0):
                    reinit(m.x, 0.0)
                    reinit(m.a, pre(m.a) + 1)
                    reinit(m.b, pre(m.b) - 1)

        compiled = CasadiBackend.compile(MultiDiscrete().flatten())
        result = compiled.simulate(tf=5.0, dt=0.01)

        a = result("a")
        b = result("b")

        # Both should change at same events
        final_a = a[-1]
        final_b = b[-1]

        # a starts at 0, increments ~5 times; b starts at 10, decrements ~5 times
        assert final_a >= 4, f"Expected a >= 4, got {final_a}"
        assert final_b <= 6, f"Expected b <= 6, got {final_b}"

        # Check they changed together: sum should stay constant (0 + 10 = 10)
        # Since a goes up by 1 and b goes down by 1, a + b should stay ~10
        assert np.abs(final_a + final_b - 10) < 2, "a + b should stay near 10"

