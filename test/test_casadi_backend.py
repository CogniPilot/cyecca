"""
Tests for the CasADi backend.
"""

import pytest
import numpy as np
from cyecca.ir import Model, Variable, VariableType, Expr, Equation, der
from cyecca.backends.casadi import CasadiBackend


def test_simple_ode():
    """Test compilation and simulation of a simple ODE: dx/dt = -k*x."""
    model = Model(name="SimpleODE")

    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=0.5))

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")
    rhs = Expr.mul(Expr.neg(k), x)
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), rhs))

    # Compile and simulate
    backend = CasadiBackend(model)
    backend.compile()

    t, sol = backend.simulate(t_final=5.0, dt=0.1)

    # Check that x decays exponentially
    assert len(t) > 0
    assert "x" in sol
    assert sol["x"][0] == pytest.approx(1.0, abs=1e-6)
    assert sol["x"][-1] < sol["x"][0]  # Should decay
    assert sol["x"][-1] == pytest.approx(np.exp(-0.5 * 5.0), abs=0.01)


def test_mass_spring_damper():
    """Test mass-spring-damper system."""
    model = Model(name="MassSpringDamper")

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

    # der(x) = v
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), v))

    # der(v) = (F - c*v - k*x) / m
    force = Expr.sub(F, Expr.mul(c, v))
    force = Expr.sub(force, Expr.mul(k, x))
    accel = Expr.div(force, m)
    model.add_equation(Equation.simple(der(Expr.var_ref("v")), accel))

    # Compile
    backend = CasadiBackend(model)
    backend.compile()

    # Simulate with no input (free oscillation)
    t, sol = backend.simulate(t_final=10.0, dt=0.01)

    # Check basic properties
    assert len(t) > 0
    assert "x" in sol
    assert "v" in sol
    assert sol["x"][0] == pytest.approx(1.0, abs=1e-6)

    # System should oscillate and decay
    assert np.abs(sol["x"][-1]) < np.abs(sol["x"][0])


def test_linearization():
    """Test linearization of a simple system."""
    model = Model(name="Linear")

    # dx/dt = A*x + B*u, where A=-1, B=1
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="u", var_type=VariableType.INPUT))

    x = Expr.var_ref("x")
    u = Expr.var_ref("u")

    # der(x) = -x + u
    rhs = Expr.add(Expr.neg(x), u)
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), rhs))

    # Compile and linearize
    backend = CasadiBackend(model)
    backend.compile()

    A, B, C, D = backend.linearize()

    # Check dimensions
    assert A.shape == (1, 1)
    assert B.shape == (1, 1)

    # Check values
    assert A[0, 0] == pytest.approx(-1.0, abs=1e-6)
    assert B[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_output_equation():
    """Test model with output equation."""
    model = Model(name="WithOutput")

    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="y", var_type=VariableType.OUTPUT))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=2.0))

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")

    # der(x) = -x
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.neg(x)))

    # y = k * x
    y = Expr.var_ref("y")
    model.add_equation(Equation.simple(y, Expr.mul(k, x)))

    backend = CasadiBackend(model)
    backend.compile()

    t, sol = backend.simulate(t_final=1.0, dt=0.1)

    # Check output is present
    assert "y" in sol
    # y should be 2*x at all times
    assert np.allclose(sol["y"], 2.0 * sol["x"], atol=1e-6)


def test_parameter_modification():
    """Test changing parameter values."""
    model = Model(name="ParamTest")

    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=1.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=1.0))

    x = Expr.var_ref("x")
    k = Expr.var_ref("k")
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), Expr.mul(Expr.neg(k), x)))

    backend = CasadiBackend(model)
    backend.compile()

    # Simulate with k=1.0
    t1, sol1 = backend.simulate(t_final=1.0, dt=0.1)

    # Change parameter
    backend.set_parameter("k", 2.0)

    # Simulate with k=2.0
    t2, sol2 = backend.simulate(t_final=1.0, dt=0.1)

    # Decay should be faster with k=2.0
    assert sol2["x"][-1] < sol1["x"][-1]


def test_trig_functions():
    """Test trigonometric function support."""
    model = Model(name="TrigTest")

    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="omega", var_type=VariableType.PARAMETER, value=1.0))

    # der(x) = sin(omega * t) - not time-dependent yet, so use sin(x)
    x = Expr.var_ref("x")
    omega = Expr.var_ref("omega")

    # der(x) = sin(omega * x)
    rhs = Expr.sin(Expr.mul(omega, x))
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), rhs))

    backend = CasadiBackend(model)
    backend.compile()

    t, sol = backend.simulate(t_final=1.0, dt=0.01)

    # Just check it runs without error
    assert len(t) > 0


def test_rumoca_bouncing_ball_integration():
    """
    Integration test: Compile bouncing ball with Rumoca, import to Cyecca, simulate with CasADi.

    This tests the full pipeline:
    1. Rumoca compiles Modelica to Base Modelica JSON
    2. Cyecca imports the JSON
    3. CasADi backend compiles and simulates the model
    """
    import tempfile
    from pathlib import Path
    from cyecca.io.base_modelica import import_base_modelica

    # Try to import rumoca (skip test if not available)
    try:
        import rumoca
    except ImportError:
        pytest.skip("Rumoca Python package not installed")

    # Define the bouncing ball model
    modelica_code = """
model BouncingBall "The 'classic' bouncing ball model"
  parameter Real e=0.8 "Coefficient of restitution";
  parameter Real h0=1.0 "Initial height";
  Real h = 1.0 "Height";
  Real v "Velocity";
  Real z;
equation
  z = 2*h + v;
  v = der(h);
  der(v) = -9.81;
  when h<0 then
    reinit(v, -e*pre(v));
  end when;
end BouncingBall;
"""

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mo_file = tmpdir / "bouncing_ball.mo"
        json_file = tmpdir / "bouncing_ball.json"

        # Write Modelica file
        mo_file.write_text(modelica_code)

        # Compile with Rumoca Python API
        try:
            print("Compiling bouncing ball model with Rumoca...")
            result = rumoca.compile(str(mo_file))
            result.export_base_modelica_json(str(json_file))
            print("✓ Rumoca compilation successful")
        except Exception as e:
            pytest.skip(f"Rumoca compilation failed: {e}")

        # Import into Cyecca
        print("Importing into Cyecca...")
        model = import_base_modelica(json_file)
        print(f"✓ Model imported: {model.name}")

        # Verify model structure (name may be "BouncingBall" or "GeneratedModel")
        assert model.name in [
            "BouncingBall",
            "GeneratedModel",
        ], f"Unexpected model name: {model.name}"

        # Check we have the expected variables
        var_names = {v.name for v in model.variables}
        print(f"Variables: {var_names}")
        assert "h" in var_names, "Should have height variable"
        assert "v" in var_names, "Should have velocity variable"
        assert "e" in var_names, "Should have coefficient of restitution parameter"
        print("✓ Variable validation passed")

        # Compile with CasADi backend
        print("\nCompiling with CasADi backend...")
        backend = CasadiBackend(model)
        backend.compile()
        print("✓ CasADi backend compilation successful")

        # Verify backend has states
        assert (
            "h" in backend.state_names or "v" in backend.state_names
        ), "Backend should recognize at least one state variable"
        print(f"States: {backend.state_names}")

        # Simulate the bouncing ball with event handling
        # Rumoca exports the when equation: when h<0 then reinit(v, -e*pre(v))
        # For now, we manually implement the event handling until the CasADi backend
        # has automatic support for when equations
        print("Running simulation with event handling...")

        # Get state indices
        h_idx = backend.state_names.index("h")
        v_idx = backend.state_names.index("v")

        # Get coefficient of restitution
        e = backend.param_values["e"]

        # Define ground collision event: h = 0
        def ground_collision(t, x):
            return x[h_idx]  # Event occurs when height crosses zero

        ground_collision.terminal = True  # Stop at event
        ground_collision.direction = -1  # Only when going down (h decreasing)

        # Simulate in segments, resetting velocity at each bounce
        from scipy.integrate import solve_ivp

        # Get initial conditions from backend
        x0 = np.array([backend._get_numeric_start_value(name) for name in backend.state_names])
        p = np.array([backend.param_values[name] for name in backend.param_names])

        print(f"Initial conditions: h={x0[h_idx]:.3f}, v={x0[v_idx]:.3f}")
        print(f"Parameters: e={e:.3f}")

        def rhs_scipy(t, x):
            u = np.zeros(len(backend.input_names))
            return backend.rhs_func(t, x, u, p).full().flatten()

        # Collect all time points and states
        t_all = []
        h_all = []
        v_all = []

        t_current = 0.0
        x_current = x0.copy()
        dt = 0.01
        t_final_sim = 5.0
        bounce_count = 0

        while t_current < t_final_sim and bounce_count < 50:  # Max 50 bounces
            # Skip if ball is at or below ground with no velocity
            if x_current[h_idx] <= 1e-6 and abs(x_current[v_idx]) < 1e-6:
                print(f"  Ball at rest on ground at t={t_current:.3f}s")
                break

            # Simulate until next bounce or end time
            sol_seg = solve_ivp(
                rhs_scipy,
                [t_current, t_final_sim],
                x_current,
                events=[ground_collision],
                dense_output=True,
                max_step=0.01,
            )

            # Check if we made progress
            if sol_seg.t[-1] <= t_current + 1e-9:
                print(f"  Warning: No progress at t={t_current:.3f}s, terminating")
                break

            # Add this segment's data
            if bounce_count == 0:
                # First segment - include initial point
                t_seg = sol_seg.t
                x_seg = sol_seg.y
            else:
                # Later segments - skip initial point to avoid duplicates
                t_seg = sol_seg.t[1:]
                x_seg = sol_seg.y[:, 1:]

            t_all.extend(t_seg)
            h_all.extend(x_seg[h_idx, :])
            v_all.extend(x_seg[v_idx, :])

            # Check if we hit the ground
            if sol_seg.status == 1 and sol_seg.t_events[0].size > 0:
                t_bounce = sol_seg.t_events[0][0]
                x_bounce = sol_seg.y_events[0][0]

                bounce_count += 1
                print(
                    f"  Bounce #{bounce_count} at t={t_bounce:.3f}s, v_before={x_bounce[v_idx]:.3f}"
                )

                # Apply reinit: v_new = -e * v_old
                x_bounce[v_idx] = -e * x_bounce[v_idx]
                print(f"    v_after={x_bounce[v_idx]:.3f}")

                # Make sure height is exactly 0 at bounce
                x_bounce[h_idx] = 0.0

                # Continue from just after bounce
                t_current = t_bounce
                x_current = x_bounce
            else:
                # No more events - reached end time
                break

        # Convert to numpy arrays
        t_result = np.array(t_all)
        h_result = np.array(h_all)
        v_result = np.array(v_all)

        # Create result dict
        sol = {"h": h_result, "v": v_result}

        print(f"✓ Simulation successful ({len(t_result)} time points, {bounce_count} bounces)")

        # Verify we have time points
        assert len(t_result) > 0, "Should have time points"

        # Check that we have solution data for key variables
        assert "h" in sol, "Should have height in solution"
        assert "v" in sol, "Should have velocity in solution"
        assert len(sol["h"]) == len(t_result), "Height solution should match time points"
        assert len(sol["v"]) == len(t_result), "Velocity solution should match time points"

        print(f"  Final h = {sol['h'][-1]:.4f}")
        print(f"  Final v = {sol['v'][-1]:.4f}")

        # Check for bouncing behavior
        # With manual event handling, the ball should:
        # 1. Hit the ground (h=0)
        # 2. Reverse velocity (v changes sign)
        # 3. Bounce back up (h increases again)
        print("\nChecking for bouncing behavior...")

        # Find if the ball stays above ground (allowing small numerical noise)
        min_h = np.min(sol["h"])
        print(f"  Minimum height: {min_h:.4f}")

        # Check if velocity ever reverses (indicating a bounce)
        v_changes_sign = np.any(np.diff(np.sign(sol["v"])) != 0)
        print(f"  Velocity changes sign: {v_changes_sign}")

        # Check that we got multiple bounces
        print(f"  Total bounces detected: {bounce_count}")

        # Validate bouncing behavior
        assert min_h >= -0.01, f"Ball fell significantly through ground: h={min_h:.4f}"
        assert v_changes_sign, "Velocity should change sign (bounce detected)"
        assert bounce_count >= 5, f"Expected multiple bounces, got {bounce_count}"

        print("✓ Bouncing behavior verified!")
        print("\n✓✓✓ Full integration test PASSED with bouncing! ✓✓✓")
