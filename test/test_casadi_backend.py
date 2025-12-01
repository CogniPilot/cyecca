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


def test_array_variable():
    """Test array state variables (vectors)."""
    model = Model(name="ArrayTest")

    # Create a 3-element state vector
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, shape=[3], start=0.0))
    model.add_variable(
        Variable(name="v", var_type=VariableType.PARAMETER, shape=[3], value=[1.0, 2.0, 3.0])
    )

    # Each element evolves at a different rate: der(x[i]) = v[i]
    # We'll use direct indexing
    for i in range(1, 4):  # Modelica 1-based indexing
        x_i = Expr.component_ref(("x", [Expr.literal(i)]))
        v_i = Expr.component_ref(("v", [Expr.literal(i)]))
        model.add_equation(Equation.simple(der(x_i), v_i))

    backend = CasadiBackend(model)
    backend.compile()

    t, sol = backend.simulate(t_final=1.0, dt=0.1)

    # Check results - x[i] should be t * v[i] at t=1
    assert len(t) > 0
    assert "x" in sol

    # Check final values (x = v * t at t=1.0)
    x_final = sol["x"][:, -1]  # Get last timestep
    assert x_final[0] == pytest.approx(1.0, abs=0.01)  # x[1] = 1.0 * 1.0
    assert x_final[1] == pytest.approx(2.0, abs=0.01)  # x[2] = 2.0 * 1.0
    assert x_final[2] == pytest.approx(3.0, abs=0.01)  # x[3] = 3.0 * 1.0


def test_array_indexing_expression():
    """Test array indexing in expressions."""
    model = Model(name="ArrayIndexTest")

    # Create scalar states
    model.add_variable(Variable(name="y", var_type=VariableType.STATE, start=0.0))
    # Create array parameter
    model.add_variable(
        Variable(name="k", var_type=VariableType.PARAMETER, shape=[2], value=[1.0, 2.0])
    )

    # der(y) = k[1] + k[2]
    k1 = Expr.component_ref(("k", [Expr.literal(1)]))
    k2 = Expr.component_ref(("k", [Expr.literal(2)]))
    rhs = Expr.add(k1, k2)
    model.add_equation(Equation.simple(der(Expr.var_ref("y")), rhs))

    backend = CasadiBackend(model)
    backend.compile()

    t, sol = backend.simulate(t_final=1.0, dt=0.1)

    # y should grow at rate k[1] + k[2] = 3.0
    y_final = sol["y"][-1]
    assert y_final == pytest.approx(3.0, abs=0.01)


def test_hierarchical_component_ref():
    """Test hierarchical component references like vehicle.engine.temp."""
    model = Model(name="HierarchicalTest")

    # Create flattened variables with dot notation
    # In real Modelica, these would come from nested components
    model.add_variable(
        Variable(name="vehicle.engine.temp", var_type=VariableType.STATE, start=20.0)
    )
    model.add_variable(Variable(name="T_ambient", var_type=VariableType.PARAMETER, value=15.0))
    model.add_variable(Variable(name="k", var_type=VariableType.PARAMETER, value=0.1))

    # der(vehicle.engine.temp) = k * (T_ambient - vehicle.engine.temp)
    temp = Expr.component_ref("vehicle", "engine", "temp")
    ambient = Expr.var_ref("T_ambient")
    k = Expr.var_ref("k")
    rhs = Expr.mul(k, Expr.sub(ambient, temp))
    model.add_equation(Equation.simple(der(temp), rhs))

    backend = CasadiBackend(model)
    backend.compile()

    # Check that the variable was recognized
    assert "vehicle.engine.temp" in backend.symbols
    assert "vehicle.engine.temp" in backend.derivatives

    # Simulate - temperature should approach ambient
    t, sol = backend.simulate(t_final=50.0, dt=0.5)

    # Should cool down towards 15.0
    temp_final = sol["vehicle.engine.temp"][-1]
    assert temp_final < 20.0  # Should have cooled
    assert temp_final > 14.0  # But not below ambient


def test_hierarchical_component_ref_with_array():
    """Test hierarchical refs with array indexing: wheels[1].pressure."""
    model = Model(name="HierarchicalArrayTest")

    # Create variables for wheels[1].pressure, wheels[2].pressure
    # In flattened form: "wheels[1].pressure", etc.
    model.add_variable(Variable(name="wheels[1].pressure", var_type=VariableType.STATE, start=32.0))
    model.add_variable(Variable(name="wheels[2].pressure", var_type=VariableType.STATE, start=31.0))
    model.add_variable(Variable(name="leak_rate", var_type=VariableType.PARAMETER, value=0.01))

    # der(wheels[i].pressure) = -leak_rate * pressure
    leak = Expr.var_ref("leak_rate")

    # Wheel 1
    p1 = Expr.component_ref(("wheels", [Expr.literal(1)]), "pressure")
    model.add_equation(Equation.simple(der(p1), Expr.mul(Expr.neg(leak), p1)))

    # Wheel 2
    p2 = Expr.component_ref(("wheels", [Expr.literal(2)]), "pressure")
    model.add_equation(Equation.simple(der(p2), Expr.mul(Expr.neg(leak), p2)))

    backend = CasadiBackend(model)
    backend.compile()

    # Check variables are recognized
    assert "wheels[1].pressure" in backend.symbols
    assert "wheels[2].pressure" in backend.symbols

    # Simulate
    t, sol = backend.simulate(t_final=10.0, dt=0.1)

    # Both should decrease due to leak
    assert sol["wheels[1].pressure"][-1] < 32.0
    assert sol["wheels[2].pressure"][-1] < 31.0


@pytest.mark.integration
def test_rumoca_bouncing_ball_integration():
    """
    Integration test: Compile bouncing ball with Rumoca, import to Cyecca, simulate with CasADi.

    This tests the full pipeline:
    1. Rumoca compiles Modelica to Base Modelica JSON
    2. Cyecca imports the JSON
    3. CasADi backend compiles and simulates the model

    Requires: rumoca package (pip install rumoca)
    """
    import tempfile
    from pathlib import Path
    from cyecca.io.dae_ir import import_dae_ir

    # Skip if rumoca is not installed
    rumoca = pytest.importorskip("rumoca", reason="Rumoca Python package not installed")

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
        model = import_dae_ir(json_file)
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
        e = backend.param_defaults["e"]

        # Define ground collision event: h = 0
        def ground_collision(t, x):
            return x[h_idx]  # Event occurs when height crosses zero

        ground_collision.terminal = True  # Stop at event
        ground_collision.direction = -1  # Only when going down (h decreasing)

        # Simulate in segments, resetting velocity at each bounce
        from scipy.integrate import solve_ivp

        # Get initial conditions from backend
        x0 = np.array([backend.state_defaults[name] for name in backend.state_names])
        p = np.array([backend.param_defaults[name] for name in backend.param_names])

        print(f"Initial conditions: h={x0[h_idx]:.3f}, v={x0[v_idx]:.3f}")
        print(f"Parameters: e={e:.3f}")

        def rhs_scipy(t, x):
            u = np.zeros(len(backend.input_names))
            return np.array(backend.f_ode(x, u, p)).flatten()

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


def test_multiple_when_clauses():
    """Test multiple when-clauses (events) in a single model.

    This tests a double-bounded oscillator: a ball bouncing between floor and ceiling.
    - When x < 0 (hits floor): reverse velocity with restitution
    - When x > 1 (hits ceiling): reverse velocity with restitution
    """
    model = Model(name="DoubleBoundedOscillator")

    # State variables
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.5))
    model.add_variable(Variable(name="v", var_type=VariableType.STATE, start=1.0))

    # Parameters
    model.add_variable(Variable(name="e", var_type=VariableType.PARAMETER, value=0.9))

    # ODEs: dx/dt = v, dv/dt = 0 (no gravity - just momentum)
    x = Expr.var_ref("x")
    v = Expr.var_ref("v")
    e = Expr.var_ref("e")

    model.add_equation(Equation.simple(der(Expr.var_ref("x")), v))
    model.add_equation(Equation.simple(der(Expr.var_ref("v")), Expr.literal(0.0)))

    # When-clause 1: Floor collision (x < 0)
    # when x < 0 then reinit(v, -e * pre(v))
    floor_condition = Expr.binary_op("<", x, Expr.literal(0.0))
    pre_v = Expr.call("pre", v)
    floor_reset = Expr.mul(Expr.neg(e), pre_v)
    floor_eq = Equation.when(floor_condition, [Equation.simple(v, floor_reset)])
    model.add_equation(floor_eq)

    # When-clause 2: Ceiling collision (x > 1)
    # when x > 1 then reinit(v, -e * pre(v))
    ceiling_condition = Expr.binary_op(">", x, Expr.literal(1.0))
    ceiling_reset = Expr.mul(Expr.neg(e), pre_v)
    ceiling_eq = Equation.when(ceiling_condition, [Equation.simple(v, ceiling_reset)])
    model.add_equation(ceiling_eq)

    # Compile with CasADi
    backend = CasadiBackend(model)
    backend.compile()

    # Check that we have two when-equations
    assert len(backend.when_equations) == 2, "Should have 2 when-equations"

    # Verify state names
    assert "x" in backend.state_names
    assert "v" in backend.state_names

    # Verify derivatives were created
    assert "x" in backend.derivatives
    assert "v" in backend.derivatives


def test_multiple_when_clauses_different_states():
    """Test multiple when-clauses affecting different state variables.

    This tests independent resets for different states:
    - When x < 0: reset x to 0
    - When y > 1: reset y to 1
    """
    model = Model(name="IndependentBounds")

    # State variables - two independent oscillators
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.5))
    model.add_variable(Variable(name="vx", var_type=VariableType.STATE, start=-0.3))
    model.add_variable(Variable(name="y", var_type=VariableType.STATE, start=0.5))
    model.add_variable(Variable(name="vy", var_type=VariableType.STATE, start=0.4))

    x = Expr.var_ref("x")
    y = Expr.var_ref("y")
    vx = Expr.var_ref("vx")
    vy = Expr.var_ref("vy")

    # ODEs: simple constant velocity motion
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), vx))
    model.add_equation(Equation.simple(der(Expr.var_ref("vx")), Expr.literal(0.0)))
    model.add_equation(Equation.simple(der(Expr.var_ref("y")), vy))
    model.add_equation(Equation.simple(der(Expr.var_ref("vy")), Expr.literal(0.0)))

    # When-clause 1: Bound x at 0 (reflect when x < 0)
    x_bound_condition = Expr.binary_op("<", x, Expr.literal(0.0))
    pre_vx = Expr.call("pre", vx)
    x_reset = Expr.neg(pre_vx)  # Simple reflection: vx = -pre(vx)
    x_eq = Equation.when(x_bound_condition, [Equation.simple(vx, x_reset)])
    model.add_equation(x_eq)

    # When-clause 2: Bound y at 1 (reflect when y > 1)
    y_bound_condition = Expr.binary_op(">", y, Expr.literal(1.0))
    pre_vy = Expr.call("pre", vy)
    y_reset = Expr.neg(pre_vy)  # Simple reflection: vy = -pre(vy)
    y_eq = Equation.when(y_bound_condition, [Equation.simple(vy, y_reset)])
    model.add_equation(y_eq)

    # Compile with CasADi
    backend = CasadiBackend(model)
    backend.compile()

    # Check that we have two when-equations
    assert len(backend.when_equations) == 2, "Should have 2 when-equations"

    # Verify all state names
    assert "x" in backend.state_names
    assert "vx" in backend.state_names
    assert "y" in backend.state_names
    assert "vy" in backend.state_names

    # Verify derivatives were created for all states
    assert "x" in backend.derivatives
    assert "vx" in backend.derivatives
    assert "y" in backend.derivatives
    assert "vy" in backend.derivatives


def test_three_when_clauses():
    """Test three when-clauses to verify scaling to more events."""
    model = Model(name="ThreeEvents")

    # State variable
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, start=0.0))
    model.add_variable(Variable(name="v", var_type=VariableType.STATE, start=1.0))

    x = Expr.var_ref("x")
    v = Expr.var_ref("v")

    # ODE: constant velocity motion
    model.add_equation(Equation.simple(der(Expr.var_ref("x")), v))
    model.add_equation(Equation.simple(der(Expr.var_ref("v")), Expr.literal(0.0)))

    # When-clause 1: Lower bound at x = -1
    lower_condition = Expr.binary_op("<", x, Expr.literal(-1.0))
    pre_v = Expr.call("pre", v)
    lower_reset = Expr.neg(pre_v)
    lower_eq = Equation.when(lower_condition, [Equation.simple(v, lower_reset)])
    model.add_equation(lower_eq)

    # When-clause 2: Middle event at x = 0 (special reset)
    middle_condition = Expr.binary_op("<", x, Expr.literal(0.0))
    middle_reset = Expr.mul(Expr.literal(0.5), pre_v)  # Slow down at middle
    middle_eq = Equation.when(middle_condition, [Equation.simple(v, middle_reset)])
    model.add_equation(middle_eq)

    # When-clause 3: Upper bound at x = 1
    upper_condition = Expr.binary_op(">", x, Expr.literal(1.0))
    upper_reset = Expr.neg(pre_v)
    upper_eq = Equation.when(upper_condition, [Equation.simple(v, upper_reset)])
    model.add_equation(upper_eq)

    # Compile with CasADi
    backend = CasadiBackend(model)
    backend.compile()

    # Check that we have three when-equations
    assert len(backend.when_equations) == 3, "Should have 3 when-equations"

    # Verify state names
    assert "x" in backend.state_names
    assert "v" in backend.state_names
