"""Tests for the dynamics module including ModelSX, ModelMX, and integrators."""

import casadi as ca
import numpy as np
import pytest
from cyecca.dynamics import (
    ModelMX,
    ModelSX,
    input_var,
    output_var,
    param,
    state,
    symbolic,
)
from cyecca.dynamics.composition import SubmodelProxy
from cyecca.dynamics.integrators import rk4, rk8, build_rk_integrator, integrate_n_steps


class TestQuickStart:
    """Test the README quick start example to ensure it works."""

    def test_readme_quickstart_example(self):
        """Verify the mass-spring-damper example from README works correctly."""
        # This is the exact code from the README Quick Start section

        @symbolic
        class States:
            x: ca.SX = state(1, 1.0, "position")  # Start at x=1
            v: ca.SX = state(1, 0.0, "velocity")

        @symbolic
        class Inputs:
            F: ca.SX = input_var(desc="force")

        @symbolic
        class Params:
            m: ca.SX = param(1.0, "mass")
            c: ca.SX = param(0.1, "damping")
            k: ca.SX = param(1.0, "spring constant")

        @symbolic
        class Outputs:
            position: ca.SX = output_var(desc="position output")
            velocity: ca.SX = output_var(desc="velocity output")

        model = ModelSX.create(States, Inputs, Params, output_type=Outputs)
        x, u, p, y = model.x, model.u, model.p, model.y

        # Mass-spring-damper: mx'' + cx' + kx = F
        f_x = ca.vertcat(x.v, (u.F - p.c * x.v - p.k * x.x) / p.m)

        # Output the full state
        f_y = ca.vertcat(x.x, x.v)

        model.build(f_x=f_x, f_y=f_y, integrator="rk4")

        # Simulate free oscillation from x0=1
        result = model.simulate(0.0, 10.0, 0.01)

        # Verify results match expected output
        final_position = result["x"][0, -1]
        final_velocity = result["x"][1, -1]

        # Check values are close to documented output
        assert abs(final_position - (-0.529209)) < 0.001
        assert abs(final_velocity - 0.323980) < 0.001

        # Verify we have the right number of timesteps
        assert len(result["t"]) == 1001  # 0 to 10 with dt=0.01

        # Verify initial conditions
        assert result["x"][0, 0] == pytest.approx(1.0)
        assert result["x"][1, 0] == pytest.approx(0.0)

        # Verify oscillatory behavior (should cross zero at least once)
        x_pos = result["x"][0, :]
        sign_changes = np.sum(np.diff(np.sign(x_pos)) != 0)
        assert sign_changes >= 2  # At least one complete oscillation

        # Verify outputs match states
        assert "out" in result
        assert np.allclose(result["out"][0, :], result["x"][0, :])  # position output
        assert np.allclose(result["out"][1, :], result["x"][1, :])  # velocity output


class TestModelCreate:
    """Test that ModelSX.create() and ModelMX.create() work correctly."""

    def test_sx_model_basic(self):
        """Test ModelSX.create() sets all expected attributes."""

        @symbolic
        class States:
            x: ca.SX = state(1, 0.0)

        @symbolic
        class Inputs:
            pass

        @symbolic
        class Params:
            m: ca.SX = param(1.0)

        model = ModelSX.create(States, Inputs, Params)
        model.build(f_x=ca.SX.zeros(1))

        # Check types are created
        assert model.state_type == States
        assert model.input_type == Inputs
        assert model.param_type == Params

        # Check symbolic instances
        assert hasattr(model, "x")
        assert hasattr(model, "p")
        assert hasattr(model, "u")

        # Check default instances
        assert hasattr(model, "x0")
        assert hasattr(model, "p0")
        assert hasattr(model, "u0")

        # Verify types
        assert model.x.x.__class__.__name__ == "SX"
        assert model.x0.x == 0.0
        assert model.p0.m == 1.0

    def test_mx_model_basic(self):
        """Test ModelMX.create() sets all expected attributes."""

        @symbolic
        class States:
            x: ca.MX = state(1, 0.0)

        @symbolic
        class Inputs:
            pass

        @symbolic
        class Params:
            m: ca.MX = param(1.0)

        model = ModelMX.create(States, Inputs, Params)
        model.build(f_x=ca.MX.zeros(1))

        assert model.state_type == States
        assert hasattr(model, "x")
        assert hasattr(model, "x0")
        assert model.x.x.__class__.__name__ == "MX"
        assert model.x0.x == 0.0
        assert model.p0.m == 1.0

    def test_vector_states(self):
        """Test with vector-valued states."""

        @symbolic
        class States:
            position: ca.SX = state(3, [1.0, 2.0, 3.0])

        @symbolic
        class Inputs:
            pass

        @symbolic
        class Params:
            k: ca.SX = param(2.5)

        model = ModelSX.create(States, Inputs, Params)
        model.build(f_x=ca.SX.zeros(3))

        # Check vector field
        assert hasattr(model.x0, "position")
        np.testing.assert_array_equal(model.x0.position, [1.0, 2.0, 3.0])

        # Check symbolic vector
        assert model.x.position.shape == (3, 1)

    def test_with_inputs(self):
        """Test with input variables."""

        @symbolic
        class States:
            x: ca.SX = state(1, 0.0)

        @symbolic
        class Inputs:
            u: ca.SX = input_var(default=0.5)

        @symbolic
        class Params:
            pass

        model = ModelSX.create(States, Inputs, Params)
        model.build(f_x=ca.SX.zeros(1))

        assert hasattr(model, "u")
        assert hasattr(model, "u0")
        assert model.u0.u == 0.5

    def test_with_outputs(self):
        """Test with output variables."""

        @symbolic
        class States:
            x: ca.SX = state(1, 0.0)

        @symbolic
        class Inputs:
            pass

        @symbolic
        class Params:
            pass

        @symbolic
        class Outputs:
            energy: ca.SX = output_var(1, 0.0, "energy calculation")

        model = ModelSX.create(States, Inputs, Params, output_type=Outputs)
        model.build(f_x=ca.SX.zeros(1), f_y=model.x.x)

        assert hasattr(model, "output_type")
        assert model.output_type == Outputs
        assert "energy" in model.output_names


class TestSimulation:
    """Test model simulation capabilities."""

    def test_simple_integration(self):
        """Test basic forward integration."""

        @symbolic
        class States:
            x: ca.SX = state(1, 1.0)

        @symbolic
        class Inputs:
            u: ca.SX = input_var(1, 0.0)

        @symbolic
        class Params:
            k: ca.SX = param(1.0, "decay rate")

        model = ModelSX.create(States, Inputs, Params)
        # dx/dt = -k*x (exponential decay)
        model.build(f_x=-model.p.k * model.x.x)

        result = model.simulate(
            t0=0.0,
            tf=1.0,
            dt=0.1,
            u_func=lambda t, x, p: model.u0.as_vec(),
            p_vec=model.p0.as_vec(),
        )

        assert "t" in result
        assert "x" in result
        assert len(result["t"]) > 0

        # result["x"] has shape (n_states, n_timesteps), so transpose
        x_traj = result["x"].T  # Now shape is (n_timesteps, n_states)

        # Final value should be less than initial (decay)
        # For dx/dt = -k*x with x(0) = 1, solution is x(t) = e^(-t)
        # At t=1, x ≈ 0.368
        final_val = float(x_traj[-1, 0])
        assert final_val < 0.5, f"Expected decay but got {final_val}"


class TestHybridFeatures:
    """Test hybrid system features with bouncing ball example.

    A bouncing ball is a classic hybrid dynamical system with:
    - Continuous dynamics: free fall under gravity
    - Discrete events: collision detection (ground contact)
    - Discrete updates: velocity reset with energy loss
    """

    @pytest.fixture
    def bouncing_ball_model(self):
        """Create a bouncing ball hybrid model.

        State:
            - h: height above ground (m)
            - v: vertical velocity (m/s)

        Discrete states:
            - num_bounces: count of ground impacts

        Events:
            - Ground contact: h <= 0

        Dynamics:
            - dh/dt = v
            - dv/dt = -g

        Reset map (at ground contact):
            - h+ = 0 (stay on ground)
            - v+ = -e * v (reverse velocity with energy loss)
            - num_bounces+ = num_bounces + 1
        """

        @symbolic
        class States:
            h: ca.SX = state(1, 10.0, "height above ground [m]")
            v: ca.SX = state(1, 0.0, "vertical velocity [m/s]")

        @symbolic
        class Inputs:
            pass  # No control inputs

        @symbolic
        class Params:
            g: ca.SX = param(9.81, "gravitational acceleration [m/s^2]")
            e: ca.SX = param(0.8, "coefficient of restitution [-]")

        @symbolic
        class DiscreteStates:
            num_bounces: ca.SX = state(1, 0.0, "number of bounces")

        @symbolic
        class EventIndicators:
            ground_contact: ca.SX = state(
                1, 1.0, "ground contact indicator (>0 = no contact)"
            )

        model = ModelSX.create(
            States,
            Inputs,
            Params,
            discrete_state_type=DiscreteStates,
            event_indicator_type=EventIndicators,
        )

        # Get symbolic variables
        x, u, p = model.x, model.u, model.p
        z = model.z  # discrete states
        c = model.c  # event indicators

        # Continuous dynamics: free fall
        f_x = ca.vertcat(x.v, -p.g)  # dh/dt = v  # dv/dt = -g

        # Event indicator: h - 0 (zero-crossing when ball hits ground)
        # Positive when h > 0 (no contact), zero/negative when h <= 0 (contact)
        f_c = x.h

        # Discrete state update at event (ground contact)
        # Increment bounce counter
        f_z = z.num_bounces + 1.0

        # Continuous state reset at event
        # Position: clamp to ground, velocity: reverse with energy loss
        f_m = ca.vertcat(
            0.0,
            -p.e * x.v,  # h+ = 0 (clamp to ground)  # v+ = -e * v (reverse and reduce)
        )

        model.build(f_x=f_x, f_c=f_c, f_z=f_z, f_m=f_m, integrator="euler")

        return model

    def test_bouncing_ball_creation(self, bouncing_ball_model):
        """Test that bouncing ball model is created properly."""
        model = bouncing_ball_model

        assert hasattr(model, "x")
        assert hasattr(model, "z")
        assert hasattr(model, "c")
        assert hasattr(model, "f_x")
        assert hasattr(model, "f_c")
        assert hasattr(model, "f_z")
        assert hasattr(model, "f_m")
        assert hasattr(model, "f_step")

        # Check initial conditions
        assert model.x0.h == 10.0
        assert model.x0.v == 0.0
        assert model.z0.num_bounces == 0.0

    def test_bouncing_ball_continuous_dynamics(self, bouncing_ball_model):
        """Test continuous dynamics (free fall) without events."""
        model = bouncing_ball_model

        # Get state and parameters
        x = ca.vertcat(model.x0.h, model.x0.v)
        z = ca.vertcat(model.z0.num_bounces)  # discrete state
        u = ca.DM.zeros(0, 1)
        p = ca.vertcat(model.p0.g, model.p0.e)

        # Evaluate dynamics at initial state
        dx_dt = model.f_x(x, z, u, p)

        # Expected: dh/dt = v = 0, dv/dt = -g = -9.81
        assert float(dx_dt[0]) == pytest.approx(0.0)
        assert float(dx_dt[1]) == pytest.approx(-9.81)

        # After some time falling, velocity should be negative
        x_falling = ca.vertcat(5.0, -10.0)  # h=5m, v=-10m/s
        dx_dt_falling = model.f_x(x_falling, z, u, p)

        assert float(dx_dt_falling[0]) == pytest.approx(-10.0)  # dh/dt = v
        assert float(dx_dt_falling[1]) == pytest.approx(-9.81)  # dv/dt = -g

    def test_bouncing_ball_event_detection(self, bouncing_ball_model):
        """Test event indicator for ground contact."""
        model = bouncing_ball_model

        u = ca.DM.zeros(0, 1)
        p = ca.vertcat(model.p0.g, model.p0.e)
        z = ca.vertcat(0.0)  # no bounces yet

        # Above ground: indicator should be positive
        x_above = ca.vertcat(5.0, -10.0)
        c_above = model.f_c(x_above, z, u, p)
        assert float(c_above) > 0

        # At ground: indicator should be zero
        x_at_ground = ca.vertcat(0.0, -10.0)
        c_at_ground = model.f_c(x_at_ground, z, u, p)
        assert float(c_at_ground) == pytest.approx(0.0)

        # Below ground (penetration): indicator should be negative
        x_below = ca.vertcat(-0.1, -10.0)
        c_below = model.f_c(x_below, z, u, p)
        assert float(c_below) < 0

    def test_bouncing_ball_reset_map(self, bouncing_ball_model):
        """Test discrete update at ground contact."""
        model = bouncing_ball_model

        u = ca.DM.zeros(0, 1)
        p = ca.vertcat(model.p0.g, model.p0.e)

        # State just before impact: h=0, v=-10 m/s
        x_before = ca.vertcat(0.0, -10.0)
        z_before = ca.vertcat(2.0)  # Already had 2 bounces

        # Discrete state update (increment bounce counter)
        z_after = model.f_z(x_before, z_before, u, p)
        assert float(z_after) == pytest.approx(3.0)

        # Continuous state reset
        # h+ = 0, v+ = -e * v = -0.8 * (-10) = 8.0 m/s
        x_after = model.f_m(x_before, z_before, u, p)
        assert float(x_after[0]) == pytest.approx(0.0)  # height clamped to ground
        assert float(x_after[1]) == pytest.approx(8.0)  # velocity reversed with loss

    def test_bouncing_ball_simulation(self):
        """Test full simulation of bouncing ball with event detection.

        Ball should:
        1. Fall from h=10m
        2. Hit ground and bounce back
        3. Reach lower height each time
        4. Eventually settle on ground
        """

        @symbolic
        class States:
            h: ca.SX = state(1, 10.0, "height [m]")
            v: ca.SX = state(1, 0.0, "velocity [m/s]")

        @symbolic
        class Inputs:
            pass

        @symbolic
        class Params:
            g: ca.SX = param(9.81, "gravity [m/s^2]")
            e: ca.SX = param(0.8, "restitution [-]")

        @symbolic
        class DiscreteStates:
            num_bounces: ca.SX = state(1, 0.0, "bounces")

        @symbolic
        class EventIndicators:
            ground_contact: ca.SX = state(1, 1.0, "contact indicator")

        model = ModelSX.create(
            States,
            Inputs,
            Params,
            discrete_state_type=DiscreteStates,
            event_indicator_type=EventIndicators,
        )

        x, u, p = model.x, model.u, model.p
        z, c = model.z, model.c

        # Dynamics
        f_x = ca.vertcat(x.v, -p.g)
        f_c = x.h  # Event when h crosses zero
        f_z = z.num_bounces + 1.0
        f_m = ca.vertcat(0.0, -p.e * x.v)

        model.build(f_x=f_x, f_c=f_c, f_z=f_z, f_m=f_m, integrator="euler")

        # Simulate for 5 seconds with event detection
        result = model.simulate(t0=0.0, tf=5.0, dt=0.01, detect_events=True)

        # Extract results
        t = result["t"]
        h = result["x"][0, :]
        v = result["x"][1, :]
        num_bounces = result["z"][0, :]

        # Verify physics
        assert len(t) > 0
        assert np.all(h >= -0.01)  # Height should stay non-negative (small tolerance)

        # Should have bounced at least once
        assert num_bounces[-1] >= 1.0

        # Find first bounce (when discrete state changes from 0 to 1)
        bounce_indices = np.where(np.diff(num_bounces) > 0.5)[0]

        if len(bounce_indices) > 0:
            first_bounce_idx = bounce_indices[0] + 1

            # Velocity before bounce should be negative
            assert v[first_bounce_idx - 1] < 0

            # Velocity after bounce should be positive (upward)
            assert v[first_bounce_idx] > 0

            # Post-bounce velocity should be less than pre-bounce (energy loss)
            assert abs(v[first_bounce_idx]) < abs(v[first_bounce_idx - 1])

            # Should approximately match coefficient of restitution
            e = model.p0.e
            expected_v_after = -e * v[first_bounce_idx - 1]
            assert v[first_bounce_idx] == pytest.approx(expected_v_after, rel=0.1)

        # Energy should decrease over time (excluding first point which is at rest)
        initial_energy = 9.81 * h[0]  # mgh (m=1)
        final_energy = 9.81 * h[-1] + 0.5 * v[-1] ** 2  # mgh + 0.5mv^2
        assert final_energy < initial_energy

        print(f"\nBouncing ball simulation:")
        print(f"  Initial height: {h[0]:.2f} m")
        print(f"  Final height: {h[-1]:.2f} m")
        print(f"  Total bounces: {int(num_bounces[-1])}")
        print(f"  Max velocity: {abs(v.min()):.2f} m/s")

    def test_bouncing_ball_energy_conservation_between_bounces(self):
        """Test that energy is conserved during continuous flight (between bounces)."""

        @symbolic
        class States:
            h: ca.SX = state(1, 5.0, "height [m]")
            v: ca.SX = state(1, 0.0, "velocity [m/s]")

        @symbolic
        class Inputs:
            pass

        @symbolic
        class Params:
            g: ca.SX = param(9.81, "gravity [m/s^2]")
            e: ca.SX = param(0.9, "restitution [-]")

        model = ModelSX.create(States, Inputs, Params)

        x, p = model.x, model.p
        f_x = ca.vertcat(x.v, -p.g)

        model.build(f_x=f_x, integrator="rk4", integrator_options={"N": 10})

        # Simulate without event detection (continuous fall)
        result = model.simulate(t0=0.0, tf=1.0, dt=0.01, detect_events=False)

        h = result["x"][0, :]
        v = result["x"][1, :]

        # Total energy: E = mgh + 0.5mv^2 (with m=1)
        g = 9.81
        energy = g * h + 0.5 * v**2

        # Energy should be approximately constant (within numerical tolerance)
        energy_initial = energy[0]
        energy_variation = np.max(np.abs(energy - energy_initial))

        assert energy_variation < 0.1  # Less than 0.1 J variation

        print(f"\nEnergy conservation test:")
        print(f"  Initial energy: {energy_initial:.4f} J")
        print(f"  Final energy: {energy[-1]:.4f} J")
        print(f"  Max variation: {energy_variation:.4f} J")


class TestModelComposition:
    """Test hierarchical model composition features."""

    def test_compose_basic(self):
        """Test basic model composition with two simple subsystems."""

        # Create first subsystem: integrator (dx/dt = u)
        @symbolic
        class IntegratorStates:
            x: ca.SX = state(1, 0.0, "position")

        @symbolic
        class IntegratorInputs:
            u: ca.SX = input_var(desc="velocity command")

        @symbolic
        class IntegratorParams:
            pass

        integrator = ModelSX.create(
            IntegratorStates, IntegratorInputs, IntegratorParams
        )
        x_int = integrator.x
        u_int = integrator.u
        integrator.build(f_x=u_int.u)

        # Create second subsystem: proportional controller (u = -k*x)
        @symbolic
        class ControllerStates:
            pass

        @symbolic
        class ControllerInputs:
            x_meas: ca.SX = input_var(desc="measured position")

        @symbolic
        class ControllerParams:
            k: ca.SX = param(2.0, "proportional gain")

        @symbolic
        class ControllerOutputs:
            u_cmd: ca.SX = output_var(desc="velocity command")

        controller = ModelSX.create(
            ControllerStates, ControllerInputs, ControllerParams, ControllerOutputs
        )
        u_ctrl = controller.u
        p_ctrl = controller.p
        controller.build(f_x=ca.SX.zeros(0), f_y=-p_ctrl.k * u_ctrl.x_meas)

        # Compose the two subsystems
        parent = ModelSX.compose({"integrator": integrator, "controller": controller})

        # Verify submodels are accessible
        assert hasattr(parent, "integrator")
        assert hasattr(parent, "controller")

        # Connect using string paths
        parent.connect("integrator.u.u", "controller.y.u_cmd")
        parent.connect("controller.u.x_meas", "integrator.x.x")

        # Build composed model
        parent.build_composed(integrator="rk4")

        # Verify composed model has combined states
        assert hasattr(parent, "x0_composed")
        assert parent.x0_composed.shape[0] == 1  # Only integrator has state

        # Verify can step the composed model
        x0 = parent.x0_composed
        x0[0] = 1.0  # Start with non-zero position

        result = parent.f_step(x=x0, u=parent.u0.as_vec(), p=parent.p0.as_vec(), dt=0.1)

        assert "x_next" in result
        # With k=2 and x0=1, u_cmd = -2, so dx/dt = -2
        # After dt=0.1, x ≈ 1 - 0.2 = 0.8
        x_next = float(result["x_next"][0])
        assert x_next < 1.0, "Position should decrease with negative feedback"
        assert x_next > 0.5, f"Expected x ≈ 0.8, got {x_next}"

    def test_compose_states_merge(self):
        """Test that compose_states properly merges multiple state types."""
        from cyecca.dynamics import compose_states

        @symbolic
        class States1:
            x: ca.SX = state(1, 0.0, "state x")
            y: ca.SX = state(1, 0.0, "state y")

        @symbolic
        class States2:
            z: ca.SX = state(1, 0.0, "state z")

        # Compose the states
        Combined = compose_states(States1, States2)

        # Verify combined type has all fields
        combined_instance = Combined.numeric()
        assert hasattr(combined_instance, "x")
        assert hasattr(combined_instance, "y")
        assert hasattr(combined_instance, "z")

        # Verify defaults are preserved
        assert combined_instance.x == 0.0
        assert combined_instance.y == 0.0
        assert combined_instance.z == 0.0

        # Verify vector size
        assert combined_instance.as_vec().shape[0] == 3

    def test_add_submodel_creates_proxy(self):
        """Test that add_submodel creates accessible proxy for connections."""

        @symbolic
        class States:
            x: ca.SX = state(1, 0.0)

        @symbolic
        class Inputs:
            u: ca.SX = input_var()

        @symbolic
        class Params:
            pass

        parent = ModelSX.create(States, Inputs, Params)
        child = ModelSX.create(States, Inputs, Params)

        # Add submodel
        parent.add_submodel("child", child)

        # Verify child proxy is accessible as attribute
        assert hasattr(parent, "child")
        assert isinstance(parent.child, SubmodelProxy)

    def test_connect_signals(self):
        """Test signal connection API."""

        @symbolic
        class States:
            x: ca.SX = state(1, 0.0)

        @symbolic
        class Inputs:
            u: ca.SX = input_var()

        @symbolic
        class Params:
            pass

        @symbolic
        class Outputs:
            y: ca.SX = output_var(desc="output")

        parent = ModelSX.create(States, Inputs, Params)
        child1 = ModelSX.create(States, Inputs, Params, Outputs)
        child2 = ModelSX.create(States, Inputs, Params)

        # Build simple dynamics
        child1.build(f_x=ca.SX.zeros(1), f_y=child1.x.x)

        parent.add_submodel("child1", child1)
        parent.add_submodel("child2", child2)

        # Test connection API using string paths
        parent.connect("child2.u.u", "child1.y.y")

        # Verify connection was stored
        assert "child2" in parent._input_connections
        assert "child2.u.u" in parent._input_connections["child2"]


class TestIntegrators:
    """Test custom Runge-Kutta integrators."""

    def test_rk4_simple_exponential_decay(self):
        """Test RK4 integrator on simple exponential decay: dx/dt = -k*x."""
        # Define dynamics: dx/dt = -k*x
        x_sym = ca.SX.sym("x", 1)
        u_sym = ca.SX.sym("u", 0)  # No inputs
        p_sym = ca.SX.sym("p", 1)  # k parameter

        f_x = -p_sym * x_sym
        f = ca.Function("f", [x_sym, u_sym, p_sym], [f_x])

        # Create RK4 integrator with step size 0.1
        h = 0.1
        rk4_step = rk4(f, h)

        # Initial conditions
        x0 = ca.DM([1.0])
        u = ca.DM([])
        k = ca.DM([1.0])

        # Integrate for 10 steps (total time = 1.0)
        x = x0
        for _ in range(10):
            x = rk4_step(x, u, k)

        # Analytical solution: x(t) = x0 * exp(-k*t)
        t_final = 1.0
        x_analytical = float(x0) * np.exp(-float(k) * t_final)

        # Check accuracy (RK4 should be quite accurate)
        assert abs(float(x) - x_analytical) < 1e-6

    def test_rk4_with_substeps(self):
        """Test RK4 with multiple substeps for improved accuracy."""
        # Same exponential decay problem
        x_sym = ca.SX.sym("x", 1)
        u_sym = ca.SX.sym("u", 0)
        p_sym = ca.SX.sym("p", 1)

        f_x = -p_sym * x_sym
        f = ca.Function("f", [x_sym, u_sym, p_sym], [f_x])

        # Create RK4 with 10 substeps
        h = 1.0
        rk4_step = rk4(f, h, N=10)

        x0 = ca.DM([1.0])
        u = ca.DM([])
        k = ca.DM([1.0])

        # Single step with substeps
        x_final = rk4_step(x0, u, k)

        # Analytical solution at t=1.0
        x_analytical = float(x0) * np.exp(-float(k) * 1.0)

        assert abs(float(x_final) - x_analytical) < 1e-6

    def test_rk4_with_inputs(self):
        """Test RK4 with inputs: dx/dt = u - k*x."""
        x_sym = ca.SX.sym("x", 1)
        u_sym = ca.SX.sym("u", 1)
        p_sym = ca.SX.sym("p", 1)

        f_x = u_sym - p_sym * x_sym
        f = ca.Function("f", [x_sym, u_sym, p_sym], [f_x])

        h = 0.1
        rk4_step = rk4(f, h)

        x0 = ca.DM([0.0])
        u = ca.DM([1.0])  # Constant input
        k = ca.DM([0.5])

        # Integrate for several steps
        x = x0
        for _ in range(20):
            x = rk4_step(x, u, k)

        # Analytical solution: x(t) = (u/k)*(1 - exp(-k*t)) for x0=0
        # With u=1, k=0.5, t=2.0: x = 2*(1 - exp(-1)) ≈ 1.264
        t_final = 2.0
        x_analytical = (float(u) / float(k)) * (1 - np.exp(-float(k) * t_final))

        assert abs(float(x) - x_analytical) < 1e-6

    def test_rk8_exponential_decay(self):
        """Test RK8 integrator on exponential decay."""
        x_sym = ca.SX.sym("x", 1)
        u_sym = ca.SX.sym("u", 0)
        p_sym = ca.SX.sym("p", 1)

        f_x = -p_sym * x_sym
        f = ca.Function("f", [x_sym, u_sym, p_sym], [f_x])

        # Use RK8 with default DOP853 tableau
        h = 0.5
        rk8_step = rk8(f, h)

        x0 = ca.DM([1.0])
        u = ca.DM([])
        k = ca.DM([1.0])

        # Single large step (RK8 should handle this well)
        x_final = rk8_step(x0, u, k)

        # Analytical solution at t=0.5
        x_analytical = float(x0) * np.exp(-float(k) * 0.5)

        # RK8 should be very accurate even with large step
        assert abs(float(x_final) - x_analytical) < 1e-8

    def test_integrate_n_steps(self):
        """Test the integrate_n_steps helper function."""
        # Simple dynamics
        x_sym = ca.SX.sym("x", 1)
        u_sym = ca.SX.sym("u", 0)
        p_sym = ca.SX.sym("p", 1)

        f_x = -p_sym * x_sym
        f = ca.Function("f", [x_sym, u_sym, p_sym], [f_x])

        # Create one-step integrator
        h = 0.1
        rk4_step = rk4(f, h)

        # Create N-step rollout
        N = 10
        rollout = integrate_n_steps(rk4_step, ca.DM([1.0]), ca.DM([]), ca.DM([1.0]), N)

        # Execute rollout
        x0 = ca.DM([1.0])
        u = ca.DM([])
        k = ca.DM([1.0])

        x_final = rollout(x0, u, k)

        # Should match 10 steps of integration
        x_analytical = float(x0) * np.exp(-float(k) * 1.0)
        assert abs(float(x_final) - x_analytical) < 1e-6

    def test_build_rk_integrator_custom_tableau(self):
        """Test build_rk_integrator with a custom tableau."""
        # Define simple Euler method as a custom tableau
        euler_tableau = {"A": [[0.0]], "b": [1.0], "c": [0.0]}

        x_sym = ca.SX.sym("x", 1)
        u_sym = ca.SX.sym("u", 0)
        p_sym = ca.SX.sym("p", 1)

        f_x = -p_sym * x_sym
        f = ca.Function("f", [x_sym, u_sym, p_sym], [f_x])

        h = 0.01
        euler_step = build_rk_integrator(f, h, euler_tableau, name="euler")

        # Take small steps with Euler method
        x = ca.DM([1.0])
        u = ca.DM([])
        k = ca.DM([1.0])

        for _ in range(100):  # 100 steps of 0.01 = t=1.0
            x = euler_step(x, u, k)

        # Euler is less accurate but should be reasonable with small steps
        x_analytical = np.exp(-1.0)
        assert abs(float(x) - x_analytical) < 0.01

    def test_rk4_multidimensional(self):
        """Test RK4 on multi-dimensional system."""
        # Harmonic oscillator: dx/dt = v, dv/dt = -k*x/m
        x_sym = ca.SX.sym("x", 2)  # [position, velocity]
        u_sym = ca.SX.sym("u", 0)
        p_sym = ca.SX.sym("p", 2)  # [k, m]

        position = x_sym[0]
        velocity = x_sym[1]
        k = p_sym[0]
        m = p_sym[1]

        f_x = ca.vertcat(velocity, -k * position / m)
        f = ca.Function("f", [x_sym, u_sym, p_sym], [f_x])

        # Create integrator
        h = 0.01
        rk4_step = rk4(f, h)

        # Initial conditions: x=1, v=0
        x0 = ca.DM([1.0, 0.0])
        u = ca.DM([])
        params = ca.DM([1.0, 1.0])  # k=1, m=1 => omega=1

        # Integrate for one period (2*pi)
        n_steps = int(2 * np.pi / h)
        x = x0
        for _ in range(n_steps):
            x = rk4_step(x, u, params)

        # After one period, should return to initial position
        assert abs(float(x[0]) - 1.0) < 0.01
        assert abs(float(x[1]) - 0.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
