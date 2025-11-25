"""Test hybrid system features with bouncing ball example.

A bouncing ball is a classic hybrid dynamical system with:
- Continuous dynamics: free fall under gravity
- Discrete events: collision detection (ground contact)
- Discrete updates: velocity reset with energy loss
"""

import casadi as ca
import numpy as np
import pytest
from cyecca.model import ModelSX, input_var, param, state, symbolic


@pytest.fixture
def bouncing_ball_model():
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
        0.0, -p.e * x.v  # h+ = 0 (clamp to ground)  # v+ = -e * v (reverse and reduce)
    )

    model.build(f_x=f_x, f_c=f_c, f_z=f_z, f_m=f_m, integrator="euler")

    return model


def test_bouncing_ball_creation(bouncing_ball_model):
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


def test_bouncing_ball_continuous_dynamics(bouncing_ball_model):
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


def test_bouncing_ball_event_detection(bouncing_ball_model):
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


def test_bouncing_ball_reset_map(bouncing_ball_model):
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


def test_bouncing_ball_simulation():
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


def test_bouncing_ball_energy_conservation_between_bounces():
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
