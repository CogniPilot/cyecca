"""Test the new unified Model class with stateful simulation API.

This tests the Modelica-like interface where:
- All variables in one namespace (model.v.theta, model.v.g, etc.)
- Time is built-in (model.t), like Modelica
- Model carries internal state (model.v0)
- simulate() uses and updates model.v0 automatically
- simulate() returns (t, data) tuple
- data supports indexing for checkpoints
- States are inferred from .dot() usage (no explicit state() declaration)
"""

import casadi as ca
import numpy as np
import pytest

from cyecca.dynamics.implicit import Model, implicit, param, var


@implicit
class Pendulum:
    """Simple pendulum with single namespace."""

    theta: float = var()  # Becomes state (has .dot())
    omega: float = var()  # Becomes state (has .dot())
    g: float = param(default=9.81)
    l: float = param(default=1.0)


def test_model_creation():
    """Test Model class initialization."""
    model = Model(Pendulum)

    # Check symbolic variables exist
    assert hasattr(model.v, "theta")
    assert hasattr(model.v, "omega")
    assert hasattr(model.v, "g")
    assert hasattr(model.v, "l")

    # Check built-in time exists
    assert hasattr(model, "t")
    assert isinstance(model.t, ca.SX)

    # Check numeric state exists
    assert hasattr(model, "v0")
    assert hasattr(model.v0, "theta")
    assert hasattr(model.v0, "omega")


def test_builtin_time():
    """Test that time is built-in like Modelica."""
    model = Model(Pendulum)

    # Time should be accessible as model.t
    assert hasattr(model, "t")
    assert isinstance(model.t, ca.SX)

    # Can use in equations (time-varying force example)
    model.eq(model.v.theta.dot() - model.v.omega)
    # Could do: model.eq(... + ca.sin(model.t))  # time-varying!
    model.eq(model.v.omega.dot() + model.v.g / model.v.l * ca.sin(model.v.theta))

    model.build()


def test_stateful_simulation():
    """Test that simulate() uses and updates internal state."""
    model = Model(Pendulum)

    # Define equations
    model.eq(model.v.theta.dot() - model.v.omega)
    model.eq(model.v.omega.dot() + model.v.g / model.v.l * ca.sin(model.v.theta))

    model.build()

    # Set initial condition
    model.v0.theta = 0.5
    model.v0.omega = 0.0

    # Simulate - returns (t, data) tuple
    t, data = model.simulate(0.0, 1.0, 0.01)

    # Check time array
    assert t.shape == (101,)
    assert abs(t[0]) < 1e-10
    assert abs(t[-1] - 1.0) < 1e-6

    # Check data trajectory
    assert data.theta.shape == (101,)
    assert data.omega.shape == (101,)

    # Check initial value preserved
    assert abs(data.theta[0] - 0.5) < 1e-10

    # Check model.v0 was updated to final state
    assert abs(model.v0.theta - data.theta[-1]) < 1e-10


def test_chained_simulation():
    """Test that simulations can be chained seamlessly."""
    model = Model(Pendulum)

    model.eq(model.v.theta.dot() - model.v.omega)
    model.eq(model.v.omega.dot() + model.v.g / model.v.l * ca.sin(model.v.theta))

    model.build()

    # First simulation
    model.v0.theta = 0.5
    t1, data1 = model.simulate(0.0, 1.0, 0.01)

    # Second simulation - should continue from where we left off
    t2, data2 = model.simulate(1.0, 2.0, 0.01)

    # Check continuity
    assert abs(data2.theta[0] - data1.theta[-1]) < 1e-10
    assert abs(t2[0] - 1.0) < 1e-6
    assert abs(t2[-1] - 2.0) < 1e-6


def test_trajectory_indexing():
    """Test that data supports indexing for checkpoints."""
    model = Model(Pendulum)

    model.eq(model.v.theta.dot() - model.v.omega)
    model.eq(model.v.omega.dot() + model.v.g / model.v.l * ca.sin(model.v.theta))

    model.build()
    model.v0.theta = 0.5

    t, data = model.simulate(0.0, 1.0, 0.01)

    # Test indexing
    final = data[-1]
    assert abs(final.theta - data.theta[-1]) < 1e-10

    initial = data[0]
    assert abs(initial.theta - 0.5) < 1e-10

    middle = data[50]
    assert abs(middle.theta - data.theta[50]) < 1e-10


def test_checkpoint_restore():
    """Test restoring model state from data checkpoint."""
    model = Model(Pendulum)

    model.eq(model.v.theta.dot() - model.v.omega)
    model.eq(model.v.omega.dot() + model.v.g / model.v.l * ca.sin(model.v.theta))

    model.build()
    model.v0.theta = 0.5

    t1, data1 = model.simulate(0.0, 1.0, 0.01)

    # Save checkpoint at timestep 50
    checkpoint = data1[50]

    # Continue to t=2
    t2, data2 = model.simulate(1.0, 2.0, 0.01)

    # Restore checkpoint
    model.v0 = checkpoint

    # Verify restoration
    assert abs(model.v0.theta - data1.theta[50]) < 1e-10

    # Simulate from checkpoint - should match
    t3, data3 = model.simulate(t1[50], 1.0, 0.01)
    assert abs(t3[-1] - 1.0) < 1e-6


def test_trajectory_is_trajectory_flag():
    """Test that data has _is_trajectory=True."""
    model = Model(Pendulum)

    model.eq(model.v.theta.dot() - model.v.omega)
    model.eq(model.v.omega.dot() + model.v.g / model.v.l * ca.sin(model.v.theta))

    model.build()
    model.v0.theta = 0.5

    t, data = model.simulate(0.0, 0.1, 0.01)

    # Data should have flag
    assert getattr(data, "_is_trajectory", False) == True

    # Indexed state should NOT have flag
    final = data[-1]
    assert getattr(final, "_is_trajectory", False) == False


@implicit
class PointMass3D:
    """3D point mass with vector states."""

    pos: float = var(shape=3)  # Becomes state (has .dot())
    vel: float = var(shape=3)  # Becomes state (has .dot())
    mass: float = param(default=1.0)
    gravity: float = param(shape=3, default=[0, 0, -9.81])


def test_vector_states_stateful():
    """Test stateful simulation with vector states."""
    model = Model(PointMass3D)

    model.eq(model.v.pos.dot() - model.v.vel)
    model.eq(model.v.vel.dot() - model.v.gravity)

    model.build()

    # Set initial conditions
    model.v0.pos = np.array([0, 0, 10])
    model.v0.vel = np.array([1, 0, 0])

    t, data = model.simulate(0.0, 1.0, 0.01)

    # Check shapes
    assert data.pos.shape == (101, 3)
    assert data.vel.shape == (101, 3)

    # Check model.v0 updated
    assert model.v0.pos.shape == (3,)

    # Check physics - should fall and move forward
    assert model.v0.pos[2] < 10  # Fell down
    assert model.v0.pos[0] > 0  # Moved forward


def test_state_field_not_allowed():
    """Test that using state() in implicit models raises an error."""
    from cyecca.dynamics.explicit.fields import state

    @implicit
    class BadModel:
        x: float = state()  # Not allowed!
        p: float = param(default=1.0)

    # Should raise when trying to create symbolic instance
    with pytest.raises(ValueError, match="not allowed in implicit models"):
        Model(BadModel)


def test_alg_field_not_allowed():
    """Test that using alg() in implicit models raises an error."""
    from cyecca.dynamics.explicit.fields import algebraic_var as alg

    @implicit
    class BadModel:
        x: float = var()
        z: float = alg()  # Not allowed!

    # Should raise when trying to create symbolic instance
    with pytest.raises(ValueError, match="not allowed in implicit models"):
        Model(BadModel)


def test_state_inference():
    """Test that states are correctly inferred from .dot() usage."""

    @implicit
    class MixedModel:
        x: float = var()  # Will become state (has .dot())
        v: float = var()  # Will become state (has .dot())
        constraint: float = var()  # Will become algebraic (no .dot())
        k: float = param(default=1.0)

    model = Model(MixedModel)

    # Define equations - x and v have derivatives, constraint doesn't
    model.eq(model.v.x.dot() - model.v.v)
    model.eq(model.v.v.dot() + model.v.k * model.v.x)
    model.eq(model.v.constraint - (model.v.x**2 + model.v.v**2 - 1))  # Algebraic constraint

    model.build()

    # Check inferred types
    assert "x" in model._state_fields
    assert "v" in model._state_fields
    assert "constraint" in model._alg_fields
