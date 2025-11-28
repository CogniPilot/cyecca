"""Test the explicit Model class with unified namespace.

This tests the unified namespace interface where:
- All variables in one class (states, inputs, params, outputs)
- Single namespace: model.v provides access to all variables
- Time is built-in (model.t)
- Model carries internal state (model.v0)
- simulate() uses and updates model.v0 automatically
- simulate() returns (t, data) tuple
- linearize() returns (A, B, C, D) state-space matrices
"""

import casadi as ca
import numpy as np
import pytest

from cyecca.dynamics.explicit import Model, explicit, input_var, output_var, param, state


@explicit
class MassSpring:
    """Simple mass-spring-damper system."""

    # States
    x: float = state()
    v: float = state()
    # Inputs
    F: float = input_var()
    # Parameters
    m: float = param(default=1.0)
    k: float = param(default=1.0)
    c: float = param(default=0.1)
    # Outputs
    position: float = output_var()
    velocity: float = output_var()


def test_model_creation():
    """Test Model class initialization."""
    model = Model(MassSpring)

    # Check symbolic variables exist in unified namespace
    assert hasattr(model.v, "x")
    assert hasattr(model.v, "v")
    assert hasattr(model.v, "F")
    assert hasattr(model.v, "m")
    assert hasattr(model.v, "position")

    # Check built-in time exists
    assert hasattr(model, "t")
    assert isinstance(model.t, ca.SX)

    # Check numeric state exists
    assert hasattr(model, "v0")

    # Check type vectors are built
    assert model.x_sym.shape[0] == 2  # x, v
    assert model.u_sym.shape[0] == 1  # F
    assert model.p_sym.shape[0] == 3  # m, k, c
    assert model.y_sym.shape[0] == 2  # position, velocity


def test_ode_definition():
    """Test ODE definition with .ode() method."""
    model = Model(MassSpring)

    # Define dynamics
    model.ode(model.v.x, model.v.v)
    model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)

    # Should have 2 ODE definitions
    assert len(model._ode_defs) == 2


def test_output_definition():
    """Test output definition with .output() method."""
    model = Model(MassSpring)

    model.ode(model.v.x, model.v.v)
    model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)

    # Define outputs
    model.output(model.v.position, model.v.x)
    model.output(model.v.velocity, model.v.v)

    assert len(model._output_defs) == 2


def test_build():
    """Test model build."""
    model = Model(MassSpring)

    model.ode(model.v.x, model.v.v)
    model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)
    model.output(model.v.position, model.v.x)
    model.output(model.v.velocity, model.v.v)

    model.build()

    # Check dynamics function created
    assert hasattr(model, "_f")
    assert hasattr(model, "_g")


def test_simulate():
    """Test simulation."""
    model = Model(MassSpring)

    model.ode(model.v.x, model.v.v)
    model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)
    model.output(model.v.position, model.v.x)

    model.build()

    # Set initial conditions
    model.v0.x = 1.0
    model.v0.v = 0.0
    model.v0.F = 0.0

    # Simulate
    t, data = model.simulate(0.0, 1.0, 0.01)

    # Check time array
    assert t.shape == (101,)
    assert abs(t[0]) < 1e-10
    assert abs(t[-1] - 1.0) < 1e-6

    # Check data trajectory
    assert data.x.shape == (101,)
    assert data.v.shape == (101,)

    # Check initial value preserved
    assert abs(data.x[0] - 1.0) < 1e-10

    # Check model.v0 was updated to final state
    assert abs(model.v0.x - data.x[-1]) < 1e-10


def test_chained_simulation():
    """Test that simulations can be chained."""
    model = Model(MassSpring)

    model.ode(model.v.x, model.v.v)
    model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)

    model.build()

    # First simulation
    model.v0.x = 1.0
    t1, data1 = model.simulate(0.0, 1.0, 0.01)

    # Second simulation - continues from final state
    t2, data2 = model.simulate(1.0, 2.0, 0.01)

    # Check continuity
    assert abs(data2.x[0] - data1.x[-1]) < 1e-10


def test_linearize():
    """Test linearization."""
    model = Model(MassSpring)

    model.ode(model.v.x, model.v.v)
    model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)
    model.output(model.v.position, model.v.x)
    model.output(model.v.velocity, model.v.v)

    model.build()

    # Linearize at origin
    A, B, C, D = model.linearize()

    # Check dimensions
    assert A.shape == (2, 2)  # 2 states
    assert B.shape == (2, 1)  # 1 input
    assert C.shape == (2, 2)  # 2 outputs, 2 states
    assert D.shape == (2, 1)  # 2 outputs, 1 input

    # Check A matrix structure for mass-spring-damper
    # dx/dt = v -> A[0,1] = 1
    # dv/dt = (F - c*v - k*x) / m -> A[1,0] = -k/m, A[1,1] = -c/m
    assert abs(A[0, 1] - 1.0) < 1e-10
    assert abs(A[1, 0] - (-1.0)) < 1e-10  # -k/m = -1/1 = -1
    assert abs(A[1, 1] - (-0.1)) < 1e-10  # -c/m = -0.1/1 = -0.1

    # Check B matrix: dv/dt depends on F/m
    assert abs(B[1, 0] - 1.0) < 1e-10  # 1/m = 1

    # Check C matrix (identity for position, velocity outputs)
    assert abs(C[0, 0] - 1.0) < 1e-10  # position = x
    assert abs(C[1, 1] - 1.0) < 1e-10  # velocity = v


def test_input_function():
    """Test simulation with time-varying input."""
    model = Model(MassSpring)

    model.ode(model.v.x, model.v.v)
    model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)

    model.build()

    model.v0.x = 0.0
    model.v0.v = 0.0

    # Apply step input at t=0.5
    def u_step(t, m):
        return {"F": 1.0} if t >= 0.5 else {"F": 0.0}

    t, data = model.simulate(0.0, 2.0, 0.01, u_func=u_step)

    # Check that system responds to input
    # Before step, should stay near zero
    idx_before = np.where(t < 0.5)[0][-1]
    assert abs(data.x[idx_before]) < 0.1

    # After step, should move
    assert abs(data.x[-1]) > 0.1


@explicit
class Pendulum:
    """Simple pendulum (nonlinear)."""

    theta: float = state()
    omega: float = state()
    g: float = param(default=9.81)
    l: float = param(default=1.0)


def test_nonlinear_system():
    """Test with nonlinear system (pendulum)."""
    model = Model(Pendulum)

    model.ode(model.v.theta, model.v.omega)
    model.ode(model.v.omega, -model.v.g / model.v.l * ca.sin(model.v.theta))

    model.build()

    # Small angle - should oscillate
    model.v0.theta = 0.1
    model.v0.omega = 0.0

    t, data = model.simulate(0.0, 2.0, 0.01)

    # Check oscillation (theta should cross zero)
    crossings = np.where(np.diff(np.sign(data.theta)))[0]
    assert len(crossings) >= 2  # At least one full oscillation


def test_properties():
    """Test n_states, n_inputs, etc. properties."""
    model = Model(MassSpring)

    assert model.n_states == 2
    assert model.n_inputs == 1
    assert model.n_params == 3
    assert model.n_outputs == 2


@explicit
class VectorSystem:
    """System with vector states."""

    pos: float = state(shape=3)
    vel: float = state(shape=3)
    force: float = input_var(shape=3)
    mass: float = param(default=1.0)


def test_vector_states():
    """Test system with vector-valued states."""
    model = Model(VectorSystem)

    # Simple point mass: dv/dt = F/m
    model.ode(model.v.pos, model.v.vel)
    model.ode(model.v.vel, model.v.force / model.v.mass)

    model.build()

    # Check dimensions
    assert model.n_states == 6  # 3 + 3
    assert model.n_inputs == 3

    # Simulate
    model.v0.pos = np.array([0.0, 0.0, 0.0])
    model.v0.vel = np.array([1.0, 0.0, 0.0])
    model.v0.force = np.array([0.0, 0.0, -9.81])

    t, data = model.simulate(0.0, 1.0, 0.01)

    # Check trajectory shapes
    assert data.pos.shape == (101, 3)
    assert data.vel.shape == (101, 3)

    # Check physics - should move in x and fall in z
    assert data.pos[-1, 0] > 0.9  # Moved ~1m in x
    assert data.pos[-1, 2] < -4  # Fell ~5m in z
