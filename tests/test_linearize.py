"""Tests for generic linearization API."""

import casadi as ca
import numpy as np
import pytest
from cyecca.dynamics import (
    ModelSX,
    input_var,
    param,
    state,
    symbolic,
    find_trim,
    linearize_dynamics,
    analyze_modes,
)


def simple_oscillator_classes():
    """Define classes for simple oscillator using @symbolic decorator."""

    @symbolic
    class States:
        x: ca.SX = state(default=0.0, desc="position (m)")
        v: ca.SX = state(default=0.0, desc="velocity (m/s)")

    @symbolic
    class Inputs:
        F: ca.SX = input_var(desc="external force (N)")

    @symbolic
    class Params:
        m: float = param(default=1.0, desc="mass (kg)")
        k: float = param(default=1.0, desc="spring constant (N/m)")
        c: float = param(default=0.1, desc="damping coefficient (Ns/m)")

    return States, Inputs, Params


def simple_oscillator():
    """Create a simple damped harmonic oscillator model.

    This is a basic 2-state system useful for testing linearization:
    - dx/dt = v
    - dv/dt = (F - k*x - c*v) / m

    Returns:
        ModelSX: Simple oscillator model
    """
    States, Inputs, Params = simple_oscillator_classes()

    model = ModelSX.create(States, Inputs, Params)

    # Dynamics: spring-mass-damper with external force
    x = model.x.x
    v = model.x.v
    F = model.u.F
    m = model.p.m
    k = model.p.k
    c = model.p.c

    dx_dt = v
    dv_dt = (F - k * x - c * v) / m

    f_x = ca.vertcat(dx_dt, dv_dt)

    model.build(f_x=f_x, integrator="rk4")

    return model


def test_simple_oscillator_creation():
    """Test that the simple oscillator model can be created."""
    model = simple_oscillator()

    assert model is not None
    assert hasattr(model, "f_x")
    assert hasattr(model, "f_step")
    assert model.x.size1() == 2
    assert model.u.size1() == 1
    assert model.p.size1() == 3


def test_find_trim_simple_oscillator():
    """Test finding equilibrium for simple oscillator."""
    model = simple_oscillator()

    # Initial guess at origin (equilibrium point)
    x_guess = np.array([0.0, 0.0])
    u_guess = np.array([0.0])

    def cost_fn(model, x_var, u_var, x_dot, p_vec):
        # Minimize state derivatives (equilibrium condition)
        return ca.sumsqr(x_dot.as_vec())

    def constraints_fn(model, x_var, u_var, x_dot, p_vec):
        # Fix force to zero to ensure equilibrium at origin
        return [u_var.F == 0.0]

    # Find trim
    x_trim, u_trim, stats = find_trim(
        model,
        x_guess,
        u_guess,
        cost_fn,
        constraints_fn,
        ipopt_print_level=0,
        print_progress=False,
        verbose=False,
    )

    # At equilibrium with no force, position should be at origin
    assert x_trim is not None
    assert u_trim is not None
    assert np.abs(x_trim[0]) < 1e-3  # Position near zero (relaxed tolerance)
    assert np.abs(x_trim[1]) < 1e-6  # Velocity near zero
    assert np.abs(u_trim[0]) < 1e-6  # Force near zero


def test_find_trim_with_force():
    """Test finding equilibrium with external force applied."""
    model = simple_oscillator()

    # With constant force, equilibrium is at displaced position
    target_force = 5.0
    expected_position = target_force / float(model.p0.k)  # F = k*x at equilibrium

    x_guess = np.array([expected_position, 0.0])
    u_guess = np.array([target_force])

    def cost_fn(model, x_var, u_var, x_dot, p_vec):
        # Minimize derivatives while keeping force constant
        return ca.sumsqr(x_dot.as_vec())

    def constraints_fn(model, x_var, u_var, x_dot, p_vec):
        # Fix the force to target value
        return [u_var.F == target_force]

    x_trim, u_trim, stats = find_trim(
        model,
        x_guess,
        u_guess,
        cost_fn,
        constraints_fn,
        ipopt_print_level=0,
        print_progress=False,
        verbose=False,
    )

    # Check equilibrium: F = k*x, v = 0
    assert np.abs(x_trim[1]) < 1e-6  # Velocity should be zero
    assert np.abs(u_trim[0] - target_force) < 1e-6  # Force should match
    assert np.abs(x_trim[0] - expected_position) < 1e-3  # Position = F/k


def test_linearize_oscillator_at_origin():
    """Test linearization of oscillator at origin."""
    model = simple_oscillator()

    # Linearize at origin with no force
    x_trim = np.array([0.0, 0.0])
    u_trim = np.array([0.0])

    A, B = linearize_dynamics(model, x_trim, u_trim)

    # Check dimensions
    assert A.shape == (2, 2)
    assert B.shape == (2, 1)

    # For spring-mass-damper system:
    # A = [[0, 1], [-k/m, -c/m]]
    m = float(model.p0.m)
    k = float(model.p0.k)
    c = float(model.p0.c)

    expected_A = np.array([[0, 1], [-k / m, -c / m]])

    # B = [[0], [1/m]]
    expected_B = np.array([[0], [1 / m]])

    np.testing.assert_allclose(A, expected_A, rtol=1e-6)
    np.testing.assert_allclose(B, expected_B, rtol=1e-6)


def test_linearize_oscillator_displaced():
    """Test that linearization is independent of operating point for linear system."""
    model = simple_oscillator()

    # Linearize at displaced position
    x_trim = np.array([2.0, 0.5])
    u_trim = np.array([3.0])

    A, B = linearize_dynamics(model, x_trim, u_trim)

    # For a linear system, A and B should be the same regardless of operating point
    m = float(model.p0.m)
    k = float(model.p0.k)
    c = float(model.p0.c)

    expected_A = np.array([[0, 1], [-k / m, -c / m]])
    expected_B = np.array([[0], [1 / m]])

    np.testing.assert_allclose(A, expected_A, rtol=1e-6)
    np.testing.assert_allclose(B, expected_B, rtol=1e-6)


def test_analyze_modes_oscillator():
    """Test modal analysis on oscillator."""
    model = simple_oscillator()

    x_trim = np.array([0.0, 0.0])
    u_trim = np.array([0.0])

    A, B = linearize_dynamics(model, x_trim, u_trim)

    state_names = ["x", "v"]
    modes = analyze_modes(A, state_names=state_names)

    # Should have 2 modes (may be complex conjugate pair)
    assert len(modes) >= 1

    # Check that modes have required fields
    for mode in modes:
        assert "eigenvalue" in mode
        assert "stable" in mode
        assert "real" in mode
        assert "imag" in mode

    # For damped oscillator, should be stable (all eigenvalues have negative real part)
    stable_modes = [m for m in modes if m["stable"]]
    assert len(stable_modes) > 0


def test_analyze_modes_stability():
    """Test stability detection in modal analysis."""
    model = simple_oscillator()

    # Create unstable system by setting negative damping
    model.p0.c = -0.1  # Negative damping = unstable

    x_trim = np.array([0.0, 0.0])
    u_trim = np.array([0.0])

    A, B = linearize_dynamics(model, x_trim, u_trim)

    state_names = ["x", "v"]
    modes = analyze_modes(A, state_names=state_names)

    # Should detect instability
    unstable_modes = [m for m in modes if not m["stable"]]
    assert len(unstable_modes) > 0

    # Eigenvalues should have positive real parts
    for mode in unstable_modes:
        assert mode["real"] > 0


def test_find_trim_with_dataclass_guess():
    """Test that find_trim accepts dataclass initial guesses."""
    model = simple_oscillator()

    # Use dataclass instances as guesses
    x_guess = model.x0  # Dataclass instance
    u_guess = model.u0  # Dataclass instance

    def cost_fn(model, x_var, u_var, x_dot, p_vec):
        return ca.sumsqr(x_dot.as_vec())

    x_trim, u_trim, stats = find_trim(
        model,
        x_guess,
        u_guess,
        cost_fn,
        None,
        ipopt_print_level=0,
        print_progress=False,
        verbose=False,
    )

    assert x_trim is not None
    assert u_trim is not None


def test_find_trim_none_guess():
    """Test that find_trim works with None guesses (uses defaults)."""
    model = simple_oscillator()

    def cost_fn(model, x_var, u_var, x_dot, p_vec):
        return ca.sumsqr(x_dot.as_vec())

    # Pass None for guesses - should use model defaults
    x_trim, u_trim, stats = find_trim(
        model,
        None,
        None,
        cost_fn,
        None,
        ipopt_print_level=0,
        print_progress=False,
        verbose=False,
    )

    assert x_trim is not None
    assert u_trim is not None

    # Should converge to origin
    assert np.abs(x_trim[0]) < 1e-6
    assert np.abs(x_trim[1]) < 1e-6


def test_default_cost_function():
    """Test that default cost function works when cost_fn=None."""
    model = simple_oscillator()

    x_guess = np.array([0.5, 0.0])
    u_guess = np.array([0.0])

    # Use default cost function
    x_trim, u_trim, stats = find_trim(
        model,
        x_guess,
        u_guess,
        None,
        None,
        ipopt_print_level=0,
        print_progress=False,
        verbose=False,
    )

    assert x_trim is not None
    assert u_trim is not None


def test_mode_time_constants():
    """Test that mode analysis includes time constants."""
    model = simple_oscillator()

    x_trim = np.array([0.0, 0.0])
    u_trim = np.array([0.0])

    A, B = linearize_dynamics(model, x_trim, u_trim)

    state_names = ["x", "v"]
    modes = analyze_modes(A, state_names=state_names)

    # Each stable mode should have a time constant
    for mode in modes:
        if mode["stable"] and mode["real"] != 0:
            assert "time_constant" in mode
            # Time constant should be positive for stable modes
            if "time_constant" in mode:
                assert mode["time_constant"] > 0


def test_linearization_preserves_dimensions():
    """Test that linearization produces correct matrix dimensions."""
    model = simple_oscillator()

    n_states = model.x.size1()
    n_inputs = model.u.size1()

    x_trim = np.array([1.0, 0.5])
    u_trim = np.array([2.0])

    A, B = linearize_dynamics(model, x_trim, u_trim)

    assert A.shape == (n_states, n_states)
    assert B.shape == (n_states, n_inputs)


def test_find_trim_return_stats():
    """Test that find_trim returns statistics dictionary."""
    model = simple_oscillator()

    x_guess = np.array([0.0, 0.0])
    u_guess = np.array([0.0])

    def cost_fn(model, x_var, u_var, x_dot, p_vec):
        return ca.sumsqr(x_dot.as_vec())

    x_trim, u_trim, stats = find_trim(
        model,
        x_guess,
        u_guess,
        cost_fn,
        None,
        ipopt_print_level=0,
        print_progress=False,
        verbose=False,
    )

    # Stats should be a dictionary with optimization results
    assert isinstance(stats, dict)
    # Check for common solver statistics fields
    assert "iter_count" in stats or "iterations" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
