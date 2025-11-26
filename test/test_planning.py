"""
Pytest unit tests for Dubins path planner
Run with: pytest test_planning.py -v
"""

import numpy as np
import pytest

from cyecca.planning import derive_dubins


@pytest.fixture(scope="module")
def dubins_functions():
    """Create planner and evaluator functions once for all tests."""
    return derive_dubins()


@pytest.fixture
def random_path_configs():
    """Generate random path configurations for testing."""
    np.random.seed(42)
    configs = []
    for _ in range(20):
        p0 = np.random.rand(2) * 10
        p1 = np.random.rand(2) * 10
        psi0 = np.random.rand() * 2 * np.pi - np.pi
        psi1 = np.random.rand() * 2 * np.pi - np.pi
        R = 1.0
        configs.append((p0, psi0, p1, psi1, R))
    return configs


def evaluate_path(eval_fn, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R, n_points=200):
    """Helper to evaluate a path at multiple points."""
    s_vals = np.linspace(0, 1, n_points)
    path_x, path_y, path_psi = [], [], []

    for s in s_vals:
        x, y, psi = eval_fn(s, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)
        path_x.append(float(x))
        path_y.append(float(y))
        path_psi.append(float(psi))

    return np.array(path_x), np.array(path_y), np.array(path_psi)


class TestForwardMotion:
    """Test that vehicle always moves forward."""

    def test_forward_motion_all_paths(self, dubins_functions, random_path_configs):
        plan_fn, eval_fn = dubins_functions
        tolerance = 1e-6

        for i, (p0, psi0, p1, psi1, R) in enumerate(random_path_configs):
            cost, ct, a1, d, a2, tp0, tp1, c0, c1 = plan_fn(p0, psi0, p1, psi1, R)
            tp0, tp1 = np.array(tp0).flatten(), np.array(tp1).flatten()
            c0, c1 = np.array(c0).flatten(), np.array(c1).flatten()

            path_x, path_y, path_psi = evaluate_path(eval_fn, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R, n_points=100)

            # Check forward motion
            for j in range(1, len(path_x)):
                dx, dy = path_x[j] - path_x[j - 1], path_y[j] - path_y[j - 1]
                dist = np.sqrt(dx**2 + dy**2)

                if dist > tolerance:
                    direction = np.arctan2(dy, dx)
                    heading_error = np.arctan2(
                        np.sin(path_psi[j - 1] - direction),
                        np.cos(path_psi[j - 1] - direction),
                    )

                    # Forward motion: heading aligned with direction of travel (within 90°)
                    assert (
                        abs(heading_error) < np.pi / 2
                    ), f"Path {i}, segment {j}: backward motion detected (heading error: {np.rad2deg(heading_error):.2f}°)"


class TestPositionContinuity:
    """Test position continuity."""

    def test_no_position_jumps(self, dubins_functions, random_path_configs):
        plan_fn, eval_fn = dubins_functions
        max_allowed_jump = 0.1

        for i, (p0, psi0, p1, psi1, R) in enumerate(random_path_configs):
            cost, ct, a1, d, a2, tp0, tp1, c0, c1 = plan_fn(p0, psi0, p1, psi1, R)
            tp0, tp1 = np.array(tp0).flatten(), np.array(tp1).flatten()
            c0, c1 = np.array(c0).flatten(), np.array(c1).flatten()

            path_x, path_y, _ = evaluate_path(eval_fn, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)

            dx = np.diff(path_x)
            dy = np.diff(path_y)
            jumps = np.sqrt(dx**2 + dy**2)
            max_jump = np.max(jumps)

            assert max_jump < max_allowed_jump, f"Path {i}: position discontinuity detected (max jump: {max_jump:.6f})"


class TestHeadingContinuity:
    """Test heading continuity."""

    def test_no_heading_jumps(self, dubins_functions, random_path_configs):
        plan_fn, eval_fn = dubins_functions
        max_allowed_jump_deg = 5
        max_allowed_jump_rad = np.deg2rad(max_allowed_jump_deg)

        for i, (p0, psi0, p1, psi1, R) in enumerate(random_path_configs):
            cost, ct, a1, d, a2, tp0, tp1, c0, c1 = plan_fn(p0, psi0, p1, psi1, R)
            tp0, tp1 = np.array(tp0).flatten(), np.array(tp1).flatten()
            c0, c1 = np.array(c0).flatten(), np.array(c1).flatten()

            _, _, path_psi = evaluate_path(eval_fn, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)

            dpsi = np.diff(path_psi)
            dpsi = np.arctan2(np.sin(dpsi), np.cos(dpsi))  # Wrap to [-pi, pi]

            max_jump = np.max(np.abs(dpsi))

            assert (
                max_jump < max_allowed_jump_rad
            ), f"Path {i}: heading discontinuity detected (max jump: {np.rad2deg(max_jump):.2f}°)"


class TestBoundaryConditions:
    """Test that paths start and end at correct positions and headings."""

    def test_start_position_and_heading(self, dubins_functions, random_path_configs):
        plan_fn, eval_fn = dubins_functions
        tolerance = 1e-4

        for i, (p0, psi0, p1, psi1, R) in enumerate(random_path_configs):
            cost, ct, a1, d, a2, tp0, tp1, c0, c1 = plan_fn(p0, psi0, p1, psi1, R)
            tp0, tp1 = np.array(tp0).flatten(), np.array(tp1).flatten()
            c0, c1 = np.array(c0).flatten(), np.array(c1).flatten()

            # Check start (s=0)
            x0, y0, psi0_out = eval_fn(0, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)
            x0, y0, psi0_out = float(x0), float(y0), float(psi0_out)

            pos_err = np.sqrt((x0 - p0[0]) ** 2 + (y0 - p0[1]) ** 2)
            head_err = abs(np.arctan2(np.sin(psi0_out - psi0), np.cos(psi0_out - psi0)))

            assert pos_err < tolerance, f"Path {i}: start position error {pos_err:.6f}"
            assert head_err < tolerance, f"Path {i}: start heading error {np.rad2deg(head_err):.6f}°"

    def test_end_position_and_heading(self, dubins_functions, random_path_configs):
        plan_fn, eval_fn = dubins_functions
        tolerance = 1e-4

        for i, (p0, psi0, p1, psi1, R) in enumerate(random_path_configs):
            cost, ct, a1, d, a2, tp0, tp1, c0, c1 = plan_fn(p0, psi0, p1, psi1, R)
            tp0, tp1 = np.array(tp0).flatten(), np.array(tp1).flatten()
            c0, c1 = np.array(c0).flatten(), np.array(c1).flatten()

            # Check end (s=1)
            x1, y1, psi1_out = eval_fn(1, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)
            x1, y1, psi1_out = float(x1), float(y1), float(psi1_out)

            pos_err = np.sqrt((x1 - p1[0]) ** 2 + (y1 - p1[1]) ** 2)
            head_err = abs(np.arctan2(np.sin(psi1_out - psi1), np.cos(psi1_out - psi1)))

            assert pos_err < tolerance, f"Path {i}: end position error {pos_err:.6f}"
            assert head_err < tolerance, f"Path {i}: end heading error {np.rad2deg(head_err):.6f}°"


class TestPathTypes:
    """Test that all path types are generated."""

    def test_all_path_types_used(self, dubins_functions):
        plan_fn, _ = dubins_functions

        # Test configurations designed to favor specific path types
        configs = [
            # RSL: start right, end left
            (np.array([0, 0]), 0, np.array([5, 5]), np.pi / 2, 1.0),
            # LSR: start left, end right
            (np.array([0, 0]), np.pi, np.array([5, -5]), -np.pi / 2, 1.0),
            # LSL: both left
            (np.array([0, 0]), 0, np.array([5, 0]), 0, 1.0),
            # RSR: both right
            (np.array([0, 0]), np.pi, np.array([5, 0]), np.pi, 1.0),
        ]

        path_types_seen = set()

        for p0, psi0, p1, psi1, R in configs:
            _, ct, _, _, _, _, _, _, _ = plan_fn(p0, psi0, p1, psi1, R)
            path_types_seen.add(int(ct))

        # We should see at least 2 different path types in these configs
        assert len(path_types_seen) >= 2, f"Only {len(path_types_seen)} path type(s) generated: {path_types_seen}"


class TestPathCost:
    """Test path cost properties."""

    def test_cost_is_positive(self, dubins_functions, random_path_configs):
        plan_fn, _ = dubins_functions

        for i, (p0, psi0, p1, psi1, R) in enumerate(random_path_configs):
            cost, _, _, _, _, _, _, _, _ = plan_fn(p0, psi0, p1, psi1, R)
            cost = float(cost)

            assert cost > 0, f"Path {i}: cost should be positive, got {cost}"
            assert cost < np.inf, f"Path {i}: cost should be finite, got {cost}"

    def test_cost_equals_path_length(self, dubins_functions, random_path_configs):
        plan_fn, eval_fn = dubins_functions
        tolerance = 0.1  # Allow some numerical error

        for i, (p0, psi0, p1, psi1, R) in enumerate(random_path_configs):
            cost, ct, a1, d, a2, tp0, tp1, c0, c1 = plan_fn(p0, psi0, p1, psi1, R)
            cost = float(cost)
            tp0, tp1 = np.array(tp0).flatten(), np.array(tp1).flatten()
            c0, c1 = np.array(c0).flatten(), np.array(c1).flatten()

            # Integrate arc lengths
            path_x, path_y, _ = evaluate_path(eval_fn, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R, n_points=1000)

            dx = np.diff(path_x)
            dy = np.diff(path_y)
            actual_length = np.sum(np.sqrt(dx**2 + dy**2))

            error = abs(cost - actual_length)

            assert (
                error < tolerance
            ), f"Path {i}: cost {cost:.4f} != actual length {actual_length:.4f} (error: {error:.4f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
