"""
Tests for NaN fixes in Lie group operations.

These tests verify that Jacobian computations don't produce NaN at critical
points like zero angular velocity, 180° rotations, and near singularities.
"""

from test.common import ProfiledTestCase, is_finite

import casadi as ca
import numpy as np
from beartype import beartype

from cyecca.lie.group_se3 import SE3Quat, se3
from cyecca.lie.group_se23 import SE23Quat, se23
from cyecca.lie.group_so3 import SO3Dcm, SO3Mrp, SO3Quat, so3


@beartype
class TestNaNFixesSO3(ProfiledTestCase):
    """Test SO(3) operations at critical points that previously caused NaN."""

    def test_zero_angular_velocity(self):
        """Test that zero angular velocity doesn't cause NaN in Jacobians."""
        # Use SX (symbolic) instead of DM (numeric)
        omega = ca.SX([0, 0, 0])
        omega_alg = so3.elem(omega)

        # Test Jacobians
        J_left = omega_alg.left_jacobian()
        J_right = omega_alg.right_jacobian()
        J_left_inv = omega_alg.left_jacobian_inv()
        J_right_inv = omega_alg.right_jacobian_inv()

        # Check for NaN - evaluate to numeric for checking
        self.assertTrue(is_finite(J_left), "NaN in left Jacobian at omega=0")
        self.assertTrue(is_finite(J_right), "NaN in right Jacobian at omega=0")
        self.assertTrue(is_finite(J_left_inv), "NaN in left Jacobian inverse at omega=0")
        self.assertTrue(is_finite(J_right_inv), "NaN in right Jacobian inverse at omega=0")

        # Check that they equal identity at zero
        J_left_val = ca.DM(ca.evalf(J_left))
        J_right_val = ca.DM(ca.evalf(J_right))
        np.testing.assert_allclose(J_left_val, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(J_right_val, np.eye(3), atol=1e-6)

    def test_near_zero_angular_velocity(self):
        """Test very small angular velocities."""
        test_omegas = [
            [1e-10, 0, 0],
            [0, 1e-10, 0],
            [0, 0, 1e-10],
            [1e-8, 1e-8, 1e-8],
        ]

        for omega_val in test_omegas:
            omega = ca.SX(omega_val)
            omega_alg = so3.elem(omega)

            J_left = omega_alg.left_jacobian()
            J_right = omega_alg.right_jacobian()

            self.assertTrue(is_finite(J_left), f"NaN in left Jacobian at {omega_val}")
            self.assertTrue(is_finite(J_right), f"NaN in right Jacobian at {omega_val}")

    def test_180_degree_rotations(self):
        """Test SO(3) operations at 180° rotations (critical points)."""
        test_omegas = [
            [np.pi, 0, 0],
            [0, np.pi, 0],
            [0, 0, np.pi],
        ]

        for omega_val in test_omegas:
            omega = ca.SX(omega_val)
            omega_alg = so3.elem(omega)

            # Test exp and log
            q = SO3Quat.exp(omega_alg)
            omega_back = SO3Quat.log(q)

            self.assertTrue(is_finite(q.param), f"NaN in quaternion at omega={omega_val}")
            self.assertTrue(is_finite(omega_back.param), f"NaN in log(exp(omega)) at {omega_val}")

    def test_quaternion_normalization(self):
        """Test quaternion normalization at edge cases."""
        test_quats = [
            [1, 0, 0, 0],  # Normal
            [1e-10, 0, 0, 0],  # Near zero (edge case)
            [0, 0, 0, 1],  # 180° rotation
        ]

        for q_val in test_quats:
            q = SO3Quat.elem(ca.SX(q_val))
            omega = SO3Quat.log(q)
            self.assertTrue(is_finite(omega.param), f"NaN in log of q={q_val}")

    def test_mrp_singularity_180_degrees(self):
        """Test MRP at its known singularity (180° rotation).

        This is the CRITICAL test - MRP has a singularity at q0 = -1.
        """
        # Create quaternions representing 180° rotations
        test_quats = [
            [0, 0, 0, 1],  # 180° about z
            [0, 0, 1, 0],  # 180° about y
            [0, 1, 0, 0],  # 180° about x
            [-0.0001, 0, 0, 1],  # Near singularity
        ]

        for q_val in test_quats:
            q = SO3Quat.elem(ca.SX(q_val))
            mrp = SO3Mrp.from_Quat(q)
            self.assertTrue(is_finite(mrp.param), f"NaN in MRP from q={q_val}")

    def test_mrp_product(self):
        """Test MRP product operation doesn't produce NaN."""
        test_pairs = [
            ([0, 0, 0], [0, 0, 0]),  # Both identity
            ([0.1, 0, 0], [0.1, 0, 0]),  # Small rotations
            ([0.5, 0, 0], [-0.5, 0, 0]),  # Opposites
        ]

        for mrp1_val, mrp2_val in test_pairs:
            mrp1 = SO3Mrp.elem(ca.SX(mrp1_val))
            mrp2 = SO3Mrp.elem(ca.SX(mrp2_val))

            mrp_prod = mrp1.group.product(mrp1, mrp2)
            self.assertTrue(
                is_finite(mrp_prod.param),
                f"NaN in product of {mrp1_val} and {mrp2_val}",
            )

    def test_rotation_matrix_to_quaternion(self):
        """Test rotation matrix to quaternion conversion at critical angles."""
        test_matrices = [
            ("Identity", np.eye(3)),
            ("180° about x", np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])),
            ("180° about y", np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])),
            ("180° about z", np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])),
            ("90° about z", np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
        ]

        for name, R_val in test_matrices:
            R = ca.SX(R_val)  # Use SX instead of DM
            q = SO3Quat.from_Matrix(R)
            self.assertTrue(is_finite(q.param), f"NaN in quaternion from {name}")

    def test_exp_log_roundtrip(self):
        """Test exp/log roundtrip doesn't produce NaN."""
        test_omegas = [
            [0, 0, 0],
            [1e-10, 0, 0],
            [0.1, 0.2, 0.3],
        ]

        for omega_val in test_omegas:
            omega1 = ca.SX(omega_val)
            omega_alg = so3.elem(omega1)

            q = SO3Quat.exp(omega_alg)
            omega2_alg = SO3Quat.log(q)
            omega2 = omega2_alg.param

            self.assertTrue(is_finite(omega2), f"NaN in roundtrip at omega={omega_val}")


@beartype
class TestNaNFixesSE3(ProfiledTestCase):
    """Test SE(3) operations at critical points."""

    def test_zero_velocity(self):
        """Test SE(3) at zero velocity."""
        xi = se3.elem(ca.SX(np.zeros(6)))

        J_left = xi.left_jacobian()
        J_right = xi.right_jacobian()

        self.assertTrue(is_finite(J_left), "NaN in SE(3) left Jacobian at zero")
        self.assertTrue(is_finite(J_right), "NaN in SE(3) right Jacobian at zero")

    def test_q_matrix_computation(self):
        """Test SE(3) Q matrix computation at various velocities."""
        test_vels = [
            np.zeros(6),
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 1e-10, 0, 0]),
            np.array([1, 2, 3, 0, 0, np.pi]),
        ]

        for vel in test_vels:
            xi = se3.elem(ca.SX(vel))

            Q_left = xi.left_Q()
            Q_right = xi.right_Q()

            self.assertTrue(is_finite(Q_left), f"NaN in left Q at vel={vel}")
            self.assertTrue(is_finite(Q_right), f"NaN in right Q at vel={vel}")


@beartype
class TestNaNFixesSE23(ProfiledTestCase):
    """Test SE₂(3) operations - critical for spacecraft dynamics."""

    def test_zero_state(self):
        """Test SE₂(3) at zero state."""
        xi = se23.elem(ca.SX(np.zeros(9)))

        J_left = xi.left_jacobian()
        J_right = xi.right_jacobian()
        J_left_inv = xi.left_jacobian_inv()
        J_right_inv = xi.right_jacobian_inv()

        self.assertTrue(is_finite(J_left), "NaN in SE₂(3) left Jacobian")
        self.assertTrue(is_finite(J_right), "NaN in SE₂(3) right Jacobian")
        self.assertTrue(is_finite(J_left_inv), "NaN in SE₂(3) left Jacobian inverse")
        self.assertTrue(is_finite(J_right_inv), "NaN in SE₂(3) right Jacobian inverse")

    def test_pure_rotation(self):
        """Test SE₂(3) with pure rotation (zero translation/acceleration)."""
        test_states = [
            np.array([0, 0, 0, 0, 0, 0, 1e-10, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, np.pi]),
            np.array([0, 0, 0, 0, 0, 0, 1, 2, 3]),
        ]

        for state in test_states:
            xi = se23.elem(ca.SX(state))
            J_left = xi.left_jacobian()
            self.assertTrue(is_finite(J_left), f"NaN in SE₂(3) Jacobian at omega={state[6:]}")

    def test_spacecraft_hover(self):
        """Test SE₂(3) in spacecraft hover condition (near-zero velocities).

        This is critical for spacecraft applications.
        """
        hover_state = ca.SX([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10])
        xi = se23.elem(hover_state)

        J_left = xi.left_jacobian()
        J_right = xi.right_jacobian()

        self.assertTrue(is_finite(J_left), "NaN in hover left Jacobian")
        self.assertTrue(is_finite(J_right), "NaN in hover right Jacobian")


@beartype
class TestTaylorSeriesEpsilon(ProfiledTestCase):
    """Test that Taylor series evaluation works with new epsilon."""

    def test_series_evaluation_near_threshold(self):
        """Test series evaluation at and near the new threshold (1e-6)."""
        from cyecca.symbolic import SQUARED_SERIES

        test_points = [
            0,
            1e-12,
            1e-9,
            1e-7,
            1e-6,  # At new threshold
            1e-5,
        ]

        for theta_sq in test_points:
            # Use SX for symbolic evaluation
            theta_sq_sym = ca.SX(theta_sq)

            val1 = SQUARED_SERIES["sin(x)/x"](theta_sq_sym)
            val2 = SQUARED_SERIES["(1 - cos(x))/x^2"](theta_sq_sym)
            val3 = SQUARED_SERIES["(x - sin(x))/x^3"](theta_sq_sym)

            self.assertTrue(is_finite(val1), f"NaN in sin(x)/x at theta_sq={theta_sq}")
            self.assertTrue(is_finite(val2), f"NaN in (1-cos(x))/x^2 at theta_sq={theta_sq}")
            self.assertTrue(is_finite(val3), f"NaN in (x-sin(x))/x^3 at theta_sq={theta_sq}")
