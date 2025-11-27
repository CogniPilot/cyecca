"""
Multi-Backend Tests: Test the same model across all available backends.

This demonstrates how to create backend-agnostic models by passing
the backend as a parameter, then verifying consistency across backends.
"""

import unittest

import numpy as np
from beartype import beartype

from cyecca.backends import JAX_AVAILABLE, SYMPY_AVAILABLE, CasADiBackend, NumPyBackend, SymbolicBackend

if SYMPY_AVAILABLE:
    from cyecca.backends import SymPyBackend

if JAX_AVAILABLE:
    from cyecca.backends import JAXBackend

from .common import ProfiledTestCase


def create_pendulum_dynamics(backend: SymbolicBackend):
    """Create a simple pendulum model using the given backend.

    The pendulum dynamics are:
        θ̈ = -g/L * sin(θ) - b * θ̇

    State: [θ, θ̇]  (angle, angular velocity)
    Parameters: g (gravity), L (length), b (damping)

    Args:
        backend: The symbolic backend to use (CasADi, SymPy, etc.)

    Returns:
        Tuple of (dynamics_func, jacobian_func, x, p, x_dot, A)
        For non-symbolic backends, functions may be None.
    """
    # State variables
    theta = backend.sym("theta")
    theta_dot = backend.sym("theta_dot")
    x = backend.vertcat(theta, theta_dot)

    # Parameters
    g = backend.sym("g")
    L = backend.sym("L")
    b = backend.sym("b")
    p = backend.vertcat(g, L, b)

    # Dynamics: dx/dt = f(x, p)
    theta_ddot = -g / L * backend.sin(theta) - b * theta_dot
    x_dot = backend.vertcat(theta_dot, theta_ddot)

    # Compute Jacobian for linearization
    A = backend.jacobian(x_dot, x)

    # Create functions (if supported)
    try:
        f = backend.function("pendulum_dynamics", [x, p], [x_dot])
        jac_f = backend.function("pendulum_jacobian", [x, p], [A])
    except (NotImplementedError, AttributeError):
        f = None
        jac_f = None

    return f, jac_f, x, p, x_dot, A


@beartype
class TestCasADiPendulum(ProfiledTestCase):
    """Test pendulum model with CasADi backend."""

    def setUp(self):
        super().setUp()
        self.backend = CasADiBackend()

    def test_create_model(self):
        """Test model creation."""
        f, jac_f, x, p, x_dot, A = create_pendulum_dynamics(self.backend)

        self.assertIsNotNone(f)
        self.assertIsNotNone(jac_f)
        self.assertEqual(x.shape, (2, 1))
        self.assertEqual(p.shape, (3, 1))
        self.assertEqual(x_dot.shape, (2, 1))
        self.assertEqual(A.shape, (2, 2))

    def test_evaluate_dynamics(self):
        """Test dynamics evaluation."""
        f, jac_f, x, p, x_dot, A = create_pendulum_dynamics(self.backend)

        # Evaluate at θ=0.5, θ̇=0, with g=9.81, L=1, b=0.1
        x0 = [0.5, 0.0]
        params = [9.81, 1.0, 0.1]

        result = f(x0, params)
        result_arr = np.array(result).flatten()

        # θ̇ should be 0 (input)
        self.assertAlmostEqual(result_arr[0], 0.0, places=10)

        # θ̈ = -g/L * sin(θ) - b * θ̇ = -9.81 * sin(0.5) - 0 ≈ -4.70
        expected_theta_ddot = -9.81 * np.sin(0.5) - 0.1 * 0.0
        self.assertAlmostEqual(result_arr[1], expected_theta_ddot, places=6)

    def test_jacobian(self):
        """Test Jacobian computation."""
        f, jac_f, x, p, x_dot, A = create_pendulum_dynamics(self.backend)

        x0 = [0.0, 0.0]  # At equilibrium
        params = [9.81, 1.0, 0.1]

        J = jac_f(x0, params)
        J_arr = np.array(J)

        # At θ=0: A = [[0, 1], [-g/L * cos(0), -b]] = [[0, 1], [-9.81, -0.1]]
        expected_A = np.array([[0, 1], [-9.81, -0.1]])
        np.testing.assert_allclose(J_arr, expected_A, rtol=1e-6)

    def test_integrator(self):
        """Test CasADi-specific integrator."""
        import casadi as ca

        f, jac_f, x, p, x_dot, A = create_pendulum_dynamics(self.backend)

        # Create integrator
        dae = {"x": x, "p": p, "ode": x_dot}
        integrator = ca.integrator("sim", "rk", dae, 0, 0.1)

        x0 = [0.5, 0.0]
        params = [9.81, 1.0, 0.1]

        result = integrator(x0=x0, p=params)
        x_next = np.array(result["xf"]).flatten()

        # After 0.1s, pendulum should have moved
        self.assertNotAlmostEqual(x_next[0], x0[0], places=3)
        self.assertNotAlmostEqual(x_next[1], x0[1], places=3)


@beartype
@unittest.skipUnless(SYMPY_AVAILABLE, "SymPy not available")
class TestSymPyPendulum(ProfiledTestCase):
    """Test pendulum model with SymPy backend."""

    def setUp(self):
        super().setUp()
        self.backend = SymPyBackend()

    def test_create_model(self):
        """Test model creation."""
        f, jac_f, x, p, x_dot, A = create_pendulum_dynamics(self.backend)

        # SymPy doesn't create CasADi-style functions
        self.assertEqual(x.shape, (2, 1))
        self.assertEqual(p.shape, (3, 1))
        self.assertEqual(x_dot.shape, (2, 1))
        self.assertEqual(A.shape, (2, 2))

    def test_symbolic_jacobian(self):
        """Test symbolic Jacobian structure."""
        import sympy

        f, jac_f, x, p, x_dot, A = create_pendulum_dynamics(self.backend)

        # Check Jacobian structure
        # A[0,0] should be 0 (∂θ̇/∂θ = 0)
        # A[0,1] should be 1 (∂θ̇/∂θ̇ = 1)
        self.assertEqual(A[0, 0], 0)
        self.assertEqual(A[0, 1], 1)

        # A[1,0] should contain cos(theta)
        self.assertTrue("cos" in str(A[1, 0]).lower())

    def test_simplification(self):
        """Test SymPy simplification."""
        theta = self.backend.sym("theta")

        # sin²(θ) + cos²(θ) = 1
        expr = self.backend.sin(theta) ** 2 + self.backend.cos(theta) ** 2
        simplified = self.backend.simplify(expr)

        import sympy

        self.assertEqual(simplified, sympy.Integer(1))

    def test_latex_output(self):
        """Test LaTeX generation."""
        theta = self.backend.sym("theta")
        g = self.backend.sym("g")
        L = self.backend.sym("L")

        expr = -g / L * self.backend.sin(theta)
        latex = self.backend.latex(expr)

        self.assertIsInstance(latex, str)
        self.assertIn("sin", latex.lower())


@beartype
@unittest.skipUnless(JAX_AVAILABLE, "JAX not available")
class TestJAXPendulum(ProfiledTestCase):
    """Test pendulum model with JAX backend."""

    def setUp(self):
        super().setUp()
        self.backend = JAXBackend()

    def test_numerical_dynamics(self):
        """Test JAX numerical dynamics."""
        import jax.numpy as jnp

        def pendulum_dynamics(state, params):
            theta, theta_dot = state[0], state[1]
            g, L, b = params[0], params[1], params[2]
            theta_ddot = -g / L * self.backend.sin(jnp.array([theta]))[0] - b * theta_dot
            return jnp.array([theta_dot, theta_ddot])

        x0 = jnp.array([0.5, 0.0])
        params = jnp.array([9.81, 1.0, 0.1])

        result = pendulum_dynamics(x0, params)

        expected_theta_ddot = -9.81 * np.sin(0.5) - 0.1 * 0.0
        np.testing.assert_allclose(result[0], 0.0, rtol=1e-6)
        np.testing.assert_allclose(result[1], expected_theta_ddot, rtol=1e-6)

    def test_jit_compilation(self):
        """Test JIT compilation."""
        import jax.numpy as jnp

        def pendulum_dynamics(state):
            theta, theta_dot = state[0], state[1]
            g, L, b = 9.81, 1.0, 0.1
            theta_ddot = -g / L * jnp.sin(theta) - b * theta_dot
            return jnp.array([theta_dot, theta_ddot])

        jit_dynamics = self.backend.jit(pendulum_dynamics)

        x0 = jnp.array([0.5, 0.0])
        result = jit_dynamics(x0)

        expected_theta_ddot = -9.81 * np.sin(0.5)
        np.testing.assert_allclose(result[1], expected_theta_ddot, rtol=1e-6)

    def test_autodiff_jacobian(self):
        """Test automatic differentiation for Jacobian."""
        import jax.numpy as jnp

        def pendulum_dynamics(state):
            theta, theta_dot = state[0], state[1]
            g, L, b = 9.81, 1.0, 0.1
            theta_ddot = -g / L * jnp.sin(theta) - b * theta_dot
            return jnp.array([theta_dot, theta_ddot])

        jac_f = self.backend.jacobian_func(pendulum_dynamics)

        x0 = jnp.array([0.0, 0.0])
        J = jac_f(x0)

        # At θ=0: A = [[0, 1], [-g/L * cos(0), -b]] = [[0, 1], [-9.81, -0.1]]
        expected_A = np.array([[0, 1], [-9.81, -0.1]])
        np.testing.assert_allclose(np.array(J), expected_A, rtol=1e-5)

    def test_vmap_batch(self):
        """Test vectorized mapping."""
        import jax.numpy as jnp

        def pendulum_accel(theta):
            g, L = 9.81, 1.0
            return -g / L * jnp.sin(theta)

        vmapped = self.backend.vmap(pendulum_accel)

        thetas = jnp.array([0.0, 0.5, 1.0, 1.5])
        results = vmapped(thetas)

        expected = -9.81 * np.sin(np.array([0.0, 0.5, 1.0, 1.5]))
        np.testing.assert_allclose(np.array(results), expected, rtol=1e-6)


@beartype
class TestBackendConsistencyPendulum(ProfiledTestCase):
    """Test that different backends give consistent results."""

    def test_casadi_vs_numpy(self):
        """Compare CasADi and NumPy results."""
        casadi = CasADiBackend()
        numpy = NumPyBackend()

        # CasADi symbolic evaluation
        f_ca, _, _, _, _, _ = create_pendulum_dynamics(casadi)
        x0 = [0.5, 0.1]
        params = [9.81, 1.0, 0.1]
        ca_result = np.array(f_ca(x0, params)).flatten()

        # NumPy direct evaluation
        theta, theta_dot = x0
        g, L, b = params
        theta_ddot = -g / L * numpy.sin(np.array([theta]))[0] - b * theta_dot
        np_result = np.array([theta_dot, theta_ddot])

        np.testing.assert_allclose(ca_result, np_result, rtol=1e-10)

    @unittest.skipUnless(SYMPY_AVAILABLE, "SymPy not available")
    def test_casadi_vs_sympy(self):
        """Compare CasADi and SymPy Jacobians."""
        casadi = CasADiBackend()
        sympy_be = SymPyBackend()

        # CasADi Jacobian
        _, jac_ca, _, _, _, _ = create_pendulum_dynamics(casadi)
        x0 = [0.0, 0.0]
        params = [9.81, 1.0, 0.1]
        ca_jac = np.array(jac_ca(x0, params))

        # SymPy Jacobian (evaluate numerically)
        _, _, x_sp, p_sp, _, A_sp = create_pendulum_dynamics(sympy_be)
        import sympy

        # Get SymPy symbols
        theta_sym = sympy_be.sym("theta")
        theta_dot_sym = sympy_be.sym("theta_dot")
        g_sym = sympy_be.sym("g")
        L_sym = sympy_be.sym("L")
        b_sym = sympy_be.sym("b")

        # Substitute values
        subs = {theta_sym: 0.0, theta_dot_sym: 0.0, g_sym: 9.81, L_sym: 1.0, b_sym: 0.1}
        sp_jac = np.array(A_sp.subs(subs)).astype(float)

        np.testing.assert_allclose(ca_jac, sp_jac, rtol=1e-10)

    @unittest.skipUnless(JAX_AVAILABLE, "JAX not available")
    def test_casadi_vs_jax(self):
        """Compare CasADi and JAX Jacobians."""
        import jax.numpy as jnp

        casadi = CasADiBackend()
        jax_be = JAXBackend()

        # CasADi Jacobian
        _, jac_ca, _, _, _, _ = create_pendulum_dynamics(casadi)
        x0 = [0.0, 0.0]
        params = [9.81, 1.0, 0.1]
        ca_jac = np.array(jac_ca(x0, params))

        # JAX autodiff Jacobian
        def pendulum_dynamics(state):
            theta, theta_dot = state[0], state[1]
            g, L, b = 9.81, 1.0, 0.1
            theta_ddot = -g / L * jnp.sin(theta) - b * theta_dot
            return jnp.array([theta_dot, theta_ddot])

        jax_jac_func = jax_be.jacobian_func(pendulum_dynamics)
        jax_jac = np.array(jax_jac_func(jnp.array([0.0, 0.0])))

        np.testing.assert_allclose(ca_jac, jax_jac, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
