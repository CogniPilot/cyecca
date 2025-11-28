"""Tests for backend abstraction layer."""

import unittest

import numpy as np
from beartype import beartype

import cyecca.sym as cy
from cyecca.backends import (
    JAX_AVAILABLE,
    SYMPY_AVAILABLE,
    CasADiBackend,
    NumPyBackend,
    SymbolicBackend,
    get_backend,
    list_backends,
    set_default_backend,
)

if SYMPY_AVAILABLE:
    from cyecca.backends import SymPyBackend

if JAX_AVAILABLE:
    from cyecca.backends import JAXBackend

from .common import ProfiledTestCase


@beartype
class TestBackendRegistry(ProfiledTestCase):
    """Test backend registry functionality."""

    def test_list_backends(self):
        """Test listing available backends."""
        backends = cy.backends()
        self.assertIn("casadi", backends)
        self.assertIn("numpy", backends)
        if SYMPY_AVAILABLE:
            self.assertIn("sympy", backends)
        if JAX_AVAILABLE:
            self.assertIn("jax", backends)

    def test_get_default_backend(self):
        """Test getting the default backend."""
        self.assertEqual(cy.backend_name(), "casadi")

    def test_set_backend(self):
        """Test setting the backend."""
        original = cy.backend_name()

        cy.set_backend("numpy")
        self.assertEqual(cy.backend_name(), "numpy")

        # Reset to casadi
        cy.set_backend("casadi")
        self.assertEqual(cy.backend_name(), "casadi")


@beartype
class TestCyNamespace(ProfiledTestCase):
    """Test cy.* namespace functionality (default CasADi backend)."""

    def setUp(self):
        super().setUp()
        cy.set_backend("casadi")

    def test_sym(self):
        """Test symbolic variable creation."""
        x = cy.sym("x")
        self.assertEqual(x.shape, (1, 1))

        v = cy.sym("v", 3)
        self.assertEqual(v.shape, (3, 1))

    def test_zeros_ones_eye(self):
        """Test matrix construction."""
        z = cy.zeros(3, 2)
        self.assertEqual(z.shape, (3, 2))

        o = cy.ones(2, 3)
        self.assertEqual(o.shape, (2, 3))

        I = cy.eye(4)
        self.assertEqual(I.shape, (4, 4))

    def test_vertcat_horzcat(self):
        """Test concatenation operations."""
        a = cy.sym("a", 2)
        b = cy.sym("b", 3)
        v = cy.vertcat(a, b)
        self.assertEqual(v.shape, (5, 1))

        c = cy.sym("c", 2)
        d = cy.sym("d", 2)
        h = cy.horzcat(c, d)
        self.assertEqual(h.shape, (2, 2))

    def test_trig_functions(self):
        """Test trigonometric functions."""
        x = cy.sym("x")

        self.assertIsNotNone(cy.sin(x))
        self.assertIsNotNone(cy.cos(x))
        self.assertIsNotNone(cy.tan(x))
        self.assertIsNotNone(cy.asin(x))
        self.assertIsNotNone(cy.acos(x))

    def test_math_functions(self):
        """Test mathematical functions."""
        x = cy.sym("x")

        self.assertIsNotNone(cy.exp(x))
        self.assertIsNotNone(cy.log(x))
        self.assertIsNotNone(cy.sqrt(x))
        self.assertIsNotNone(cy.fabs(x))

    def test_jacobian(self):
        """Test Jacobian computation."""
        x = cy.sym("x", 3)
        y = cy.vertcat(x[0] ** 2, x[1] * x[2], cy.sin(x[0]))

        J = cy.jacobian(y, x)
        self.assertEqual(J.shape, (3, 3))

    def test_function_creation(self):
        """Test function creation and evaluation."""
        x = cy.sym("x", 2)
        y = cy.vertcat(x[0] ** 2 + x[1], cy.sin(x[0]))

        f = cy.function("f", [x], [y])
        result = f([1.0, 2.0])

        result_arr = np.array(result)
        expected = np.array([[3.0], [np.sin(1.0)]])
        np.testing.assert_allclose(result_arr, expected, rtol=1e-10)

    def test_numeric_evaluation(self):
        """Test numerical evaluation via function."""
        x = cy.sym("x")
        expr = cy.sin(x) + x**2

        f = cy.function("f", [x], [expr])
        result = f(1.0)
        expected = np.sin(1.0) + 1.0
        self.assertAlmostEqual(float(result[0]), expected, places=10)


@beartype
class TestCasADiBackend(ProfiledTestCase):
    """Test CasADi backend directly."""

    def setUp(self):
        super().setUp()
        self.backend = CasADiBackend()

    def test_name(self):
        """Test backend name."""
        self.assertEqual(self.backend.name, "casadi")

    def test_sym(self):
        """Test symbolic variable creation."""
        x = self.backend.sym("x")
        self.assertEqual(x.shape, (1, 1))

        v = self.backend.sym("v", 3)
        self.assertEqual(v.shape, (3, 1))

    def test_linear_algebra(self):
        """Test linear algebra operations."""
        A = self.backend.sym("A", 3)  # 3x1 vector
        b = self.backend.sym("b", 3)

        # Test dot product
        result = self.backend.dot(A, b)
        self.assertEqual(result.shape, (1, 1))

        # Test transpose
        At = self.backend.transpose(A)
        self.assertEqual(At.shape, (1, 3))


@beartype
class TestNumPyBackend(ProfiledTestCase):
    """Test NumPy backend functionality."""

    def setUp(self):
        super().setUp()
        self.backend = NumPyBackend()

    def test_name(self):
        """Test backend name."""
        self.assertEqual(self.backend.name, "numpy")

    def test_zeros_ones_eye(self):
        """Test matrix construction."""
        z = self.backend.zeros(3, 2)
        self.assertEqual(z.shape, (3, 2))
        np.testing.assert_array_equal(z, np.zeros((3, 2)))

        o = self.backend.ones(2, 3)
        self.assertEqual(o.shape, (2, 3))
        np.testing.assert_array_equal(o, np.ones((2, 3)))

        I = self.backend.eye(4)
        self.assertEqual(I.shape, (4, 4))
        np.testing.assert_array_equal(I, np.eye(4))

    def test_vertcat_horzcat(self):
        """Test concatenation operations."""
        a = np.array([1, 2]).reshape(-1, 1)
        b = np.array([3, 4, 5]).reshape(-1, 1)
        v = self.backend.vertcat(a, b)
        self.assertEqual(v.size, 5)  # Check element count, shape may vary

        c = np.array([1, 2]).reshape(-1, 1)
        d = np.array([3, 4]).reshape(-1, 1)
        h = self.backend.horzcat(c, d)
        self.assertEqual(h.shape, (2, 2))

    def test_trig_functions(self):
        """Test trigonometric functions."""
        x = np.array([0.5])

        np.testing.assert_allclose(self.backend.sin(x), np.sin(x))
        np.testing.assert_allclose(self.backend.cos(x), np.cos(x))
        np.testing.assert_allclose(self.backend.tan(x), np.tan(x))

    def test_math_functions(self):
        """Test mathematical functions."""
        x = np.array([2.0])

        np.testing.assert_allclose(self.backend.exp(x), np.exp(x))
        np.testing.assert_allclose(self.backend.log(x), np.log(x))
        np.testing.assert_allclose(self.backend.sqrt(x), np.sqrt(x))

    def test_linear_algebra(self):
        """Test linear algebra operations."""
        A = np.array([[1, 2], [3, 4]])
        b = np.array([[1], [2]])

        result = self.backend.mtimes(A, b)
        expected = A @ b
        np.testing.assert_allclose(result, expected)

        At = self.backend.transpose(A)
        np.testing.assert_array_equal(At, A.T)

    def test_no_symbolic_operations(self):
        """Test that symbolic operations raise errors."""
        # NumPy sym() doesn't raise error, it creates placeholder array
        # But jacobian should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.backend.jacobian(np.array([1]), np.array([1]))


@beartype
@unittest.skipUnless(SYMPY_AVAILABLE, "SymPy not available")
class TestSymPyBackend(ProfiledTestCase):
    """Test SymPy backend functionality."""

    def setUp(self):
        super().setUp()
        self.backend = SymPyBackend()

    def test_name(self):
        """Test backend name."""
        self.assertEqual(self.backend.name, "sympy")

    def test_sym(self):
        """Test symbolic variable creation."""
        import sympy

        x = self.backend.sym("x")
        self.assertIsInstance(x, sympy.Symbol)

        v = self.backend.sym("v", 3)
        self.assertIsInstance(v, sympy.Matrix)
        self.assertEqual(v.shape, (3, 1))

    def test_zeros_ones_eye(self):
        """Test matrix construction."""
        import sympy

        z = self.backend.zeros(3, 2)
        self.assertIsInstance(z, sympy.Matrix)
        self.assertEqual(z.shape, (3, 2))

        o = self.backend.ones(2, 3)
        self.assertEqual(o.shape, (2, 3))

        I = self.backend.eye(4)
        self.assertEqual(I.shape, (4, 4))

    def test_trig_functions(self):
        """Test trigonometric functions."""
        import sympy

        x = self.backend.sym("x")

        self.assertIsInstance(self.backend.sin(x), sympy.Basic)
        self.assertIsInstance(self.backend.cos(x), sympy.Basic)
        self.assertIsInstance(self.backend.tan(x), sympy.Basic)

    def test_jacobian(self):
        """Test Jacobian computation."""
        x = self.backend.sym("x", 3)
        y = self.backend.vertcat(x[0] ** 2, x[1] * x[2], self.backend.sin(x[0]))

        J = self.backend.jacobian(y, x)
        self.assertEqual(J.shape, (3, 3))

    def test_latex_output(self):
        """Test LaTeX output capability."""
        x = self.backend.sym("x")
        expr = self.backend.sin(x) ** 2 + self.backend.cos(x) ** 2

        latex = self.backend.latex(expr)
        self.assertIsInstance(latex, str)
        self.assertIn("sin", latex.lower())

    def test_simplify(self):
        """Test expression simplification."""
        import sympy

        x = self.backend.sym("x")
        # sin^2(x) + cos^2(x) should simplify to 1
        expr = self.backend.sin(x) ** 2 + self.backend.cos(x) ** 2
        simplified = self.backend.simplify(expr)

        # Check it simplifies to 1
        self.assertEqual(simplified, sympy.Integer(1))


@beartype
@unittest.skipUnless(JAX_AVAILABLE, "JAX not available")
class TestJAXBackend(ProfiledTestCase):
    """Test JAX backend functionality."""

    def setUp(self):
        super().setUp()
        self.backend = JAXBackend()

    def test_name(self):
        """Test backend name."""
        self.assertEqual(self.backend.name, "jax")

    def test_zeros_ones_eye(self):
        """Test matrix construction."""
        import jax.numpy as jnp

        z = self.backend.zeros(3, 2)
        self.assertEqual(z.shape, (3, 2))
        np.testing.assert_array_equal(np.array(z), np.zeros((3, 2)))

        o = self.backend.ones(2, 3)
        self.assertEqual(o.shape, (2, 3))
        np.testing.assert_array_equal(np.array(o), np.ones((2, 3)))

        I = self.backend.eye(4)
        self.assertEqual(I.shape, (4, 4))
        np.testing.assert_array_equal(np.array(I), np.eye(4))

    def test_vertcat_horzcat(self):
        """Test concatenation operations."""
        import jax.numpy as jnp

        a = jnp.array([1.0, 2.0]).reshape(-1, 1)
        b = jnp.array([3.0, 4.0, 5.0]).reshape(-1, 1)
        v = self.backend.vertcat(a, b)
        self.assertEqual(v.size, 5)  # Check element count, shape may vary

        c = jnp.array([1.0, 2.0]).reshape(-1, 1)
        d = jnp.array([3.0, 4.0]).reshape(-1, 1)
        h = self.backend.horzcat(c, d)
        self.assertEqual(h.shape, (2, 2))

    def test_trig_functions(self):
        """Test trigonometric functions."""
        import jax.numpy as jnp

        x = jnp.array([0.5])

        np.testing.assert_allclose(np.array(self.backend.sin(x)), np.sin(0.5), rtol=1e-6)
        np.testing.assert_allclose(np.array(self.backend.cos(x)), np.cos(0.5), rtol=1e-6)
        np.testing.assert_allclose(np.array(self.backend.tan(x)), np.tan(0.5), rtol=1e-6)

    def test_math_functions(self):
        """Test mathematical functions."""
        import jax.numpy as jnp

        x = jnp.array([2.0])

        np.testing.assert_allclose(np.array(self.backend.exp(x)), np.exp(2.0), rtol=1e-6)
        np.testing.assert_allclose(np.array(self.backend.log(x)), np.log(2.0), rtol=1e-6)
        np.testing.assert_allclose(np.array(self.backend.sqrt(x)), np.sqrt(2.0), rtol=1e-6)

    def test_linear_algebra(self):
        """Test linear algebra operations."""
        import jax.numpy as jnp

        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[1.0], [2.0]])

        result = self.backend.mtimes(A, b)
        expected = np.array([[5.0], [11.0]])
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-6)

        At = self.backend.transpose(A)
        np.testing.assert_array_equal(np.array(At), np.array(A).T)

    def test_grad_func(self):
        """Test gradient function creation."""
        import jax.numpy as jnp

        def f(x):
            return jnp.sum(x**2)

        grad_f = self.backend.grad_func(f)

        x = jnp.array([1.0, 2.0, 3.0])
        grad_result = grad_f(x)

        # Gradient of sum(x^2) is 2*x
        expected = 2.0 * np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.array(grad_result), expected, rtol=1e-6)

    def test_jacobian_func(self):
        """Test Jacobian function creation."""
        import jax.numpy as jnp

        def f(x):
            return jnp.array([x[0] ** 2, x[0] * x[1], jnp.sin(x[1])])

        jac_f = self.backend.jacobian_func(f)

        x = jnp.array([2.0, 1.0])
        jac_result = jac_f(x)

        # Check shape
        self.assertEqual(jac_result.shape, (3, 2))

    def test_jit(self):
        """Test JIT compilation."""
        import jax.numpy as jnp

        def slow_func(x):
            return jnp.sum(x**2)

        jit_func = self.backend.jit(slow_func)

        x = jnp.array([1.0, 2.0, 3.0])
        result = jit_func(x)
        expected = 14.0

        np.testing.assert_allclose(float(result), expected, rtol=1e-6)

    def test_vmap(self):
        """Test vectorized mapping."""
        import jax.numpy as jnp

        def single_func(x):
            return jnp.sum(x**2)

        vmapped = self.backend.vmap(single_func)

        # Batch of inputs
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        results = vmapped(batch)

        expected = np.array([5.0, 25.0, 61.0])
        np.testing.assert_allclose(np.array(results), expected, rtol=1e-6)


@beartype
class TestBackendConsistency(ProfiledTestCase):
    """Test consistency across backends for common operations."""

    def tearDown(self):
        """Reset to default backend after each test."""
        super().tearDown()
        cy.set_backend("casadi")

    def test_trig_consistency_cy(self):
        """Test trig functions via cy namespace with different backends."""
        test_val = 0.7

        # CasADi backend
        cy.set_backend("casadi")
        x_ca = cy.sym("x")
        ca_expr = cy.sin(x_ca) + cy.cos(x_ca) ** 2
        f_ca = cy.function("f", [x_ca], [ca_expr])
        ca_result = float(f_ca(test_val)[0])

        # NumPy backend (direct evaluation)
        cy.set_backend("numpy")
        np_result = float(np.sin(test_val) + np.cos(test_val) ** 2)

        np.testing.assert_allclose(ca_result, np_result, rtol=1e-10)

    @unittest.skipUnless(SYMPY_AVAILABLE, "SymPy not available")
    def test_sympy_casadi_consistency(self):
        """Test consistency between SymPy and CasADi via cy namespace."""
        import sympy

        test_val = 0.7

        # CasADi backend
        cy.set_backend("casadi")
        x_ca = cy.sym("x")
        ca_expr = cy.sin(x_ca) + cy.cos(x_ca) ** 2
        f = cy.function("f", [x_ca], [ca_expr])
        ca_result = float(f(test_val)[0])

        # SymPy backend
        cy.set_backend("sympy")
        x_sp = cy.sym("x")
        sp_expr = cy.sin(x_sp) + cy.cos(x_sp) ** 2
        sp_result = float(sp_expr.subs(x_sp, test_val))

        np.testing.assert_allclose(ca_result, sp_result, rtol=1e-10)

    @unittest.skipUnless(JAX_AVAILABLE, "JAX not available")
    def test_jax_numpy_consistency(self):
        """Test consistency between JAX and NumPy backends."""
        import jax.numpy as jnp

        jax_backend = JAXBackend()
        numpy_backend = NumPyBackend()

        test_array = np.array([1.0, 2.0, 3.0])
        jax_array = jnp.array(test_array)

        # Test various operations
        np.testing.assert_allclose(np.array(jax_backend.sin(jax_array)), numpy_backend.sin(test_array), rtol=1e-6)

        np.testing.assert_allclose(np.array(jax_backend.exp(jax_array)), numpy_backend.exp(test_array), rtol=1e-6)

        np.testing.assert_allclose(np.array(jax_backend.sqrt(jax_array)), numpy_backend.sqrt(test_array), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
