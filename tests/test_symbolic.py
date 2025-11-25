"""Tests for symbolic conversions between SymPy and CasADi."""

from cyecca.symbolic import sympy_to_casadi, casadi_to_sympy
from beartype import beartype
import sympy
import casadi as ca
import numpy as np
from .common import ProfiledTestCase, SX_close


@beartype
class Test_Symbolic(ProfiledTestCase):
    """Test symbolic expression conversions between SymPy and CasADi."""

    def setUp(self):
        super().setUp()

    def test_sympy_to_casadi(self):
        """Test basic SymPy to CasADi conversion."""
        x = sympy.symbols("x")
        y = sympy.sin(x) + 2
        result, symbols = sympy_to_casadi(y)
        assert isinstance(result, ca.SX)
        assert 'x' in symbols

    def test_casadi_to_sympy(self):
        """Test basic CasADi to SymPy conversion."""
        x = ca.SX.sym("x")
        y = ca.tan(x) + ca.cos(x) * ca.sin(x) + 2
        result = casadi_to_sympy(y)
        assert isinstance(result, sympy.Basic)

    def test_sympy_matrix_to_casadi_vector(self):
        """Test SymPy 6x1 Matrix to CasADi SX vector conversion."""
        # Create symbolic variables
        x, y, z = sympy.symbols("x y z")

        # Create a 6x1 SymPy matrix with complex expressions
        sympy_matrix = sympy.Matrix([
            sympy.sin(x) + sympy.cos(y),
            x**2 + y**2,
            sympy.exp(z),
            sympy.sqrt(x**2 + y**2 + z**2),
            sympy.atan2(y, x),
            x * y * z
        ])

        # Convert to CasADi
        casadi_vec, symbols = sympy_to_casadi(sympy_matrix)

        # Verify it's a CasADi SX vector with correct shape
        assert isinstance(casadi_vec, ca.SX)
        assert casadi_vec.shape == (6, 1)

        # Test with numeric values
        x_val, y_val, z_val = 1.5, 2.3, 0.7

        # Create CasADi function for evaluation
        casadi_func = ca.Function("f",
                                  [symbols['x'], symbols['y'], symbols['z']],
                                  [casadi_vec])
        casadi_result = casadi_func(x_val, y_val, z_val)

        # Compute expected result with SymPy
        expected = sympy_matrix.subs({x: x_val, y: y_val, z: z_val})
        expected_numeric = np.array([float(val) for val in expected])

        # Compare results
        assert np.allclose(np.array(casadi_result).flatten(),
                          expected_numeric.flatten(),
                          rtol=1e-10)

    def test_casadi_vector_to_sympy_matrix(self):
        """Test CasADi SX vector to SymPy Matrix conversion and back."""
        # Create CasADi symbolic variables
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        z = ca.SX.sym("z")

        # Create a 6x1 CasADi vector with complex expressions
        casadi_vec = ca.vertcat(
            ca.sin(x) + ca.cos(y),
            x**2 + y**2,
            ca.exp(z),
            ca.sqrt(x**2 + y**2 + z**2),
            ca.atan2(y, x),
            x * y * z
        )

        # Convert to SymPy
        sympy_result = casadi_to_sympy(casadi_vec)

        # Verify it's a SymPy Matrix with correct shape
        assert isinstance(sympy_result, sympy.Matrix)
        assert sympy_result.shape == (6, 1)

        # Convert back to CasADi
        casadi_roundtrip, symbols_rt = sympy_to_casadi(sympy_result)

        # Verify shape is preserved
        assert casadi_roundtrip.shape == casadi_vec.shape

        # Test with numeric values
        x_val, y_val, z_val = 1.5, 2.3, 0.7

        # Evaluate both expressions
        func_original = ca.Function("f_orig", [x, y, z], [casadi_vec])
        func_roundtrip = ca.Function("f_round",
                                     [symbols_rt['x'], symbols_rt['y'], symbols_rt['z']],
                                     [casadi_roundtrip])

        result_original = func_original(x_val, y_val, z_val)
        result_roundtrip = func_roundtrip(x_val, y_val, z_val)

        # Should be very close after roundtrip
        assert np.allclose(np.array(result_original),
                          np.array(result_roundtrip),
                          rtol=1e-10)

    def test_matrix_operations_conversion(self):
        """Test conversion of matrix operations."""
        # SymPy matrix operations
        x = sympy.symbols("x")
        A = sympy.Matrix([[sympy.sin(x), sympy.cos(x)],
                         [sympy.cos(x), -sympy.sin(x)]])
        v = sympy.Matrix([x, x**2])

        # Matrix-vector multiplication
        result_sympy = A * v

        # Convert to CasADi
        result_casadi, symbols = sympy_to_casadi(result_sympy)

        # Verify result
        assert isinstance(result_casadi, ca.SX)
        assert result_casadi.shape == (2, 1)

        # Test numeric evaluation
        x_val = 0.5
        func = ca.Function("f", [symbols['x']], [result_casadi])
        numeric_result = func(x_val)

        # Compute expected
        expected = result_sympy.subs(x, x_val)
        expected_numeric = np.array([float(expected[0]), float(expected[1])])

        assert np.allclose(np.array(numeric_result).flatten(),
                          expected_numeric,
                          rtol=1e-10)
