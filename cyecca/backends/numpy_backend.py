"""NumPy backend implementation.

This module provides a NumPy implementation of the SymbolicBackend interface.
NumPy is useful for pure numerical computation without symbolic overhead.

Use this backend for:
- Fast numerical evaluation (no symbolic overhead)
- When you don't need symbolic differentiation
- Prototyping with numeric values
- Comparing symbolic vs numeric results

NOT recommended for:
- Symbolic manipulation
- Automatic differentiation (use CasADi or JAX)
- Symbolic analysis
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .base import SymbolicBackend


class NumpyArray:
    """Wrapper to provide .sym() method for NumPy arrays.

    This makes NumPy arrays compatible with the backend interface.
    Note: These are NOT symbolic - they create numeric arrays.
    """

    @classmethod
    def sym(cls, name: str, n: int = 1):
        """Create a named array of zeros (placeholder for symbolic).

        Args:
            name: Variable name (stored as metadata, not used)
            n: Number of elements

        Returns:
            NumPy array of zeros
        """
        arr = np.zeros(n)
        # Store name as attribute for debugging
        arr = arr.view(NamedArray)
        arr._name = name
        return arr

    @classmethod
    def zeros(cls, rows: int, cols: int = 1):
        """Create zero matrix."""
        if cols == 1:
            return np.zeros(rows)
        return np.zeros((rows, cols))

    @classmethod
    def ones(cls, rows: int, cols: int = 1):
        """Create ones matrix."""
        if cols == 1:
            return np.ones(rows)
        return np.ones((rows, cols))

    @classmethod
    def eye(cls, n: int):
        """Create identity matrix."""
        return np.eye(n)


class NamedArray(np.ndarray):
    """NumPy array with a name attribute for debugging."""

    _name: str = ""

    def __repr__(self):
        if hasattr(self, "_name") and self._name:
            return f"NamedArray('{self._name}', {np.ndarray.__repr__(self)})"
        return np.ndarray.__repr__(self)


class NumpyFunction:
    """Wrapper to make NumPy functions behave like CasADi Function.

    Since NumPy doesn't have symbolic expressions, we store a callable
    that operates on numeric arrays.
    """

    def __init__(
        self,
        name: str,
        func: callable,
        n_inputs: int,
        n_outputs: int,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ):
        """Create a NumPy function wrapper.

        Args:
            name: Function name
            func: Callable that takes inputs and returns outputs
            n_inputs: Number of inputs
            n_outputs: Number of outputs
            input_names: Optional names for inputs
            output_names: Optional names for outputs
        """
        self.name = name
        self._func = func
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self.input_names = input_names or [f"i{i}" for i in range(n_inputs)]
        self.output_names = output_names or [f"o{i}" for i in range(n_outputs)]

    def __call__(self, *args):
        """Evaluate the function."""
        return self._func(*args)

    def __repr__(self):
        return f"NumpyFunction('{self.name}')"

    def size_in(self, i: int) -> int:
        """Get size of input i."""
        return -1  # Unknown without actual inputs

    def size_out(self, i: int) -> Tuple[int, int]:
        """Get size of output i."""
        return (-1, -1)  # Unknown without actual inputs


class NumPyBackend(SymbolicBackend):
    """NumPy implementation of the symbolic backend.

    This backend provides pure numerical operations using NumPy.
    It does NOT support symbolic computation - all operations are numeric.

    Best suited for:
    - Fast numerical evaluation
    - When symbolic overhead is not needed
    - Prototyping and testing

    Limitations:
    - No symbolic variables (sym() creates numeric placeholders)
    - No automatic differentiation (jacobian returns numeric approx)
    - No symbolic simplification
    """

    @property
    def name(self) -> str:
        return "numpy"

    # ========== Symbol Types ==========

    @property
    def SX(self) -> type:
        """NumPy array type (no symbolic support)."""
        return NumpyArray

    @property
    def MX(self) -> type:
        """NumPy array type (same as SX for NumPy)."""
        return NumpyArray

    @property
    def DM(self) -> type:
        """NumPy array type."""
        return NumpyArray

    # ========== Symbol Creation ==========

    def sym(self, name: str, shape: int = 1, sym_type=None) -> np.ndarray:
        """Create a named numeric array (NOT symbolic).

        Note: NumPy doesn't support symbolic computation.
        This creates a zero array as a placeholder.
        """
        arr = np.zeros(shape)
        arr = arr.view(NamedArray)
        arr._name = name
        return arr

    def zeros(self, rows: int, cols: int = 1, sym_type=None) -> np.ndarray:
        if cols == 1:
            return np.zeros(rows)
        return np.zeros((rows, cols))

    def ones(self, rows: int, cols: int = 1, sym_type=None) -> np.ndarray:
        if cols == 1:
            return np.ones(rows)
        return np.ones((rows, cols))

    def eye(self, n: int, sym_type=None) -> np.ndarray:
        return np.eye(n)

    # ========== Matrix Operations ==========

    def vertcat(self, *args) -> np.ndarray:
        """Vertical concatenation."""
        if not args:
            return np.array([])

        parts = []
        for arg in args:
            arr = np.atleast_1d(np.asarray(arg))
            parts.append(arr.flatten())

        return np.concatenate(parts)

    def horzcat(self, *args) -> np.ndarray:
        """Horizontal concatenation."""
        if not args:
            return np.array([[]])

        parts = []
        for arg in args:
            arr = np.atleast_2d(np.asarray(arg))
            if arr.shape[0] == 1:
                arr = arr.T
            parts.append(arr)

        return np.hstack(parts)

    def reshape(self, x: Any, shape: Tuple[int, int]) -> np.ndarray:
        return np.reshape(x, shape)

    def transpose(self, x: Any) -> np.ndarray:
        return np.transpose(x)

    # ========== Differentiation ==========

    def jacobian(self, expr: Any, x: Any) -> np.ndarray:
        """Compute Jacobian numerically using finite differences.

        Note: NumPy doesn't support symbolic differentiation.
        This uses numerical finite differences.
        """
        raise NotImplementedError(
            "NumPy backend does not support symbolic Jacobian. " "Use CasADi or SymPy backend for differentiation."
        )

    def gradient(self, expr: Any, x: Any) -> np.ndarray:
        """Compute gradient numerically."""
        raise NotImplementedError(
            "NumPy backend does not support symbolic gradient. " "Use CasADi or SymPy backend for differentiation."
        )

    def hessian(self, expr: Any, x: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Hessian numerically."""
        raise NotImplementedError(
            "NumPy backend does not support symbolic Hessian. " "Use CasADi or SymPy backend for differentiation."
        )

    def jtimes(self, expr: Any, x: Any, v: Any) -> np.ndarray:
        """Jacobian-times-vector product."""
        raise NotImplementedError(
            "NumPy backend does not support symbolic jtimes. " "Use CasADi or SymPy backend for differentiation."
        )

    # ========== Function Creation ==========

    def function(
        self,
        name: str,
        inputs: List[Any],
        outputs: List[Any],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> NumpyFunction:
        """Create a function (limited support for NumPy).

        Note: For NumPy backend, outputs should be numeric arrays,
        not symbolic expressions. This is mainly for API compatibility.
        """

        # For numeric outputs, just return them directly
        def func(*args):
            # If outputs are already numeric, return them
            if len(outputs) == 1:
                return np.asarray(outputs[0])
            return [np.asarray(o) for o in outputs]

        return NumpyFunction(name, func, len(inputs), len(outputs), input_names, output_names)

    # ========== Math Functions ==========

    def sin(self, x: Any) -> np.ndarray:
        return np.sin(x)

    def cos(self, x: Any) -> np.ndarray:
        return np.cos(x)

    def tan(self, x: Any) -> np.ndarray:
        return np.tan(x)

    def exp(self, x: Any) -> np.ndarray:
        return np.exp(x)

    def log(self, x: Any) -> np.ndarray:
        return np.log(x)

    def sqrt(self, x: Any) -> np.ndarray:
        return np.sqrt(x)

    def power(self, x: Any, y: Any) -> np.ndarray:
        return np.power(x, y)

    def abs(self, x: Any) -> np.ndarray:
        return np.abs(x)

    def atan2(self, y: Any, x: Any) -> np.ndarray:
        return np.arctan2(y, x)

    def asin(self, x: Any) -> np.ndarray:
        return np.arcsin(x)

    def acos(self, x: Any) -> np.ndarray:
        return np.arccos(x)

    def tanh(self, x: Any) -> np.ndarray:
        return np.tanh(x)

    # ========== Vector/Matrix Math ==========

    def sumsqr(self, x: Any) -> float:
        """Sum of squares of all elements."""
        return np.sum(np.square(x))

    def sum1(self, x: Any) -> np.ndarray:
        """Sum along rows (returns column vector)."""
        arr = np.atleast_2d(x)
        return np.sum(arr, axis=1, keepdims=True)

    def sum2(self, x: Any) -> np.ndarray:
        """Sum along columns (returns row vector)."""
        arr = np.atleast_2d(x)
        return np.sum(arr, axis=0, keepdims=True)

    def fmin(self, x: Any, y: Any) -> np.ndarray:
        """Element-wise minimum."""
        return np.minimum(x, y)

    def fmax(self, x: Any, y: Any) -> np.ndarray:
        """Element-wise maximum."""
        return np.maximum(x, y)

    # ========== Conditional Operations ==========

    def if_else(self, cond: Any, if_true: Any, if_false: Any) -> np.ndarray:
        return np.where(cond, if_true, if_false)

    # ========== Numeric Conversion ==========

    def to_numpy(self, x: Any) -> np.ndarray:
        return np.asarray(x)

    def from_numpy(self, x: np.ndarray) -> np.ndarray:
        return x

    def is_symbolic(self, x: Any) -> bool:
        """NumPy arrays are never symbolic."""
        return False

    # ========== Shape/Size Operations ==========

    def size(self, x: Any, dim: int = 0) -> int:
        arr = np.atleast_1d(x)
        if dim < len(arr.shape):
            return arr.shape[dim]
        return 1

    def numel(self, x: Any) -> int:
        return np.size(x)

    # ========== Additional NumPy-Specific Methods ==========

    def solve(self, A: Any, b: Any) -> np.ndarray:
        """Solve linear system A @ x = b."""
        return np.linalg.solve(A, b)

    def inv(self, A: Any) -> np.ndarray:
        """Matrix inverse."""
        return np.linalg.inv(A)

    def det(self, A: Any) -> float:
        """Matrix determinant."""
        return np.linalg.det(A)

    def eig(self, A: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalue decomposition."""
        return np.linalg.eig(A)

    def svd(self, A: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Singular value decomposition."""
        return np.linalg.svd(A)

    def mtimes(self, A: Any, B: Any) -> np.ndarray:
        """Matrix multiplication."""
        return np.matmul(A, B)

    def dot(self, a: Any, b: Any) -> float:
        """Dot product."""
        return np.dot(a, b)

    def cross(self, a: Any, b: Any) -> np.ndarray:
        """Cross product."""
        return np.cross(a, b)

    def norm(self, x: Any, ord=None) -> float:
        """Vector/matrix norm."""
        return np.linalg.norm(x, ord=ord)

    def norm_2(self, x: Any) -> float:
        """Euclidean norm."""
        return np.linalg.norm(x, ord=2)

    def norm_1(self, x: Any) -> float:
        """1-norm."""
        return np.linalg.norm(x, ord=1)

    def norm_inf(self, x: Any) -> float:
        """Infinity norm."""
        return np.linalg.norm(x, ord=np.inf)

    def diag(self, x: Any) -> np.ndarray:
        """Diagonal matrix or extract diagonal."""
        return np.diag(x)

    def trace(self, x: Any) -> float:
        """Matrix trace."""
        return np.trace(x)

    def sum(self, x: Any, axis=None) -> Any:
        """Sum elements."""
        return np.sum(x, axis=axis)

    def mean(self, x: Any, axis=None) -> Any:
        """Mean of elements."""
        return np.mean(x, axis=axis)

    def std(self, x: Any, axis=None) -> Any:
        """Standard deviation."""
        return np.std(x, axis=axis)

    def min(self, x: Any, axis=None) -> Any:
        """Minimum."""
        return np.min(x, axis=axis)

    def max(self, x: Any, axis=None) -> Any:
        """Maximum."""
        return np.max(x, axis=axis)


# Singleton instance for convenience
_backend = None


def get_numpy_backend() -> NumPyBackend:
    """Get the singleton NumPy backend instance."""
    global _backend
    if _backend is None:
        _backend = NumPyBackend()
    return _backend
