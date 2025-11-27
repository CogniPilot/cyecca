"""Abstract base class for symbolic backends.

This module defines the interface that all symbolic backends must implement.
The interface is designed to support the operations needed by cyecca's
dynamics modeling framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class SymbolicBackend(ABC):
    """Abstract base class for symbolic math backends.

    This interface defines all operations needed for:
    - Symbol creation (scalar, vector, matrix)
    - Matrix operations (vertcat, horzcat, etc.)
    - Differentiation (jacobian, gradient, hessian)
    - Function creation and evaluation
    - Numerical conversions
    - Integration (DAE solvers)

    Implementations must provide concrete versions of these operations
    for their specific framework (CasADi, JAX, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""
        pass

    # ========== Symbol Types ==========

    @property
    @abstractmethod
    def SX(self) -> type:
        """Sparse symbolic type (like CasADi SX).

        This is the primary symbolic type for scalar expressions.
        """
        pass

    @property
    @abstractmethod
    def MX(self) -> type:
        """Matrix symbolic type (like CasADi MX).

        This is used for larger expressions with graph-based representation.
        """
        pass

    @property
    @abstractmethod
    def DM(self) -> type:
        """Dense numeric matrix type (like CasADi DM).

        This is used for numeric values in computations.
        """
        pass

    # ========== Symbol Creation ==========

    @abstractmethod
    def sym(self, name: str, shape: int = 1, sym_type=None) -> Any:
        """Create a symbolic variable.

        Args:
            name: Variable name for display/debugging
            shape: Number of elements (scalar if 1)
            sym_type: Symbol type (SX, MX). If None, uses SX.

        Returns:
            Symbolic variable
        """
        pass

    @abstractmethod
    def zeros(self, rows: int, cols: int = 1, sym_type=None) -> Any:
        """Create a zero matrix/vector.

        Args:
            rows: Number of rows
            cols: Number of columns
            sym_type: Output type (SX, MX, DM). If None, uses DM.

        Returns:
            Zero matrix
        """
        pass

    @abstractmethod
    def ones(self, rows: int, cols: int = 1, sym_type=None) -> Any:
        """Create a matrix of ones.

        Args:
            rows: Number of rows
            cols: Number of columns
            sym_type: Output type (SX, MX, DM). If None, uses DM.

        Returns:
            Matrix of ones
        """
        pass

    @abstractmethod
    def eye(self, n: int, sym_type=None) -> Any:
        """Create an identity matrix.

        Args:
            n: Matrix size (n x n)
            sym_type: Output type (SX, MX, DM). If None, uses DM.

        Returns:
            Identity matrix
        """
        pass

    # ========== Matrix Operations ==========

    @abstractmethod
    def vertcat(self, *args) -> Any:
        """Vertical concatenation of vectors/matrices.

        Args:
            *args: Vectors or matrices to concatenate vertically

        Returns:
            Concatenated result
        """
        pass

    @abstractmethod
    def horzcat(self, *args) -> Any:
        """Horizontal concatenation of vectors/matrices.

        Args:
            *args: Vectors or matrices to concatenate horizontally

        Returns:
            Concatenated result
        """
        pass

    @abstractmethod
    def reshape(self, x: Any, shape: Tuple[int, int]) -> Any:
        """Reshape a matrix.

        Args:
            x: Matrix to reshape
            shape: New shape (rows, cols)

        Returns:
            Reshaped matrix
        """
        pass

    @abstractmethod
    def transpose(self, x: Any) -> Any:
        """Matrix transpose.

        Args:
            x: Matrix to transpose

        Returns:
            Transposed matrix
        """
        pass

    # ========== Differentiation ==========

    @abstractmethod
    def jacobian(self, expr: Any, x: Any) -> Any:
        """Compute Jacobian matrix.

        Args:
            expr: Expression to differentiate (m x 1)
            x: Variable to differentiate with respect to (n x 1)

        Returns:
            Jacobian matrix (m x n)
        """
        pass

    @abstractmethod
    def gradient(self, expr: Any, x: Any) -> Any:
        """Compute gradient (transpose of Jacobian for scalar expr).

        Args:
            expr: Scalar expression
            x: Variable to differentiate with respect to

        Returns:
            Gradient vector
        """
        pass

    @abstractmethod
    def hessian(self, expr: Any, x: Any) -> Tuple[Any, Any]:
        """Compute Hessian matrix.

        Args:
            expr: Scalar expression
            x: Variable to differentiate with respect to

        Returns:
            Tuple of (Hessian, gradient)
        """
        pass

    @abstractmethod
    def jtimes(self, expr: Any, x: Any, v: Any) -> Any:
        """Jacobian-times-vector product (forward mode AD).

        Args:
            expr: Expression
            x: Variable
            v: Vector to multiply

        Returns:
            J @ v
        """
        pass

    # ========== Function Creation ==========

    @abstractmethod
    def function(
        self,
        name: str,
        inputs: List[Any],
        outputs: List[Any],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> Any:
        """Create a callable function from symbolic expressions.

        Args:
            name: Function name
            inputs: List of symbolic input variables
            outputs: List of symbolic output expressions
            input_names: Optional names for inputs
            output_names: Optional names for outputs

        Returns:
            Callable function object
        """
        pass

    # ========== Math Functions ==========

    @abstractmethod
    def sin(self, x: Any) -> Any:
        """Sine function."""
        pass

    @abstractmethod
    def cos(self, x: Any) -> Any:
        """Cosine function."""
        pass

    @abstractmethod
    def tan(self, x: Any) -> Any:
        """Tangent function."""
        pass

    @abstractmethod
    def exp(self, x: Any) -> Any:
        """Exponential function."""
        pass

    @abstractmethod
    def log(self, x: Any) -> Any:
        """Natural logarithm."""
        pass

    @abstractmethod
    def sqrt(self, x: Any) -> Any:
        """Square root."""
        pass

    @abstractmethod
    def power(self, x: Any, y: Any) -> Any:
        """Power function x^y."""
        pass

    @abstractmethod
    def abs(self, x: Any) -> Any:
        """Absolute value."""
        pass

    @abstractmethod
    def atan2(self, y: Any, x: Any) -> Any:
        """Two-argument arctangent."""
        pass

    @abstractmethod
    def asin(self, x: Any) -> Any:
        """Arcsine."""
        pass

    @abstractmethod
    def acos(self, x: Any) -> Any:
        """Arccosine."""
        pass

    @abstractmethod
    def tanh(self, x: Any) -> Any:
        """Hyperbolic tangent."""
        pass

    # ========== Vector/Matrix Math ==========

    @abstractmethod
    def norm_2(self, x: Any) -> Any:
        """L2 norm (Euclidean norm) of a vector."""
        pass

    @abstractmethod
    def dot(self, x: Any, y: Any) -> Any:
        """Dot product of two vectors."""
        pass

    @abstractmethod
    def cross(self, x: Any, y: Any) -> Any:
        """Cross product of two 3-vectors."""
        pass

    @abstractmethod
    def inv(self, x: Any) -> Any:
        """Matrix inverse."""
        pass

    @abstractmethod
    def sumsqr(self, x: Any) -> Any:
        """Sum of squares of all elements."""
        pass

    @abstractmethod
    def sum1(self, x: Any) -> Any:
        """Sum along rows (returns column vector)."""
        pass

    @abstractmethod
    def sum2(self, x: Any) -> Any:
        """Sum along columns (returns row vector)."""
        pass

    @abstractmethod
    def fmin(self, x: Any, y: Any) -> Any:
        """Element-wise minimum."""
        pass

    @abstractmethod
    def fmax(self, x: Any, y: Any) -> Any:
        """Element-wise maximum."""
        pass

    @abstractmethod
    def diag(self, x: Any) -> Any:
        """Extract diagonal or create diagonal matrix."""
        pass

    # ========== Conditional Operations ==========

    @abstractmethod
    def if_else(self, cond: Any, if_true: Any, if_false: Any) -> Any:
        """Conditional expression.

        Args:
            cond: Boolean condition
            if_true: Value if condition is true
            if_false: Value if condition is false

        Returns:
            Conditional result
        """
        pass

    # ========== Numeric Conversion ==========

    @abstractmethod
    def to_numpy(self, x: Any) -> np.ndarray:
        """Convert to numpy array.

        Args:
            x: Symbolic or numeric expression

        Returns:
            numpy array
        """
        pass

    @abstractmethod
    def from_numpy(self, x: np.ndarray) -> Any:
        """Convert from numpy array to DM.

        Args:
            x: numpy array

        Returns:
            DM matrix
        """
        pass

    @abstractmethod
    def is_symbolic(self, x: Any) -> bool:
        """Check if expression contains symbolic variables.

        Args:
            x: Expression to check

        Returns:
            True if symbolic
        """
        pass

    # ========== Shape/Size Operations ==========

    @abstractmethod
    def size(self, x: Any, dim: int = 0) -> int:
        """Get size along a dimension.

        Args:
            x: Matrix
            dim: Dimension (0 for rows, 1 for cols)

        Returns:
            Size along dimension
        """
        pass

    @abstractmethod
    def numel(self, x: Any) -> int:
        """Get total number of elements.

        Args:
            x: Matrix

        Returns:
            Number of elements
        """
        pass

    # ========== Integration (Optional) ==========

    def integrator(
        self,
        name: str,
        method: str,
        dae: Dict[str, Any],
        t0: float,
        tf: Union[float, Sequence[float]],
        options: Optional[Dict] = None,
    ) -> Any:
        """Create an ODE/DAE integrator.

        Args:
            name: Integrator name
            method: Integration method ('cvodes', 'idas', 'rk', etc.)
            dae: DAE dictionary with keys 'x', 'z', 'p', 'ode', 'alg'
            t0: Initial time
            tf: Final time or grid of times
            options: Solver options

        Returns:
            Integrator function

        Note:
            Not all backends may support this. Raise NotImplementedError
            if not supported.
        """
        raise NotImplementedError(f"Backend '{self.name}' does not support integrator()")

    # ========== Optimization (Optional) ==========

    def nlpsol(
        self,
        name: str,
        solver: str,
        nlp: Dict[str, Any],
        options: Optional[Dict] = None,
    ) -> Any:
        """Create a nonlinear programming solver.

        Args:
            name: Solver name
            solver: Solver type ('ipopt', 'sqpmethod', etc.)
            nlp: NLP dictionary with keys 'x', 'f', 'g'
            options: Solver options

        Returns:
            NLP solver function

        Note:
            Not all backends may support this. Raise NotImplementedError
            if not supported.
        """
        raise NotImplementedError(f"Backend '{self.name}' does not support nlpsol()")


# Type alias for use in type hints
Expr = Any  # Symbolic expression type
