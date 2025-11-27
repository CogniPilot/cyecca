"""Cyecca symbolic math front-end.

This module provides a clean, concise API for symbolic operations,
similar to how numpy uses `np.` This is the recommended way to use
cyecca's symbolic capabilities.

Usage:
    import cyecca.sym as cy
    
    # Create symbols
    x = cy.sym('x', 3)
    y = cy.sym('y')
    
    # Math operations
    expr = cy.sin(x[0]) + cy.cos(x[1])
    
    # Differentiation
    jac = cy.jacobian(expr, x)
    grad = cy.gradient(expr, x)
    
    # Create functions
    f = cy.function('f', [x], [expr])
    
    # Matrix operations
    z = cy.vertcat(x, y)
    
    # Change backend globally
    cy.set_backend('sympy')  # For symbolic analysis
    cy.set_backend('casadi')  # Back to default

The default backend is CasADi. All operations delegate to the
current backend, so switching backends changes behavior globally.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .backends import SymbolicBackend, get_backend, list_backends, set_default_backend

# For IDE autocomplete, we use CasADi types since it's the default backend
# At runtime, Any is used for backend flexibility
if TYPE_CHECKING:
    import casadi as ca

    # Type aliases for better autocomplete
    SXType = ca.SX
    MXType = ca.MX
    DMType = ca.DM
    FunctionType = ca.Function
else:
    SXType = Any
    MXType = Any
    DMType = Any
    FunctionType = Any

# Current backend instance (lazy initialization)
_backend: Optional[SymbolicBackend] = None


def _get_backend() -> SymbolicBackend:
    """Get current backend, initializing if needed."""
    global _backend
    if _backend is None:
        _backend = get_backend()
    return _backend


def set_backend(name: str):
    """Set the backend for all cy.* operations.

    Args:
        name: Backend name ('casadi', 'sympy', etc.)

    Example:
        >>> import cyecca.sym as cy
        >>> cy.set_backend('sympy')
        >>> cy.backend_name()
        'sympy'
        >>> cy.set_backend('casadi')
    """
    global _backend
    set_default_backend(name)
    _backend = get_backend(name)


def backend_name() -> str:
    """Get current backend name."""
    return _get_backend().name


def backends() -> List[str]:
    """List available backends."""
    return list_backends()


# ========== Symbol Types ==========


@property
def SX():
    """Sparse symbolic type (CasADi SX or equivalent)."""
    return _get_backend().SX


@property
def MX():
    """Matrix symbolic type (CasADi MX or equivalent)."""
    return _get_backend().MX


@property
def DM():
    """Dense numeric matrix type."""
    return _get_backend().DM


# ========== Symbol Creation ==========


def sym(name: str, shape: int = 1, sym_type=None) -> SXType:
    """Create a symbolic variable.

    Args:
        name: Variable name
        shape: Number of elements (default 1 for scalar)
        sym_type: Symbol type (SX, MX). If None, uses SX.

    Returns:
        Symbolic variable

    Example:
        >>> import cyecca.sym as cy
        >>> x = cy.sym('x')      # Scalar
        >>> v = cy.sym('v', 3)   # 3-vector
    """
    return _get_backend().sym(name, shape, sym_type)


def zeros(rows: int, cols: int = 1, sym_type=None) -> DMType:
    """Create a zero matrix."""
    return _get_backend().zeros(rows, cols, sym_type)


def ones(rows: int, cols: int = 1, sym_type=None) -> DMType:
    """Create a matrix of ones."""
    return _get_backend().ones(rows, cols, sym_type)


def eye(n: int, sym_type=None) -> DMType:
    """Create an identity matrix."""
    return _get_backend().eye(n, sym_type)


# ========== Matrix Operations ==========


def vertcat(*args) -> SXType:
    """Vertical concatenation.

    Example:
        >>> import cyecca.sym as cy
        >>> x = cy.sym('x', 2)
        >>> y = cy.sym('y', 3)
        >>> z = cy.vertcat(x, y)  # 5-vector
    """
    return _get_backend().vertcat(*args)


def horzcat(*args) -> SXType:
    """Horizontal concatenation."""
    return _get_backend().horzcat(*args)


def reshape(x: SXType, shape: Tuple[int, int]) -> SXType:
    """Reshape a matrix."""
    return _get_backend().reshape(x, shape)


def transpose(x: SXType) -> SXType:
    """Matrix transpose."""
    return _get_backend().transpose(x)


# ========== Differentiation ==========


def jacobian(expr: SXType, x: SXType) -> SXType:
    """Compute Jacobian matrix.

    Args:
        expr: Expression to differentiate (m x 1)
        x: Variable to differentiate with respect to (n x 1)

    Returns:
        Jacobian matrix (m x n)

    Example:
        >>> import cyecca.sym as cy
        >>> x = cy.sym('x', 2)
        >>> expr = x[0]**2 + x[1]**3
        >>> J = cy.jacobian(expr, x)
    """
    return _get_backend().jacobian(expr, x)


def gradient(expr: SXType, x: SXType) -> SXType:
    """Compute gradient (for scalar expressions).

    Args:
        expr: Scalar expression
        x: Variable to differentiate with respect to

    Returns:
        Gradient vector
    """
    return _get_backend().gradient(expr, x)


def hessian(expr: SXType, x: SXType) -> Tuple[SXType, SXType]:
    """Compute Hessian matrix.

    Args:
        expr: Scalar expression
        x: Variable to differentiate with respect to

    Returns:
        Tuple of (Hessian, gradient)
    """
    return _get_backend().hessian(expr, x)


def jtimes(expr: SXType, x: SXType, v: SXType) -> SXType:
    """Jacobian-times-vector product (forward mode AD)."""
    return _get_backend().jtimes(expr, x, v)


# ========== Function Creation ==========


def function(
    name: str,
    inputs: List[SXType],
    outputs: List[SXType],
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
) -> FunctionType:
    """Create a callable function from symbolic expressions.

    Args:
        name: Function name
        inputs: List of symbolic input variables
        outputs: List of symbolic output expressions
        input_names: Optional names for inputs
        output_names: Optional names for outputs

    Returns:
        Callable function

    Example:
        >>> import cyecca.sym as cy
        >>> x = cy.sym('x', 2)
        >>> y = x[0]**2 + x[1]
        >>> f = cy.function('f', [x], [y])
    """
    return _get_backend().function(name, inputs, outputs, input_names, output_names)


# ========== Math Functions ==========


def sin(x: SXType) -> SXType:
    """Sine function."""
    return _get_backend().sin(x)


def cos(x: SXType) -> SXType:
    """Cosine function."""
    return _get_backend().cos(x)


def tan(x: SXType) -> SXType:
    """Tangent function."""
    return _get_backend().tan(x)


def exp(x: SXType) -> SXType:
    """Exponential function."""
    return _get_backend().exp(x)


def log(x: SXType) -> SXType:
    """Natural logarithm."""
    return _get_backend().log(x)


def sqrt(x: SXType) -> SXType:
    """Square root."""
    return _get_backend().sqrt(x)


def power(x: SXType, y: SXType) -> SXType:
    """Power function x^y."""
    return _get_backend().power(x, y)


def fabs(x: SXType) -> SXType:
    """Absolute value."""
    return _get_backend().abs(x)


def atan2(y: SXType, x: SXType) -> SXType:
    """Two-argument arctangent."""
    return _get_backend().atan2(y, x)


def asin(x: SXType) -> SXType:
    """Arcsine."""
    return _get_backend().asin(x)


def acos(x: SXType) -> SXType:
    """Arccosine."""
    return _get_backend().acos(x)


def tanh(x: SXType) -> SXType:
    """Hyperbolic tangent."""
    return _get_backend().tanh(x)


# ========== Vector/Matrix Math ==========


def norm_2(x: SXType) -> SXType:
    """L2 (Euclidean) norm of a vector."""
    return _get_backend().norm_2(x)


def dot(x: SXType, y: SXType) -> SXType:
    """Dot product of two vectors."""
    return _get_backend().dot(x, y)


def cross(x: SXType, y: SXType) -> SXType:
    """Cross product of two 3-vectors."""
    return _get_backend().cross(x, y)


def inv(x: SXType) -> SXType:
    """Matrix inverse."""
    return _get_backend().inv(x)


def sumsqr(x: SXType) -> SXType:
    """Sum of squares of all elements."""
    return _get_backend().sumsqr(x)


def sum1(x: SXType) -> SXType:
    """Sum along rows (returns column vector)."""
    return _get_backend().sum1(x)


def sum2(x: SXType) -> SXType:
    """Sum along columns (returns row vector)."""
    return _get_backend().sum2(x)


def fmin(x: SXType, y: SXType) -> SXType:
    """Element-wise minimum."""
    return _get_backend().fmin(x, y)


def fmax(x: SXType, y: SXType) -> SXType:
    """Element-wise maximum."""
    return _get_backend().fmax(x, y)


def diag(x: SXType) -> SXType:
    """Extract diagonal or create diagonal matrix."""
    return _get_backend().diag(x)


# ========== Conditional Operations ==========


def if_else(cond: SXType, if_true: SXType, if_false: SXType) -> SXType:
    """Conditional expression."""
    return _get_backend().if_else(cond, if_true, if_false)


# ========== Numeric Conversion ==========


def to_numpy(x: Any) -> np.ndarray:
    """Convert to numpy array."""
    return _get_backend().to_numpy(x)


def from_numpy(x: np.ndarray) -> Any:
    """Convert from numpy array."""
    return _get_backend().from_numpy(x)


def is_symbolic(x: Any) -> bool:
    """Check if expression contains symbolic variables."""
    return _get_backend().is_symbolic(x)


# ========== Shape/Size Operations ==========


def size(x: Any, dim: int = 0) -> int:
    """Get size along a dimension."""
    return _get_backend().size(x, dim)


def numel(x: Any) -> int:
    """Get total number of elements."""
    return _get_backend().numel(x)


# ========== Integration (CasADi-specific) ==========


def integrator(
    name: str,
    method: str,
    dae: Dict[str, Any],
    t0: float,
    tf: Union[float, Sequence[float]],
    options: Optional[Dict] = None,
) -> Any:
    """Create an ODE/DAE integrator."""
    return _get_backend().integrator(name, method, dae, t0, tf, options)


# ========== Optimization (CasADi-specific) ==========


def nlpsol(
    name: str,
    solver: str,
    nlp: Dict[str, Any],
    options: Optional[Dict] = None,
) -> Any:
    """Create a nonlinear programming solver."""
    return _get_backend().nlpsol(name, solver, nlp, options)


# ========== Backend-specific access ==========


def get_raw_backend() -> SymbolicBackend:
    """Get the raw backend instance for advanced operations.

    Use this when you need backend-specific methods not exposed
    through the cy.* API.

    Example:
        >>> import cyecca.sym as cy
        >>> backend = cy.get_raw_backend()
        >>> if backend.name == 'sympy':
        ...     latex_str = backend.latex(expr)
    """
    return _get_backend()
