"""JAX backend implementation.

This module provides a JAX implementation of the SymbolicBackend interface.
JAX provides automatic differentiation and GPU/TPU acceleration.

Use this backend for:
- GPU/TPU accelerated computation
- Automatic differentiation (forward and reverse mode)
- JIT compilation for speed
- Vectorization with vmap
- Machine learning integration

Note: JAX uses a different paradigm than CasADi:
- No symbolic variables - uses tracing for autodiff
- Functions must be pure (no side effects)
- Arrays are immutable

Installation:
    pip install jax jaxlib
    # For GPU: pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import grad
    from jax import hessian as jax_hessian
    from jax import jacobian as jax_jacobian
    from jax import jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from .base import SymbolicBackend

if JAX_AVAILABLE:

    class JaxArray:
        """Wrapper to provide .sym() method for JAX arrays.

        JAX doesn't have symbolic variables like CasADi. Instead, it uses
        tracing - you define functions, and JAX traces through them to
        compute derivatives.

        This wrapper creates placeholder arrays that can be used as
        function inputs for tracing.
        """

        @classmethod
        def sym(cls, name: str, n: int = 1):
            """Create a named array placeholder.

            Args:
                name: Variable name (for documentation)
                n: Number of elements

            Returns:
                JAX array of zeros (placeholder for tracing)
            """
            return jnp.zeros(n)

        @classmethod
        def zeros(cls, rows: int, cols: int = 1):
            """Create zero matrix."""
            if cols == 1:
                return jnp.zeros(rows)
            return jnp.zeros((rows, cols))

        @classmethod
        def ones(cls, rows: int, cols: int = 1):
            """Create ones matrix."""
            if cols == 1:
                return jnp.ones(rows)
            return jnp.ones((rows, cols))

        @classmethod
        def eye(cls, n: int):
            """Create identity matrix."""
            return jnp.eye(n)

    class JaxFunction:
        """Wrapper for JAX-traced functions.

        Unlike CasADi where you build symbolic expressions, JAX traces
        through Python functions. This wrapper provides a similar interface.
        """

        def __init__(
            self,
            name: str,
            func: callable,
            input_shapes: List[Tuple[int, ...]],
            output_shapes: List[Tuple[int, ...]],
            input_names: Optional[List[str]] = None,
            output_names: Optional[List[str]] = None,
            use_jit: bool = True,
        ):
            """Create a JAX function wrapper.

            Args:
                name: Function name
                func: Python function to wrap
                input_shapes: Shapes of inputs
                output_shapes: Shapes of outputs
                input_names: Optional names for inputs
                output_names: Optional names for outputs
                use_jit: Whether to JIT compile the function
            """
            self.name = name
            self._input_shapes = input_shapes
            self._output_shapes = output_shapes
            self.input_names = input_names or [f"i{i}" for i in range(len(input_shapes))]
            self.output_names = output_names or [f"o{i}" for i in range(len(output_shapes))]

            if use_jit:
                self._func = jit(func)
            else:
                self._func = func

        def __call__(self, *args):
            """Evaluate the function."""
            # Convert inputs to JAX arrays
            jax_args = [jnp.asarray(arg) for arg in args]
            result = self._func(*jax_args)

            # Convert outputs to numpy for compatibility
            if isinstance(result, (list, tuple)):
                return [np.asarray(r) for r in result]
            return np.asarray(result)

        def __repr__(self):
            return f"JaxFunction('{self.name}')"

        def size_in(self, i: int) -> int:
            """Get size of input i."""
            shape = self._input_shapes[i]
            return int(np.prod(shape))

        def size_out(self, i: int) -> Tuple[int, int]:
            """Get size of output i."""
            return self._output_shapes[i]

    class JAXBackend(SymbolicBackend):
        """JAX implementation of the symbolic backend.

        JAX provides automatic differentiation through function tracing,
        not symbolic expression building. Key differences from CasADi:

        1. No symbolic variables - use placeholder arrays and trace functions
        2. Differentiation is done on Python functions, not expressions
        3. GPU/TPU acceleration via XLA compilation
        4. JIT compilation for speed

        Best suited for:
        - Machine learning and neural networks
        - GPU-accelerated computation
        - Large-scale automatic differentiation
        - Vectorized operations

        Example:
            import cyecca.sym as cy
            cy.set_backend('jax')

            # Define a function (not symbolic expressions)
            def f(x):
                return cy.sin(x[0]) + cy.cos(x[1])

            # Get Jacobian function
            df = cy.get_raw_backend().jacobian_func(f)

            # Evaluate
            x = jnp.array([1.0, 2.0])
            print(df(x))
        """

        def __init__(self):
            if not JAX_AVAILABLE:
                raise ImportError("JAX is not installed. Install it with: pip install jax jaxlib")

        @property
        def name(self) -> str:
            return "jax"

        # ========== Symbol Types ==========

        @property
        def SX(self) -> type:
            """JAX array type."""
            return JaxArray

        @property
        def MX(self) -> type:
            """JAX array type (same as SX for JAX)."""
            return JaxArray

        @property
        def DM(self) -> type:
            """JAX array type."""
            return JaxArray

        # ========== Symbol Creation ==========

        def sym(self, name: str, shape: int = 1, sym_type=None) -> Any:
            """Create a placeholder array.

            Note: JAX doesn't have symbolic variables. This creates a
            zero array that can be used for function tracing.
            """
            return jnp.zeros(shape)

        def zeros(self, rows: int, cols: int = 1, sym_type=None) -> Any:
            if cols == 1:
                return jnp.zeros(rows)
            return jnp.zeros((rows, cols))

        def ones(self, rows: int, cols: int = 1, sym_type=None) -> Any:
            if cols == 1:
                return jnp.ones(rows)
            return jnp.ones((rows, cols))

        def eye(self, n: int, sym_type=None) -> Any:
            return jnp.eye(n)

        # ========== Matrix Operations ==========

        def vertcat(self, *args) -> Any:
            """Vertical concatenation."""
            if not args:
                return jnp.array([])

            parts = []
            for arg in args:
                arr = jnp.atleast_1d(jnp.asarray(arg))
                parts.append(arr.flatten())

            return jnp.concatenate(parts)

        def horzcat(self, *args) -> Any:
            """Horizontal concatenation."""
            if not args:
                return jnp.array([[]])

            parts = []
            for arg in args:
                arr = jnp.atleast_2d(jnp.asarray(arg))
                if arr.shape[0] == 1:
                    arr = arr.T
                parts.append(arr)

            return jnp.hstack(parts)

        def reshape(self, x: Any, shape: Tuple[int, int]) -> Any:
            return jnp.reshape(x, shape)

        def transpose(self, x: Any) -> Any:
            return jnp.transpose(x)

        # ========== Differentiation ==========

        def jacobian(self, expr: Any, x: Any) -> Any:
            """Compute Jacobian.

            Note: JAX differentiation works on functions, not expressions.
            For expression-based Jacobian, we create a lambda.

            This is mainly for API compatibility. For best results,
            use jacobian_func() to get a Jacobian function.
            """
            raise NotImplementedError(
                "JAX backend computes Jacobians of functions, not expressions. "
                "Use jacobian_func(f) to get the Jacobian of a function f, "
                "or use CasADi/SymPy for expression-based differentiation."
            )

        def gradient(self, expr: Any, x: Any) -> Any:
            """Compute gradient."""
            raise NotImplementedError(
                "JAX backend computes gradients of functions, not expressions. "
                "Use grad_func(f) to get the gradient of a function f."
            )

        def hessian(self, expr: Any, x: Any) -> Tuple[Any, Any]:
            """Compute Hessian."""
            raise NotImplementedError(
                "JAX backend computes Hessians of functions, not expressions. "
                "Use hessian_func(f) to get the Hessian of a function f."
            )

        def jtimes(self, expr: Any, x: Any, v: Any) -> Any:
            """Jacobian-times-vector product."""
            raise NotImplementedError(
                "JAX backend computes JVPs of functions, not expressions. "
                "Use jax.jvp() directly for JVP computation."
            )

        # ========== JAX-specific differentiation ==========

        def grad_func(self, f: callable, argnums: int = 0) -> callable:
            """Get gradient function (reverse-mode AD).

            Args:
                f: Scalar-valued function
                argnums: Which argument to differentiate w.r.t.

            Returns:
                Gradient function
            """
            return grad(f, argnums=argnums)

        def jacobian_func(self, f: callable, argnums: int = 0) -> callable:
            """Get Jacobian function.

            Args:
                f: Vector-valued function
                argnums: Which argument to differentiate w.r.t.

            Returns:
                Jacobian function
            """
            return jax_jacobian(f, argnums=argnums)

        def hessian_func(self, f: callable, argnums: int = 0) -> callable:
            """Get Hessian function.

            Args:
                f: Scalar-valued function
                argnums: Which argument to differentiate w.r.t.

            Returns:
                Hessian function
            """
            return jax_hessian(f, argnums=argnums)

        def jvp(self, f: callable, primals: tuple, tangents: tuple) -> Tuple[Any, Any]:
            """Jacobian-vector product (forward-mode AD).

            Args:
                f: Function to differentiate
                primals: Input values
                tangents: Tangent vectors

            Returns:
                Tuple of (output, output_tangent)
            """
            return jax.jvp(f, primals, tangents)

        def vjp(self, f: callable, *primals) -> Tuple[Any, callable]:
            """Vector-Jacobian product (reverse-mode AD).

            Args:
                f: Function to differentiate
                *primals: Input values

            Returns:
                Tuple of (output, vjp_function)
            """
            return jax.vjp(f, *primals)

        # ========== Function Creation ==========

        def function(
            self,
            name: str,
            inputs: List[Any],
            outputs: List[Any],
            input_names: Optional[List[str]] = None,
            output_names: Optional[List[str]] = None,
        ) -> JaxFunction:
            """Create a function wrapper.

            Note: For JAX, this is mainly for API compatibility.
            Better to define Python functions directly.
            """
            # Infer shapes from inputs/outputs
            input_shapes = [jnp.asarray(inp).shape for inp in inputs]
            output_shapes = [jnp.asarray(out).shape for out in outputs]

            # Create a simple function that returns the outputs
            def func(*args):
                if len(outputs) == 1:
                    return outputs[0]
                return outputs

            return JaxFunction(name, func, input_shapes, output_shapes, input_names, output_names)

        # ========== Math Functions ==========

        def sin(self, x: Any) -> Any:
            return jnp.sin(x)

        def cos(self, x: Any) -> Any:
            return jnp.cos(x)

        def tan(self, x: Any) -> Any:
            return jnp.tan(x)

        def exp(self, x: Any) -> Any:
            return jnp.exp(x)

        def log(self, x: Any) -> Any:
            return jnp.log(x)

        def sqrt(self, x: Any) -> Any:
            return jnp.sqrt(x)

        def power(self, x: Any, y: Any) -> Any:
            return jnp.power(x, y)

        def abs(self, x: Any) -> Any:
            return jnp.abs(x)

        def atan2(self, y: Any, x: Any) -> Any:
            return jnp.arctan2(y, x)

        def asin(self, x: Any) -> Any:
            return jnp.arcsin(x)

        def acos(self, x: Any) -> Any:
            return jnp.arccos(x)

        def tanh(self, x: Any) -> Any:
            return jnp.tanh(x)

        # ========== Vector/Matrix Math ==========

        def sumsqr(self, x: Any) -> Any:
            """Sum of squares of all elements."""
            return jnp.sum(jnp.square(x))

        def sum1(self, x: Any) -> Any:
            """Sum along rows (returns column vector)."""
            arr = jnp.atleast_2d(x)
            return jnp.sum(arr, axis=1, keepdims=True)

        def sum2(self, x: Any) -> Any:
            """Sum along columns (returns row vector)."""
            arr = jnp.atleast_2d(x)
            return jnp.sum(arr, axis=0, keepdims=True)

        def fmin(self, x: Any, y: Any) -> Any:
            """Element-wise minimum."""
            return jnp.minimum(x, y)

        def fmax(self, x: Any, y: Any) -> Any:
            """Element-wise maximum."""
            return jnp.maximum(x, y)

        # ========== Conditional Operations ==========

        def if_else(self, cond: Any, if_true: Any, if_false: Any) -> Any:
            return jnp.where(cond, if_true, if_false)

        # ========== Numeric Conversion ==========

        def to_numpy(self, x: Any) -> np.ndarray:
            return np.asarray(x)

        def from_numpy(self, x: np.ndarray) -> Any:
            return jnp.asarray(x)

        def is_symbolic(self, x: Any) -> bool:
            """JAX arrays are traced, not symbolic in the CasADi sense."""
            return False

        # ========== Shape/Size Operations ==========

        def size(self, x: Any, dim: int = 0) -> int:
            arr = jnp.atleast_1d(x)
            if dim < len(arr.shape):
                return arr.shape[dim]
            return 1

        def numel(self, x: Any) -> int:
            return int(jnp.size(x))

        # ========== JAX-Specific Methods ==========

        def jit(self, f: callable) -> callable:
            """JIT compile a function for speed."""
            return jit(f)

        def vmap(self, f: callable, in_axes=0, out_axes=0) -> callable:
            """Vectorize a function."""
            return vmap(f, in_axes=in_axes, out_axes=out_axes)

        def solve(self, A: Any, b: Any) -> Any:
            """Solve linear system A @ x = b."""
            return jnp.linalg.solve(A, b)

        def inv(self, A: Any) -> Any:
            """Matrix inverse."""
            return jnp.linalg.inv(A)

        def det(self, A: Any) -> Any:
            """Matrix determinant."""
            return jnp.linalg.det(A)

        def eig(self, A: Any) -> Tuple[Any, Any]:
            """Eigenvalue decomposition."""
            return jnp.linalg.eig(A)

        def svd(self, A: Any) -> Tuple[Any, Any, Any]:
            """Singular value decomposition."""
            return jnp.linalg.svd(A)

        def mtimes(self, A: Any, B: Any) -> Any:
            """Matrix multiplication."""
            return jnp.matmul(A, B)

        def dot(self, a: Any, b: Any) -> Any:
            """Dot product."""
            return jnp.dot(a, b)

        def cross(self, a: Any, b: Any) -> Any:
            """Cross product."""
            return jnp.cross(a, b)

        def norm(self, x: Any, ord=None) -> Any:
            """Vector/matrix norm."""
            return jnp.linalg.norm(x, ord=ord)

        def norm_2(self, x: Any) -> Any:
            """Euclidean norm."""
            return jnp.linalg.norm(x, ord=2)

        def norm_1(self, x: Any) -> Any:
            """1-norm."""
            return jnp.linalg.norm(x, ord=1)

        def norm_inf(self, x: Any) -> Any:
            """Infinity norm."""
            return jnp.linalg.norm(x, ord=jnp.inf)

        def diag(self, x: Any) -> Any:
            """Diagonal matrix or extract diagonal."""
            return jnp.diag(x)

        def trace(self, x: Any) -> Any:
            """Matrix trace."""
            return jnp.trace(x)

        def sum(self, x: Any, axis=None) -> Any:
            """Sum elements."""
            return jnp.sum(x, axis=axis)

        def mean(self, x: Any, axis=None) -> Any:
            """Mean of elements."""
            return jnp.mean(x, axis=axis)


# Singleton instance for convenience
_backend = None


def get_jax_backend() -> "JAXBackend":
    """Get the singleton JAX backend instance."""
    global _backend
    if _backend is None:
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not installed. Install it with: pip install jax jaxlib")
        _backend = JAXBackend()
    return _backend
