"""SymPy backend implementation.

This module provides a SymPy implementation of the SymbolicBackend interface.
SymPy is useful for symbolic mathematics, analytical solutions, and 
LaTeX output.

Note: SymPy is primarily for symbolic manipulation and analysis.
It does not support efficient numerical integration like CasADi.
Use this backend for:
- Symbolic analysis and simplification
- LaTeX equation generation
- Analytical derivatives
- Educational/documentation purposes

NOT recommended for:
- Large-scale simulation
- Numerical optimization
- Real-time control
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import sympy as sp
    from sympy import (
        Abs,
        Function,
        Matrix,
        Piecewise,
        acos,
        asin,
        atan2,
        cos,
        exp,
        lambdify,
        log,
        sin,
        sqrt,
        symbols,
        tan,
    )
    from sympy.matrices import eye, ones, zeros

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

from .base import SymbolicBackend


class SympySymbol:
    """Wrapper to make SymPy symbols behave more like CasADi SX.

    Provides a .sym() class method for creating symbols.
    """

    @classmethod
    def sym(cls, name: str, n: int = 1):
        """Create symbolic variable(s).

        Args:
            name: Variable name
            n: Number of elements

        Returns:
            SymPy symbol or Matrix of symbols
        """
        if n == 1:
            return sp.Symbol(name, real=True)
        else:
            syms = [sp.Symbol(f"{name}_{i}", real=True) for i in range(n)]
            return sp.Matrix(syms)

    @classmethod
    def zeros(cls, rows: int, cols: int = 1):
        """Create zero matrix."""
        return sp.zeros(rows, cols)

    @classmethod
    def ones(cls, rows: int, cols: int = 1):
        """Create ones matrix."""
        return sp.ones(rows, cols)

    @classmethod
    def eye(cls, n: int):
        """Create identity matrix."""
        return sp.eye(n)


class SympyDM:
    """Wrapper to make SymPy matrices behave like CasADi DM."""

    def __new__(cls, data):
        """Convert data to SymPy Matrix."""
        if isinstance(data, sp.Matrix):
            return data
        elif isinstance(data, np.ndarray):
            return sp.Matrix(data.tolist())
        elif isinstance(data, (list, tuple)):
            return sp.Matrix(data)
        elif np.isscalar(data):
            return sp.Matrix([[data]])
        else:
            return sp.Matrix(data)

    @classmethod
    def zeros(cls, rows: int, cols: int = 1):
        """Create zero matrix."""
        return sp.zeros(rows, cols)

    @classmethod
    def ones(cls, rows: int, cols: int = 1):
        """Create ones matrix."""
        return sp.ones(rows, cols)

    @classmethod
    def eye(cls, n: int):
        """Create identity matrix."""
        return sp.eye(n)


class SympyFunction:
    """Wrapper to make SymPy lambdified functions behave like CasADi Function."""

    def __init__(
        self,
        name: str,
        inputs: List[Any],
        outputs: List[Any],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ):
        """Create a callable function from symbolic expressions.

        Args:
            name: Function name
            inputs: List of symbolic input variables
            outputs: List of symbolic output expressions
            input_names: Optional names for inputs
            output_names: Optional names for outputs
        """
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.input_names = input_names or [f"i{i}" for i in range(len(inputs))]
        self.output_names = output_names or [f"o{i}" for i in range(len(outputs))]

        # Flatten inputs for lambdify
        flat_inputs = []
        for inp in inputs:
            if isinstance(inp, sp.Matrix):
                flat_inputs.extend(list(inp))
            elif isinstance(inp, (list, tuple)):
                flat_inputs.extend(inp)
            else:
                flat_inputs.append(inp)

        # Create lambdified functions for each output
        self._funcs = []
        for out in outputs:
            if isinstance(out, sp.Matrix):
                # For matrix outputs, lambdify returns a matrix
                f = sp.lambdify(flat_inputs, out, modules=["numpy"])
            else:
                f = sp.lambdify(flat_inputs, out, modules=["numpy"])
            self._funcs.append(f)

        self._flat_input_symbols = flat_inputs

    def __call__(self, *args):
        """Evaluate the function.

        Args:
            *args: Input values (can be scalars, arrays, or matrices)

        Returns:
            Output value(s)
        """
        # Flatten input arguments
        flat_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                flat_args.extend(arg.flatten().tolist())
            elif isinstance(arg, sp.Matrix):
                flat_args.extend([float(x) for x in arg])
            elif hasattr(arg, "__iter__") and not isinstance(arg, str):
                flat_args.extend(list(arg))
            else:
                flat_args.append(float(arg))

        # Evaluate each output function
        results = []
        for f in self._funcs:
            try:
                result = f(*flat_args)
                if isinstance(result, np.ndarray):
                    results.append(result)
                elif isinstance(result, (list, tuple)):
                    results.append(np.array(result))
                else:
                    results.append(np.atleast_1d(result))
            except Exception as e:
                raise RuntimeError(f"Error evaluating function '{self.name}': {e}")

        if len(results) == 1:
            return results[0]
        return results

    def __repr__(self):
        return f"SympyFunction('{self.name}')"

    def size_in(self, i: int) -> int:
        """Get size of input i."""
        inp = self.inputs[i]
        if isinstance(inp, sp.Matrix):
            return inp.shape[0] * inp.shape[1]
        elif hasattr(inp, "__len__"):
            return len(inp)
        return 1

    def size_out(self, i: int) -> Tuple[int, int]:
        """Get size of output i."""
        out = self.outputs[i]
        if isinstance(out, sp.Matrix):
            return out.shape
        return (1, 1)


class SymPyBackend(SymbolicBackend):
    """SymPy implementation of the symbolic backend.

    This backend provides symbolic mathematics capabilities using SymPy.
    Best suited for:
    - Symbolic analysis and simplification
    - LaTeX equation generation
    - Analytical derivatives
    - Educational purposes

    Not recommended for numerical simulation (use CasADi instead).
    """

    def __init__(self):
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is not installed. Install it with: pip install sympy")

    @property
    def name(self) -> str:
        return "sympy"

    # ========== Symbol Types ==========

    @property
    def SX(self) -> type:
        """SymPy symbol type (analogous to CasADi SX)."""
        return SympySymbol

    @property
    def MX(self) -> type:
        """SymPy symbol type (same as SX for SymPy)."""
        return SympySymbol

    @property
    def DM(self) -> type:
        """SymPy numeric matrix type."""
        return SympyDM

    # ========== Symbol Creation ==========

    def sym(self, name: str, shape: int = 1, sym_type=None) -> Any:
        if shape == 1:
            return sp.Symbol(name, real=True)
        else:
            syms = [sp.Symbol(f"{name}_{i}", real=True) for i in range(shape)]
            return sp.Matrix(syms)

    def zeros(self, rows: int, cols: int = 1, sym_type=None) -> Any:
        return sp.zeros(rows, cols)

    def ones(self, rows: int, cols: int = 1, sym_type=None) -> Any:
        return sp.ones(rows, cols)

    def eye(self, n: int, sym_type=None) -> Any:
        return sp.eye(n)

    # ========== Matrix Operations ==========

    def vertcat(self, *args) -> Any:
        """Vertical concatenation."""
        if not args:
            return sp.Matrix([])

        parts = []
        for arg in args:
            if isinstance(arg, sp.Matrix):
                parts.append(arg)
            elif isinstance(arg, (sp.Basic, sp.Expr)):
                parts.append(sp.Matrix([arg]))
            elif isinstance(arg, (list, tuple)):
                parts.append(sp.Matrix(arg))
            elif isinstance(arg, np.ndarray):
                parts.append(sp.Matrix(arg.tolist()))
            else:
                parts.append(sp.Matrix([arg]))

        if len(parts) == 1:
            return parts[0]

        # Stack vertically
        result = parts[0]
        for p in parts[1:]:
            result = result.col_join(p)
        return result

    def horzcat(self, *args) -> Any:
        """Horizontal concatenation."""
        if not args:
            return sp.Matrix([])

        parts = []
        for arg in args:
            if isinstance(arg, sp.Matrix):
                parts.append(arg)
            elif isinstance(arg, (sp.Basic, sp.Expr)):
                parts.append(sp.Matrix([arg]))
            else:
                parts.append(sp.Matrix([arg]))

        if len(parts) == 1:
            return parts[0]

        result = parts[0]
        for p in parts[1:]:
            result = result.row_join(p)
        return result

    def reshape(self, x: Any, shape: Tuple[int, int]) -> Any:
        if isinstance(x, sp.Matrix):
            return x.reshape(shape[0], shape[1])
        return sp.Matrix([x]).reshape(shape[0], shape[1])

    def transpose(self, x: Any) -> Any:
        if isinstance(x, sp.Matrix):
            return x.T
        return x

    # ========== Differentiation ==========

    def jacobian(self, expr: Any, x: Any) -> Any:
        """Compute Jacobian matrix."""
        if isinstance(expr, sp.Matrix):
            expr_vec = expr
        else:
            expr_vec = sp.Matrix([expr])

        if isinstance(x, sp.Matrix):
            x_vec = list(x)
        elif isinstance(x, (list, tuple)):
            x_vec = list(x)
        else:
            x_vec = [x]

        return expr_vec.jacobian(x_vec)

    def gradient(self, expr: Any, x: Any) -> Any:
        """Compute gradient (for scalar expressions)."""
        if isinstance(x, sp.Matrix):
            x_vec = list(x)
        elif isinstance(x, (list, tuple)):
            x_vec = list(x)
        else:
            x_vec = [x]

        return sp.Matrix([sp.diff(expr, xi) for xi in x_vec])

    def hessian(self, expr: Any, x: Any) -> Tuple[Any, Any]:
        """Compute Hessian matrix."""
        if isinstance(x, sp.Matrix):
            x_vec = list(x)
        else:
            x_vec = [x]

        grad = sp.Matrix([sp.diff(expr, xi) for xi in x_vec])
        hess = grad.jacobian(x_vec)
        return hess, grad

    def jtimes(self, expr: Any, x: Any, v: Any) -> Any:
        """Jacobian-times-vector product."""
        jac = self.jacobian(expr, x)
        if isinstance(v, sp.Matrix):
            return jac * v
        return jac * sp.Matrix([v])

    # ========== Function Creation ==========

    def function(
        self,
        name: str,
        inputs: List[Any],
        outputs: List[Any],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> SympyFunction:
        return SympyFunction(name, inputs, outputs, input_names, output_names)

    # ========== Math Functions ==========

    def sin(self, x: Any) -> Any:
        return sp.sin(x)

    def cos(self, x: Any) -> Any:
        return sp.cos(x)

    def tan(self, x: Any) -> Any:
        return sp.tan(x)

    def exp(self, x: Any) -> Any:
        return sp.exp(x)

    def log(self, x: Any) -> Any:
        return sp.log(x)

    def sqrt(self, x: Any) -> Any:
        return sp.sqrt(x)

    def power(self, x: Any, y: Any) -> Any:
        return x**y

    def abs(self, x: Any) -> Any:
        return sp.Abs(x)

    def atan2(self, y: Any, x: Any) -> Any:
        return sp.atan2(y, x)

    def asin(self, x: Any) -> Any:
        return sp.asin(x)

    def acos(self, x: Any) -> Any:
        return sp.acos(x)

    def tanh(self, x: Any) -> Any:
        return sp.tanh(x)

    # ========== Vector/Matrix Math ==========

    def norm_2(self, x: Any) -> Any:
        """L2 (Euclidean) norm."""
        if isinstance(x, sp.Matrix):
            return x.norm()
        return sp.Abs(x)

    def sumsqr(self, x: Any) -> Any:
        """Sum of squares of all elements."""
        if isinstance(x, sp.Matrix):
            return sum(xi**2 for xi in x)
        return x**2

    def sum1(self, x: Any) -> Any:
        """Sum along rows (returns column vector)."""
        if isinstance(x, sp.Matrix):
            rows, cols = x.shape
            return sp.Matrix([sum(x[i, j] for j in range(cols)) for i in range(rows)])
        return x

    def sum2(self, x: Any) -> Any:
        """Sum along columns (returns row vector)."""
        if isinstance(x, sp.Matrix):
            rows, cols = x.shape
            return sp.Matrix([[sum(x[i, j] for i in range(rows)) for j in range(cols)]])
        return x

    def fmin(self, x: Any, y: Any) -> Any:
        """Element-wise minimum."""
        return sp.Min(x, y)

    def fmax(self, x: Any, y: Any) -> Any:
        """Element-wise maximum."""
        return sp.Max(x, y)

    def diag(self, x: Any) -> Any:
        """Extract diagonal or create diagonal matrix."""
        if isinstance(x, sp.Matrix):
            if x.shape[0] == 1 or x.shape[1] == 1:
                # Vector -> diagonal matrix
                return sp.diag(*list(x))
            else:
                # Square matrix -> extract diagonal
                n = min(x.shape)
                return sp.Matrix([x[i, i] for i in range(n)])
        return sp.Matrix([[x]])

    # ========== Conditional Operations ==========

    def if_else(self, cond: Any, if_true: Any, if_false: Any) -> Any:
        return sp.Piecewise((if_true, cond), (if_false, True))

    # ========== Numeric Conversion ==========

    def to_numpy(self, x: Any) -> np.ndarray:
        if isinstance(x, sp.Matrix):
            return np.array(x.tolist(), dtype=float)
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array([[float(x)]])

    def from_numpy(self, x: np.ndarray) -> Any:
        return sp.Matrix(x.tolist())

    def is_symbolic(self, x: Any) -> bool:
        if isinstance(x, sp.Matrix):
            return any(xi.free_symbols for xi in x)
        elif isinstance(x, sp.Basic):
            return bool(x.free_symbols)
        return False

    # ========== Shape/Size Operations ==========

    def size(self, x: Any, dim: int = 0) -> int:
        if isinstance(x, sp.Matrix):
            return x.shape[dim]
        return 1

    def numel(self, x: Any) -> int:
        if isinstance(x, sp.Matrix):
            return x.shape[0] * x.shape[1]
        return 1

    # ========== Additional SymPy-Specific Methods ==========

    def simplify(self, expr: Any) -> Any:
        """Simplify expression."""
        if isinstance(expr, sp.Matrix):
            return expr.applyfunc(sp.simplify)
        return sp.simplify(expr)

    def expand(self, expr: Any) -> Any:
        """Expand expression."""
        if isinstance(expr, sp.Matrix):
            return expr.applyfunc(sp.expand)
        return sp.expand(expr)

    def latex(self, expr: Any) -> str:
        """Convert to LaTeX string."""
        return sp.latex(expr)

    def substitute(self, expr: Any, old: Any, new: Any) -> Any:
        """Substitute old with new in expr."""
        if isinstance(expr, sp.Matrix):
            return expr.subs(old, new)
        return expr.subs(old, new)

    def solve(self, equations: Any, variables: Any) -> Any:
        """Solve equations for variables."""
        return sp.solve(equations, variables)

    def det(self, A: Any) -> Any:
        """Matrix determinant."""
        return A.det()

    def inv(self, A: Any) -> Any:
        """Matrix inverse."""
        return A.inv()

    def eigenvals(self, A: Any) -> Dict:
        """Eigenvalues."""
        return A.eigenvals()

    def eigenvects(self, A: Any) -> List:
        """Eigenvectors."""
        return A.eigenvects()

    def mtimes(self, A: Any, B: Any) -> Any:
        """Matrix multiplication."""
        return A * B

    def dot(self, a: Any, b: Any) -> Any:
        """Dot product."""
        if isinstance(a, sp.Matrix) and isinstance(b, sp.Matrix):
            return (a.T * b)[0, 0]
        return a * b

    def cross(self, a: Any, b: Any) -> Any:
        """Cross product (3D vectors)."""
        if isinstance(a, sp.Matrix) and isinstance(b, sp.Matrix):
            return a.cross(b)
        raise ValueError("Cross product requires 3D vectors as Matrix")

    def norm(self, x: Any) -> Any:
        """Euclidean norm."""
        if isinstance(x, sp.Matrix):
            return x.norm()
        return sp.Abs(x)


# Singleton instance for convenience
_backend = None


def get_sympy_backend() -> SymPyBackend:
    """Get the singleton SymPy backend instance."""
    global _backend
    if _backend is None:
        _backend = SymPyBackend()
    return _backend
