"""CasADi backend implementation.

This module provides the CasADi implementation of the SymbolicBackend interface.
CasADi is the primary and default backend for cyecca.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import casadi as ca
import numpy as np

from .base import SymbolicBackend


class CasADiBackend(SymbolicBackend):
    """CasADi implementation of the symbolic backend.

    This is the primary backend for cyecca, providing full feature support
    including symbolic computation, differentiation, function creation,
    ODE/DAE integration, and optimization.
    """

    @property
    def name(self) -> str:
        return "casadi"

    # ========== Symbol Types ==========

    @property
    def SX(self) -> type:
        return ca.SX

    @property
    def MX(self) -> type:
        return ca.MX

    @property
    def DM(self) -> type:
        return ca.DM

    # ========== Symbol Creation ==========

    def sym(self, name: str, shape: int = 1, sym_type=None) -> Any:
        if sym_type is None:
            sym_type = ca.SX
        return sym_type.sym(name, shape)

    def zeros(self, rows: int, cols: int = 1, sym_type=None) -> Any:
        if sym_type is None:
            sym_type = ca.DM
        return sym_type.zeros(rows, cols)

    def ones(self, rows: int, cols: int = 1, sym_type=None) -> Any:
        if sym_type is None:
            sym_type = ca.DM
        return sym_type.ones(rows, cols)

    def eye(self, n: int, sym_type=None) -> Any:
        if sym_type is None:
            sym_type = ca.DM
        return sym_type.eye(n)

    # ========== Matrix Operations ==========

    def vertcat(self, *args) -> Any:
        return ca.vertcat(*args)

    def horzcat(self, *args) -> Any:
        return ca.horzcat(*args)

    def reshape(self, x: Any, shape: Tuple[int, int]) -> Any:
        return ca.reshape(x, shape[0], shape[1])

    def transpose(self, x: Any) -> Any:
        return ca.transpose(x)

    # ========== Differentiation ==========

    def jacobian(self, expr: Any, x: Any) -> Any:
        return ca.jacobian(expr, x)

    def gradient(self, expr: Any, x: Any) -> Any:
        return ca.gradient(expr, x)

    def hessian(self, expr: Any, x: Any) -> Tuple[Any, Any]:
        return ca.hessian(expr, x)

    def jtimes(self, expr: Any, x: Any, v: Any) -> Any:
        return ca.jtimes(expr, x, v)

    # ========== Function Creation ==========

    def function(
        self,
        name: str,
        inputs: List[Any],
        outputs: List[Any],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> ca.Function:
        if input_names is not None and output_names is not None:
            return ca.Function(name, inputs, outputs, input_names, output_names)
        elif input_names is not None:
            return ca.Function(name, inputs, outputs, input_names)
        else:
            return ca.Function(name, inputs, outputs)

    # ========== Math Functions ==========

    def sin(self, x: Any) -> Any:
        return ca.sin(x)

    def cos(self, x: Any) -> Any:
        return ca.cos(x)

    def tan(self, x: Any) -> Any:
        return ca.tan(x)

    def exp(self, x: Any) -> Any:
        return ca.exp(x)

    def log(self, x: Any) -> Any:
        return ca.log(x)

    def sqrt(self, x: Any) -> Any:
        return ca.sqrt(x)

    def power(self, x: Any, y: Any) -> Any:
        return ca.power(x, y)

    def abs(self, x: Any) -> Any:
        return ca.fabs(x)

    def atan2(self, y: Any, x: Any) -> Any:
        return ca.atan2(y, x)

    def asin(self, x: Any) -> Any:
        return ca.asin(x)

    def acos(self, x: Any) -> Any:
        return ca.acos(x)

    def tanh(self, x: Any) -> Any:
        return ca.tanh(x)

    # ========== Vector/Matrix Math ==========

    def sumsqr(self, x: Any) -> Any:
        return ca.sumsqr(x)

    def fmin(self, x: Any, y: Any) -> Any:
        return ca.fmin(x, y)

    def fmax(self, x: Any, y: Any) -> Any:
        return ca.fmax(x, y)

    # ========== Conditional Operations ==========

    def if_else(self, cond: Any, if_true: Any, if_false: Any) -> Any:
        return ca.if_else(cond, if_true, if_false)

    # ========== Numeric Conversion ==========

    def to_numpy(self, x: Any) -> np.ndarray:
        return np.array(ca.DM(x))

    def from_numpy(self, x: np.ndarray) -> Any:
        return ca.DM(x)

    def is_symbolic(self, x: Any) -> bool:
        return isinstance(x, (ca.SX, ca.MX))

    # ========== Shape/Size Operations ==========

    def size(self, x: Any, dim: int = 0) -> int:
        if dim == 0:
            return x.size1()
        else:
            return x.size2()

    def numel(self, x: Any) -> int:
        return x.numel()

    # ========== Integration ==========

    def integrator(
        self,
        name: str,
        method: str,
        dae: Dict[str, Any],
        t0: float,
        tf: Union[float, Sequence[float]],
        options: Optional[Dict] = None,
    ) -> Any:
        if options is None:
            options = {}
        return ca.integrator(name, method, dae, t0, tf, options)

    # ========== Optimization ==========

    def nlpsol(
        self,
        name: str,
        solver: str,
        nlp: Dict[str, Any],
        options: Optional[Dict] = None,
    ) -> Any:
        if options is None:
            options = {}
        return ca.nlpsol(name, solver, nlp, options)

    # ========== Additional CasADi-Specific Methods ==========
    # These are provided for convenience but should not be relied upon
    # in backend-agnostic code.

    def substitute(self, expr: Any, old: Any, new: Any) -> Any:
        """Substitute old with new in expr."""
        return ca.substitute(expr, old, new)

    def depends_on(self, expr: Any, x: Any) -> bool:
        """Check if expr depends on x."""
        return ca.depends_on(expr, x)

    def solve(self, A: Any, b: Any) -> Any:
        """Solve linear system A @ x = b."""
        return ca.solve(A, b)

    def inv(self, A: Any) -> Any:
        """Matrix inverse."""
        return ca.inv(A)

    def det(self, A: Any) -> Any:
        """Matrix determinant."""
        return ca.det(A)

    def eig_symbolic(self, A: Any) -> Any:
        """Symbolic eigenvalue computation."""
        return ca.eig_symbolic(A)

    def mtimes(self, A: Any, B: Any) -> Any:
        """Matrix multiplication."""
        return ca.mtimes(A, B)

    def norm_2(self, x: Any) -> Any:
        """Euclidean norm."""
        return ca.norm_2(x)

    def norm_inf(self, x: Any) -> Any:
        """Infinity norm."""
        return ca.norm_inf(x)

    def norm_1(self, x: Any) -> Any:
        """1-norm."""
        return ca.norm_1(x)

    def dot(self, a: Any, b: Any) -> Any:
        """Dot product."""
        return ca.dot(a, b)

    def cross(self, a: Any, b: Any) -> Any:
        """Cross product."""
        return ca.cross(a, b)

    def diag(self, x: Any) -> Any:
        """Diagonal matrix or extract diagonal."""
        return ca.diag(x)

    def trace(self, x: Any) -> Any:
        """Matrix trace."""
        return ca.trace(x)

    def sum1(self, x: Any) -> Any:
        """Sum along columns (axis 0)."""
        return ca.sum1(x)

    def sum2(self, x: Any) -> Any:
        """Sum along rows (axis 1)."""
        return ca.sum2(x)


# Singleton instance for convenience
_backend = CasADiBackend()


def get_casadi_backend() -> CasADiBackend:
    """Get the singleton CasADi backend instance."""
    return _backend
