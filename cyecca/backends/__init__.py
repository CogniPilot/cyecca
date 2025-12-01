"""
Backend implementations for code generation and execution.

Backends convert the IR to executable code in different frameworks:
- CasADi: Fast numerical optimization and simulation
- SymPy: Symbolic manipulation and analysis
- JAX: Auto-diff and GPU acceleration (future)
- Algorithm: Numeric algorithm section execution
"""

from cyecca.backends.base import Backend
from cyecca.backends.casadi import CasadiBackend
from cyecca.backends.algorithm import (
    AlgorithmExecutor,
    NumericAlgorithmExecutor,
    ExecutionContext,
    execute_algorithm,
)

# SymPy backend is optional (requires sympy package)
try:
    from cyecca.backends.sympy import SympyBackend

    __all__ = [
        "Backend",
        "CasadiBackend",
        "SympyBackend",
        "AlgorithmExecutor",
        "NumericAlgorithmExecutor",
        "ExecutionContext",
        "execute_algorithm",
    ]
except ImportError:
    __all__ = [
        "Backend",
        "CasadiBackend",
        "AlgorithmExecutor",
        "NumericAlgorithmExecutor",
        "ExecutionContext",
        "execute_algorithm",
    ]
