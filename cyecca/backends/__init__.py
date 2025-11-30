"""
Backend implementations for code generation and execution.

Backends convert the IR to executable code in different frameworks:
- CasADi: Fast numerical optimization and simulation
- SymPy: Symbolic manipulation and analysis
- JAX: Auto-diff and GPU acceleration (future)
"""

from cyecca.backends.base import Backend
from cyecca.backends.casadi import CasadiBackend

# SymPy backend is optional (requires sympy package)
try:
    from cyecca.backends.sympy import SympyBackend

    __all__ = ["Backend", "CasadiBackend", "SympyBackend"]
except ImportError:
    __all__ = ["Backend", "CasadiBackend"]
