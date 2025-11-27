"""Backend abstraction layer for cyecca.

This module provides an abstraction layer for symbolic math backends,
allowing cyecca to work with different symbolic/autodiff frameworks.

Currently supported backends:
- casadi (default): CasADi symbolic framework
- sympy: SymPy symbolic mathematics (for analysis, not simulation)
- numpy: Pure numerical operations (no symbolic support)
- jax: JAX for GPU acceleration and autodiff ecosystem

Usage:
    from cyecca.backends import get_backend, set_default_backend
    
    # Get the default backend (casadi)
    backend = get_backend()
    
    # Get a specific backend
    backend = get_backend('casadi')
    backend = get_backend('sympy')
    backend = get_backend('numpy')
    backend = get_backend('jax')
    
    # Use in Model
    from cyecca.dynamics.explicit import Model
    model = Model(MyClass, backend='sympy')  # For symbolic analysis
"""

from .base import SymbolicBackend
from .casadi_backend import CasADiBackend
from .numpy_backend import NumPyBackend

# Try to import SymPy backend (optional dependency)
try:
    from .sympy_backend import SymPyBackend

    SYMPY_AVAILABLE = True
except ImportError:
    SymPyBackend = None
    SYMPY_AVAILABLE = False

# Try to import JAX backend (optional dependency)
try:
    from .jax_backend import JAXBackend

    JAX_AVAILABLE = True
except ImportError:
    JAXBackend = None
    JAX_AVAILABLE = False

# Registry of available backends
_BACKENDS = {
    "casadi": CasADiBackend,
    "numpy": NumPyBackend,
}

# Register SymPy if available
if SYMPY_AVAILABLE:
    _BACKENDS["sympy"] = SymPyBackend

# Register JAX if available
if JAX_AVAILABLE:
    _BACKENDS["jax"] = JAXBackend

_default_backend = "casadi"


def get_backend(name: str | None = None) -> SymbolicBackend:
    """Get a backend instance by name.

    Args:
        name: Backend name ('casadi', 'jax', etc.).
              If None, returns the default backend.

    Returns:
        SymbolicBackend instance

    Raises:
        ValueError: If the backend is not found
    """
    if name is None:
        name = _default_backend

    if name not in _BACKENDS:
        available = list(_BACKENDS.keys())
        raise ValueError(f"Backend '{name}' not found. Available backends: {available}")

    return _BACKENDS[name]()


def set_default_backend(name: str):
    """Set the default backend.

    Args:
        name: Backend name to set as default

    Raises:
        ValueError: If the backend is not found
    """
    global _default_backend

    if name not in _BACKENDS:
        available = list(_BACKENDS.keys())
        raise ValueError(f"Backend '{name}' not found. Available backends: {available}")

    _default_backend = name


def register_backend(name: str, backend_class: type):
    """Register a new backend.

    Args:
        name: Name for the backend
        backend_class: Backend class (must inherit from SymbolicBackend)
    """
    if not issubclass(backend_class, SymbolicBackend):
        raise TypeError(f"Backend class must inherit from SymbolicBackend, got {backend_class}")

    _BACKENDS[name] = backend_class


def list_backends() -> list[str]:
    """List available backend names."""
    return list(_BACKENDS.keys())


__all__ = [
    "SymbolicBackend",
    "CasADiBackend",
    "SymPyBackend",
    "NumPyBackend",
    "JAXBackend",
    "get_backend",
    "set_default_backend",
    "register_backend",
    "list_backends",
    "SYMPY_AVAILABLE",
    "JAX_AVAILABLE",
]
