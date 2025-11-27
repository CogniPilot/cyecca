"""Field descriptors for implicit DAE models.

Provides factory functions for creating Modelica-style variable fields
in implicit DAE models where F(ẋ, x, z, p, t) = 0.

Field types:
- var: Generic variable (state or algebraic - inferred from .dot() usage)
- param: Constant parameter during simulation

The key difference from explicit models is that you don't declare whether
a variable is a state or algebraic. The model automatically infers this:
- Variables with .dot() called become states (differential variables)
- Variables without .dot() become algebraic (constraint variables)

This matches Modelica semantics where you write equations naturally and
the compiler determines causality.
"""

from dataclasses import field
from typing import Any

__all__ = [
    "VarDescriptor",
    "var",
    "param",
]


class VarDescriptor:
    """Descriptor for implicit DAE variables."""

    def __init__(self, var_type: str, shape: int = 1, default: Any = None, desc: str = ""):
        """Initialize variable descriptor.

        Args:
            var_type: Type of variable ('var' or 'parameter')
            shape: Shape of variable (1 for scalar, N for vector)
            default: Default value
            desc: Human-readable description
        """
        self.var_type = var_type
        self.shape = shape
        self.default = default if default is not None else (0.0 if shape == 1 else [0.0] * shape)
        self.desc = desc


def var(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Declare a variable (state or algebraic - automatically inferred).

    In implicit/Modelica-style models, you don't declare whether a variable
    is a state or algebraic. The model infers this based on usage:
    - Variables with .dot() called become states
    - Variables without .dot() become algebraic

    Args:
        shape: Number of elements (1 for scalar, N for vector). Default: 1
        default: Default initial value
        desc: Human-readable description

    Returns:
        Field descriptor for use in @implicit dataclass

    Example:
        >>> from cyecca.dynamics.implicit import implicit, var, param
        >>> @implicit
        ... class Pendulum:
        ...     theta: float = var()   # Becomes state (has .dot() in equations)
        ...     omega: float = var()   # Becomes state (has .dot() in equations)
        ...     g: float = param(default=9.81)
        >>> p = Pendulum.numeric()
        >>> p.theta
        0.0
    """
    return field(default_factory=lambda: VarDescriptor("var", shape=shape, default=default, desc=desc))


def param(default: float = 0.0, shape: int = 1, desc: str = "") -> Any:
    """Declare a parameter (constant during simulation).

    Parameters are fixed values that don't change during integration.

    Args:
        default: Default parameter value. Default: 0.0
        shape: Number of elements (1 for scalar, N for vector). Default: 1
        desc: Human-readable description

    Returns:
        Field descriptor for use in @implicit dataclass

    Example:
        >>> from cyecca.dynamics.implicit import implicit, var, param
        >>> @implicit
        ... class Model:
        ...     x: float = var()
        ...     mass: float = param(default=1.0, desc="mass (kg)")
        ...     gravity: float = param(default=9.81, desc="gravity (m/s^2)")
        >>> m = Model.numeric()
        >>> m.mass
        1.0
        >>> m.gravity
        9.81
    """
    return field(default_factory=lambda: VarDescriptor("parameter", shape=shape, default=default, desc=desc))
