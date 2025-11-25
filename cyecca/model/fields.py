"""Field creation helpers for the modeling framework.

Provides factory functions for creating dataclass fields with proper metadata
for different variable types (states, inputs, parameters, etc.).
"""

from dataclasses import field
from typing import Union

__all__ = [
    "state",
    "algebraic_var",
    "dependent_var",
    "quadrature_var",
    "discrete_state",
    "discrete_var",
    "event_indicator",
    "param",
    "input_var",
    "output_var",
]


def state(dim: int = 1, default: Union[float, list, None] = None, desc: str = ""):
    """Create a continuous state variable field (dx/dt in equations).

    Parameters
    ----------
    dim : int, optional
        Dimension (1 for scalar, >1 for vector), default=1
    default : float, list, or None, optional
        Default initial value (scalar or list)
    desc : str, optional
        Human-readable description

    Returns
    -------
    field
        Dataclass field with state metadata

    Examples
    --------
    >>> import casadi as ca
    >>> from cyecca.model import state, symbolic
    >>> @symbolic
    ... class States:
    ...     x: ca.SX = state(1, 0.0, "position (m)")
    ...     v: ca.SX = state(1, 1.0, "velocity (m/s)")
    >>> s = States.numeric()
    >>> s.x
    0.0
    >>> s.v
    1.0
    """
    if default is None:
        default = 0.0 if dim == 1 else [0.0] * dim
    elif isinstance(default, (int, float)) and dim > 1:
        default = [float(default)] * dim
    return field(
        default=None,
        metadata={"dim": dim, "default": default, "desc": desc, "type": "state"},
    )


def algebraic_var(
    dim: int = 1, default: Union[float, list, None] = None, desc: str = ""
):
    """Create algebraic variable for DAE constraints: 0 = g(x, z_alg, u, p).

    Used for implicit constraints in DAE systems (contact forces, Lagrange
    multipliers, kinematic loops, etc.).

    Args:
        dim: Dimension
        default: Initial guess for DAE solver
        desc: Description
    """
    if default is None:
        default = 0.0 if dim == 1 else [0.0] * dim
    elif isinstance(default, (int, float)) and dim > 1:
        default = [float(default)] * dim
    return field(
        default=None,
        metadata={"dim": dim, "default": default, "desc": desc, "type": "algebraic"},
    )


def dependent_var(
    dim: int = 1, default: Union[float, list, None] = None, desc: str = ""
):
    """Create dependent variable: y = f_y(x, u, p) (computed, not stored).

    For quantities computed from states but not integrated (energy, forces, etc.).

    Args:
        dim: Dimension
        default: Default value for initialization
        desc: Description
    """
    if default is None:
        default = 0.0 if dim == 1 else [0.0] * dim
    elif isinstance(default, (int, float)) and dim > 1:
        default = [float(default)] * dim
    return field(
        default=None,
        metadata={"dim": dim, "default": default, "desc": desc, "type": "dependent"},
    )


def quadrature_var(
    dim: int = 1, default: Union[float, list, None] = None, desc: str = ""
):
    """Create quadrature state: dq/dt = integrand(x, u, p) (for cost functions).

    Used for tracking accumulated quantities (cost, energy, etc.) that don't
    feed back into dynamics.

    Args:
        dim: Dimension
        default: Initial value q(0)
        desc: Description
    """
    if default is None:
        default = 0.0 if dim == 1 else [0.0] * dim
    elif isinstance(default, (int, float)) and dim > 1:
        default = [float(default)] * dim
    return field(
        default=None,
        metadata={"dim": dim, "default": default, "desc": desc, "type": "quadrature"},
    )


def discrete_state(
    dim: int = 1, default: Union[float, list, None] = None, desc: str = ""
):
    """Create discrete state z(tₑ): updated only at events, constant between.

    For variables that jump at events (bounce counter, mode switches, etc.).

    Args:
        dim: Dimension
        default: Initial value z(t₀)
        desc: Description
    """
    if default is None:
        default = 0.0 if dim == 1 else [0.0] * dim
    elif isinstance(default, (int, float)) and dim > 1:
        default = [float(default)] * dim
    return field(
        default=None,
        metadata={
            "dim": dim,
            "default": default,
            "desc": desc,
            "type": "discrete_state",
        },
    )


def discrete_var(default: int = 0, desc: str = ""):
    """Create discrete variable m(tₑ): integer/boolean updated at events.

    For discrete-valued quantities (flags, modes, counters).

    Args:
        default: Initial integer value
        desc: Description
    """
    return field(
        default=None,
        metadata={
            "dim": 1,
            "default": float(default),
            "desc": desc,
            "type": "discrete_var",
        },
    )


def event_indicator(dim: int = 1, desc: str = ""):
    """Create event indicator c: event occurs when c crosses zero.

    Zero-crossing detection for hybrid systems.

    Args:
        dim: Number of indicators
        desc: Description
    """
    return field(
        default=None,
        metadata={"dim": dim, "default": 0.0, "desc": desc, "type": "event_indicator"},
    )


def param(default: float, desc: str = ""):
    """Create a parameter field (time-independent constant).

    Args:
        default: Default numeric value
        desc: Description string

    Example:
        m: ca.SX = param(1.5, "mass (kg)")
        g: ca.SX = param(9.81, "gravity (m/s^2)")
    """
    return field(
        default=None,
        metadata={
            "dim": 1,
            "default": float(default),
            "desc": desc,
            "type": "parameter",
        },
    )


def input_var(dim: int = 1, default: Union[float, list, None] = None, desc: str = ""):
    """Create an input variable field (control signal).

    Args:
        dim: Dimension of the input (default: 1 for scalar)
        default: Default numeric value (scalar or list matching dim)
        desc: Description string

    Example:
        thrust: ca.SX = input_var(1, 0.0, "thrust command (N)")
        quaternion: ca.SX = input_var(4, desc="orientation [w,x,y,z]")
        steering: ca.SX = input_var(desc="steering angle (rad)")
    """
    if default is None:
        default = 0.0 if dim == 1 else [0.0] * dim
    return field(
        default=None,
        metadata={"dim": dim, "default": default, "desc": desc, "type": "input"},
    )


def output_var(dim: int = 1, default: Union[float, list, None] = None, desc: str = ""):
    """Create an output variable field (observable).

    Args:
        dim: Dimension
        default: Default value
        desc: Description

    Example:
        speed: ca.SX = output_var(1, 0.0, "ground speed (m/s)")
        forces: ca.SX = output_var(3, desc="force vector (N)")
    """
    if default is None:
        default = 0.0 if dim == 1 else [0.0] * dim
    elif isinstance(default, (int, float)) and dim > 1:
        default = [float(default)] * dim
    return field(
        default=None,
        metadata={"dim": dim, "default": default, "desc": desc, "type": "output"},
    )
