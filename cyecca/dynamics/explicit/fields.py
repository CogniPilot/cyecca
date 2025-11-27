"""Field descriptors for explicit ODE models.

Provides factory functions for creating typed state/input/parameter/output fields
in explicit ODE models where dx/dt = f(x, u, p, t).

Field types:
- state: Continuous state variable with time derivative
- input_var: Control input signal
- param: Constant parameter during simulation  
- output_var: Observable output (computed from states)
- algebraic_var: DAE constraint variable (for semi-explicit DAEs)
- dependent_var: Computed quantity (not integrated)
- quadrature_var: Accumulated quantity (cost functions)
- discrete_state: Event-updated state
- discrete_var: Integer/boolean discrete variable
- event_indicator: Zero-crossing event detector
"""

from dataclasses import field
from typing import Any

__all__ = [
    "VarDescriptor",
    "state",
    "input_var",
    "param",
    "output_var",
    "algebraic_var",
    "dependent_var",
    "quadrature_var",
    "discrete_state",
    "discrete_var",
    "event_indicator",
]


class VarDescriptor:
    """Descriptor for explicit ODE variables."""

    def __init__(self, var_type: str, shape: int = 1, default: Any = None, desc: str = ""):
        """Initialize variable descriptor.

        Args:
            var_type: Type of variable ('state', 'input', 'parameter', 'output', etc.)
            shape: Shape of variable (1 for scalar, N for vector)
            default: Default value
            desc: Human-readable description
        """
        self.var_type = var_type
        self.shape = shape
        self.default = default if default is not None else (0.0 if shape == 1 else [0.0] * shape)
        self.desc = desc


def state(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Declare a state variable (continuous, has time derivative).

    States are integrated by the ODE solver. Define their derivatives
    using model.ode().

    Args:
        shape: Number of elements (1 for scalar, N for vector). Default: 1
        default: Default initial value
        desc: Human-readable description

    Returns:
        Field descriptor for use in @explicit dataclass

    Example:
        >>> from cyecca.dynamics.explicit import explicit, state
        >>> @explicit
        ... class Model:
        ...     position: float = state(desc="position (m)")
        ...     velocity: float = state(desc="velocity (m/s)")
        >>> m = Model.numeric()
        >>> m.position
        0.0
    """
    return field(default_factory=lambda: VarDescriptor("state", shape=shape, default=default, desc=desc))


def input_var(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create an input variable field (control signal).

    Inputs are external signals that can be set during simulation.

    Args:
        shape: Dimension of the input (1 for scalar, N for vector)
        default: Default numeric value
        desc: Human-readable description

    Example:
        >>> from cyecca.dynamics.explicit import explicit, state, input_var
        >>> @explicit
        ... class Model:
        ...     x: float = state()
        ...     thrust: float = input_var(desc="thrust command (N)")
        >>> m = Model.numeric()
        >>> m.thrust
        0.0
    """
    return field(default_factory=lambda: VarDescriptor("input", shape=shape, default=default, desc=desc))


def param(default: float = 0.0, shape: int = 1, desc: str = "") -> Any:
    """Declare a parameter (constant during simulation).

    Parameters are fixed values that don't change during integration.

    Args:
        default: Default parameter value. Default: 0.0
        shape: Number of elements (1 for scalar, N for vector). Default: 1
        desc: Human-readable description

    Returns:
        Field descriptor for use in @explicit dataclass

    Example:
        >>> from cyecca.dynamics.explicit import explicit, state, param
        >>> @explicit
        ... class Model:
        ...     x: float = state()
        ...     mass: float = param(default=1.0, desc="mass (kg)")
        ...     gravity: float = param(default=9.81, desc="gravity (m/s^2)")
        >>> m = Model.numeric()
        >>> m.mass
        1.0
        >>> m.gravity
        9.81
    """
    return field(default_factory=lambda: VarDescriptor("parameter", shape=shape, default=default, desc=desc))


def output_var(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create an output variable field (observable).

    Outputs are computed from states and inputs using model.output().

    Args:
        shape: Dimension (1 for scalar, N for vector)
        default: Default value
        desc: Human-readable description

    Example:
        >>> from cyecca.dynamics.explicit import explicit, state, output_var
        >>> @explicit
        ... class Model:
        ...     x: float = state()
        ...     v: float = state()
        ...     speed: float = output_var(desc="speed (m/s)")
        >>> m = Model.numeric()
        >>> m.speed
        0.0
    """
    return field(default_factory=lambda: VarDescriptor("output", shape=shape, default=default, desc=desc))


def algebraic_var(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create algebraic variable for DAE constraints: 0 = g(x, z_alg, u, p).

    Used for implicit constraints in semi-explicit DAE systems
    (contact forces, Lagrange multipliers, kinematic loops, etc.).

    Args:
        shape: Dimension (1 for scalar, N for vector)
        default: Initial guess for DAE solver
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor("algebraic", shape=shape, default=default, desc=desc))


def dependent_var(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create dependent variable: y = f_y(x, u, p) (computed, not stored).

    For quantities computed from states but not integrated (energy, forces, etc.).

    Args:
        shape: Dimension (1 for scalar, N for vector)
        default: Default value for initialization
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor("dependent", shape=shape, default=default, desc=desc))


def quadrature_var(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create quadrature state: dq/dt = integrand(x, u, p) (for cost functions).

    Used for tracking accumulated quantities (cost, energy, etc.) that don't
    feed back into dynamics.

    Args:
        shape: Dimension (1 for scalar, N for vector)
        default: Initial value q(0)
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor("quadrature", shape=shape, default=default, desc=desc))


def discrete_state(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create discrete state z(t_e): updated only at events, constant between.

    For variables that jump at events (bounce counter, mode switches, etc.).

    Args:
        shape: Dimension (1 for scalar, N for vector)
        default: Initial value z(t_0)
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor("discrete_state", shape=shape, default=default, desc=desc))


def discrete_var(default: int = 0, desc: str = "") -> Any:
    """Create discrete variable m(t_e): integer/boolean updated at events.

    For discrete-valued quantities (flags, modes, counters).

    Args:
        default: Initial integer value
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor("discrete_var", shape=1, default=float(default), desc=desc))


def event_indicator(shape: int = 1, desc: str = "") -> Any:
    """Create event indicator c: event occurs when c crosses zero.

    Zero-crossing detection for hybrid systems.

    Args:
        shape: Number of indicators
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor("event_indicator", shape=shape, default=0.0, desc=desc))
