"""Shared field descriptors for both explicit and implicit models.

Provides a unified VarDescriptor class and field factory functions
that work across both modeling paradigms.
"""

from dataclasses import field
from typing import Any

__all__ = [
    "VarDescriptor",
    "var",
    "param",
    # Legacy/explicit model support
    "time",
    "state",
    "alg",
    "algebraic_var",
    "input_var",
    "output_var",
    "dependent_var",
    "quadrature_var",
    "discrete_state",
    "discrete_var",
    "event_indicator",
]


class VarDescriptor:
    """Unified descriptor for both explicit ODE and implicit DAE variables."""
    
    def __init__(self, var_type: str, shape: int = 1, default: Any = None, desc: str = ""):
        """Initialize variable descriptor.
        
        Args:
            var_type: Type of variable ('var', 'state', 'alg', 'input', 'parameter', 'output', etc.)
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
        >>> @implicit
        >>> class Pendulum:
        >>>     theta: float = var()   # Becomes state (has .dot() in equations)
        >>>     omega: float = var()   # Becomes state (has .dot() in equations)
        >>>     g: float = param(default=9.81)
    """
    return field(default_factory=lambda: VarDescriptor('var', shape=shape, default=default, desc=desc))


def time(desc: str = "simulation time") -> Any:
    """Declare the time variable (for legacy/explicit models).
    
    Note: In the new implicit Model class, time is built-in as model.t.
    This function is kept for backward compatibility with explicit models.
    
    Args:
        desc: Human-readable description. Default: "simulation time"
        
    Returns:
        Field descriptor for use in @implicit dataclass
    """
    return field(default_factory=lambda: VarDescriptor('time', shape=1, default=0.0, desc=desc))


def state(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Declare a state variable (continuous, has time derivative).
    
    Args:
        shape: Number of elements (1 for scalar, N for vector). Default: 1
        default: Default initial value (for explicit models)
        desc: Human-readable description
        
    Returns:
        Field descriptor for use in @explicit or @implicit dataclass
        
    Example:
        >>> @explicit
        >>> class Model:
        >>>     position: float = state(desc="position (m)")
        >>>     velocity: float = state(desc="velocity (m/s)")
    """
    return field(default_factory=lambda: VarDescriptor('state', shape=shape, default=default, desc=desc))


def alg(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Declare an algebraic variable (no time derivative, DAE constraint).
    
    Args:
        shape: Number of elements (1 for scalar, N for vector). Default: 1
        default: Initial guess for DAE solver
        desc: Human-readable description
        
    Returns:
        Field descriptor for use in @implicit dataclass
        
    Example:
        >>> @implicit
        >>> class Model:
        >>>     lambda_force: float = alg(desc="constraint force")
    """
    return field(default_factory=lambda: VarDescriptor('alg', shape=shape, default=default, desc=desc))


# Alias for explicit models
def algebraic_var(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create algebraic variable for DAE constraints: 0 = g(x, z_alg, u, p).

    Used for implicit constraints in DAE systems (contact forces, Lagrange
    multipliers, kinematic loops, etc.).

    Args:
        shape: Dimension (1 for scalar, N for vector)
        default: Initial guess for DAE solver
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor('algebraic', shape=shape, default=default, desc=desc))


def param(default: float = 0.0, shape: int = 1, desc: str = "") -> Any:
    """Declare a parameter (constant during simulation).
    
    Args:
        default: Default parameter value. Default: 0.0
        shape: Number of elements (1 for scalar, N for vector). Default: 1
        desc: Human-readable description
        
    Returns:
        Field descriptor for use in @explicit or @implicit dataclass
        
    Example:
        >>> @explicit
        >>> class Model:
        >>>     mass: float = param(default=1.0, desc="mass (kg)")
        >>>     gravity: float = param(default=9.81, desc="gravity (m/s^2)")
    """
    return field(default_factory=lambda: VarDescriptor('parameter', shape=shape, default=default, desc=desc))


def input_var(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create an input variable field (control signal).

    Args:
        shape: Dimension of the input (1 for scalar, N for vector)
        default: Default numeric value
        desc: Human-readable description

    Example:
        thrust: float = input_var(desc="thrust command (N)")
        quaternion: float = input_var(shape=4, desc="orientation [w,x,y,z]")
        steering: float = input_var(desc="steering angle (rad)")
    """
    return field(default_factory=lambda: VarDescriptor('input', shape=shape, default=default, desc=desc))


def output_var(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create an output variable field (observable).

    Args:
        shape: Dimension (1 for scalar, N for vector)
        default: Default value
        desc: Human-readable description

    Example:
        speed: float = output_var(desc="ground speed (m/s)")
        forces: float = output_var(shape=3, desc="force vector (N)")
    """
    return field(default_factory=lambda: VarDescriptor('output', shape=shape, default=default, desc=desc))


def dependent_var(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create dependent variable: y = f_y(x, u, p) (computed, not stored).

    For quantities computed from states but not integrated (energy, forces, etc.).

    Args:
        shape: Dimension (1 for scalar, N for vector)
        default: Default value for initialization
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor('dependent', shape=shape, default=default, desc=desc))


def quadrature_var(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create quadrature state: dq/dt = integrand(x, u, p) (for cost functions).

    Used for tracking accumulated quantities (cost, energy, etc.) that don't
    feed back into dynamics.

    Args:
        shape: Dimension (1 for scalar, N for vector)
        default: Initial value q(0)
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor('quadrature', shape=shape, default=default, desc=desc))


def discrete_state(shape: int = 1, default: Any = None, desc: str = "") -> Any:
    """Create discrete state z(t_e): updated only at events, constant between.

    For variables that jump at events (bounce counter, mode switches, etc.).

    Args:
        shape: Dimension (1 for scalar, N for vector)
        default: Initial value z(t_0)
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor('discrete_state', shape=shape, default=default, desc=desc))


def discrete_var(default: int = 0, desc: str = "") -> Any:
    """Create discrete variable m(t_e): integer/boolean updated at events.

    For discrete-valued quantities (flags, modes, counters).

    Args:
        default: Initial integer value
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor('discrete_var', shape=1, default=float(default), desc=desc))


def event_indicator(shape: int = 1, desc: str = "") -> Any:
    """Create event indicator c: event occurs when c crosses zero.

    Zero-crossing detection for hybrid systems.

    Args:
        shape: Number of indicators
        desc: Human-readable description
    """
    return field(default_factory=lambda: VarDescriptor('event_indicator', shape=shape, default=0.0, desc=desc))
