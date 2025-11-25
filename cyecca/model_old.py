"""Type-safe CasADi modeling framework with full hybrid DAE support.

Provides declarative API for building hybrid dynamical systems without dynamic
class generation. Full autocomplete and type safety throughout.

Features:
- Continuous states (x): dx/dt = f_x(...)
- Algebraic variables (z_alg): 0 = g(x, z_alg, u, p) for DAE constraints
- Dependent variables (dep): dep = f_dep(x, u, p) for computed quantities
- Quadrature states (q): dq/dt = f_q(x, u, p) for path integrals
- Discrete states (z): z⁺ = f_z(...) updated at events
- Discrete variables (m): m⁺ = f_m(...) for integers/booleans
- Event indicators (c): event when c crosses zero
- Inputs (u): control signals
- Parameters (p): time-independent constants
- Outputs (y): y = f_y(x, u, p) observables/diagnostics

Example:
    @symbolic
    class States:
        h: ca.SX = state(1, 10.0, "height (m)")
        v: ca.SX = state(1, 0.0, "velocity (m/s)")

    @symbolic
    class Inputs:
        thrust: ca.SX = input_var(desc="thrust (N)")

    @symbolic
    class Params:
        m: ca.SX = param(1.0, "mass (kg)")
        g: ca.SX = param(9.81, "gravity (m/s^2)")

    model = ModelSX.create(States, Inputs, Params)

    x = model.x()
    u = model.u()
    p = model.p()

    f_x = ca.vertcat(x.v, u.thrust / p.m - p.g)
    model.build(f_x=f_x, integrator='rk4')
"""

import copy
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Generic, TypeVar, Union

import casadi as ca
import numpy as np
from beartype import beartype

__all__ = [
    "symbolic",
    "state",
    "param",
    "input_var",
    "output_var",
    "algebraic_var",
    "dependent_var",
    "quadrature_var",
    "discrete_state",
    "discrete_var",
    "event_indicator",
    "ModelSX",
    "ModelMX",
]

# ============================================================================
# Field Creation Helpers
# ============================================================================


def state(dim: int = 1, default: Union[float, list, None] = None, desc: str = ""):
    """Create a continuous state variable field (dx/dt in equations).

    Args:
        dim: Dimension (1 for scalar, >1 for vector)
        default: Default value (scalar or list)
        desc: Description string

    Returns:
        dataclass field with metadata

    Example:
        p: ca.SX = state(3, [0, 0, 10], "position (m)")
        v: ca.SX = state(1, 0.0, "velocity (m/s)")
        w: ca.SX = state(3, desc="angular velocity")  # defaults to zeros
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


# ============================================================================
# Symbolic Dataclass Decorator
# ============================================================================


def compose_states(*state_types):
    """Compose multiple state types into a single combined state type.

    This programmatically creates a new state class that includes all fields
    from the input state types, preserving type safety and field metadata.

    Args:
        *state_types: State type classes to combine

    Returns:
        New combined state class with all fields from input types

    Example:
        >>> ClosedLoopStates = compose_states(PlantStates, ControllerStates)
    """
    from dataclasses import MISSING, dataclass
    from dataclasses import fields as get_fields

    # Collect all fields from input types, preserving metadata
    combined_annotations = {}
    combined_fields = {}

    for state_type in state_types:
        # Ensure the type is a dataclass
        if not hasattr(state_type, "__dataclass_fields__"):
            continue

        # Get fields from the dataclass
        for fld in get_fields(state_type):
            if fld.name.startswith("_"):
                continue  # Skip private fields
            combined_annotations[fld.name] = fld.type
            # Create a new field with the same metadata
            if fld.default is not MISSING:
                combined_fields[fld.name] = field(
                    default=fld.default, metadata=fld.metadata
                )
            elif fld.default_factory is not MISSING:
                combined_fields[fld.name] = field(
                    default_factory=fld.default_factory, metadata=fld.metadata
                )
            else:
                combined_fields[fld.name] = field(default=None, metadata=fld.metadata)

    # Create new class dynamically
    combined_class = type(
        "ComposedStates",
        (),
        {"__annotations__": combined_annotations, **combined_fields},
    )

    # Apply @dataclass first, then @symbolic
    combined_class = dataclass(combined_class)
    return symbolic(combined_class)


def symbolic(cls):
    """Combined decorator: applies @dataclass and adds CasADi symbolic methods.

    Adds methods:
        - cls.symbolic(sym_type=ca.SX) -> instance with symbolic variables
        - cls.numeric() -> instance with numeric defaults
        - instance.as_vec() -> ca.SX/MX column vector
        - cls.from_vec(vec) -> instance from vector
        - instance.size1() -> number of rows
        - instance.size2() -> number of columns (always 1)

    Usage:
        @symbolic
        class States:
            x: ca.SX = state(1, 0.0, "position")
            v: ca.SX = state(1, 0.0, "velocity")

        # Create symbolic instance
        x_sym = States.symbolic()  # x_sym.x is ca.SX.sym('x')

        # Create numeric instance
        x0 = States.numeric()  # x0.x == 0.0

        # Convert to vector
        vec = x_sym.as_vec()  # ca.vertcat(x_sym.x, x_sym.v)

        # Access instances directly
        x = model.x  # Access as attribute
    """
    # Apply @dataclass if not already applied
    if not hasattr(cls, "__dataclass_fields__"):
        cls = dataclass(cls)

    # Extract field metadata
    field_info = {}
    for f in fields(cls):
        meta = f.metadata or {}
        dim = meta.get("dim", 1)
        raw_default = meta.get("default", None)

        # Ensure default is properly formatted for dimension
        if raw_default is None:
            default = 0.0 if dim == 1 else [0.0] * dim
        elif isinstance(raw_default, (int, float)):
            default = float(raw_default) if dim == 1 else [float(raw_default)] * dim
        else:
            default = raw_default

        desc = meta.get("desc", "")
        var_type = meta.get("type", "unknown")
        field_info[f.name] = {
            "dim": dim,
            "default": default,
            "desc": desc,
            "type": var_type,
            "casadi_type": f.type,
        }

    # Create symbolic instance factory
    @classmethod
    def create_symbolic(cls_obj, sym_type=ca.SX):
        """Create instance with symbolic CasADi variables."""
        kwargs = {}
        for name, info in field_info.items():
            dim = info["dim"]
            if dim == 1:
                kwargs[name] = sym_type.sym(name)
            else:
                kwargs[name] = sym_type.sym(name, dim)
        return cls_obj(**kwargs)

    # Create numeric instance factory
    @classmethod
    def create_numeric(cls_obj):
        """Create instance with numeric default values."""
        kwargs = {}
        for name, info in field_info.items():
            default = info["default"]
            if isinstance(default, (list, tuple)):
                kwargs[name] = np.array(default, dtype=float)
            else:
                kwargs[name] = float(default)
        return cls_obj(**kwargs)

    # Vector conversion
    def as_vec(self):
        """Convert to CasADi column vector."""
        parts = []
        for f in fields(self.__class__):
            val = getattr(self, f.name)
            parts.append(ca.vec(val))
        return ca.vertcat(*parts) if parts else ca.DM.zeros(0, 1)

    # Reconstruct from vector
    @classmethod
    def from_vec(cls_obj, vec):
        """Reconstruct from CasADi vector (works with both numeric and symbolic).

        For numeric vectors (DM), converts to float/numpy arrays.
        For symbolic vectors (SX/MX), preserves symbolic expressions.
        """
        # Handle CasADi Function dict outputs
        if isinstance(vec, dict):
            if len(vec) == 1:
                vec = list(vec.values())[0]
            else:
                raise ValueError(
                    f"from_vec() received dict with multiple outputs: {list(vec.keys())}"
                )

        # Check if this is a symbolic type (SX or MX)
        is_symbolic = hasattr(vec, "__class__") and vec.__class__.__name__ in (
            "SX",
            "MX",
        )

        # Convert to proper vector (only for numeric)
        if not is_symbolic and not hasattr(vec, "shape"):
            vec = ca.DM(vec)

        # Ensure column vector
        if hasattr(vec, "shape") and len(vec.shape) == 2 and vec.shape[1] != 1:
            vec = vec.T

        kwargs = {}
        offset = 0
        for f in fields(cls_obj):
            name = f.name
            dim = field_info[name]["dim"]

            if dim == 1:
                val = vec[offset]
                # Convert DM scalar to float (numeric only)
                if not is_symbolic and hasattr(val, "numel") and val.numel() == 1:
                    val = float(val)
                kwargs[name] = val
                offset += 1
            else:
                val = vec[offset : offset + dim]
                # Convert DM vector to numpy array (numeric only)
                if (
                    not is_symbolic
                    and hasattr(val, "__class__")
                    and val.__class__.__name__ == "DM"
                ):
                    val = np.array(val, dtype=float).flatten()
                kwargs[name] = val
                offset += dim

        return cls_obj(**kwargs)

    # Matrix reconstruction (for trajectories)
    @classmethod
    def from_matrix(cls_obj, matrix):
        """Create instances from matrix where columns are timesteps.

        Args:
            matrix: CasADi matrix or NumPy array with shape (n_vars, n_steps)

        Returns:
            List of instances, one per timestep
        """
        # Convert to NumPy
        if hasattr(matrix, "__class__") and matrix.__class__.__name__ in (
            "SX",
            "MX",
            "DM",
        ):
            matrix = np.array(ca.DM(matrix))
        else:
            matrix = np.asarray(matrix)

        n_steps = matrix.shape[1]
        instances = []

        for i in range(n_steps):
            col = matrix[:, i : i + 1]
            instances.append(cls_obj.from_vec(col))

        return instances

    # CasADi compatibility methods
    def size1(self):
        """Get number of rows when converted to vector."""
        return self.as_vec().size1()

    def size2(self):
        """Get number of columns (always 1)."""
        return 1

    # Custom repr with descriptions
    def custom_repr(self):
        """String representation with field descriptions."""
        parts = []
        for f in fields(self.__class__):
            val = getattr(self, f.name)
            desc = field_info[f.name].get("desc", "")
            if desc:
                parts.append(f"{f.name}={val!r}  # {desc}")
            else:
                parts.append(f"{f.name}={val!r}")
        return f"{cls.__name__}(" + ", ".join(parts) + ")"

    # Attach methods to class
    cls.symbolic = create_symbolic
    cls.numeric = create_numeric
    cls.as_vec = as_vec
    cls.from_vec = from_vec
    cls.from_matrix = from_matrix
    cls.size1 = size1
    cls.size2 = size2
    cls.__repr__ = custom_repr
    cls._field_info = field_info

    return cls


# ============================================================================
# Type-Safe Model with Full Hybrid DAE Support
# ============================================================================

TState = TypeVar("TState")
TInput = TypeVar("TInput")
TParam = TypeVar("TParam")
TOutput = TypeVar("TOutput")
TAlgebraic = TypeVar("TAlgebraic")
TDependent = TypeVar("TDependent")
TQuadrature = TypeVar("TQuadrature")
TDiscreteState = TypeVar("TDiscreteState")
TDiscreteVar = TypeVar("TDiscreteVar")
TEventIndicator = TypeVar("TEventIndicator")


@beartype
class ModelSX(Generic[TState, TInput, TParam]):
    """Type-safe SX model with full hybrid DAE support.

    Supports:
    - Continuous states (x): dx/dt = f_x(...)
    - Algebraic variables (z_alg): 0 = g(x, z_alg, u, p)
    - Outputs (y): y = f_y(x, u, p)
    - Dependent variables (dep): dep = f_dep(x, u, p)
    - Quadratures (q): dq/dt = f_q(x, u, p)
    - Discrete states (z): z⁺ = f_z(...) at events
    - Discrete variables (m): m⁺ = f_m(...) at events
    - Event indicators (c): event when c = 0

    Example:
        model = ModelSX.create(States, Inputs, Params)

        x = model.x()
        u = model.u()
        p = model.p()

        f_x = ca.vertcat(x.v, u.thrust / p.m - p.g)
        model.build(f_x=f_x, integrator='rk4')
    """

    # Required types
    x: TState
    u: TInput
    p: TParam
    x0: TState
    u0: TInput
    p0: TParam

    # Optional types (set if provided)
    z_alg: Any = None  # Algebraic variables
    y: Any = None  # Dependent variables
    q: Any = None  # Quadrature states
    z: Any = None  # Discrete states
    m: Any = None  # Discrete variables
    c: Any = None  # Event indicators
    out: Any = None  # Outputs

    def __init__(
        self,
        state_type: type[TState],
        input_type: type[TInput],
        param_type: type[TParam],
        output_type: type[TOutput] | None = None,
        algebraic_type: type[TAlgebraic] | None = None,
        dependent_type: type[TDependent] | None = None,
        quadrature_type: type[TQuadrature] | None = None,
        discrete_state_type: type[TDiscreteState] | None = None,
        discrete_var_type: type[TDiscreteVar] | None = None,
        event_indicator_type: type[TEventIndicator] | None = None,
    ):
        """Initialize typed hybrid DAE model.

        Prefer using ModelSX.create() for automatic type inference.
        """
        self._sym = ca.SX
        self.state_type = state_type
        self.input_type = input_type
        self.param_type = param_type
        self.output_type = output_type
        self.algebraic_type = algebraic_type
        self.dependent_type = dependent_type
        self.quadrature_type = quadrature_type
        self.discrete_state_type = discrete_state_type
        self.discrete_var_type = discrete_var_type
        self.event_indicator_type = event_indicator_type

        # Create required symbolic instances
        self.x = state_type.symbolic(ca.SX)
        self.u = input_type.symbolic(ca.SX)
        self.p = param_type.symbolic(ca.SX)

        # Create required numeric defaults
        self.x0 = state_type.numeric()
        self.u0 = input_type.numeric()
        self.p0 = param_type.numeric()

        # Create optional types if provided
        if output_type:
            self.y = output_type.symbolic(ca.SX)
            self.y0 = output_type.numeric()

        if algebraic_type:
            self.z_alg = algebraic_type.symbolic(ca.SX)
            self.z_alg0 = algebraic_type.numeric()

        if dependent_type:
            self.dep = dependent_type.symbolic(ca.SX)
            self.dep0 = dependent_type.numeric()

        if quadrature_type:
            self.q = quadrature_type.symbolic(ca.SX)
            self.q0 = quadrature_type.numeric()

        if discrete_state_type:
            self.z = discrete_state_type.symbolic(ca.SX)
            self.z0 = discrete_state_type.numeric()

        if discrete_var_type:
            self.m = discrete_var_type.symbolic(ca.SX)
            self.m0 = discrete_var_type.numeric()

        if event_indicator_type:
            self.c = event_indicator_type.symbolic(ca.SX)

        # Create connection helpers for autocomplete
        # These are separate from the actual u/y objects used for building
        class ConnectionHelper:
            """Helper for accessing signals in connect() calls."""

            def __init__(self, prefix: str, obj):
                self._prefix = prefix
                self._obj = obj

            def __getattr__(self, attr: str):
                if hasattr(self._obj, attr):
                    return ModelSX.SignalRef(self._prefix, attr)
                raise AttributeError(f"Signal '{attr}' not found")

        self.inputs = ConnectionHelper("u", self.u)
        if output_type:
            self.outputs = ConnectionHelper("y", self.y)

    @classmethod
    def create(
        cls,
        state_type: type[TState],
        input_type: type[TInput],
        param_type: type[TParam],
        output_type: type[TOutput] = None,
        **kwargs,
    ):
        """Create a fully-typed model instance with automatic type inference.

        This factory method ensures proper type inference for IDE autocomplete.

        Args:
            state_type: Dataclass decorated with @symbolic
            input_type: Dataclass decorated with @symbolic
            param_type: Dataclass decorated with @symbolic
            output_type: Optional output type dataclass
            **kwargs: Optional types (algebraic_type, dependent_type, etc.)

        Returns:
            Fully-typed ModelSX instance

        Usage:
            model = ModelSX.create(States, Inputs, Params, Outputs)
            # IDE now knows exact types for autocomplete!

            x = model.x()  # x has full autocomplete
            u = model.u()  # u has full autocomplete
            p = model.p()  # p has full autocomplete
        """
        return cls(
            state_type, input_type, param_type, output_type=output_type, **kwargs
        )

    @classmethod
    def compose(
        cls,
        submodels: dict[str, "ModelSX"],
        state_type: type[TState] = None,
        input_type: type[TInput] = None,
        param_type: type[TParam] = None,
        output_type: type[TOutput] = None,
    ):
        """Compose multiple submodels into a single parent model.

        Creates a parent model with submodels already attached, ready for connections.
        Automatically composes states from all submodels if not specified.

        Args:
            submodels: Dictionary of {name: model} pairs
            state_type: Parent state type (auto-composed from submodels if None)
            input_type: Parent input type (empty if None)
            param_type: Parent parameter type (empty if None)
            output_type: Parent output type (empty if None)

        Returns:
            Parent model with submodels attached as attributes

        Example:
            >>> plant = sportcub()
            >>> controller = autolevel_controller()
            >>> parent = ModelSX.compose({"plant": plant, "controller": controller})
            >>> parent.connect(controller.u.q, plant.x.r)
        """
        # Auto-compose state type if not provided
        if state_type is None:
            state_types = [sub.state_type for sub in submodels.values()]
            state_type = compose_states(*state_types)

        # Create empty types if not provided
        if input_type is None:

            @symbolic
            class EmptyInputs:
                pass

            input_type = EmptyInputs

        if param_type is None:

            @symbolic
            class EmptyParams:
                pass

            param_type = EmptyParams

        if output_type is None:

            @symbolic
            class EmptyOutputs:
                pass

            output_type = EmptyOutputs

        # Create parent model
        parent = cls.create(state_type, input_type, param_type, output_type)

        # Add all submodels and create proxies for connection API
        for name, submodel in submodels.items():
            parent.add_submodel(name, submodel)
            # Submodel proxy is already created by add_submodel

        return parent

    @beartype
    def build(
        self,
        f_x: ca.SX | ca.MX,
        f_y: ca.SX | ca.MX | None = None,
        f_dep: ca.SX | ca.MX | None = None,
        f_alg: ca.SX | ca.MX | None = None,
        f_q: ca.SX | ca.MX | None = None,
        f_z: ca.SX | ca.MX | None = None,
        f_m: ca.SX | ca.MX | None = None,
        f_c: ca.SX | ca.MX | None = None,
        integrator: str = "rk4",
        integrator_options: dict = None,
    ):
        """Build model with all evolution functions.

        Args:
            f_x: Continuous dynamics dx/dt (required)
            f_y: Output expressions y = f_y(x, u, p)
            f_dep: Dependent variable expressions (less commonly used)
            f_alg: Algebraic constraints (for DAE)
            f_q: Quadrature dynamics dq/dt
            f_z: Discrete state update z⁺
            f_m: Discrete variable update m⁺
            f_c: Event indicator functions
            integrator: Integration method ("rk4", "euler", "idas")
            integrator_options: Integrator settings (e.g., {'N': 10})
        """
        if integrator_options is None:
            integrator_options = {}

        # Build f_x (required)
        self._build_f_x(f_x)

        # Build optional functions
        if f_y is not None:
            self._build_f_y(f_y)
        if f_dep is not None:
            self._build_f_dep(f_dep)
        if f_alg is not None:
            self._build_f_alg(f_alg)
        if f_q is not None:
            self._build_f_q(f_q)
        if f_z is not None:
            self._build_f_z(f_z)
        if f_m is not None:
            self._build_f_m(f_m)
        if f_c is not None:
            self._build_f_c(f_c)

        # Build integrator
        if integrator == "rk4":
            self._build_rk4_integrator(integrator_options)
        elif integrator == "euler":
            self._build_euler_integrator()
        elif integrator == "idas":
            self._build_idas_integrator(integrator_options)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        # Build index maps
        self._build_index_maps()

    def _build_f_x(self, f_x_expr):
        """Build continuous dynamics function.

        Note: The 'dep' input here refers to dependent variables (if present),
        not outputs. Dependent variables are computed values that feed into
        dynamics but are not integrated states.
        """
        inputs = [self.x.as_vec()]
        input_names = ["x"]

        if hasattr(self, "dep") and self.dep is not None:
            inputs.append(self.dep.as_vec())
            input_names.append("dep")
        if self.z is not None:
            inputs.append(self.z.as_vec())
            input_names.append("z")
        if self.m is not None:
            inputs.append(self.m.as_vec())
            input_names.append("m")

        inputs.append(self.u.as_vec())
        inputs.append(self.p.as_vec())
        input_names.extend(["u", "p"])

        self.f_x = ca.Function("f_x", inputs, [f_x_expr], input_names, ["dx_dt"])

    def _build_f_y(self, f_y_expr):
        """Build output function.

        Outputs (y) are observables/diagnostics computed from states.
        """
        inputs, names = self._get_standard_inputs()
        self.f_y = ca.Function("f_y", inputs, [f_y_expr], names, ["y"])

    def _build_f_dep(self, f_dep_expr):
        """Build dependent variable function."""
        inputs, names = self._get_standard_inputs(include_dep=False)
        self.f_dep = ca.Function("f_dep", inputs, [f_dep_expr], names, ["dep"])

    def _build_f_alg(self, f_alg_expr):
        """Build algebraic constraint function: 0 = g(x, z_alg, u, p)."""
        inputs = [
            self.x.as_vec(),
            self.z_alg.as_vec(),
            self.u.as_vec(),
            self.p.as_vec(),
        ]
        names = ["x", "z_alg", "u", "p"]
        self.f_alg = ca.Function("f_alg", inputs, [f_alg_expr], names, ["residual"])

    def _build_f_q(self, f_q_expr):
        """Build quadrature dynamics."""
        inputs, names = self._get_standard_inputs()
        self.f_q = ca.Function("f_q", inputs, [f_q_expr], names, ["dq_dt"])

    def _build_f_z(self, f_z_expr):
        """Build discrete state update function."""
        inputs, names = self._get_standard_inputs()
        self.f_z = ca.Function("f_z", inputs, [f_z_expr], names, ["z_plus"])

    def _build_f_m(self, f_m_expr):
        """Build discrete variable update function."""
        inputs, names = self._get_standard_inputs()
        self.f_m = ca.Function("f_m", inputs, [f_m_expr], names, ["m_plus"])

    def _build_f_c(self, f_c_expr):
        """Build event indicator function."""
        inputs, names = self._get_standard_inputs()
        self.f_c = ca.Function("f_c", inputs, [f_c_expr], names, ["indicators"])

    def _get_standard_inputs(self, include_dep=True):
        """Get standard input list for functions."""
        inputs = [self.x.as_vec()]
        names = ["x"]

        if include_dep and hasattr(self, "dep") and self.dep is not None:
            inputs.append(self.dep.as_vec())
            names.append("dep")
        if self.z is not None:
            inputs.append(self.z.as_vec())
            names.append("z")
        if self.m is not None:
            inputs.append(self.m.as_vec())
            names.append("m")

        inputs.append(self.u.as_vec())
        inputs.append(self.p.as_vec())
        names.extend(["u", "p"])

        return inputs, names

    def _build_rk4_integrator(self, options: dict):
        """Build RK4 integrator."""
        from . import integrators

        dt_sym = ca.SX.sym("dt")
        N = options.get("N", 10)

        rk4_step = integrators.rk4(self.f_x, dt_sym, name="rk4", N=N)

        # Extract inputs to handle SX vs MX
        if rk4_step.is_a("SXFunction"):
            xin = rk4_step.sx_in(0)
            uin = rk4_step.sx_in(1)
            pin = rk4_step.sx_in(2)
        else:
            xin = rk4_step.mx_in(0)
            uin = rk4_step.mx_in(1)
            pin = rk4_step.mx_in(2)

        self.f_step = ca.Function(
            "f_step",
            [xin, uin, pin, dt_sym],
            [rk4_step(xin, uin, pin, dt_sym)],
            ["x", "u", "p", "dt"],
            ["x_next"],
        )

    def _build_euler_integrator(self):
        """Build Euler integrator."""
        dt_sym = ca.SX.sym("dt")

        # For composed models, use the combined state size
        if hasattr(self, "_composed") and self._composed:
            x_sym = ca.SX.sym("x", self._total_composed_states)
        else:
            x_sym = self.x.as_vec()

        u_sym = self.u.as_vec()
        p_sym = self.p.as_vec()

        # Build argument list matching f_x signature
        f_x_args = [x_sym]

        # Add dependent variables if present (not currently used in integrators)
        if hasattr(self, "dep") and self.dep is not None:
            # For now, use zero/default values for dep in integrator
            # In a full implementation, would evaluate f_dep here
            f_x_args.append(self.dep0.as_vec())

        # Add discrete states if present (constant during integration)
        if self.z is not None:
            z_sym = ca.SX.sym("z", self.z.size1())
            f_x_args.append(z_sym)
        else:
            z_sym = None

        # Add discrete variables if present (constant during integration)
        if self.m is not None:
            m_sym = ca.SX.sym("m", self.m.size1())
            f_x_args.append(m_sym)
        else:
            m_sym = None

        f_x_args.extend([u_sym, p_sym])

        # Evaluate dynamics
        dx_dt = self.f_x(*f_x_args)
        x_next = x_sym + dt_sym * dx_dt

        # Build f_step with appropriate inputs
        step_inputs = [x_sym]
        step_names = ["x"]

        if z_sym is not None:
            step_inputs.append(z_sym)
            step_names.append("z")
        if m_sym is not None:
            step_inputs.append(m_sym)
            step_names.append("m")

        step_inputs.extend([u_sym, p_sym, dt_sym])
        step_names.extend(["u", "p", "dt"])

        self.f_step = ca.Function(
            "f_step", step_inputs, [x_next], step_names, ["x_next"]
        )

    def _build_idas_integrator(self, options: dict):
        """Build IDAS DAE integrator."""
        # TODO: Implement IDAS for DAE systems with algebraic constraints
        print("Warning: IDAS not yet implemented, falling back to RK4")
        self._build_rk4_integrator(options)

    def _build_index_maps(self):
        """Build index maps for state/input/parameter/output access (internal use only)."""
        # State indices (private - internal use only)
        self._state_index = {}
        offset = 0
        for fname, finfo in self.state_type._field_info.items():
            dim = finfo["dim"]
            if dim == 1:
                self._state_index[fname] = offset
                offset += 1
            else:
                for i in range(dim):
                    self._state_index[f"{fname}[{i}]"] = offset + i
                offset += dim

        # Input indices (private - internal use only)
        self._input_index = {}
        offset = 0
        for fname, finfo in self.input_type._field_info.items():
            dim = finfo["dim"]
            if dim == 1:
                self._input_index[fname] = offset
                offset += 1
            else:
                for i in range(dim):
                    self._input_index[f"{fname}[{i}]"] = offset + i
                offset += dim

        # Parameter indices (private - internal use only)
        self._parameter_index = {}
        offset = 0
        for fname, finfo in self.param_type._field_info.items():
            dim = finfo["dim"]
            if dim == 1:
                self._parameter_index[fname] = offset
                offset += 1
            else:
                for i in range(dim):
                    self._parameter_index[f"{fname}[{i}]"] = offset + i
                offset += dim

        # Output indices (if outputs exist) (private - internal use only)
        if self.output_type is not None:
            self._output_index = {}
            offset = 0
            for fname, finfo in self.output_type._field_info.items():
                dim = finfo["dim"]
                self._output_index[fname] = offset
                if dim > 1:
                    for i in range(dim):
                        self._output_index[f"{fname}[{i}]"] = offset + i
                offset += dim

        # Create ordered name lists
        self.state_names = [
            n for n, _ in sorted(self._state_index.items(), key=lambda kv: kv[1])
        ]
        self.input_names = (
            [n for n, _ in sorted(self._input_index.items(), key=lambda kv: kv[1])]
            if self._input_index
            else []
        )
        self.parameter_names = (
            [n for n, _ in sorted(self._parameter_index.items(), key=lambda kv: kv[1])]
            if self._parameter_index
            else []
        )
        self.output_names = (
            [n for n, _ in sorted(self._output_index.items(), key=lambda kv: kv[1])]
            if hasattr(self, "_output_index")
            else []
        )

    def simulate(
        self,
        t0: float,
        tf: float,
        dt: float,
        u_func: Callable = None,
        p_vec=None,
        x0_vec=None,
        detect_events: bool = False,
    ):
        """Simulate model from t0 to tf with event detection.

        Args:
            t0: Initial time
            tf: Final time
            dt: Timestep
            u_func: Optional control function (t, x, p) -> u_vec
            p_vec: Optional parameter vector (uses p0 if None)
            x0_vec: Optional initial state vector (uses x0 if None)
            detect_events: Whether to detect and handle zero-crossings

        Returns:
            Dictionary with 't', 'x', and optionally 'z', 'm', 'q', 'out' arrays
        """
        if not hasattr(self, "f_step"):
            raise ValueError("Model not built. Call build() first.")

        # Handle composed models
        if hasattr(self, "_composed") and self._composed:
            p_vec = self.p0.as_vec() if p_vec is None else p_vec
            x_curr = self.x0_composed if x0_vec is None else x0_vec
        else:
            p_vec = self.p0.as_vec() if p_vec is None else p_vec
            x_curr = self.x0.as_vec() if x0_vec is None else x0_vec

        z_curr = self.z0.as_vec() if self.z is not None else None
        m_curr = self.m0.as_vec() if self.m is not None else None
        q_curr = self.q0.as_vec() if self.q is not None else None

        t_hist = [t0]
        x_hist = [x_curr]
        if self.z is not None:
            z_hist = [z_curr]
        if self.m is not None:
            m_hist = [m_curr]
        if self.q is not None:
            q_hist = [q_curr]
        out_hist = [] if hasattr(self, "f_y") else None

        # Track previous event indicator for zero-crossing detection
        c_prev = None
        if detect_events and hasattr(self, "f_c"):
            args = self._build_eval_args(
                x_curr, z_curr, m_curr, self.u0.as_vec(), p_vec
            )
            c_prev = float(self.f_c(*args))

        t = t0
        while t < tf - dt / 2:
            # Get control
            if u_func is not None:
                u_curr = u_func(t, x_curr, p_vec)
            else:
                u_curr = self.u0.as_vec()

            # Evaluate outputs before step
            if out_hist is not None:
                args = self._build_eval_args(x_curr, z_curr, m_curr, u_curr, p_vec)
                out_val = self.f_y(*args)
                out_hist.append(out_val)

            # Integration step - build arguments for f_step
            step_args = [x_curr]
            if self.z is not None:
                step_args.append(z_curr)
            if self.m is not None:
                step_args.append(m_curr)
            step_args.extend([u_curr, p_vec, dt])

            x_next = self.f_step(*step_args)

            # Check for events
            if detect_events and hasattr(self, "f_c"):
                args = self._build_eval_args(x_next, z_curr, m_curr, u_curr, p_vec)
                c_curr = float(self.f_c(*args))

                # Detect zero-crossing
                if c_prev is not None and c_prev > 0 and c_curr <= 0:
                    # Event occurred!
                    if hasattr(self, "f_z") and z_curr is not None:
                        z_curr = self.f_z(*args)
                    if hasattr(self, "f_m"):
                        # f_m can reset either discrete variables or continuous states
                        # Check output dimension to determine which
                        m_reset = self.f_m(*args)
                        if m_curr is not None and m_reset.size1() == m_curr.size1():
                            # Reset discrete variables
                            m_curr = m_reset
                        elif m_reset.size1() == x_next.size1():
                            # Reset continuous states
                            x_next = m_reset
                        else:
                            # Assume it's for continuous states if sizes don't match m
                            x_next = m_reset

                c_prev = c_curr

            # Integrate quadratures if present
            if self.q is not None and hasattr(self, "f_q"):
                args = self._build_eval_args(x_curr, z_curr, m_curr, u_curr, p_vec)
                dq_dt = self.f_q(*args)
                q_curr = q_curr + dt * dq_dt

            # Store
            t += dt
            t_hist.append(t)
            x_hist.append(x_next)
            if self.z is not None:
                z_hist.append(z_curr)
            if self.m is not None:
                m_hist.append(m_curr)
            if self.q is not None:
                q_hist.append(q_curr)

            x_curr = x_next

        # Final output
        if out_hist is not None:
            args = self._build_eval_args(x_curr, z_curr, m_curr, u_curr, p_vec)
            out_val = self.f_y(*args)
            out_hist.append(out_val)

        # Build result dictionary
        result = {"t": np.array(t_hist), "x": ca.hcat(x_hist).full()}
        if self.z is not None:
            result["z"] = ca.hcat(z_hist).full()
        if self.m is not None:
            result["m"] = ca.hcat(m_hist).full()
        if self.q is not None:
            result["q"] = ca.hcat(q_hist).full()
        if out_hist is not None:
            result["out"] = ca.hcat(out_hist).full()

        return result

    def _build_eval_args(self, x, z, m, u, p):
        """Build argument list for function evaluation.

        This matches the signature of functions built with _get_standard_inputs(),
        which may include dependent variables (dep) if they exist.
        """
        args = [x]
        if hasattr(self, "dep") and self.dep is not None:
            # Evaluate dependent variables and add to argument list
            # f_dep has signature (x, u, p) -> dep
            if hasattr(self, "f_dep"):
                dep_val = self.f_dep(x, u, p)
                args.append(dep_val)
            else:
                # If dep exists but f_dep not built, this is an error
                raise ValueError("Model has dep but f_dep was not built")
        if z is not None:
            args.append(z)
        if m is not None:
            args.append(m)
        args.append(u)
        args.append(p)
        return args  # ============================================================================

    # Hierarchical Model Composition
    # ============================================================================
    # Hierarchical Composition
    # ============================================================================

    class SignalRef:
        """Reference to a signal in a model for autocomplete-friendly connections."""

        def __init__(self, model_name: str, signal_name: str):
            self.model_name = model_name
            self.signal_name = signal_name
            self._path = f"{model_name}.{signal_name}"

        def __str__(self):
            return self._path

        def __repr__(self):
            return f"SignalRef('{self._path}')"

    class SubmodelProxy:
        """Proxy object that allows attribute access to submodel signals."""

        def __init__(self, name: str, model: "ModelSX"):
            self._name = name
            self._model = model

        def __getattr__(self, attr: str):
            # Check if attribute exists in submodel's state, input, or output
            if hasattr(self._model, "x") and hasattr(self._model.x, attr):
                return ModelSX.SignalRef(self._name, attr)
            elif hasattr(self._model, "u") and hasattr(self._model.u, attr):
                return ModelSX.SignalRef(self._name, attr)
            elif hasattr(self._model, "y") and hasattr(self._model.y, attr):
                return ModelSX.SignalRef(self._name, attr)
            else:
                raise AttributeError(
                    f"Signal '{attr}' not found in submodel '{self._name}'"
                )

    def add_submodel(
        self,
        name: str,
        submodel: "ModelSX",
        state_connections: dict[str, str] = None,
        input_connections: dict[str, str] = None,
        output_connections: dict[str, str] = None,
    ):
        """Add a submodel for hierarchical composition.

        This allows building complex systems from simpler components. Submodels
        are stored and their states/inputs/outputs can be connected to the parent
        model or other submodels.

        Args:
            name: Unique identifier for this submodel
            submodel: ModelSX instance to add as a component
            state_connections: Map submodel states to parent states
                Example: {"controller.i_pitch": "x.pitch_integral"}
            input_connections: Map submodel inputs to parent inputs or submodel outputs
                Example: {"controller.pitch_ref": "u.pitch_cmd",
                         "controller.pitch_meas": "aircraft.pitch"}
            output_connections: Map submodel outputs to parent outputs
                Example: {"controller.elevator": "y.elevator_cmd"}

        Example:
            >>> parent = ModelSX.create(ParentStates, ParentInputs, ParentParams)
            >>> aircraft = sportcub()
            >>> controller = pid_controller()
            >>> parent.add_submodel("aircraft", aircraft,
            ...     input_connections={"aircraft.thr": "controller.thr"})
            >>> parent.add_submodel("controller", controller,
            ...     input_connections={"controller.pitch_meas": "aircraft.theta"})
        """
        if not hasattr(self, "_submodels"):
            self._submodels = {}
            self._state_connections = {}
            self._input_connections = {}
            self._output_connections = {}

        if name in self._submodels:
            raise ValueError(f"Submodel '{name}' already exists")

        self._submodels[name] = submodel

        # Create proxy for autocomplete-friendly signal access
        setattr(self, name, self.SubmodelProxy(name, submodel))

        if state_connections:
            self._state_connections[name] = state_connections
        if input_connections:
            self._input_connections[name] = input_connections
        if output_connections:
            self._output_connections[name] = output_connections

    def connect(self, target, source):
        """Add a connection between signals in a more readable way.

        Accepts either string paths or SignalRef objects for autocomplete support.

        Args:
            target: Target signal (SignalRef or string like "controller.u.q" or "y.ail")
            source: Source signal (SignalRef or string like "plant.x.r" or "u.ail_manual")

        Example:
            >>> parent.connect("controller.u.q", "plant.x.r")
            >>> parent.connect(parent.controller.u.q, parent.plant.x.r)
        """
        # Import casadi to check for symbolic types
        import casadi as ca

        # Helper to convert signal references to path strings
        def to_path_string(sig):
            if isinstance(sig, self.SignalRef):
                return str(sig)
            elif isinstance(sig, (ca.SX, ca.MX, ca.DM)):
                # CasADi symbols don't carry model/signal path information
                raise TypeError(
                    f"Cannot connect CasADi symbol directly: {sig}. "
                    f"Use string paths like 'controller.u.q' or SignalRef via parent.controller.u.q"
                )
            else:
                return sig

        target_str = to_path_string(target)
        source_str = to_path_string(source)

        if not hasattr(self, "_submodels"):
            self._submodels = {}
            self._state_connections = {}
            self._input_connections = {}
            self._output_connections = {}

        # Parse target to determine connection type
        # Supports both formats:
        # - "model.signal" (legacy 2-part for direct signal access)
        # - "model.type.field" (3-part for structured signals like controller.u.q)
        target_parts = target_str.split(".")
        source_parts = source_str.split(".")

        if len(target_parts) < 2:
            raise ValueError(
                f"Invalid target format: {target_str}. Expected 'model.signal' or 'model.type.field'"
            )

        # Determine connection type based on first part of target
        target_model = target_parts[0]

        # Determine connection type based on target
        if target_model == "y":
            # Output connection: submodel output -> parent output
            # Format: "y.field" or "y.field" <- "model.y.field"
            source_model = source_parts[0]
            if source_model not in self._output_connections:
                self._output_connections[source_model] = {}
            self._output_connections[source_model][source_str] = target_str
        elif target_model == "x":
            # State connection (rarely used)
            source_model = source_parts[0]
            if source_model not in self._state_connections:
                self._state_connections[source_model] = {}
            self._state_connections[source_model][source_str] = target_str
        else:
            # Input connection: anything -> submodel input
            # Format: "model.u.field" <- "other.x.field" or "u.field"
            if target_model not in self._input_connections:
                self._input_connections[target_model] = {}
            self._input_connections[target_model][target_str] = source_str

    def build_composed(self, integrator: str = "rk4", integrator_options: dict = None):
        """Build a composed model from added submodels.

        This creates a unified dynamics function that integrates all submodels,
        routing signals according to the connection maps specified in add_submodel().

        The composed model will have:
        - States: concatenation of all submodel states
        - Inputs: parent inputs (submodel inputs are internal)
        - Outputs: parent outputs mapped from submodel outputs
        - Dynamics: combined f_x from all submodels with signal routing

        Example:
            >>> parent = ModelSX.create(ParentStates, ParentInputs, ParentParams)
            >>> parent.add_submodel("aircraft", aircraft, ...)
            >>> parent.add_submodel("controller", controller, ...)
            >>> parent.build_composed(integrator="rk4")
            >>> # Now parent.f_step() integrates both aircraft and controller
        """
        if not hasattr(self, "_submodels") or not self._submodels:
            raise ValueError("No submodels added. Use add_submodel() first.")

        if integrator_options is None:
            integrator_options = {}

        # Build index maps for parent model (needed for connection resolution)
        self._build_index_maps()

        # Build combined state vector from all submodels
        submodel_state_slices = {}
        offset = 0
        for name, submodel in self._submodels.items():
            n_states = submodel.x.size1()
            submodel_state_slices[name] = (offset, offset + n_states)
            offset += n_states

        total_states = offset

        # Create combined dynamics function
        x_combined = ca.SX.sym("x_combined", total_states)
        u_parent = self.u.as_vec()
        p_parent = self.p.as_vec()

        # Extract submodel states from combined vector
        submodel_states = {}
        for name, (start, end) in submodel_state_slices.items():
            submodel_states[name] = x_combined[start:end]

        # Build input vectors for each submodel by resolving connections
        # We need to do this iteratively because outputs depend on inputs
        # For now, we'll do a simple two-pass approach:
        # Pass 1: Resolve state connections
        # Pass 2: Resolve output connections after evaluating outputs with state-connected inputs

        submodel_inputs = {}
        submodel_outputs = {}

        # First pass: Build preliminary inputs with only state connections resolved
        for name, submodel in self._submodels.items():
            u_sub_dict = {}
            input_conns = self._input_connections.get(name, {})

            for field_obj in fields(submodel.u):
                field_name = field_obj.name
                full_path = f"{name}.u.{field_name}"

                if full_path in input_conns:
                    source = input_conns[full_path]
                    parts = source.split(".", 2)

                    if len(parts) >= 3 and parts[1] == "x":
                        # State connection - resolve immediately
                        source_model, source_type, source_field = parts
                        if source_model in submodel_states:
                            source_submodel = self._submodels[source_model]
                            if hasattr(source_submodel.x, source_field):
                                source_x = source_submodel.x
                                x_vec = submodel_states[source_model]

                                offset = 0
                                for fld in fields(source_x):
                                    if fld.name == source_field:
                                        field_val = getattr(source_x, fld.name)
                                        field_size = (
                                            field_val.shape[0]
                                            if hasattr(field_val, "shape")
                                            else 1
                                        )
                                        u_sub_dict[field_name] = x_vec[
                                            offset : offset + field_size
                                        ]
                                        break
                                    else:
                                        fld_val = getattr(source_x, fld.name)
                                        fld_size = (
                                            fld_val.shape[0]
                                            if hasattr(fld_val, "shape")
                                            else 1
                                        )
                                        offset += fld_size
                                else:
                                    u_sub_dict[field_name] = getattr(
                                        submodel.u0, field_name
                                    )
                            else:
                                u_sub_dict[field_name] = getattr(
                                    submodel.u0, field_name
                                )
                        else:
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                    else:
                        # Output or parent input connection - use default for now
                        u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                else:
                    u_sub_dict[field_name] = getattr(submodel.u0, field_name)

            # Build preliminary input vector
            u_sub_list = [u_sub_dict[f.name] for f in fields(submodel.u)]
            submodel_inputs[name] = (
                ca.vertcat(*u_sub_list) if u_sub_list else ca.DM.zeros(0, 1)
            )

        # Evaluate all submodel outputs with preliminary inputs
        for name, submodel in self._submodels.items():
            if hasattr(submodel, "f_y"):
                x_sub = submodel_states[name]
                u_sub = submodel_inputs[name]
                p_sub = submodel.p0.as_vec()
                y_sub = submodel.f_y(x_sub, u_sub, p_sub)
                submodel_outputs[name] = y_sub

        # Second pass: Resolve output connections now that outputs are available
        for name, submodel in self._submodels.items():
            u_sub_dict = {}
            input_conns = self._input_connections.get(name, {})

            for field_obj in fields(submodel.u):
                field_name = field_obj.name
                full_path = f"{name}.u.{field_name}"

                if full_path in input_conns:
                    source = input_conns[full_path]

                    if source.startswith("u."):
                        # Connected to parent input
                        parent_field = source[2:]
                        if hasattr(self.u, parent_field):
                            u_sub_dict[field_name] = getattr(self.u, parent_field)
                        else:
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)

                    elif "." in source:
                        parts = source.split(".", 2)
                        if len(parts) >= 3:
                            source_model, source_type, source_field = parts

                            if source_type == "y":
                                # Output connection
                                if source_model in submodel_outputs:
                                    source_submodel = self._submodels[source_model]
                                    if hasattr(source_submodel.y, source_field):
                                        source_y = source_submodel.y
                                        y_vec = submodel_outputs[source_model]

                                        offset = 0
                                        for fld in fields(source_y):
                                            if fld.name == source_field:
                                                field_val = getattr(source_y, fld.name)
                                                field_size = (
                                                    field_val.shape[0]
                                                    if hasattr(field_val, "shape")
                                                    else 1
                                                )
                                                u_sub_dict[field_name] = y_vec[
                                                    offset : offset + field_size
                                                ]
                                                break
                                            else:
                                                fld_val = getattr(source_y, fld.name)
                                                fld_size = (
                                                    fld_val.shape[0]
                                                    if hasattr(fld_val, "shape")
                                                    else 1
                                                )
                                                offset += fld_size
                                        else:
                                            u_sub_dict[field_name] = getattr(
                                                submodel.u0, field_name
                                            )
                                    else:
                                        u_sub_dict[field_name] = getattr(
                                            submodel.u0, field_name
                                        )
                                else:
                                    u_sub_dict[field_name] = getattr(
                                        submodel.u0, field_name
                                    )

                            elif source_type == "x":
                                # State connection - already resolved in first pass
                                source_submodel = self._submodels[source_model]
                                if hasattr(source_submodel.x, source_field):
                                    source_x = source_submodel.x
                                    x_vec = submodel_states[source_model]

                                    offset = 0
                                    for fld in fields(source_x):
                                        if fld.name == source_field:
                                            field_val = getattr(source_x, fld.name)
                                            field_size = (
                                                field_val.shape[0]
                                                if hasattr(field_val, "shape")
                                                else 1
                                            )
                                            u_sub_dict[field_name] = x_vec[
                                                offset : offset + field_size
                                            ]
                                            break
                                        else:
                                            fld_val = getattr(source_x, fld.name)
                                            fld_size = (
                                                fld_val.shape[0]
                                                if hasattr(fld_val, "shape")
                                                else 1
                                            )
                                            offset += fld_size
                                    else:
                                        u_sub_dict[field_name] = getattr(
                                            submodel.u0, field_name
                                        )
                                else:
                                    u_sub_dict[field_name] = getattr(
                                        submodel.u0, field_name
                                    )
                            else:
                                u_sub_dict[field_name] = getattr(
                                    submodel.u0, field_name
                                )
                        else:
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                    else:
                        u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                else:
                    u_sub_dict[field_name] = getattr(submodel.u0, field_name)

            # Build final input vector
            u_sub_list = [u_sub_dict[f.name] for f in fields(submodel.u)]
            submodel_inputs[name] = (
                ca.vertcat(*u_sub_list) if u_sub_list else ca.DM.zeros(0, 1)
            )

        # Evaluate each submodel's dynamics with resolved inputs
        f_x_parts = []
        for name, submodel in self._submodels.items():
            x_sub = submodel_states[name]
            u_sub = submodel_inputs[name]
            p_sub = submodel.p0.as_vec()

            # Evaluate submodel dynamics
            dx_sub = submodel.f_x(x_sub, u_sub, p_sub)
            f_x_parts.append(dx_sub)

        # Combine all submodel derivatives
        f_x_combined = ca.vertcat(*f_x_parts)

        # Create composed dynamics function
        self.f_x = ca.Function(
            "f_x_composed",
            [x_combined, u_parent, p_parent],
            [f_x_combined],
            ["x", "u", "p"],
            ["dx_dt"],
        )

        # Build composed output function if parent has output type
        if self.output_type is not None and hasattr(self, "y"):
            # Build parent output by evaluating submodel outputs
            # We need fresh symbolic variables for the output function
            x_out = ca.SX.sym("x", total_states)
            u_out = ca.SX.sym("u", u_parent.shape[0])
            p_out = ca.SX.sym("p", p_parent.shape[0])

            # Build parent output vector by evaluating submodel outputs
            parent_y_parts = []

            # Track which parent output fields have been set
            parent_y_dict = {}
            for field_obj in fields(self.y):
                field_val = getattr(self.y, field_obj.name)
                field_size = field_val.shape[0] if hasattr(field_val, "shape") else 1
                parent_y_dict[field_obj.name] = ca.DM.zeros(field_size, 1)

            # Build dependency graph for topological sorting
            # Find which submodels depend on outputs from other submodels
            output_dependencies = {}  # submodel -> set of submodels it depends on
            for source_model in self._submodels.keys():
                deps = set()
                input_conns = self._input_connections.get(source_model, {})
                for full_path, source in input_conns.items():
                    if "." in source:
                        parts = source.split(".", 2)
                        if len(parts) >= 2 and parts[1] == "y":
                            # This submodel depends on another submodel's output
                            deps.add(parts[0])
                output_dependencies[source_model] = deps

            # Topological sort to get evaluation order
            def topological_sort(dependencies):
                """Return list of nodes in topological order."""
                result = []
                visited = set()
                temp_mark = set()

                def visit(node):
                    if node in temp_mark:
                        raise ValueError(
                            f"Circular dependency detected involving {node}"
                        )
                    if node not in visited:
                        temp_mark.add(node)
                        for dep in dependencies.get(node, set()):
                            visit(dep)
                        temp_mark.remove(node)
                        visited.add(node)
                        result.append(node)

                for node in dependencies.keys():
                    if node not in visited:
                        visit(node)
                return result

            eval_order = topological_sort(output_dependencies)

            # Store submodel outputs for use in connections
            submodel_outputs = {}  # submodel_name -> output_vector

            # Evaluate each submodel's output in dependency order
            for source_model in eval_order:
                connections = self._output_connections.get(source_model, {})
                submodel = self._submodels.get(source_model)
                if submodel is None or not hasattr(submodel, "f_y"):
                    continue

                # Extract submodel state
                start, end = submodel_state_slices[source_model]
                x_sub = x_out[start:end]

                # Build submodel input (resolve connections)
                u_sub_dict = {}
                input_conns = self._input_connections.get(source_model, {})

                for field_obj in fields(submodel.u):
                    field_name = field_obj.name
                    full_path = f"{source_model}.u.{field_name}"

                    if full_path in input_conns:
                        source = input_conns[full_path]

                        if source.startswith("u."):
                            # Connected to parent input - extract from u_out
                            parent_field = source[2:]
                            offset = 0
                            for fld in fields(self.u):
                                if fld.name == parent_field:
                                    fld_val = getattr(self.u, fld.name)
                                    fld_size = (
                                        fld_val.shape[0]
                                        if hasattr(fld_val, "shape")
                                        else 1
                                    )
                                    u_sub_dict[field_name] = u_out[
                                        offset : offset + fld_size
                                    ]
                                    break
                                else:
                                    fld_val = getattr(self.u, fld.name)
                                    fld_size = (
                                        fld_val.shape[0]
                                        if hasattr(fld_val, "shape")
                                        else 1
                                    )
                                    offset += fld_size
                            else:
                                u_sub_dict[field_name] = getattr(
                                    submodel.u0, field_name
                                )
                        elif "." in source:
                            parts = source.split(".", 2)
                            if len(parts) >= 3:
                                src_model, src_type, src_field = parts

                                if src_type == "x":
                                    # State connection - extract from x_out
                                    src_start, src_end = submodel_state_slices.get(
                                        src_model, (0, 0)
                                    )
                                    src_x_vec = x_out[src_start:src_end]
                                    src_submodel = self._submodels[src_model]

                                    offset = 0
                                    for fld in fields(src_submodel.x):
                                        if fld.name == src_field:
                                            fld_val = getattr(src_submodel.x, fld.name)
                                            fld_size = (
                                                fld_val.shape[0]
                                                if hasattr(fld_val, "shape")
                                                else 1
                                            )
                                            u_sub_dict[field_name] = src_x_vec[
                                                offset : offset + fld_size
                                            ]
                                            break
                                        else:
                                            fld_val = getattr(src_submodel.x, fld.name)
                                            fld_size = (
                                                fld_val.shape[0]
                                                if hasattr(fld_val, "shape")
                                                else 1
                                            )
                                            offset += fld_size
                                    else:
                                        u_sub_dict[field_name] = getattr(
                                            submodel.u0, field_name
                                        )
                                elif src_type == "y":
                                    # Output connection - extract from previously computed submodel output
                                    if src_model in submodel_outputs:
                                        src_y_vec = submodel_outputs[src_model]
                                        src_submodel = self._submodels[src_model]

                                        offset = 0
                                        for fld in fields(src_submodel.y):
                                            if fld.name == src_field:
                                                fld_val = getattr(
                                                    src_submodel.y, fld.name
                                                )
                                                fld_size = (
                                                    fld_val.shape[0]
                                                    if hasattr(fld_val, "shape")
                                                    else 1
                                                )
                                                u_sub_dict[field_name] = src_y_vec[
                                                    offset : offset + fld_size
                                                ]
                                                break
                                            else:
                                                fld_val = getattr(
                                                    src_submodel.y, fld.name
                                                )
                                                fld_size = (
                                                    fld_val.shape[0]
                                                    if hasattr(fld_val, "shape")
                                                    else 1
                                                )
                                                offset += fld_size
                                        else:
                                            u_sub_dict[field_name] = getattr(
                                                submodel.u0, field_name
                                            )
                                    else:
                                        u_sub_dict[field_name] = getattr(
                                            submodel.u0, field_name
                                        )
                                else:
                                    u_sub_dict[field_name] = getattr(
                                        submodel.u0, field_name
                                    )
                        else:
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                    else:
                        u_sub_dict[field_name] = getattr(submodel.u0, field_name)

                # Build submodel input vector
                u_sub_list = [u_sub_dict[f.name] for f in fields(submodel.u)]
                u_sub = ca.vertcat(*u_sub_list) if u_sub_list else ca.DM.zeros(0, 1)

                # Build submodel parameter vector
                p_sub = submodel.p0.as_vec()

                # Evaluate submodel output
                y_sub_vec = submodel.f_y(x_sub, u_sub, p_sub)

                # Store output for use by other submodels
                submodel_outputs[source_model] = y_sub_vec

                # Extract fields and assign to parent output
                for source_str, target_str in connections.items():
                    target_parts = target_str.split(".")
                    if len(target_parts) < 2:
                        continue
                    to_field = target_parts[1]

                    source_parts = source_str.split(".")
                    if len(source_parts) < 3:
                        continue
                    source_field = source_parts[2]

                    # Find the field in the submodel output vector
                    offset = 0
                    for fld in fields(submodel.y):
                        if fld.name == source_field:
                            fld_val = getattr(submodel.y, fld.name)
                            fld_size = (
                                fld_val.shape[0] if hasattr(fld_val, "shape") else 1
                            )
                            parent_y_dict[to_field] = y_sub_vec[
                                offset : offset + fld_size
                            ]
                            break
                        else:
                            fld_val = getattr(submodel.y, fld.name)
                            fld_size = (
                                fld_val.shape[0] if hasattr(fld_val, "shape") else 1
                            )
                            offset += fld_size

            # Assemble parent output vector
            parent_y_list = [parent_y_dict[f.name] for f in fields(self.y)]
            parent_y_vec = (
                ca.vertcat(*parent_y_list) if parent_y_list else ca.DM.zeros(0, 1)
            )

            self.f_y = ca.Function(
                "f_y_composed",
                [x_out, u_out, p_out],
                [parent_y_vec],
                ["x", "u", "p"],
                ["y"],
            )

        # Store composition metadata BEFORE building integrator
        # (integrator builders check _composed and _total_composed_states)
        self._composed = True
        self._submodel_state_slices = submodel_state_slices
        self._total_composed_states = total_states

        # Build integrator
        if integrator == "rk4":
            self._build_rk4_integrator(integrator_options)
        elif integrator == "euler":
            self._build_euler_integrator()
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        # Update x0 to be concatenated initial states
        x0_parts = []
        for name in self._submodels.keys():
            x0_parts.append(self._submodels[name].x0.as_vec())

        # Create a compatible x0 structure - flat numeric vector
        self.x0_composed = ca.vertcat(*x0_parts) if x0_parts else ca.DM.zeros(0, 1)

        # Create a structured initial state that provides field access to submodel states
        # This allows sim.py to access states like: x.plant.p, x.controller.i_p, etc.
        from dataclasses import make_dataclass, field as dc_field

        # Build fields list for the composite dataclass
        composite_fields = []
        for name, submodel in self._submodels.items():
            # Each field holds a copy of the submodel's state dataclass
            composite_fields.append(
                (
                    name,
                    type(submodel.x0),
                    dc_field(default_factory=lambda sm=submodel: copy.deepcopy(sm.x0)),
                )
            )

        # Create composite state dataclass with as_vec() and from_vec() methods
        def _as_vec(self_state):
            """Convert structured state to flat vector."""
            return ca.vertcat(
                *[getattr(self_state, name).as_vec() for name in self._submodels.keys()]
            )

        def _from_vec(cls, x_vec):
            """Convert flat vector to structured state."""
            kwargs = {}
            for name, (start, end) in self._submodel_state_slices.items():
                submodel = self._submodels[name]
                x_sub_vec = x_vec[start:end]
                # Use submodel's state type to reconstruct from vector
                kwargs[name] = type(submodel.x0).from_vec(x_sub_vec)
            return cls(**kwargs)

        # Create the composite state dataclass
        ComposedState = make_dataclass(
            "ComposedState",
            composite_fields,
            namespace={"as_vec": _as_vec, "from_vec": classmethod(_from_vec)},
        )

        # Create initial state instance
        self.x0 = ComposedState()

        # Store helpers for backwards compatibility
        self._state_to_vec = lambda x_struct: x_struct.as_vec()
        self._vec_to_state = lambda x_vec: ComposedState.from_vec(x_vec)

    # Utility methods
    def saturate(self, val, lower, upper):
        """Saturate value between bounds."""
        return ca.fmin(ca.fmax(val, lower), upper)

    # Backwards compatibility properties for legacy API
    @property
    def State(self):
        """Legacy API: access to state type class."""
        return self.state_type

    @property
    def Input(self):
        """Legacy API: access to input type class."""
        return self.input_type

    @property
    def Parameters(self):
        """Legacy API: access to parameter type class."""
        return self.param_type

    @property
    def output_sizes(self):
        """Legacy API: dictionary mapping output names to their dimensions."""
        if self.output_type is None:
            return {}
        return {
            fname: finfo["dim"] for fname, finfo in self.output_type._field_info.items()
        }


@beartype
class ModelMX(ModelSX[TState, TInput, TParam]):
    """Type-safe MX model for large-scale optimization.

    Same API as ModelSX but uses MX symbolic type for better
    performance with large-scale optimization problems.

    Usage:
        model = ModelMX.create(States, Inputs, Params)

        x = model.x()
        u = model.u()
        p = model.p()

        f_x = ca.vertcat(x.v, u.thrust / p.m - p.g)
        model.build(f_x=f_x, integrator='rk4')
    """

    @classmethod
    def create(
        cls,
        state_type: type[TState],
        input_type: type[TInput],
        param_type: type[TParam],
        **kwargs,
    ):
        """Create a fully-typed MX model instance with automatic type inference."""
        return cls(state_type, input_type, param_type, **kwargs)

    def __init__(
        self,
        state_type: type[TState],
        input_type: type[TInput],
        param_type: type[TParam],
        **kwargs,
    ):
        """Initialize MX-based model."""
        # Set MX before calling parent __init__
        self._sym = ca.MX

        # Store types
        self.state_type = state_type
        self.input_type = input_type
        self.param_type = param_type
        self.output_type = kwargs.get("output_type")
        self.algebraic_type = kwargs.get("algebraic_type")
        self.dependent_type = kwargs.get("dependent_type")
        self.quadrature_type = kwargs.get("quadrature_type")
        self.discrete_state_type = kwargs.get("discrete_state_type")
        self.discrete_var_type = kwargs.get("discrete_var_type")
        self.event_indicator_type = kwargs.get("event_indicator_type")

        # Create symbolic instances with MX
        self.x = state_type.symbolic(ca.MX)
        self.u = input_type.symbolic(ca.MX)
        self.p = param_type.symbolic(ca.MX)

        # Create numeric defaults
        self.x0 = state_type.numeric()
        self.u0 = input_type.numeric()
        self.p0 = param_type.numeric()

        # Create optional types if provided
        if self.output_type:
            self.y = self.output_type.symbolic(ca.MX)
            self.y0 = self.output_type.numeric()

        if self.algebraic_type:
            self.z_alg = self.algebraic_type.symbolic(ca.MX)
            self.z_alg0 = self.algebraic_type.numeric()

        if self.dependent_type:
            self.dep = self.dependent_type.symbolic(ca.MX)
            self.dep0 = self.dependent_type.numeric()

        if self.quadrature_type:
            self.q = self.quadrature_type.symbolic(ca.MX)
            self.q0 = self.quadrature_type.numeric()

        if self.discrete_state_type:
            self.z = self.discrete_state_type.symbolic(ca.MX)
            self.z0 = self.discrete_state_type.numeric()

        if self.discrete_var_type:
            self.m = self.discrete_var_type.symbolic(ca.MX)
            self.m0 = self.discrete_var_type.numeric()

        if self.event_indicator_type:
            self.c = self.event_indicator_type.symbolic(ca.MX)

    def _build_rk4_integrator(self, options: dict):
        """Build RK4 integrator with MX."""
        from . import integrators

        dt_sym = ca.MX.sym("dt")
        N = options.get("N", 10)

        rk4_step = integrators.rk4(self.f_x, dt_sym, name="rk4", N=N)

        xin = rk4_step.mx_in(0)
        uin = rk4_step.mx_in(1)
        pin = rk4_step.mx_in(2)

        self.f_step = ca.Function(
            "f_step",
            [xin, uin, pin, dt_sym],
            [rk4_step(xin, uin, pin, dt_sym)],
            ["x", "u", "p", "dt"],
            ["x_next"],
        )

    def _build_euler_integrator(self):
        """Build Euler integrator with MX."""
        dt_sym = ca.MX.sym("dt")

        # For composed models, use the combined state size
        if hasattr(self, "_composed") and self._composed:
            x_sym = ca.MX.sym("x", self._total_composed_states)
        else:
            x_sym = self.x.as_vec()

        u_sym = self.u.as_vec()
        p_sym = self.p.as_vec()

        dx_dt = self.f_x(x_sym, u_sym, p_sym)
        x_next = x_sym + dt_sym * dx_dt

        self.f_step = ca.Function(
            "f_step",
            [x_sym, u_sym, p_sym, dt_sym],
            [x_next],
            ["x", "u", "p", "dt"],
            ["x_next"],
        )
