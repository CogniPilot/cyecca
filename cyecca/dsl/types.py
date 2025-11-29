"""
Field descriptors for the Cyecca DSL.

Provides var() and submodel() descriptors for declaring model variables
in a Modelica-like style.

================================================================================
PROTOTYPE MODE - API IS IN FLUX
================================================================================

This DSL is in active prototype development. The API may change significantly
between versions. Do NOT maintain backward compatibility - iterate rapidly.

================================================================================
DESIGN PRINCIPLES - DO NOT REMOVE OR IGNORE
================================================================================

1. MODELICA CONFORMANCE: This DSL conforms as closely as possible to the
   Modelica Language Specification v3.7-dev.
   Reference: https://specification.modelica.org/master/
   
   - Variable types (Real, Integer, Boolean) follow Modelica predefined types
   - Variable prefixes (parameter, input, output, constant) follow Modelica semantics
   - Attributes (start, fixed, min, max, nominal, unit) match Modelica definitions
   - Equation-based modeling with der() operator
   - Variable classification (state vs algebraic) is automatic based on der() usage
   - Arrays are N-dimensional with fixed rank (MLS Chapter 10)
   - Connector/connection semantics (future) will follow Modelica Chapter 9

2. TYPE SAFETY: All functions MUST use beartype for runtime type checking.
   - All public functions decorated with @beartype
   - All classes use proper type hints for full IDE autocomplete
   - Do NOT use Any where a more specific type is possible
   - Use Union types for values that can be scalar or array
   - WHEN ADDING NEW FUNCTIONS: Always add @beartype decorator
   - Type aliases (Shape, Indices) are defined here - use them consistently

3. IMMUTABILITY: Field descriptors are dataclasses (frozen where possible).

================================================================================

Variable Types (Modelica Predefined Types - MLS Chapter 4.9)
============================================================

Real    - Floating point number (default)
Integer - Integer number
Boolean - True/False

Variable Classification (Modelica Variability - MLS Chapter 4.5)
================================================================

Variables are classified based on their flags and equation usage:

1. **constant** (constant=True): Compile-time constant, cannot be modified
2. **parameter** (parameter=True): Constant during simulation, can change between runs
3. **discrete** (discrete=True): Piecewise constant, changes only at events (pre() allowed)
4. **input** (input=True): Externally controlled signal
5. **output** (output=True): Computed value exposed for external use
6. **state**: Variable whose derivative der(x) appears in equations (automatic)
7. **algebraic**: Computed variable with no derivative (automatic)

Visibility (Modelica MLS 4.1)
=============================

- **public** (default): Visible from outside, part of the interface
- **protected** (protected=True): Internal implementation, not part of interface

For blocks, all public non-parameter variables must have input or output prefix.

The sorting is done automatically by analyzing the equations.

Array Support (Modelica MLS Chapter 10)
=======================================

Modelica arrays are N-dimensional with fixed rank (dimensionality) at declaration.
The shape is specified as a tuple:
- () : scalar (zero dimensions)
- (3,) : 1D vector with 3 elements
- (3, 3) : 2D matrix 3x3
- (2, 3, 4) : 3D array

Example:
    pos = var(shape=(3,))           # 3D position vector
    rot = var(shape=(3, 3))         # Rotation matrix
    tensor = var(shape=(3, 3, 3))   # 3D tensor

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, Union

import numpy as np

if TYPE_CHECKING:
    pass


# Shape type: () for scalar, (n,) for vector, (n,m) for matrix, etc.
Shape = Tuple[int, ...]

# Indices type: tracks current indexing into an array
Indices = Tuple[int, ...]


class DType(Enum):
    """
    Modelica predefined types (MLS Chapter 4.9).

    These correspond to the built-in Modelica types.
    """

    REAL = auto()  # Floating point (default for continuous variables)
    INTEGER = auto()  # Integer
    BOOLEAN = auto()  # Boolean (True/False)
    # STRING = auto()  # String (not commonly used in dynamic models)


class VarKind(Enum):
    """Classification of variables (determined automatically or by flags)."""

    CONSTANT = auto()  # constant=True: compile-time constant
    PARAMETER = auto()  # parameter=True: constant during simulation
    DISCRETE = auto()  # discrete=True: changes only at events (MLS 4.5)
    INPUT = auto()  # input=True: externally controlled
    OUTPUT = auto()  # output=True: computed, exposed externally
    STATE = auto()  # der(var) appears in equations (automatic)
    ALGEBRAIC = auto()  # computed from equations, no derivative (automatic)


# Type alias for numeric values that can be scalar or array
NumericValue = Optional[Union[float, int, bool, List[float], np.ndarray]]


@dataclass
class Var:
    """
    Universal variable descriptor for the Cyecca DSL.

    Following Modelica, variables have a type (Real, Integer, Boolean) and
    optional variability prefixes (parameter, input, output, constant).

    The classification into state vs algebraic is automatic based on whether
    der(var) appears in equations.

    Arrays (MLS Chapter 10)
    -----------------------
    Variables can be N-dimensional arrays with shape specified as a tuple:
    - shape=() : scalar (default)
    - shape=(3,) : 1D vector with 3 elements
    - shape=(3,3) : 2D matrix 3x3
    - shape=(2,3,4) : 3D array 2x3x4

    Attributes
    ----------
    dtype : DType
        Variable type: REAL (default), INTEGER, or BOOLEAN
    default : NumericValue
        Default value (used if start not specified)
    shape : tuple of int
        Array shape: () for scalar, (n,) for vector, (m,n) for matrix, etc.
    unit : str, optional
        Physical unit (e.g., "m", "rad/s") - only for Real
    desc : str
        Description
    start : NumericValue, optional
        Initial/start value (Modelica-style)
    fixed : bool
        If True, start value is used as initial equation
    min : float, optional
        Minimum bound for the variable
    max : float, optional
        Maximum bound for the variable
    nominal : float, optional
        Nominal value for scaling (only for Real)

    Variability/Causality Prefixes (Modelica MLS 4.5)
    -------------------------------------------------
    In Modelica, these are just prefixes on variables that indicate variability
    or causality. They are NOT separate equation categories.

    parameter : bool
        If True, variable is constant during simulation
    discrete : bool
        If True, variable is piecewise constant (changes only at events).
        Can use pre() to access previous value. Typical for counters, modes.
    input : bool
        If True, variable value is provided externally (causality: input)
    output : bool
        If True, variable value is computed internally and exposed (causality: output)
    constant : bool
        If True, variable is a compile-time constant

    Visibility (Modelica MLS 4.1)
    -----------------------------
    protected : bool
        If True, variable is in the protected section (internal implementation).
        Protected variables are not part of the public interface.
        For blocks, only public (non-protected) variables must have input/output.

    Note: Input and output prefixes indicate causality - how the variable
    interfaces with the outside world. Equations defining outputs are written
    in the same equations() method as all other equations.

    Internal
    --------
    name : str, optional
        Variable name (set by @model decorator)
    kind : VarKind, optional
        Classification (set during model analysis)

    Example
    -------
    >>> from cyecca.dsl import model, var  # doctest: +SKIP
    >>> @model  # doctest: +SKIP
    ... class Pendulum:
    ...     # Real parameters (dtype=DType.REAL is default)
    ...     g = var(9.81, parameter=True)
    ...     l = var(1.0, parameter=True)
    ...
    ...     # Real states (auto-detected via der() usage)
    ...     theta = var(start=0.5)
    ...     omega = var()
    ...
    ...     # Vector state (3D position)
    ...     pos = var(shape=(3,))
    ...     vel = var(shape=(3,))
    ...
    ...     # Matrix parameter (rotation)
    ...     R = var(shape=(3, 3), parameter=True)
    ...
    ...     # Integer parameter
    ...     n_segments = var(10, dtype=DType.INTEGER, parameter=True)
    ...
    ...     # Boolean
    ...     is_active = var(True, dtype=DType.BOOLEAN, parameter=True)
    """

    # Type (Modelica predefined type)
    dtype: DType = DType.REAL

    # Value attributes
    default: NumericValue = None
    shape: Shape = ()  # () = scalar, (n,) = vector, (m,n) = matrix, etc.
    unit: Optional[str] = None
    desc: str = ""
    start: NumericValue = None
    fixed: bool = False
    min: Optional[Union[float, int]] = None
    max: Optional[Union[float, int]] = None
    nominal: Optional[Union[float, int]] = None

    # Variability prefixes (Modelica-style)
    parameter: bool = False
    discrete: bool = False
    input: bool = False
    output: bool = False
    constant: bool = False

    # Visibility (Modelica-style)
    protected: bool = False  # If True, variable is in protected section

    # Connector prefixes (Modelica MLS Chapter 9)
    flow: bool = False  # If True, variable uses sum-to-zero semantics in connections

    # Internal (set by @model decorator and analysis)
    name: Optional[str] = None
    kind: Optional[VarKind] = None  # Set during equation analysis

    @property
    def size(self) -> int:
        """Total number of scalar elements (product of shape dimensions)."""
        if not self.shape:
            return 1
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def ndim(self) -> int:
        """Number of dimensions (0 for scalar, 1 for vector, 2 for matrix, etc.)."""
        return len(self.shape)

    def is_scalar(self) -> bool:
        """Return True if this is a scalar (shape=())."""
        return self.shape == ()

    def __repr__(self) -> str:
        parts = []
        if self.dtype != DType.REAL:
            parts.append(f"dtype={self.dtype.name}")
        if self.default is not None:
            parts.append(f"default={self.default}")
        if self.shape != ():
            parts.append(f"shape={self.shape}")
        if self.start is not None:
            parts.append(f"start={self.start}")
        if self.fixed:
            parts.append("fixed=True")
        if self.min is not None:
            parts.append(f"min={self.min}")
        if self.max is not None:
            parts.append(f"max={self.max}")
        # Variability prefixes
        if self.constant:
            parts.append("constant=True")
        if self.parameter:
            parts.append("parameter=True")
        if self.discrete:
            parts.append("discrete=True")
        if self.input:
            parts.append("input=True")
        if self.output:
            parts.append("output=True")
        # Visibility
        if self.protected:
            parts.append("protected=True")
        if self.flow:
            parts.append("flow=True")
        return f"var({', '.join(parts)})" if parts else "var()"

    def get_initial_value(self) -> NumericValue:
        """Get the initial/default value for this variable."""
        if self.start is not None:
            return self.start
        return self.default


@dataclass
class SubmodelField:
    """A submodel (nested model instance)."""

    model_class: Type[Any]
    name: Optional[str] = None  # Set by @model decorator

    def __repr__(self) -> str:
        return f"submodel({self.model_class.__name__})"
