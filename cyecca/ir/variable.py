"""
IRVariable - Clean variable representation for the IR.

This module provides a simple, explicit variable type without
the decorator magic of the DSL layer.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import numpy as np
from beartype import beartype


class VariableKind(Enum):
    """Classification of variables in a dynamic system."""

    # Continuous-time variables
    STATE = auto()  # Appears differentiated (has der())
    ALGEBRAIC = auto()  # Continuous, not differentiated

    # Discrete-time variables
    DISCRETE = auto()  # Changes only at events

    # External interface
    INPUT = auto()  # Provided externally
    OUTPUT = auto()  # Computed internally, exposed externally

    # Constants
    PARAMETER = auto()  # Constant during simulation
    CONSTANT = auto()  # Compile-time constant


class DataType(Enum):
    """Data types for variables."""

    REAL = auto()  # Floating point
    INTEGER = auto()  # Integer
    BOOLEAN = auto()  # True/False


# Type aliases
Shape = Tuple[int, ...]
NumericArray = Union[List[float], List[int], np.ndarray]
NumericValue = Union[float, int, bool, str, NumericArray, None]


@beartype
@dataclass
class IRVariable:
    """
    Immutable variable definition for the IR.

    This is a clean, explicit representation of a variable without
    the magic of the decorator-based DSL.

    Parameters
    ----------
    name : str
        Variable name (must be valid identifier)
    dtype : DataType
        Data type (REAL, INTEGER, BOOLEAN, STRING)
    shape : tuple of int
        Array shape: () for scalar, (n,) for vector, (m,n) for matrix
    start : numeric, optional
        Initial value
    unit : str, optional
        Physical unit (e.g., "m/s", "rad")
    description : str
        Human-readable description
    min : numeric, optional
        Minimum bound
    max : numeric, optional
        Maximum bound
    nominal : numeric, optional
        Nominal value for scaling
    fixed : bool
        If True, start value is used as fixed initial condition
    parameter : bool
        If True, constant during simulation
    discrete : bool
        If True, changes only at events
    input : bool
        If True, externally provided
    output : bool
        If True, computed and exposed
    constant : bool
        If True, compile-time constant
    flow : bool
        If True, uses sum-to-zero connection semantics

    Example
    -------
    >>> theta = IRVariable(
    ...     name="theta",
    ...     dtype=DataType.REAL,
    ...     start=0.5,
    ...     unit="rad",
    ...     description="Pendulum angle",
    ... )
    """

    name: Optional[str] = None
    dtype: DataType = DataType.REAL
    shape: Shape = ()
    default: NumericValue = None
    start: NumericValue = None
    unit: Optional[str] = None
    description: str = ""
    min: NumericValue = None
    max: NumericValue = None
    nominal: NumericValue = None
    fixed: bool = False
    parameter: bool = False
    discrete: bool = False
    input: bool = False
    output: bool = False
    constant: bool = False
    flow: bool = False
    protected: bool = False
    kind: Optional[VariableKind] = None
    desc: InitVar[Optional[str]] = None

    def __post_init__(self, desc: Optional[str] = None) -> None:
        """Validate variable definition."""
        if self.name is not None and not self._is_valid_name(self.name):
            raise ValueError(f"Invalid variable name: {self.name!r}")
        if desc is not None:
            self.description = desc

    def is_scalar(self) -> bool:
        """Return True if this is a scalar variable."""
        return self.shape == ()

    @property
    def size(self) -> int:
        """Total number of scalar elements."""
        if self.shape == ():
            return 1
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    def get_kind(self) -> VariableKind:
        """Determine the variable kind from flags."""
        if self.constant:
            return VariableKind.CONSTANT
        if self.parameter:
            return VariableKind.PARAMETER
        if self.input:
            return VariableKind.INPUT
        if self.output:
            return VariableKind.OUTPUT
        if self.discrete:
            return VariableKind.DISCRETE
        # Default: will be classified as STATE or ALGEBRAIC based on equations
        return VariableKind.ALGEBRAIC

    def with_name(self, name: str) -> "IRVariable":
        """Return a copy with a different name."""
        return IRVariable(
            name=name,
            dtype=self.dtype,
            shape=self.shape,
            default=self.default,
            start=self.start,
            unit=self.unit,
            description=self.description,
            min=self.min,
            max=self.max,
            nominal=self.nominal,
            fixed=self.fixed,
            parameter=self.parameter,
            discrete=self.discrete,
            input=self.input,
            output=self.output,
            constant=self.constant,
            flow=self.flow,
            protected=self.protected,
            kind=self.kind,
        )

    def __repr__(self) -> str:
        parts: List[str] = []
        if self.dtype != DataType.REAL:
            parts.append(f"dtype={self.dtype.name}")
        if self.default is not None:
            parts.append(f"default={self.default}")
        if self.shape != ():
            parts.append(f"shape={self.shape}")
        if self.unit:
            parts.append(f"unit={self.unit!r}")
        if self.description:
            parts.append(f"desc={self.description!r}")
        if self.start is not None:
            parts.append(f"start={self.start}")
        if self.fixed:
            parts.append("fixed=True")
        if self.min is not None:
            parts.append(f"min={self.min}")
        if self.max is not None:
            parts.append(f"max={self.max}")
        if self.nominal is not None:
            parts.append(f"nominal={self.nominal}")
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
        if self.protected:
            parts.append("protected=True")
        if self.flow:
            parts.append("flow=True")
        return f"var({', '.join(parts)})" if parts else "var()"

    def get_initial_value(self) -> NumericValue:
        """Return the preferred initial value (start overrides default)."""
        if self.start is not None:
            return self.start
        return self.default

    @staticmethod
    def _is_valid_name(name: str) -> bool:
        """Return True for identifiers or dotted submodel paths."""
        if not name:
            return False
        return all(part.isidentifier() for part in name.split("."))


def _get_desc(self: IRVariable) -> str:
    """Backward-compatible alias for description."""
    return self.description


IRVariable.desc = property(_get_desc)
