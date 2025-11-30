"""
Type definitions for the IR.
"""

from enum import Enum, auto


class VariableType(Enum):
    """Type of variable in the model."""

    STATE = auto()  # Continuous state (has derivative)
    DER_STATE = auto()  # Derivative of state (automatically created)
    DISCRETE_STATE = auto()  # Discrete state (updated at events)
    ALGEBRAIC = auto()  # Algebraic variable (no derivative, no update)
    INPUT = auto()  # External input
    OUTPUT = auto()  # Output variable
    PARAMETER = auto()  # Constant parameter
    CONSTANT = auto()  # Compile-time constant


class Causality(Enum):
    """Causality of a variable (FMI terminology)."""

    LOCAL = auto()  # Internal variable
    PARAMETER = auto()  # Fixed parameter
    CALCULATED_PARAMETER = auto()  # Computed from other parameters
    INPUT = auto()  # Set from outside
    OUTPUT = auto()  # Computed and exposed
    INDEPENDENT = auto()  # Independent variable (usually time)


class Variability(Enum):
    """How often a variable can change (FMI terminology)."""

    CONSTANT = auto()  # Never changes
    FIXED = auto()  # Fixed after initialization
    TUNABLE = auto()  # Can change between events
    DISCRETE = auto()  # Changes only at events
    CONTINUOUS = auto()  # Can change continuously


class PrimitiveType(Enum):
    """Primitive data types (Modelica 3.7)."""

    REAL = auto()
    INTEGER = auto()
    BOOLEAN = auto()
    STRING = auto()
