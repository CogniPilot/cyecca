"""
Variable representation in the IR.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from cyecca.ir.types import VariableType, Causality, Variability, PrimitiveType


@dataclass
class Variable:
    """
    Represents a variable in the model.

    This is what Rumoca will generate for each Modelica variable.
    """

    name: str
    var_type: VariableType
    primitive_type: PrimitiveType = PrimitiveType.REAL
    causality: Optional[Causality] = None
    variability: Optional[Variability] = None

    # Initial value / parameter value
    start: Optional[float] = None
    value: Optional[Any] = None

    # Array dimensions (None = scalar, [n] = vector, [n,m] = matrix)
    shape: Optional[list[int]] = None

    # Metadata
    description: str = ""
    unit: str = ""
    comment: str = ""  # Modelica comment string (similar to description)
    display_unit: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    nominal: Optional[float] = None  # Nominal value for scaling

    # For derivative states: reference to the state variable
    state_ref: Optional[str] = None

    # Lie-group metadata (for SE(2,3) and other manifolds)
    lie_group_type: Optional[str] = None  # "SO3", "SE23", "R3", etc.
    manifold_chart: Optional[str] = None  # "exp", "cayley", "quaternion", etc.

    # Generic metadata for extensions
    metadata: Optional[dict] = None

    def __post_init__(self):
        """Set default causality and variability based on var_type."""
        if self.causality is None:
            self.causality = self._default_causality()
        if self.variability is None:
            self.variability = self._default_variability()

    def _default_causality(self) -> Causality:
        """Get default causality for this variable type."""
        mapping = {
            VariableType.STATE: Causality.LOCAL,
            VariableType.DER_STATE: Causality.LOCAL,
            VariableType.DISCRETE_STATE: Causality.LOCAL,
            VariableType.ALGEBRAIC: Causality.LOCAL,
            VariableType.INPUT: Causality.INPUT,
            VariableType.OUTPUT: Causality.OUTPUT,
            VariableType.PARAMETER: Causality.PARAMETER,
            VariableType.CONSTANT: Causality.PARAMETER,
        }
        return mapping[self.var_type]

    def _default_variability(self) -> Variability:
        """Get default variability for this variable type."""
        mapping = {
            VariableType.STATE: Variability.CONTINUOUS,
            VariableType.DER_STATE: Variability.CONTINUOUS,
            VariableType.DISCRETE_STATE: Variability.DISCRETE,
            VariableType.ALGEBRAIC: Variability.CONTINUOUS,
            VariableType.INPUT: Variability.CONTINUOUS,
            VariableType.OUTPUT: Variability.CONTINUOUS,
            VariableType.PARAMETER: Variability.FIXED,
            VariableType.CONSTANT: Variability.CONSTANT,
        }
        return mapping[self.var_type]

    @property
    def is_state(self) -> bool:
        """True if this is a state variable."""
        return self.var_type == VariableType.STATE

    @property
    def is_derivative(self) -> bool:
        """True if this is a derivative variable."""
        return self.var_type == VariableType.DER_STATE

    @property
    def is_discrete(self) -> bool:
        """True if this is a discrete state."""
        return self.var_type == VariableType.DISCRETE_STATE

    @property
    def is_algebraic(self) -> bool:
        """True if this is an algebraic variable."""
        return self.var_type == VariableType.ALGEBRAIC

    @property
    def is_input(self) -> bool:
        """True if this is an input."""
        return self.var_type == VariableType.INPUT

    @property
    def is_output(self) -> bool:
        """True if this is an output."""
        return self.var_type == VariableType.OUTPUT

    @property
    def is_parameter(self) -> bool:
        """True if this is a parameter or constant."""
        return self.var_type in (VariableType.PARAMETER, VariableType.CONSTANT)

    @property
    def is_scalar(self) -> bool:
        """True if this is a scalar variable."""
        return self.shape is None

    @property
    def is_array(self) -> bool:
        """True if this is an array variable."""
        return self.shape is not None
