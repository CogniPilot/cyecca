"""
Model representation in the IR.

A Model is a collection of variables and equations that define a dynamical system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any

from cyecca.ir.variable import Variable
from cyecca.ir.equation import Equation
from cyecca.ir.types import VariableType
from cyecca.ir.algorithm import AlgorithmSection
from cyecca.ir.event import Event


@dataclass
class Model:
    """
    Represents a complete model.

    This is what Rumoca generates as output. It contains:
    - Variables (states, inputs, outputs, parameters)
    - Equations (derivatives, algebraic, discrete updates)
    - Initial equations/algorithms (only hold at t=0)
    - Algorithms (imperative code sections)
    - Events (discrete state changes)
    - Metadata (name, description, Lie-group annotations, etc.)
    """

    name: str
    variables: list[Variable] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)

    # Initial sections (only executed at t=0)
    initial_equations: list[Equation] = field(default_factory=list)

    # Algorithm sections (imperative, sequential)
    algorithms: list["AlgorithmSection"] = field(default_factory=list)
    initial_algorithms: list["AlgorithmSection"] = field(default_factory=list)

    # Events (discrete state changes)
    events: list["Event"] = field(default_factory=list)

    # Metadata
    description: str = ""
    version: str = ""
    author: str = ""

    # Model-level metadata (Lie groups, annotations, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Variable indices (built automatically)
    _var_dict: dict[str, Variable] = field(default_factory=dict, init=False, repr=False)

    def add_variable(self, var: Variable) -> None:
        """Add a variable to the model."""
        if var.name in self._var_dict:
            raise ValueError(f"Variable '{var.name}' already exists in model")
        self.variables.append(var)
        self._var_dict[var.name] = var

    def add_equation(self, eq: Equation) -> None:
        """Add an equation to the model."""
        self.equations.append(eq)

    def get_variable(self, name: str) -> Optional[Variable]:
        """Get a variable by name."""
        return self._var_dict.get(name)

    def has_variable(self, name: str) -> bool:
        """Check if a variable exists."""
        return name in self._var_dict

    def get_variables_by_type(self, var_type: VariableType) -> list[Variable]:
        """Get all variables of a specific type."""
        return [v for v in self.variables if v.var_type == var_type]

    @property
    def states(self) -> list[Variable]:
        """Get all state variables."""
        return self.get_variables_by_type(VariableType.STATE)

    @property
    def der_states(self) -> list[Variable]:
        """Get all derivative state variables."""
        return self.get_variables_by_type(VariableType.DER_STATE)

    @property
    def discrete_states(self) -> list[Variable]:
        """Get all discrete state variables."""
        return self.get_variables_by_type(VariableType.DISCRETE_STATE)

    @property
    def algebraic_vars(self) -> list[Variable]:
        """Get all algebraic variables."""
        return self.get_variables_by_type(VariableType.ALGEBRAIC)

    @property
    def inputs(self) -> list[Variable]:
        """Get all input variables."""
        return self.get_variables_by_type(VariableType.INPUT)

    @property
    def outputs(self) -> list[Variable]:
        """Get all output variables."""
        return self.get_variables_by_type(VariableType.OUTPUT)

    @property
    def parameters(self) -> list[Variable]:
        """Get all parameters and constants."""
        return [
            v
            for v in self.variables
            if v.var_type in (VariableType.PARAMETER, VariableType.CONSTANT)
        ]

    @property
    def n_states(self) -> int:
        """Number of continuous states."""
        return len(self.states)

    @property
    def n_discrete_states(self) -> int:
        """Number of discrete states."""
        return len(self.discrete_states)

    @property
    def n_algebraic(self) -> int:
        """Number of algebraic variables."""
        return len(self.algebraic_vars)

    @property
    def n_inputs(self) -> int:
        """Number of inputs."""
        return len(self.inputs)

    @property
    def n_outputs(self) -> int:
        """Number of outputs."""
        return len(self.outputs)

    @property
    def n_parameters(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

    def validate(
        self,
        check_undefined_vars: bool = True,
        check_array_bounds: bool = True,
        check_missing_values: bool = True,
        check_derivatives: bool = True,
        check_balance: bool = True,
    ) -> list[str]:
        """
        Validate the model and return a list of error messages.

        This method performs comprehensive validation including:
        - Undefined variable detection in expressions
        - Array index bounds checking
        - Missing start/parameter value detection
        - Derivative equation consistency
        - Equation/variable balance checking

        Note: BLT analysis is handled by Rumoca, not Cyecca.

        Args:
            check_undefined_vars: Check for undefined variable references
            check_array_bounds: Check array index bounds
            check_missing_values: Check for missing start/parameter values
            check_derivatives: Check derivative equations for states
            check_balance: Check equation/variable balance

        Returns:
            Empty list if valid, otherwise list of error messages.
            For detailed validation results, use validate_detailed() instead.
        """
        from cyecca.ir.validation import validate_model

        result = validate_model(
            self,
            check_undefined_vars=check_undefined_vars,
            check_array_bounds=check_array_bounds,
            check_missing_values=check_missing_values,
            check_derivatives=check_derivatives,
            check_balance=check_balance,
        )

        # Return only error messages for backward compatibility
        return [str(issue) for issue in result.errors]

    def validate_detailed(
        self,
        check_undefined_vars: bool = True,
        check_array_bounds: bool = True,
        check_missing_values: bool = True,
        check_derivatives: bool = True,
        check_balance: bool = True,
    ):
        """
        Perform detailed validation of the model.

        Returns a ValidationResult object with all issues (errors, warnings, info)
        categorized by type and severity.

        Note: BLT analysis is handled by Rumoca, not Cyecca.

        Args:
            check_undefined_vars: Check for undefined variable references
            check_array_bounds: Check array index bounds
            check_missing_values: Check for missing start/parameter values
            check_derivatives: Check derivative equations for states
            check_balance: Check equation/variable balance

        Returns:
            ValidationResult with all issues found
        """
        from cyecca.ir.validation import validate_model

        return validate_model(
            self,
            check_undefined_vars=check_undefined_vars,
            check_array_bounds=check_array_bounds,
            check_missing_values=check_missing_values,
            check_derivatives=check_derivatives,
            check_balance=check_balance,
        )

    def add_algorithm(self, algo: "AlgorithmSection") -> None:
        """Add an algorithm section to the model."""
        self.algorithms.append(algo)

    def add_event(self, event: "Event") -> None:
        """Add an event to the model."""
        self.events.append(event)

    def __str__(self):
        """String representation of the model."""
        lines = [f"Model: {self.name}"]
        if self.description:
            lines.append(f"  Description: {self.description}")
        lines.append(f"  States: {self.n_states}")
        lines.append(f"  Discrete States: {self.n_discrete_states}")
        lines.append(f"  Algebraic Vars: {self.n_algebraic}")
        lines.append(f"  Inputs: {self.n_inputs}")
        lines.append(f"  Outputs: {self.n_outputs}")
        lines.append(f"  Parameters: {self.n_parameters}")
        lines.append(f"  Equations: {len(self.equations)}")
        if self.initial_equations:
            lines.append(f"  Initial Equations: {len(self.initial_equations)}")
        if self.algorithms:
            lines.append(f"  Algorithms: {len(self.algorithms)}")
        if self.initial_algorithms:
            lines.append(f"  Initial Algorithms: {len(self.initial_algorithms)}")
        if self.events:
            lines.append(f"  Events: {len(self.events)}")
        if self.metadata:
            lines.append(f"  Metadata: {list(self.metadata.keys())}")
        return "\n".join(lines)

    def __repr__(self):
        """Detailed representation of the model (same as __str__)."""
        return self.__str__()
