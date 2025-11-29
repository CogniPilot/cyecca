"""
ModelInstance and SubmodelProxy for the Cyecca DSL.

This module contains the runtime model instance class and submodel proxy.

================================================================================
PROTOTYPE MODE - API IS IN FLUX
================================================================================

This DSL is in active prototype development. The API may change significantly
between versions. Do NOT maintain backward compatibility - iterate rapidly.

================================================================================
DESIGN PRINCIPLES - DO NOT REMOVE OR IGNORE
================================================================================

1. MODELICA CONFORMANCE: This DSL conforms to Modelica Language Spec v3.7-dev.
2. TYPE SAFETY: All functions MUST use beartype for runtime type checking.
3. SELF-CONTAINED: NO external compute libraries (CasADi, JAX) in core DSL.
4. IMMUTABILITY: Prefer immutable data structures where possible.
5. EXPLICIT > IMPLICIT: All behavior should be explicit and documented.

================================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Type, Union

from cyecca.dsl.equations import ArrayEquation, Equation, WhenClause
from cyecca.dsl.variables import SymbolicVar, TimeVar
from cyecca.ir.flat_model import FlatModel

if TYPE_CHECKING:
    from cyecca.dsl.decorators import ModelMetadata
    from cyecca.ir import IRModel


class SubmodelProxy:
    """Proxy for accessing submodel variables with dot notation.

    Supports nested submodels: m.resistor.pin.v
    """

    def __init__(self, name: str, instance: "ModelInstance", parent: "ModelInstance"):
        self._name = name
        self._instance = instance
        self._parent = parent

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            raise AttributeError(attr)

        # First check if this is a nested submodel
        if attr in self._instance._submodels:
            nested_instance = self._instance._submodels[attr]
            nested_name = f"{self._name}.{attr}"
            return SubmodelProxy(nested_name, nested_instance, self._parent)

        # Access submodel's symbolic variable with prefixed name
        full_name = f"{self._name}.{attr}"
        if full_name in self._parent._sym_vars:
            return self._parent._sym_vars[full_name]
        raise AttributeError(f"Submodel '{self._name}' has no attribute '{attr}'")


class ModelInstance:
    """
    Runtime instance of a model for building equations.

    Created when a @model decorated class is instantiated.
    """

    _dsl_metadata: "ModelMetadata"  # Set by @model decorator

    def __init__(self, model_class: Type[Any], name: str = ""):
        self._model_class = model_class
        self._name = name or model_class.__name__
        self._metadata: "ModelMetadata" = model_class._dsl_metadata

        # Symbolic storage - unified
        self._sym_vars: Dict[str, SymbolicVar] = {}
        self._submodels: Dict[str, "ModelInstance"] = {}
        self._param_overrides: Dict[str, Any] = {}  # Parameter overrides from submodel()

        self._time = TimeVar()  # Time variable (Modelica built-in)

        self._create_symbols()

    def _create_symbols(self) -> None:
        """Create symbolic variables for all fields."""
        from cyecca.dsl.symbol_table import create_symbols, flatten_submodel_symbols

        # Create symbols for all variables
        self._sym_vars = create_symbols(self._metadata, self)

        # Submodels - create nested instances and flatten their symbols
        for name, subfld in self._metadata.submodels.items():
            # Instantiate the submodel class (which is already decorated with @model)
            # This creates a ModelClass instance with proper equations() method
            sub_instance = subfld.model_class(name=name)
            # Store the overrides on the instance for use during flattening
            sub_instance._param_overrides = subfld.overrides
            self._submodels[name] = sub_instance

        # Flatten submodel symbols into parent with prefixed names
        flatten_submodel_symbols(self._submodels, self._sym_vars, self)

    def __getattr__(self, name: str) -> Any:
        """Provide access to symbolic variables and submodels."""
        if name.startswith("_"):
            raise AttributeError(name)

        # Check submodels first
        if name in self._submodels:
            return SubmodelProxy(name, self._submodels[name], self)

        # Check variables (unified storage)
        if name in self._sym_vars:
            return self._sym_vars[name]

        raise AttributeError(f"'{self._model_class.__name__}' has no attribute '{name}'")

    @property
    def time(self) -> TimeVar:
        """Time variable (Modelica built-in)."""
        return self._time

    # Subclasses store their @equations methods here
    _equations_methods: List[Callable] = []

    def get_equations(self) -> List[Union[Equation, ArrayEquation, WhenClause]]:
        """
        Execute all @equations methods and collect equations.

        This is overridden by the @model decorator to execute the
        actual @equations decorated methods.

        Returns
        -------
        List[Union[Equation, ArrayEquation, WhenClause]]
            All equations collected from @equations methods
        """
        return []

    def get_algorithm(self) -> List[Assignment]:
        """
        Override this method to define algorithm sections.

        Algorithm sections contain imperative assignments that are executed
        in order, unlike equations which are declarative. Use @ for assignments.

        Algorithm sections are useful for:
        - Computing intermediate values
        - Implementing control logic with if/else
        - Breaking complex expressions into readable steps

        Notes
        -----
        In Modelica, algorithm sections use := for assignment (vs == for equations).
        In this DSL, we use @ since := is Python's walrus operator.
        """
        return []

    def get_initial_equations(self) -> List[Equation]:
        """
        Override this method to define initial equations.

        Initial equations (Modelica: `initial equation` section) are used to
        specify initial conditions for simulation. They are solved once at
        t=0 to determine initial values of states and algebraic variables.

        This provides more flexibility than just using `start` values:
        - Can specify relationships between initial values
        - Can use equations rather than just fixed values
        - Can leave some variables to be computed from others

        Notes
        -----
        Modelica Spec: Section 8.6 - Initialization, Initial Equation, and Initial Algorithm
        """
        return []

    def flatten(self, expand_arrays: bool = True) -> FlatModel:
        """
        Flatten the model into a backend-agnostic representation.

        This delegates to IRModel.flatten() which is the single source of
        truth for flattening logic.

        Parameters
        ----------
        expand_arrays : bool, default=True
            If True, array equations are expanded to scalar equations.

        Returns
        -------
        FlatModel
            A flattened model ready for backend compilation.
        """
        return self.to_ir().flatten(expand_arrays=expand_arrays)

    def to_ir(self) -> "IRModel":
        """
        Convert this model instance to an IRModel.

        The IRModel is the clean intermediate representation that can be
        built either via this DSL or via the direct API. Both paths
        produce the same IR structure.

        Returns
        -------
        IRModel
            The intermediate representation of this model.

        Example
        -------
        .. code-block:: python

            from cyecca.dsl import Real, der, equations, model

            @model
            class Ball:
                h = Real(start=1.0)
                v = Real()

                @equations
                def _(m):
                    der(m.h) == m.v
                    der(m.v) == -9.81

            ball = Ball()
            ir = ball.to_ir()
            print(ir.summary())
        """
        from cyecca.ir import (
            DataType,
            IRAssignment,
            IREquation,
            IRInitialEquation,
            IRModel,
            IRReinit,
            IRVariable,
            IRWhenClause,
        )

        # Create the IRModel
        ir = IRModel(name=self._name)

        # Convert Var to IRVariable
        def var_to_ir_var(name: str, v: Var) -> IRVariable:
            # Map DType to DataType
            from cyecca.ir.types import DType

            dtype_map = {
                DType.REAL: DataType.REAL,
                DType.INTEGER: DataType.INTEGER,
                DType.BOOLEAN: DataType.BOOLEAN,
            }
            return IRVariable(
                name=name,
                dtype=dtype_map.get(v.dtype, DataType.REAL),
                shape=v.shape,
                start=v.get_initial_value(),
                unit=v.unit,
                description=v.desc,
                min=v.min,
                max=v.max,
                nominal=v.nominal,
                fixed=v.fixed,
                parameter=v.parameter,
                discrete=v.discrete,
                input=v.input,
                output=v.output,
                constant=v.constant,
                flow=v.flow,
            )

        # Add variables from this model
        for name, v in self._metadata.variables.items():
            ir.add_variable(var_to_ir_var(name, v))

        # Add equations
        from cyecca.dsl.equations import IfEquation, Reinit

        for eq in self.get_equations():
            if isinstance(eq, Equation):
                ir.add_equation(IREquation(lhs=eq.lhs, rhs=eq.rhs))
            elif isinstance(eq, ArrayEquation):
                # Expand array equations to scalar equations
                for scalar_eq in eq.expand():
                    ir.add_equation(IREquation(lhs=scalar_eq.lhs, rhs=scalar_eq.rhs))
            elif isinstance(eq, WhenClause):
                reinits = []
                for item in eq.body:
                    if isinstance(item, Reinit):
                        reinits.append(IRReinit(var_name=item.var_name, expr=item.expr))
                ir.add_when_clause(
                    IRWhenClause(
                        condition=eq.condition,
                        reinits=reinits,
                    )
                )
            elif isinstance(eq, IfEquation):
                # Expand if-equations to regular equations with conditional expressions
                for expanded_eq in eq.expand():
                    ir.add_equation(IREquation(lhs=expanded_eq.lhs, rhs=expanded_eq.rhs))

        # Add initial equations
        for eq in self.get_initial_equations():
            if isinstance(eq, Equation):
                ir.add_initial_equation(IRInitialEquation(lhs=eq.lhs, rhs=eq.rhs))
            elif isinstance(eq, ArrayEquation):
                for scalar_eq in eq.expand():
                    ir.add_initial_equation(IRInitialEquation(lhs=scalar_eq.lhs, rhs=scalar_eq.rhs))

        # Add algorithm assignments
        algorithm_section = []
        for assign in self.get_algorithm():
            algorithm_section.append(IRAssignment(var_name=assign.target, expr=assign.expr))
        if algorithm_section:
            ir.algorithms.append(algorithm_section)

        # Add submodels (recursively)
        for sub_name, sub_instance in self._submodels.items():
            sub_ir = sub_instance.to_ir()
            ir.add_submodel(sub_name, sub_ir)

        return ir


# Alias for type hints
Model = ModelInstance
