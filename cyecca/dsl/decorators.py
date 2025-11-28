"""
Model decorators for the Cyecca DSL.

This module contains:
- @model: Main decorator for creating models
- @block: Decorator for signal-flow blocks
- @function: Decorator for Modelica functions
- ModelMetadata: Metadata container for models
- var(): Factory function for variable declarations
- submodel(): Factory function for submodel declarations

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
from beartype import beartype

from cyecca.dsl.context import (
    execute_algorithm_method,
    execute_equations_method,
    execute_initial_equations_method,
    is_algorithm_method,
    is_equations_method,
    is_initial_equations_method,
)
from cyecca.dsl.equations import ArrayEquation, Assignment, Equation, WhenClause
from cyecca.dsl.instance import ModelInstance
from cyecca.dsl.types import DType, Shape, SubmodelField, Var, VarKind

if TYPE_CHECKING:
    pass


# =============================================================================
# Model metadata
# =============================================================================


@dataclass
class ModelMetadata:
    """Metadata extracted from a model class by the @model decorator."""

    variables: Dict[str, Var] = field(default_factory=dict)
    submodels: Dict[str, SubmodelField] = field(default_factory=dict)


# =============================================================================
# Function metadata
# =============================================================================


@dataclass
class FunctionMetadata:
    """Metadata for a function, extracted from FlatModel."""

    name: str
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    protected_names: List[str] = field(default_factory=list)
    algorithm_assignments: List[Assignment] = field(default_factory=list)
    algorithm_locals: List[str] = field(default_factory=list)


# =============================================================================
# Factory function for var() - the unified variable declaration
# =============================================================================


@beartype
def var(
    default: Optional[Union[float, int, List[float], np.ndarray]] = None,
    dtype: DType = DType.REAL,
    shape: Shape = (),
    unit: Optional[str] = None,
    desc: str = "",
    start: Optional[Union[float, int, List[float], np.ndarray]] = None,
    fixed: bool = False,
    min: Optional[Union[float, int]] = None,
    max: Optional[Union[float, int]] = None,
    nominal: Optional[Union[float, int]] = None,
    # Variability flags (Modelica-style)
    parameter: bool = False,
    discrete: bool = False,
    input: bool = False,
    output: bool = False,
    constant: bool = False,
    # Visibility (Modelica-style)
    protected: bool = False,
) -> Var:
    """
    Declare a variable in a Cyecca model.

    This is the unified way to declare all types of variables. The classification
    (state, algebraic, parameter, input, output) is determined automatically based
    on the flags and equation analysis.

    Parameters
    ----------
    default : float or array, optional
        Default value (used if start not specified)
    dtype : DType, optional
        Variable type (REAL, INTEGER, BOOLEAN). Default: REAL
    shape : tuple of int, optional
        Array shape: () for scalar, (n,) for vector, (m,n) for matrix, etc.
    unit : str, optional
        Physical unit (e.g., "m", "rad/s")
    desc : str, optional
        Description
    start : float or array, optional
        Initial value (Modelica-style, takes precedence over default)
    fixed : bool, optional
        If True, start value is used as fixed initial condition
    min : float, optional
        Minimum bound for the variable
    max : float, optional
        Maximum bound for the variable
    nominal : float, optional
        Nominal value for scaling

    Variability/Causality Prefixes (Modelica-style)
    ------------------------------------------------
    parameter : bool, optional
        If True, variable is constant during simulation
    discrete : bool, optional
        If True, variable is piecewise constant (changes only at events)
    input : bool, optional
        If True, variable value is provided externally
    output : bool, optional
        If True, variable value is computed internally and exposed
    constant : bool, optional
        If True, variable is a compile-time constant

    Visibility (Modelica-style)
    ---------------------------
    protected : bool, optional
        If True, variable is internal (not part of public interface)

    Automatic Classification
    ------------------------
    If none of the flags are True:
    - If der(var) appears in equations → state variable
    - Otherwise → algebraic variable
    """
    return Var(
        default=default,
        dtype=dtype,
        shape=shape,
        unit=unit,
        desc=desc,
        start=start,
        fixed=fixed,
        min=min,
        max=max,
        nominal=nominal,
        parameter=parameter,
        discrete=discrete,
        input=input,
        output=output,
        constant=constant,
        protected=protected,
    )


@beartype
def submodel(model_class: Type) -> SubmodelField:
    """
    Declare a submodel (nested model).

    Submodels allow hierarchical composition of models.

    Example
    -------
    >>> from cyecca.dsl import model, var, submodel
    >>> @model
    ... class Controller:
    ...     gain = var(1.0, parameter=True)
    >>> @model
    ... class System:
    ...     ctrl = submodel(Controller)
    """
    return SubmodelField(model_class=model_class)


# =============================================================================
# @model decorator
# =============================================================================


@beartype
def model(cls: Type[Any]) -> Type[Any]:
    """
    Decorator to convert a class into a Cyecca model.

    Processes field descriptors (var, submodel) and sets up the class
    for equation-based modeling with automatic variable classification.

    Example
    -------
    >>> from cyecca.dsl import model, var, der, sin, equations
    >>> @model
    ... class Pendulum:
    ...     g = var(9.81, parameter=True)
    ...     l = var(1.0, parameter=True)
    ...     theta = var(start=0.5)
    ...     omega = var()
    ...     x = var(output=True)
    ...
    ...     @equations
    ...     def _(m):
    ...         der(m.theta) == m.omega
    ...         der(m.omega) == -m.g / m.l * sin(m.theta)
    ...         m.x == m.l * sin(m.theta)
    """
    metadata = ModelMetadata()

    # Process class attributes
    for name, value in list(vars(cls).items()):
        if isinstance(value, Var):
            value.name = name
            metadata.variables[name] = value
        elif isinstance(value, SubmodelField):
            value.name = name
            metadata.submodels[name] = value

    # Store metadata on the class
    cls._dsl_metadata = metadata

    # Find decorated methods
    equations_methods: List[Callable] = []
    initial_equations_methods: List[Callable] = []
    algorithm_methods: List[Callable] = []
    for name, value in vars(cls).items():
        if callable(value) and is_equations_method(value):
            equations_methods.append(value)
        if callable(value) and is_initial_equations_method(value):
            initial_equations_methods.append(value)
        if callable(value) and is_algorithm_method(value):
            algorithm_methods.append(value)

    # Validate: equations/initial_equations/algorithm must use decorators
    equations_attr = getattr(cls, "equations", None)
    if equations_attr is not None and not is_equations_method(equations_attr):
        raise TypeError(f"Model '{cls.__name__}': equations() must use @equations decorator.")

    initial_equations_attr = getattr(cls, "initial_equations", None)
    if initial_equations_attr is not None and not is_initial_equations_method(initial_equations_attr):
        raise TypeError(f"Model '{cls.__name__}': initial_equations() must use @initial_equations decorator.")

    algorithm_attr = getattr(cls, "algorithm", None)
    if algorithm_attr is not None and not is_algorithm_method(algorithm_attr):
        raise TypeError(f"Model '{cls.__name__}': algorithm() must use @algorithm decorator.")

    class ModelClass(ModelInstance):
        __doc__ = cls.__doc__
        __name__ = cls.__name__
        __qualname__ = cls.__qualname__
        __module__ = cls.__module__

        _equations_methods = equations_methods
        _initial_equations_methods = initial_equations_methods
        _algorithm_methods = algorithm_methods

        def __init__(self, name: str = ""):
            super().__init__(cls, name=name or cls.__name__)

        def get_equations(self) -> List[Union[Equation, ArrayEquation, WhenClause]]:
            """Execute all @equations methods and collect equations."""
            all_equations: List[Union[Equation, ArrayEquation, WhenClause]] = []
            for method in self._equations_methods:
                eqs = execute_equations_method(method, self)
                all_equations.extend(eqs)
            return all_equations

        def get_initial_equations(self) -> List[Equation]:
            """Execute all @initial_equations methods and collect equations."""
            all_equations: List[Equation] = []
            for method in self._initial_equations_methods:
                eqs = execute_initial_equations_method(method, self)
                all_equations.extend(eqs)
            return all_equations

        def get_algorithm(self) -> List[Assignment]:
            """Execute all @algorithm methods and collect assignments."""
            all_assignments: List[Assignment] = []
            for method in self._algorithm_methods:
                assigns = execute_algorithm_method(method, self)
                all_assignments.extend(assigns)
            return all_assignments

    ModelClass._dsl_metadata = metadata

    return ModelClass


# =============================================================================
# @function decorator - Modelica functions (Ch. 12)
# =============================================================================


@beartype
def function(cls: Type[Any]) -> Type[Any]:
    """
    Decorator to convert a class into a Cyecca function.

    A function is a restricted model (Modelica Ch. 12) that:
    - Uses only algorithm sections (no equations)
    - All public non-parameter variables must be input or output
    - Cannot have states (no der())
    - Is evaluated once when called, not continuously simulated

    Raises
    ------
    TypeError
        If a public non-parameter variable lacks input or output prefix.
        If equations() method is defined (functions use algorithm only).
    """
    errors = []

    # Check for @equations methods (not allowed in functions)
    for name, value in vars(cls).items():
        if callable(value) and is_equations_method(value):
            errors.append("Functions cannot have @equations - use @algorithm only")
            break

    # Validate all public non-parameter variables have input/output
    for name, value in vars(cls).items():
        if isinstance(value, Var):
            if value.protected:
                continue
            if value.parameter or value.constant:
                continue
            if not value.input and not value.output:
                errors.append(
                    f"  - '{name}': must have input=True or output=True "
                    f"(or use protected=True for intermediate variables)"
                )

    if errors:
        error_msg = f"Function '{cls.__name__}' violates Modelica function constraints.\n" + "\n".join(errors)
        raise TypeError(error_msg)

    # Check for @algorithm methods
    has_algorithm = any(callable(value) and is_algorithm_method(value) for name, value in vars(cls).items())
    if not has_algorithm:
        raise TypeError(f"Function '{cls.__name__}' must have an @algorithm method")

    model_cls = model(cls)
    model_cls._is_function = True

    def get_function_metadata(self) -> FunctionMetadata:
        """Get function-specific metadata."""
        flat = self.flatten()
        protected_names = [name for name, v in self._metadata.variables.items() if v.protected]

        return FunctionMetadata(
            name=flat.name,
            input_names=flat.input_names,
            output_names=flat.output_names,
            param_names=flat.param_names,
            protected_names=protected_names,
            algorithm_assignments=flat.algorithm_assignments,
            algorithm_locals=flat.algorithm_locals,
        )

    model_cls.get_function_metadata = get_function_metadata

    return model_cls


# =============================================================================
# @block decorator - Modelica block (signal-flow)
# =============================================================================


@beartype
def block(cls: Type[Any]) -> Type[Any]:
    """
    Decorator to convert a class into a Cyecca block.

    A block is a specialized model for signal-flow (causal) modeling.
    In Modelica, blocks have a key restriction: all public (non-protected)
    variables that are not parameters must have input or output prefix.

    Raises
    ------
    TypeError
        If a public non-parameter variable lacks input or output prefix.
    """
    errors = []
    for name, value in vars(cls).items():
        if isinstance(value, Var):
            if value.protected:
                continue
            if value.parameter or value.constant:
                continue
            if not value.input and not value.output:
                errors.append(
                    f"  - '{name}': public variable must have input=True or output=True "
                    f"(or use protected=True for internal variables)"
                )

    if errors:
        error_msg = (
            f"Block '{cls.__name__}' violates Modelica block constraints.\n"
            f"All public non-parameter variables must have input or output prefix:\n" + "\n".join(errors)
        )
        raise TypeError(error_msg)

    model_cls = model(cls)
    model_cls._is_block = True

    return model_cls
