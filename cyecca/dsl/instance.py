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

from cyecca.dsl.equations import ArrayEquation, Assignment, Equation, IfEquation, WhenClause
from cyecca.dsl.expr import Expr, ExprKind, find_derivatives, format_indices, is_array_state
from cyecca.dsl.flat_model import FlatModel
from cyecca.dsl.types import Var, VarKind
from cyecca.dsl.variables import SymbolicVar, TimeVar

if TYPE_CHECKING:
    from cyecca.dsl.decorators import ModelMetadata


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
        md = self._metadata

        # Create symbols for all variables
        for name, v in md.variables.items():
            self._sym_vars[name] = SymbolicVar(name, v, self)

        # Submodels - create nested instances and flatten their symbols
        for name, subfld in md.submodels.items():
            # Instantiate the submodel class (which is already decorated with @model)
            # This creates a ModelClass instance with proper equations() method
            sub_instance = subfld.model_class(name=name)
            # Store the overrides on the instance for use during flattening
            sub_instance._param_overrides = subfld.overrides
            self._submodels[name] = sub_instance

            # Flatten submodel symbols into parent with prefixed names
            for var_name, sym_var in sub_instance._sym_vars.items():
                full_name = f"{name}.{var_name}"
                self._sym_vars[full_name] = SymbolicVar(full_name, sym_var._var, self)

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

        This method performs automatic variable classification:
        1. Variables with parameter=True → parameters
        2. Variables with constant=True → constants (treated as parameters)
        3. Variables with discrete=True → discrete (piecewise constant)
        4. Variables with input=True → inputs
        5. Variables with output=True → outputs
        6. Variables whose der() appears in equations → states
        7. Remaining variables → algebraic

        Parameters
        ----------
        expand_arrays : bool, default=True
            If True, array equations like `der(pos) == vel` are expanded to
            scalar equations.
            If False, array equations are kept as-is for MX backend.

        Returns
        -------
        FlatModel
            A flattened model with all variables and equations.
        """
        md = self._metadata

        # Collect equations using the new @equations-based approach
        raw_equations = self.get_equations()
        equations: List[Equation] = []
        array_equations: List[ArrayEquation] = []
        when_clauses_from_equations: List[WhenClause] = []
        if_equations: List[IfEquation] = []

        for eq in raw_equations:
            if isinstance(eq, ArrayEquation):
                if expand_arrays:
                    equations.extend(eq.expand())
                else:
                    array_equations.append(eq)
            elif isinstance(eq, Equation):
                equations.append(eq)
            elif isinstance(eq, WhenClause):
                when_clauses_from_equations.append(eq)
            elif isinstance(eq, IfEquation):
                if_equations.append(eq)
            else:
                raise TypeError(f"Expected Equation, ArrayEquation, WhenClause, or IfEquation, got {type(eq)}")

        # Collect equations from submodels (with prefixed variable names)
        for sub_name, sub_instance in self._submodels.items():
            for eq in sub_instance.get_equations():
                if isinstance(eq, Equation):
                    prefixed_eq = eq._prefix_names(sub_name)
                    equations.append(prefixed_eq)
                elif isinstance(eq, ArrayEquation):
                    if expand_arrays:
                        for scalar_eq in eq.expand():
                            prefixed_eq = scalar_eq._prefix_names(sub_name)
                            equations.append(prefixed_eq)
                    else:
                        array_equations.append(eq)
                elif isinstance(eq, WhenClause):
                    prefixed_wc = eq._prefix_names(sub_name)
                    when_clauses_from_equations.append(prefixed_wc)
                elif isinstance(eq, IfEquation):
                    prefixed_if = eq._prefix_names(sub_name)
                    if_equations.append(prefixed_if)

        # Expand if-equations into regular equations with conditional expressions
        # This converts: if cond then y==e1 else y==e2 end if
        # Into: y == if_then_else(cond, e1, e2)
        for if_eq in if_equations:
            expanded = if_eq.expand()
            equations.extend(expanded)

        # Collect algorithm assignments
        algorithm_assignments: List[Assignment] = []
        algorithm_locals: List[str] = []

        for assign in self.get_algorithm():
            if isinstance(assign, Assignment):
                algorithm_assignments.append(assign)
                if assign.is_local and assign.target not in algorithm_locals:
                    algorithm_locals.append(assign.target)
            else:
                raise TypeError(f"Expected Assignment in algorithm, got {type(assign)}")

        # Collect initial equations
        initial_equations_list: List[Equation] = []

        for eq in self.get_initial_equations():
            if isinstance(eq, Equation):
                initial_equations_list.append(eq)
            elif isinstance(eq, ArrayEquation):
                initial_equations_list.extend(eq.expand())
            else:
                raise TypeError(f"Expected Equation in initial_equations, got {type(eq)}")

        # Collect initial equations from submodels
        for sub_name, sub_instance in self._submodels.items():
            for eq in sub_instance.get_initial_equations():
                if isinstance(eq, Equation):
                    prefixed_eq = eq._prefix_names(sub_name)
                    initial_equations_list.append(prefixed_eq)
                elif isinstance(eq, ArrayEquation):
                    for scalar_eq in eq.expand():
                        prefixed_eq = scalar_eq._prefix_names(sub_name)
                        initial_equations_list.append(prefixed_eq)

        when_clauses_list: List[WhenClause] = list(when_clauses_from_equations)

        # Find all derivatives (der(x)) used in equations to identify states
        derivatives_used: set[str] = set()
        for eq in equations:
            derivatives_used.update(find_derivatives(eq.lhs))
            derivatives_used.update(find_derivatives(eq.rhs))

        # Also check array equations for derivatives
        array_state_names: set[str] = set()
        for arr_eq in array_equations:
            if arr_eq.is_derivative:
                base_name = arr_eq.lhs_var.base_name
                array_state_names.add(base_name)

        # All equations (stored as-is, backend does residual conversion)
        all_equations: List[Equation] = []
        array_equations_map: Dict[str, Any] = {}
        output_equations_map: Dict[str, Expr] = {}

        # Classify variables
        state_names: List[str] = []
        state_vars: Dict[str, Var] = {}
        state_defaults: Dict[str, Any] = {}

        param_names: List[str] = []
        param_vars: Dict[str, Var] = {}
        param_defaults: Dict[str, Any] = {}

        input_names: List[str] = []
        input_vars: Dict[str, Var] = {}
        input_defaults: Dict[str, Any] = {}

        discrete_names: List[str] = []
        discrete_vars: Dict[str, Var] = {}
        discrete_defaults: Dict[str, Any] = {}

        output_names: List[str] = []
        output_vars: Dict[str, Var] = {}

        algebraic_names: List[str] = []
        algebraic_vars: Dict[str, Var] = {}

        # First pass: classify based on flags
        output_name_set: set[str] = set()
        for name, v in md.variables.items():
            if v.constant or v.parameter:
                v.kind = VarKind.CONSTANT if v.constant else VarKind.PARAMETER
                param_names.append(name)
                param_vars[name] = v
                if v.default is not None:
                    param_defaults[name] = v.default
                elif v.start is not None:
                    param_defaults[name] = v.start
            elif v.input:
                v.kind = VarKind.INPUT
                input_names.append(name)
                input_vars[name] = v
                val = v.get_initial_value()
                if val is not None:
                    input_defaults[name] = val
            elif v.discrete:
                v.kind = VarKind.DISCRETE
                discrete_names.append(name)
                discrete_vars[name] = v
                val = v.get_initial_value()
                if val is not None:
                    discrete_defaults[name] = val
            elif v.output:
                v.kind = VarKind.OUTPUT
                output_names.append(name)
                output_vars[name] = v
                output_name_set.add(name)
            elif (
                name in derivatives_used or is_array_state(name, v.shape, derivatives_used) or name in array_state_names
            ):
                v.kind = VarKind.STATE
                state_names.append(name)
                state_vars[name] = v
                val = v.get_initial_value()
                if val is not None:
                    state_defaults[name] = val
            else:
                v.kind = VarKind.ALGEBRAIC
                algebraic_names.append(name)
                algebraic_vars[name] = v

        # Classify scalar equations - store all, extract outputs for convenience
        # For array outputs, collect elements by their indices to build aggregate expression
        array_output_elements: Dict[str, Dict[tuple, Expr]] = {}  # {base_name: {indices: rhs}}
        
        for eq in equations:
            # Extract output equations for convenience (output_var == expr)
            if eq.lhs.kind == ExprKind.VARIABLE and eq.lhs.name in output_name_set:
                base_name = eq.lhs.name
                indices = eq.lhs.indices
                
                if indices:
                    # Array element equation - collect for aggregation
                    if base_name not in array_output_elements:
                        array_output_elements[base_name] = {}
                    array_output_elements[base_name][indices] = eq.rhs
                else:
                    # Scalar output equation
                    output_equations_map[base_name] = eq.rhs
            # All equations go into the flat list
            all_equations.append(eq)
        
        # Build aggregate ARRAY_LITERAL expressions for array outputs
        for base_name, elements in array_output_elements.items():
            if base_name in output_vars:
                shape = output_vars[base_name].shape
                if shape:
                    # Collect elements in row-major order
                    from cyecca.dsl.expr import iter_indices
                    rhs_elements = []
                    for indices in iter_indices(shape):
                        if indices in elements:
                            rhs_elements.append(elements[indices])
                        else:
                            # Missing element - use zero as default
                            rhs_elements.append(Expr(ExprKind.CONSTANT, value=0.0))
                    # Create ARRAY_LITERAL with the collected elements
                    output_equations_map[base_name] = Expr(ExprKind.ARRAY_LITERAL, tuple(rhs_elements))

        # Classify array equations
        for arr_eq in array_equations:
            if arr_eq.is_derivative:
                base_name = arr_eq.lhs_var.base_name
                array_equations_map[base_name] = {
                    "shape": arr_eq.lhs_var.shape,
                    "rhs": arr_eq.rhs,
                }

        # Helper function to recursively collect submodel variables
        def collect_submodel_vars(
            sub_instance: "ModelInstance",
            prefix: str,
            all_derivatives: set[str],
        ) -> None:
            """Recursively collect variables from a submodel and its nested submodels."""
            # Get parameter overrides for this submodel instance
            overrides = getattr(sub_instance, "_param_overrides", {})

            # Collect derivatives from this submodel's equations
            sub_derivatives: set[str] = set()
            for eq in sub_instance.get_equations():
                if isinstance(eq, Equation):
                    for der_name in find_derivatives(eq.lhs):
                        sub_derivatives.add(f"{prefix}.{der_name}")
                    for der_name in find_derivatives(eq.rhs):
                        sub_derivatives.add(f"{prefix}.{der_name}")

            # Add this submodel's direct variables
            for name, v in sub_instance._metadata.variables.items():
                full_name = f"{prefix}.{name}"

                if (
                    full_name in state_vars
                    or full_name in input_vars
                    or full_name in output_vars
                    or full_name in param_vars
                    or full_name in algebraic_vars
                ):
                    continue

                sub_v = Var(
                    dtype=v.dtype,
                    default=v.default,
                    shape=v.shape,
                    unit=v.unit,
                    desc=v.desc,
                    start=v.start,
                    fixed=v.fixed,
                    min=v.min,
                    max=v.max,
                    nominal=v.nominal,
                    parameter=v.parameter,
                    discrete=v.discrete,
                    input=v.input,
                    output=v.output,
                    constant=v.constant,
                    protected=v.protected,
                    name=full_name,
                    flow=v.flow,
                )

                if v.constant or v.parameter:
                    sub_v.kind = VarKind.CONSTANT if v.constant else VarKind.PARAMETER
                    param_names.append(full_name)
                    param_vars[full_name] = sub_v
                    # Apply override if present, otherwise use default/start
                    if name in overrides:
                        param_defaults[full_name] = overrides[name]
                    elif v.default is not None:
                        param_defaults[full_name] = v.default
                    elif v.start is not None:
                        param_defaults[full_name] = v.start
                elif v.input:
                    sub_v.kind = VarKind.INPUT
                    input_names.append(full_name)
                    input_vars[full_name] = sub_v
                    val = sub_v.get_initial_value()
                    if val is not None:
                        input_defaults[full_name] = val
                elif v.discrete:
                    sub_v.kind = VarKind.DISCRETE
                    discrete_names.append(full_name)
                    discrete_vars[full_name] = sub_v
                    val = sub_v.get_initial_value()
                    if val is not None:
                        discrete_defaults[full_name] = val
                elif v.output:
                    sub_v.kind = VarKind.OUTPUT
                    output_names.append(full_name)
                    output_vars[full_name] = sub_v
                elif full_name in sub_derivatives or full_name in all_derivatives:
                    sub_v.kind = VarKind.STATE
                    state_names.append(full_name)
                    state_vars[full_name] = sub_v
                    val = sub_v.get_initial_value()
                    if val is not None:
                        state_defaults[full_name] = val
                else:
                    sub_v.kind = VarKind.ALGEBRAIC
                    algebraic_names.append(full_name)
                    algebraic_vars[full_name] = sub_v

            # Recursively process nested submodels
            for nested_name, nested_sub in sub_instance._submodels.items():
                nested_prefix = f"{prefix}.{nested_name}"
                collect_submodel_vars(nested_sub, nested_prefix, all_derivatives)

        # Collect all derivatives from all equations (including nested)
        all_derivatives: set[str] = set(derivatives_used)

        # Add submodel variables (recursively)
        for sub_name, sub in self._submodels.items():
            collect_submodel_vars(sub, sub_name, all_derivatives)

        return FlatModel(
            name=self._name,
            state_names=state_names,
            input_names=input_names,
            output_names=output_names,
            param_names=param_names,
            discrete_names=discrete_names,
            algebraic_names=algebraic_names,
            state_vars=state_vars,
            input_vars=input_vars,
            output_vars=output_vars,
            param_vars=param_vars,
            discrete_vars=discrete_vars,
            algebraic_vars=algebraic_vars,
            equations=all_equations,
            array_equations=array_equations_map,
            output_equations=output_equations_map,
            state_defaults=state_defaults,
            input_defaults=input_defaults,
            discrete_defaults=discrete_defaults,
            param_defaults=param_defaults,
            initial_equations=initial_equations_list,
            when_clauses=when_clauses_list,
            if_equations=if_equations,
            algorithm_assignments=algorithm_assignments,
            algorithm_locals=algorithm_locals,
            expand_arrays=expand_arrays,
        )


# Alias for type hints
Model = ModelInstance
