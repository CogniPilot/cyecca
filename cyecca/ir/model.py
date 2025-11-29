"""
IRModel - The core intermediate representation for models.

This is the canonical representation that can be created either
via the DSL (@model decorator) or via the direct API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from beartype import beartype

from cyecca.ir.equation import IRAssignment, IREquation, IRInitialEquation, IRReinit, IRWhenClause
from cyecca.ir.flat_model import FlatModel
from cyecca.ir.variable import IRVariable, NumericValue, VariableKind


@beartype
@dataclass
class IRModel:
    """
    The Intermediate Representation of a model.

    IRModel is the core data structure that represents a model before
    flattening and compilation. It can be created either:

    1. Via the decorator DSL (convenient, Modelica-like syntax):

        from cyecca.dsl import model, Real, der, equations

        @model
        class Ball:
            h = Real(start=1.0)
            v = Real()

            @equations
            def _(m):
                der(m.h) == m.v
                der(m.v) == -9.81

        ir = Ball().to_ir()

    2. Via the direct API (explicit, no magic):

        from cyecca.ir import IRModel, IRVariable, IREquation

        m = IRModel(name="Ball")
        m.add_variable(IRVariable("h", start=1.0))
        m.add_variable(IRVariable("v"))
        # Add equations...

    Both approaches produce the same IRModel, which can then be
    flattened and compiled to backends.

    Parameters
    ----------
    name : str
        Name of the model
    description : str
        Optional description

    Attributes
    ----------
    variables : dict
        Name -> IRVariable mapping
    equations : list
        List of IREquation
    when_clauses : list
        List of IRWhenClause for event handling
    initial_equations : list
        List of IRInitialEquation (solved at t=0)
    submodels : dict
        Name -> IRModel for hierarchical composition
    """

    name: str
    description: str = ""

    # Core model components
    variables: Dict[str, IRVariable] = field(default_factory=dict)
    equations: List[IREquation] = field(default_factory=list)
    when_clauses: List[IRWhenClause] = field(default_factory=list)
    initial_equations: List[IRInitialEquation] = field(default_factory=list)

    # Hierarchical composition
    submodels: Dict[str, "IRModel"] = field(default_factory=dict)

    # Connectors and connections (for component-based modeling)
    connectors: Dict[str, "IRConnector"] = field(default_factory=dict)
    connections: List[Tuple[str, str]] = field(default_factory=list)

    # Algorithm sections (for procedural code)
    algorithms: List[List[IRAssignment]] = field(default_factory=list)

    # --------------------------------------------------------------------------
    # Variable management
    # --------------------------------------------------------------------------

    def add_variable(self, var: IRVariable) -> IRVariable:
        """
        Add a variable to the model.

        Parameters
        ----------
        var : IRVariable
            Variable to add

        Returns
        -------
        IRVariable
            The added variable (for chaining)

        Raises
        ------
        ValueError
            If a variable with the same name already exists
        """
        if var.name in self.variables:
            raise ValueError(f"Variable '{var.name}' already exists in model '{self.name}'")
        self.variables[var.name] = var
        return var

    def get_variable(self, name: str) -> Optional[IRVariable]:
        """Get a variable by name, or None if not found."""
        return self.variables.get(name)

    def get_variables_by_kind(self, kind: VariableKind) -> List[IRVariable]:
        """Get all variables of a specific kind."""
        return [v for v in self.variables.values() if v.get_kind() == kind]

    @property
    def states(self) -> List[IRVariable]:
        """Get all state variables."""
        return self.get_variables_by_kind(VariableKind.STATE)

    @property
    def algebraic(self) -> List[IRVariable]:
        """Get all algebraic variables."""
        return self.get_variables_by_kind(VariableKind.ALGEBRAIC)

    @property
    def parameters(self) -> List[IRVariable]:
        """Get all parameters."""
        return self.get_variables_by_kind(VariableKind.PARAMETER)

    @property
    def inputs(self) -> List[IRVariable]:
        """Get all input variables."""
        return self.get_variables_by_kind(VariableKind.INPUT)

    @property
    def outputs(self) -> List[IRVariable]:
        """Get all output variables."""
        return self.get_variables_by_kind(VariableKind.OUTPUT)

    @property
    def discrete(self) -> List[IRVariable]:
        """Get all discrete variables."""
        return self.get_variables_by_kind(VariableKind.DISCRETE)

    # --------------------------------------------------------------------------
    # Equation management
    # --------------------------------------------------------------------------

    def add_equation(self, eq: IREquation) -> IREquation:
        """
        Add an equation to the model.

        Parameters
        ----------
        eq : IREquation
            Equation to add

        Returns
        -------
        IREquation
            The added equation (for chaining)
        """
        self.equations.append(eq)
        return eq

    def add_when_clause(self, when: IRWhenClause) -> IRWhenClause:
        """
        Add a when-clause for event handling.

        Parameters
        ----------
        when : IRWhenClause
            When-clause to add

        Returns
        -------
        IRWhenClause
            The added when-clause (for chaining)
        """
        self.when_clauses.append(when)
        return when

    def add_initial_equation(self, eq: IRInitialEquation) -> IRInitialEquation:
        """
        Add an initial equation (solved at t=0).

        Parameters
        ----------
        eq : IRInitialEquation
            Initial equation to add

        Returns
        -------
        IRInitialEquation
            The added equation (for chaining)
        """
        self.initial_equations.append(eq)
        return eq

    # --------------------------------------------------------------------------
    # Submodel management (hierarchical composition)
    # --------------------------------------------------------------------------

    def add_submodel(self, name: str, submodel: "IRModel") -> "IRModel":
        """
        Add a submodel (component) to this model.

        Parameters
        ----------
        name : str
            Instance name for the submodel
        submodel : IRModel
            The submodel to add

        Returns
        -------
        IRModel
            The added submodel (for chaining)
        """
        if name in self.submodels:
            raise ValueError(f"Submodel '{name}' already exists in model '{self.name}'")
        self.submodels[name] = submodel
        return submodel

    def add_connection(self, connector_a: str, connector_b: str) -> None:
        """
        Connect two connectors.

        Parameters
        ----------
        connector_a : str
            Path to first connector (e.g., "motor.flange")
        connector_b : str
            Path to second connector
        """
        self.connections.append((connector_a, connector_b))

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"IRModel('{self.name}', "
            f"vars={len(self.variables)}, "
            f"eqs={len(self.equations)}, "
            f"when={len(self.when_clauses)}, "
            f"sub={len(self.submodels)})"
        )

    def summary(self) -> str:
        """Return a human-readable summary of the model."""
        lines = [f"Model: {self.name}"]
        if self.description:
            lines.append(f"  {self.description}")
        lines.append(f"  States: {len(self.states)}")
        lines.append(f"  Algebraic: {len(self.algebraic)}")
        lines.append(f"  Parameters: {len(self.parameters)}")
        lines.append(f"  Inputs: {len(self.inputs)}")
        lines.append(f"  Outputs: {len(self.outputs)}")
        lines.append(f"  Discrete: {len(self.discrete)}")
        lines.append(f"  Equations: {len(self.equations)}")
        lines.append(f"  When-clauses: {len(self.when_clauses)}")
        lines.append(f"  Submodels: {len(self.submodels)}")
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # Flattening
    # --------------------------------------------------------------------------

    def flatten(self, expand_arrays: bool = True) -> "FlatModel":
        """
        Flatten the model into a backend-agnostic representation.

        This method:
        1. Recursively flattens submodels with prefixed names
        2. Classifies variables (state/algebraic/param/input/output/discrete)
        3. Collects all equations into a flat list

        Variable Classification:
        - parameter=True or constant=True → parameters
        - discrete=True → discrete
        - input=True → inputs
        - output=True → outputs
        - der(x) appears in equations → states
        - Otherwise → algebraic

        Parameters
        ----------
        expand_arrays : bool, default=True
            If True, array equations are expanded to scalar equations.

        Returns
        -------
        FlatModel
            A flattened model ready for backend compilation.
        """
        from cyecca.ir.expr import Expr, ExprKind, find_derivatives, is_array_state, iter_indices, prefix_expr
        from cyecca.ir.flat_model import FlatModel
        from cyecca.ir.types import Var, VarKind

        # Collect all equations (with prefixed names for submodels)
        all_equations: List[IREquation] = []
        all_initial_equations: List[IRInitialEquation] = []
        all_when_clauses: List[IRWhenClause] = []
        all_assignments: List[IRAssignment] = []

        def collect_from_model(model: "IRModel", prefix: str = "") -> None:
            """Recursively collect equations from model and submodels."""
            for eq in model.equations:
                if prefix:
                    prefixed_eq = IREquation(
                        lhs=prefix_expr(eq.lhs, prefix),
                        rhs=prefix_expr(eq.rhs, prefix),
                        description=eq.description,
                    )
                    all_equations.append(prefixed_eq)
                else:
                    all_equations.append(eq)

            for eq in model.initial_equations:
                if prefix:
                    prefixed_eq = IRInitialEquation(
                        lhs=prefix_expr(eq.lhs, prefix),
                        rhs=prefix_expr(eq.rhs, prefix),
                    )
                    all_initial_equations.append(prefixed_eq)
                else:
                    all_initial_equations.append(eq)

            for wc in model.when_clauses:
                if prefix:
                    prefixed_reinits = [
                        IRReinit(var_name=f"{prefix}.{r.var_name}", expr=prefix_expr(r.expr, prefix))
                        for r in wc.reinits
                    ]
                    prefixed_wc = IRWhenClause(
                        condition=prefix_expr(wc.condition, prefix),
                        reinits=prefixed_reinits,
                        description=wc.description,
                    )
                    all_when_clauses.append(prefixed_wc)
                else:
                    all_when_clauses.append(wc)

            for algo_section in model.algorithms:
                for assign in algo_section:
                    if prefix:
                        all_assignments.append(
                            IRAssignment(
                                var_name=f"{prefix}.{assign.var_name}",
                                expr=prefix_expr(assign.expr, prefix),
                            )
                        )
                    else:
                        all_assignments.append(assign)

            # Recurse into submodels
            for sub_name, sub_model in model.submodels.items():
                sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
                collect_from_model(sub_model, sub_prefix)

        collect_from_model(self)

        # Find all derivatives to identify state variables
        derivatives_used: set[str] = set()
        for eq in all_equations:
            derivatives_used.update(find_derivatives(eq.lhs))
            derivatives_used.update(find_derivatives(eq.rhs))

        # Collect all variables (with prefixed names for submodels)
        all_vars: Dict[str, Tuple[IRVariable, str]] = {}  # name -> (var, prefix)

        def collect_vars_from_model(model: "IRModel", prefix: str = "") -> None:
            for name, var in model.variables.items():
                full_name = f"{prefix}.{name}" if prefix else name
                all_vars[full_name] = (var, prefix)

            for sub_name, sub_model in model.submodels.items():
                sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
                collect_vars_from_model(sub_model, sub_prefix)

        collect_vars_from_model(self)

        # Classify variables
        state_names: List[str] = []
        state_vars: Dict[str, Var] = {}
        state_defaults: Dict[str, NumericValue] = {}

        param_names: List[str] = []
        param_vars: Dict[str, Var] = {}
        param_defaults: Dict[str, NumericValue] = {}

        input_names: List[str] = []
        input_vars: Dict[str, Var] = {}
        input_defaults: Dict[str, NumericValue] = {}

        discrete_names: List[str] = []
        discrete_vars: Dict[str, Var] = {}
        discrete_defaults: Dict[str, NumericValue] = {}

        output_names: List[str] = []
        output_vars: Dict[str, Var] = {}

        algebraic_names: List[str] = []
        algebraic_vars: Dict[str, Var] = {}

        output_name_set: set[str] = set()

        for full_name, (ir_var, _prefix) in all_vars.items():
            # Convert IRVariable to Var for FlatModel
            v = Var(
                name=full_name,
                shape=ir_var.shape,
                start=ir_var.start,
                fixed=ir_var.fixed,
                min=ir_var.min,
                max=ir_var.max,
                nominal=ir_var.nominal,
                unit=ir_var.unit,
                desc=ir_var.description,
                parameter=ir_var.parameter,
                discrete=ir_var.discrete,
                input=ir_var.input,
                output=ir_var.output,
                constant=ir_var.constant,
                flow=ir_var.flow,
            )

            if ir_var.constant or ir_var.parameter:
                v.kind = VarKind.CONSTANT if ir_var.constant else VarKind.PARAMETER
                param_names.append(full_name)
                param_vars[full_name] = v
                if ir_var.start is not None:
                    param_defaults[full_name] = ir_var.start
            elif ir_var.input:
                v.kind = VarKind.INPUT
                input_names.append(full_name)
                input_vars[full_name] = v
                if ir_var.start is not None:
                    input_defaults[full_name] = ir_var.start
            elif ir_var.discrete:
                v.kind = VarKind.DISCRETE
                discrete_names.append(full_name)
                discrete_vars[full_name] = v
                if ir_var.start is not None:
                    discrete_defaults[full_name] = ir_var.start
            elif ir_var.output:
                v.kind = VarKind.OUTPUT
                output_names.append(full_name)
                output_vars[full_name] = v
                output_name_set.add(full_name)
            elif full_name in derivatives_used or is_array_state(full_name, ir_var.shape, derivatives_used):
                v.kind = VarKind.STATE
                state_names.append(full_name)
                state_vars[full_name] = v
                if ir_var.start is not None:
                    state_defaults[full_name] = ir_var.start
            else:
                v.kind = VarKind.ALGEBRAIC
                algebraic_names.append(full_name)
                algebraic_vars[full_name] = v

        # Extract output equations (output_var == expr)
        output_equations: Dict[str, Expr] = {}
        array_output_elements: Dict[str, Dict[tuple, Expr]] = {}

        for eq in all_equations:
            if eq.lhs.kind == ExprKind.VARIABLE and eq.lhs.name in output_name_set:
                base_name = eq.lhs.name
                indices = eq.lhs.indices

                if indices:
                    if base_name not in array_output_elements:
                        array_output_elements[base_name] = {}
                    array_output_elements[base_name][indices] = eq.rhs
                else:
                    output_equations[base_name] = eq.rhs

        # Build aggregate ARRAY_LITERAL for array outputs
        for base_name, elements in array_output_elements.items():
            if base_name in output_vars:
                shape = output_vars[base_name].shape
                if shape:
                    rhs_elements = []
                    for indices in iter_indices(shape):
                        if indices in elements:
                            rhs_elements.append(elements[indices])
                        else:
                            rhs_elements.append(Expr(ExprKind.CONSTANT, value=0.0))
                    output_equations[base_name] = Expr(ExprKind.ARRAY_LITERAL, tuple(rhs_elements))

        return FlatModel(
            name=self.name,
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
            output_equations=output_equations,
            state_defaults=state_defaults,
            input_defaults=input_defaults,
            discrete_defaults=discrete_defaults,
            param_defaults=param_defaults,
            initial_equations=all_initial_equations,
            when_clauses=all_when_clauses,
            algorithm_assignments=all_assignments,
            expand_arrays=expand_arrays,
        )


@beartype
@dataclass
class IRConnector:
    """
    A connector type for component-based modeling.

    Connectors define the interface between components.
    They contain "potential" and "flow" variables that are
    handled specially during connection.

    Parameters
    ----------
    name : str
        Name of the connector type
    potentials : list
        Variables whose values must be equal when connected
    flows : list
        Variables whose sum must be zero when connected
    """

    name: str
    potentials: List[IRVariable] = field(default_factory=list)
    flows: List[IRVariable] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"IRConnector('{self.name}', pot={len(self.potentials)}, flow={len(self.flows)})"
