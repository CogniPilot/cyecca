"""
FlatModel - Backend-agnostic representation of a flattened model.

This is the output of the DSL that backends compile into executable functions.

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
from typing import Any, Dict, List

from cyecca.dsl.equations import Assignment, Equation, WhenClause
from cyecca.dsl.expr import Expr
from cyecca.dsl.types import Var


@dataclass
class FlatModel:
    """
    Flattened model representation - the output of the DSL.

    This is a backend-agnostic representation of the model that contains:
    - All variables (states, inputs, outputs, params) with metadata
    - All equations as expression trees
    - Default values for initialization

    Variable Classification (Modelica-conformant)
    ---------------------------------------------
    In Modelica, `input` and `output` are just **prefixes** on variables that
    indicate causality (how the variable interfaces with the outside world).
    They are NOT separate equation categories.

    - parameter=True → parameter (constant during simulation)
    - discrete=True → discrete (piecewise constant, changes at events)
    - input=True → input (value provided externally)
    - output=True → output (value computed internally, exposed externally)
    - der(var) in equations → state
    - otherwise → algebraic

    All equations come from a single equations() method. The `output_equations`
    field is populated by extracting equations of the form `output_var == expr`
    for backend convenience.

    Backends (CasADi, JAX, etc.) compile this into executable functions.
    """

    name: str

    # Variable lists (ordered)
    state_names: List[str]
    input_names: List[str]
    output_names: List[str]
    param_names: List[str]
    discrete_names: List[str]
    algebraic_names: List[str]

    # Variable metadata (using unified Var type)
    state_vars: Dict[str, Var]
    input_vars: Dict[str, Var]
    output_vars: Dict[str, Var]
    param_vars: Dict[str, Var]
    discrete_vars: Dict[str, Var]
    algebraic_vars: Dict[str, Var]

    # Equations
    # NOTE: In Modelica, all equations are in one section. The separation here
    # is for backend convenience only. Output equations are extracted from
    # equations that define output variables (var with output=True).
    derivative_equations: Dict[str, Expr]  # state_name -> rhs expression for der(state)
    output_equations: Dict[str, Expr]  # output_name -> rhs (extracted from equations())
    algebraic_equations: List[Equation]  # 0 = f(x, z) equations

    # Default values
    state_defaults: Dict[str, Any]
    input_defaults: Dict[str, Any]
    discrete_defaults: Dict[str, Any]
    param_defaults: Dict[str, Any]

    # Initial equations (Modelica: initial equation section)
    # These are solved once at t=0 to determine initial values
    initial_equations: List[Equation] = field(default_factory=list)

    # When-clauses (Modelica: when equations, MLS 8.5)
    # Event-driven equations with conditions and reinit statements
    when_clauses: List[WhenClause] = field(default_factory=list)

    # Array derivative equations (when expand_arrays=False)
    # For CasADi MX backend: keeps array structure for efficient matrix operations
    # Key is base variable name (e.g., 'pos'), value is {'shape': (3,), 'rhs': SymbolicVar}
    array_derivative_equations: Dict[str, Any] = field(default_factory=dict)

    # Algorithm section
    # Ordered list of assignments from algorithm() method
    algorithm_assignments: List[Assignment] = field(default_factory=list)
    # Local variables declared in algorithm section
    algorithm_locals: List[str] = field(default_factory=list)

    # Flattening mode
    expand_arrays: bool = True  # If False, array equations are kept as-is for MX backend

    def __repr__(self) -> str:
        parts = [f"'{self.name}'"]
        if self.state_names:
            parts.append(f"states={self.state_names}")
        if self.discrete_names:
            parts.append(f"discrete={self.discrete_names}")
        if self.input_names:
            parts.append(f"inputs={self.input_names}")
        if self.output_names:
            parts.append(f"outputs={self.output_names}")
        if self.param_names:
            parts.append(f"params={self.param_names}")
        if self.when_clauses:
            parts.append(f"when_clauses={len(self.when_clauses)}")
        return f"FlatModel({', '.join(parts)})"
