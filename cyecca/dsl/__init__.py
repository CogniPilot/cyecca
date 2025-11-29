"""
Cyecca DSL - A Modelica-inspired Domain Specific Language for Python.

This module provides a high-level, declarative API for defining dynamic systems
that closely mirrors Modelica syntax while remaining fully Pythonic.

================================================================================
MODELICA CONFORMANCE
================================================================================

For detailed conformance analysis, see: MODELICA_CONFORMANCE.md

Reference: https://specification.modelica.org/master/

Overall Conformance: ~25-30% of Modelica Language Specification
DAE Representation (Appendix B): ~60% complete

We implement a semi-explicit index-1 DAE:

    der(x) = f(x, z, u, p, t)           # Explicit ODE
    0 = g(x, z, u, p, t)                # Algebraic constraints
    y = h(x, z, u, p, t)                # Outputs
    when condition: reinit(x, x_new)    # Events

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
   
   - Variable flags (parameter, input, output, constant) follow Modelica semantics
   - Attributes (start, fixed, min, max, nominal, unit) match Modelica definitions
   - Equation-based modeling with der() operator
   - Automatic state/algebraic classification based on der() usage
   - Connector/connection semantics (future) will follow Modelica Chapter 9

2. TYPE SAFETY: All functions MUST use beartype for runtime type checking.
   - All public functions decorated with @beartype
   - All classes use proper type hints for full IDE autocomplete
   - Do NOT use Any where a more specific type is possible
   - Do NOT remove beartype decorators
   - WHEN ADDING NEW FUNCTIONS: Always add @beartype decorator
   - Import and use type aliases (Shape, Indices) from types.py

3. SELF-CONTAINED: This DSL module uses NO external compute libraries except:
   - Python standard library
   - beartype (for runtime type checking)
   - numpy (for NumericValue type hints only)
   
   The DSL builds an abstract model representation (expression trees, equation
   graphs) that can be compiled by separate backends (CasADi, JAX, etc.).
   
   DO NOT import CasADi, JAX, or other compute backends in the core DSL modules.

4. IMMUTABILITY: Prefer immutable data structures where possible.

5. EXPLICIT > IMPLICIT: All behavior should be explicit and documented.

================================================================================

Example
-------
>>> import matplotlib.pyplot as plt
>>> from cyecca.dsl import model, Real, sin, cos, der, equations
>>> from cyecca.dsl.backends import CasadiBackend
>>>
>>> @model
... class Pendulum:
...     '''Simple pendulum model.'''
...     # Parameters (constant during simulation)
...     g = Real(9.81, parameter=True, unit="m/s^2", desc="Gravity")
...     l = Real(1.0, parameter=True, unit="m", desc="Length")
...     
...     # States (automatically detected via der() usage)
...     theta = Real(start=0.5, unit="rad", desc="Angle")
...     omega = Real(unit="rad/s", desc="Angular velocity")
...     
...     # Outputs (explicitly flagged)
...     x = Real(output=True, unit="m", desc="Horizontal position")
...     y = Real(output=True, unit="m", desc="Vertical position")
...
...     @equations
...     def _(m):  # Use 'm' for model namespace (cleaner than 'self')
...         der(m.theta) == m.omega
...         der(m.omega) == -m.g / m.l * sin(m.theta)
...         m.x == m.l * sin(m.theta)
...         m.y == -m.l * cos(m.theta)
>>>
>>> # Create, compile, and simulate
>>> pend = Pendulum()
>>> compiled = CasadiBackend.compile(pend.flatten())
>>> result = compiled.simulate(tf=10.0)
>>>
>>> # Plot using model variables (autocomplete works on pend.theta)
>>> _ = plt.figure()  # doctest: +SKIP
>>> _ = plt.plot(result.t, result(pend.theta), label='theta')  # doctest: +SKIP
>>> _ = plt.plot(result.t, result(pend.omega), label='omega')  # doctest: +SKIP
>>> _ = plt.legend()  # doctest: +SKIP
>>> plt.show()  # doctest: +SKIP

Variable Types (Modelica-style)
-------------------------------
Use type constructors for Modelica-like syntax:

- Real(start=1.0, parameter=True)  → floating point variable
- Integer(5, parameter=True)       → integer variable
- Boolean(True, discrete=True)     → boolean variable
- String("name", parameter=True)   → string variable (limited support)

Variable Classification
-----------------------
Variables are classified automatically based on flags and equation analysis:

1. Real(parameter=True) → parameter (constant during simulation)
2. Real(constant=True) → constant (compile-time constant)
3. Real(input=True) → input (externally controlled)
4. Real(output=True) → output (computed, exposed)
5. If der(var) appears in equations → state
6. Otherwise → algebraic variable

Architecture
------------
The DSL is structured in two layers:

1. DSL Layer (no external compute libraries):
   - model.py: @model decorator, var(), expression trees
   - simulation.py: SimulationResult with plotting utilities
   - operators.py: Math functions that return Expr nodes

2. Backend Layer:
   - backends/casadi.py: Compiles FlatModel to CasADi functions
   - (future) backends/jax.py: Compiles FlatModel to JAX functions
"""

# Algorithm support
from cyecca.dsl.algorithm import AlgorithmVar, assign, local

# Causality analysis
from cyecca.dsl.causality import ImplicitBlock, SolvedEquation, SortedSystem, analyze_causality

# Context and when-clause support
from cyecca.dsl.context import algorithm, connect, else_eq, elseif_eq, equations, if_eq, initial_equations, reinit, when

# Decorators and var() factory
from cyecca.dsl.decorators import (
    Boolean,
    FunctionMetadata,
    Integer,
    ModelMetadata,
    Real,
    String,
    block,
    connector,
    function,
    model,
    submodel,
    var,
)

# Equations and statements
from cyecca.dsl.equations import Assignment, Equation, IfEquation, IfEquationBranch, Reinit, WhenClause

# Expression tree
from cyecca.dsl.expr import Expr, ExprKind

# Flat model representation
from cyecca.dsl.flat_model import FlatModel

# Model instance and alias
from cyecca.dsl.instance import Model, ModelInstance
from cyecca.dsl.operators import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    ceil,
    cos,
    cosh,
    exp,
    floor,
    log,
    log10,
    max,
    min,
    mod,
    sign,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)

# Core operators
from cyecca.dsl.operators_core import (
    and_,
    change,
    der,
    edge,
    eq,
    if_then_else,
    initial,
    ne,
    not_,
    or_,
    pre,
    sample,
    terminal,
)
from cyecca.dsl.simulation import SimulationResult, Simulator
from cyecca.dsl.types import DType, Indices, Shape, Var, VarKind

# Variables
from cyecca.dsl.variables import ArrayDerivativeExpr, DerivativeExpr, SymbolicVar, TimeVar

__all__ = [
    # Decorators
    "model",
    "block",
    "function",
    "connector",
    "equations",
    "initial_equations",
    "algorithm",
    # Variable declaration
    "var",
    "Real",
    "Integer",
    "Boolean",
    "String",
    "Var",
    "VarKind",
    "DType",
    "Shape",
    "Indices",
    "submodel",
    # Free functions (continuous)
    "der",
    # Free functions (discrete/event)
    "pre",
    "edge",
    "change",
    # Special operators
    "initial",
    "terminal",
    "sample",
    # Connectors
    "connect",
    # When-clauses (hybrid systems)
    "when",
    "reinit",
    "WhenClause",
    "Reinit",
    # If-equations (conditional equations)
    "if_eq",
    "elseif_eq",
    "else_eq",
    "IfEquation",
    "IfEquationBranch",
    # Boolean operators
    "and_",
    "or_",
    "not_",
    # Comparison operators (for use in expressions)
    "eq",
    "ne",
    # Conditional expression
    "if_then_else",
    # Algorithm section
    "local",
    "assign",
    "Assignment",
    # Function support
    "FunctionMetadata",
    # Base class
    "Model",
    # Model representation
    "FlatModel",
    "Expr",
    "ExprKind",
    "Equation",
    # Causality analysis
    "analyze_causality",
    "SortedSystem",
    "SolvedEquation",
    "ImplicitBlock",
    # Simulation
    "SimulationResult",
    "Simulator",
    # Math functions
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sqrt",
    "exp",
    "log",
    "log10",
    "abs",
    "sign",
    "floor",
    "ceil",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "mod",
    "min",
    "max",
]
