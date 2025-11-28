"""
Cyecca DSL - A Modelica-inspired Domain Specific Language for Python.

This module provides a high-level, declarative API for defining dynamic systems
that closely mirrors Modelica syntax while remaining fully Pythonic.

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
>>> from cyecca.dsl import model, var, sin, cos, der, equations
>>> from cyecca.dsl.backends import CasadiBackend
>>>
>>> @model
... class Pendulum:
...     '''Simple pendulum model.'''
...     # Parameters (constant during simulation)
...     g = var(9.81, parameter=True)
...     l = var(1.0, parameter=True)
...     
...     # States (automatically detected via der() usage)
...     theta = var(start=0.5)
...     omega = var()
...     
...     # Outputs (explicitly flagged)
...     x = var(output=True)
...     y = var(output=True)
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

Variable Classification
-----------------------
Variables are classified automatically based on flags and equation analysis:

1. var(parameter=True) → parameter (constant during simulation)
2. var(constant=True) → constant (compile-time constant)
3. var(input=True) → input (externally controlled)
4. var(output=True) → output (computed, exposed)
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

# Context and when-clause support
from cyecca.dsl.context import equations, reinit, when

# Decorators and var() factory
from cyecca.dsl.decorators import FunctionMetadata, ModelMetadata, block, function, model, submodel, var

# Equations and statements
from cyecca.dsl.equations import Assignment, Equation, Reinit, WhenClause

# Expression tree
from cyecca.dsl.expr import Expr, ExprKind

# Flat model representation
from cyecca.dsl.flat_model import FlatModel

# Model instance and alias
from cyecca.dsl.instance import Model, ModelInstance
from cyecca.dsl.operators import (
    abs,
    acos,
    asin,
    atan,
    atan2,
    ceil,
    cos,
    cosh,
    exp,
    floor,
    log,
    log10,
    max,
    min,
    sign,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)

# Core operators
from cyecca.dsl.operators_core import and_, change, der, edge, if_then_else, not_, or_, pre
from cyecca.dsl.simulation import SimulationResult, Simulator
from cyecca.dsl.types import DType, Indices, Shape, Var, VarKind

# Variables
from cyecca.dsl.variables import ArrayDerivativeExpr, DerivativeExpr, SymbolicVar, TimeVar

__all__ = [
    # Decorators
    "model",
    "block",
    "function",
    "equations",
    # Variable declaration
    "var",
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
    # When-clauses (hybrid systems)
    "when",
    "reinit",
    "WhenClause",
    "Reinit",
    # Boolean operators
    "and_",
    "or_",
    "not_",
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
    "min",
    "max",
]
