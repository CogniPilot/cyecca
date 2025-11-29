"""
Cyecca IR - Clean Intermediate Representation for Dynamic Systems.

This module provides a clean, explicit API for defining dynamic systems
without decorators or magic. The decorator-based DSL (`@model`, `Real()`, etc.)
in `cyecca.dsl` is built on top of this IR.

Architecture
============

```
┌─────────────────────────────────────┐
│     cyecca.dsl                      │  ← Decorator DSL (optional)
│  (@model, Real(), @equations, etc)  │
└─────────────────┬───────────────────┘
                  │ .to_ir()
                  ▼
┌─────────────────────────────────────┐
│     cyecca.ir                       │  ← Clean IR (can be built directly)
│  (IRModel, IRVariable, IREquation)  │
└─────────────────┬───────────────────┘
                  │ .flatten()
                  ▼
┌─────────────────────────────────────┐
│     FlatModel                       │  ← Backend-ready representation
└─────────────────┬───────────────────┘
                  │ compile()
                  ▼
┌─────────────────────────────────────┐
│     cyecca.backends                 │  ← CasADi, SymPy, etc.
└─────────────────────────────────────┘
```

Two Ways to Define Models
=========================

1. **Direct IR API** (clean, explicit, no decorators):

    ```python
    from cyecca.ir import IRModel, IRVariable, IREquation, DataType

    m = IRModel("Ball")
    m.add_variable(IRVariable("h", start=1.0))
    m.add_variable(IRVariable("v"))
    # Add equations using Expr trees...
    ```

2. **Decorator DSL** (convenient, IDE-friendly):

    ```python
    from cyecca.dsl import model, Real, der, equations

    @model
    class Ball:
        h = Real(start=1.0)
        v = Real()

        @equations
        def _(m):
            der(m.h) == m.v
            der(m.v) == -9.81

    # Get the underlying IRModel
    ir = Ball().to_ir()
    ```

Both produce the same `IRModel`.

Design Principles
=================

1. **Explicit over implicit**: All behavior is visible in the code
2. **No decorators required**: Pure function/method calls
3. **Type safe**: Full beartype support, IDE autocomplete works
4. **Immutable where possible**: IR objects are frozen dataclasses
5. **Backend agnostic**: Produces FlatModel for any backend
6. **No DSL dependency**: This package has NO imports from cyecca.dsl
"""

from cyecca.ir.causality import SortedSystem, analyze_causality
from cyecca.ir.equation import IRAssignment, IREquation, IRInitialEquation, IRReinit, IRWhenClause
from cyecca.ir.expr import (
    Expr,
    ExprKind,
    find_derivatives,
    format_indices,
    get_base_name,
    is_array_state,
    iter_indices,
    parse_indices,
    prefix_expr,
)

# Flattening and causality analysis
from cyecca.ir.flat_model import FlatModel
from cyecca.ir.model import IRConnector, IRModel

# Types
from cyecca.ir.types import DType, Indices, Shape, Var, VarKind
from cyecca.ir.variable import DataType, IRVariable, VariableKind

__all__ = [
    # Expression tree
    "Expr",
    "ExprKind",
    "find_derivatives",
    "prefix_expr",
    "get_base_name",
    "parse_indices",
    "format_indices",
    "iter_indices",
    "is_array_state",
    # Core model
    "IRModel",
    "IRConnector",
    # Variables
    "IRVariable",
    "VariableKind",
    "DataType",
    # Equations
    "IREquation",
    "IRWhenClause",
    "IRReinit",
    "IRInitialEquation",
    "IRAssignment",
    # Flattening
    "FlatModel",
    "SortedSystem",
    "analyze_causality",
    # Types
    "Var",
    "VarKind",
    "DType",
    "Shape",
    "Indices",
]
