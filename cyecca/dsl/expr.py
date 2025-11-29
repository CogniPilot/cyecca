"""
Expression tree representation for the Cyecca DSL.

This module re-exports the core Expr from cyecca.ir and adds
DSL-specific functionality like to_expr() which handles
SymbolicVar, DerivativeExpr, etc.

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

from typing import TYPE_CHECKING, List, Union

import numpy as np
from beartype import beartype

# Re-export everything from cyecca.ir.expr
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

# Re-export type aliases
from cyecca.ir.types import Indices, Shape

if TYPE_CHECKING:
    from cyecca.dsl.algorithm import AlgorithmVar
    from cyecca.dsl.variables import DerivativeExpr, SymbolicVar, TimeVar

# Type alias for anything that can be converted to an Expr
# Used in operators and functions that accept expressions
# Note: At runtime, beartype uses object (accepting anything) and to_expr() validates.
# For static type checking, the Union provides proper type hints.
if TYPE_CHECKING:
    ExprLike = Union[
        "Expr",
        "SymbolicVar",
        "DerivativeExpr",
        "TimeVar",
        "AlgorithmVar",
        float,
        int,
        List["ExprLike"],
        np.ndarray,
    ]
else:
    # At runtime, accept any object - to_expr() will validate and convert
    ExprLike = object


@beartype
def to_expr(x: ExprLike) -> Expr:
    """Convert various types to Expr, including DSL types."""
    if isinstance(x, Expr):
        return x
    # Import here to avoid circular imports
    from cyecca.dsl.algorithm import AlgorithmVar
    from cyecca.dsl.variables import DerivativeExpr, SymbolicVar, TimeVar

    if isinstance(x, SymbolicVar):
        return x._expr
    if isinstance(x, DerivativeExpr):
        return x._expr
    if isinstance(x, TimeVar):
        return x._expr
    if isinstance(x, AlgorithmVar):
        return x._expr
    if isinstance(x, (int, float)):
        return Expr(ExprKind.CONSTANT, value=float(x))
    if isinstance(x, np.ndarray) and x.size == 1:
        return Expr(ExprKind.CONSTANT, value=float(x.flat[0]))
    if isinstance(x, list):
        # Convert list to array literal - recursively convert each element
        children = tuple(to_expr(elem) for elem in x)
        return Expr(ExprKind.ARRAY_LITERAL, children=children)
    raise TypeError(f"Cannot convert {type(x)} to Expr")


# Export all
__all__ = [
    # From IR
    "Expr",
    "ExprKind",
    "find_derivatives",
    "prefix_expr",
    "get_base_name",
    "parse_indices",
    "format_indices",
    "iter_indices",
    "is_array_state",
    # DSL-specific
    "to_expr",
    "ExprLike",
]
