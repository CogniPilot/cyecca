"""
Core operators for the Cyecca DSL.

This module contains:
- der(): Derivative operator for state variables
- pre(), edge(), change(): Discrete operators
- and_(), or_(), not_(): Boolean operators
- if_then_else(): Conditional expression

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

from typing import TYPE_CHECKING, Any, Union

from beartype import beartype

from cyecca.dsl.expr import Expr, ExprKind, to_expr
from cyecca.dsl.variables import ArrayDerivativeExpr, DerivativeExpr, SymbolicVar

# =============================================================================
# Free function: der()
# =============================================================================


@beartype
def der(var: SymbolicVar) -> Union[DerivativeExpr, ArrayDerivativeExpr]:
    """
    Return the derivative of a state variable.

    This is a free function for use in equations. Works with both scalar
    and array variables:

        @equations
        def _(m):
            der(m.theta) == m.omega      # Scalar
            der(m.pos) == m.vel          # Array (element-wise)
            der(m.pos[0]) == m.vel[0]    # Single element
            der(m.R[0,0]) == m.R_dot[0,0]  # Matrix element

    Parameters
    ----------
    var : SymbolicVar
        The state variable (scalar or array, possibly indexed)

    Returns
    -------
    DerivativeExpr or ArrayDerivativeExpr
        An expression representing the derivative
    """
    if not isinstance(var, SymbolicVar):
        raise TypeError(f"der() expects a SymbolicVar, got {type(var)}")

    if var.is_scalar():
        # Fully indexed or scalar variable
        return DerivativeExpr(var._name)
    else:
        # Array with remaining dimensions
        return ArrayDerivativeExpr(var)


# =============================================================================
# Discrete operators: pre(), edge(), change()
# =============================================================================


@beartype
def pre(var: SymbolicVar) -> Expr:
    """
    Return the previous value of a discrete variable.

    In Modelica, pre(x) returns the value of x at the previous event instant.
    This is only valid for discrete variables (variables with discrete=True
    or variables assigned in when-equations).

    Parameters
    ----------
    var : SymbolicVar
        A discrete variable

    Returns
    -------
    Expr
        An expression representing pre(var)
    """
    if not isinstance(var, SymbolicVar):
        raise TypeError(f"pre() expects a SymbolicVar, got {type(var)}")
    if not var.is_scalar():
        raise TypeError(f"pre() currently only supports scalar variables, got shape {var.shape}")
    return Expr(ExprKind.PRE, name=var._name)


@beartype
def edge(var: SymbolicVar) -> Expr:
    """
    Return True when a Boolean variable changes from False to True.

    Equivalent to: `var and not pre(var)`

    Parameters
    ----------
    var : SymbolicVar
        A Boolean discrete variable

    Returns
    -------
    Expr
        An expression representing edge(var)
    """
    if not isinstance(var, SymbolicVar):
        raise TypeError(f"edge() expects a SymbolicVar, got {type(var)}")
    if not var.is_scalar():
        raise TypeError(f"edge() currently only supports scalar variables")
    return Expr(ExprKind.EDGE, name=var._name)


@beartype
def change(var: SymbolicVar) -> Expr:
    """
    Return True when a variable changes its value.

    Equivalent to: `var != pre(var)`

    Parameters
    ----------
    var : SymbolicVar
        A discrete variable

    Returns
    -------
    Expr
        An expression representing change(var)
    """
    if not isinstance(var, SymbolicVar):
        raise TypeError(f"change() expects a SymbolicVar, got {type(var)}")
    if not var.is_scalar():
        raise TypeError(f"change() currently only supports scalar variables")
    return Expr(ExprKind.CHANGE, name=var._name)


# =============================================================================
# Boolean operators: and_, or_, not_
# =============================================================================
# NOTE: Python's `and`, `or`, `not` are keywords and cannot be overloaded.
# We use trailing underscores per PEP 8 convention for avoiding conflicts.


@beartype
def and_(a: Any, b: Any) -> Expr:
    """
    Logical AND of two Boolean expressions.

    Since Python's `and` keyword cannot be overloaded, use this function
    for Boolean conjunction in model equations.

    Parameters
    ----------
    a : Expr or SymbolicVar or bool
        First Boolean operand
    b : Expr or SymbolicVar or bool
        Second Boolean operand

    Returns
    -------
    Expr
        Boolean expression representing `a and b`
    """
    return Expr(ExprKind.AND, (to_expr(a), to_expr(b)))


@beartype
def or_(a: Any, b: Any) -> Expr:
    """
    Logical OR of two Boolean expressions.

    Since Python's `or` keyword cannot be overloaded, use this function
    for Boolean disjunction in model equations.

    Parameters
    ----------
    a : Expr or SymbolicVar or bool
        First Boolean operand
    b : Expr or SymbolicVar or bool
        Second Boolean operand

    Returns
    -------
    Expr
        Boolean expression representing `a or b`
    """
    return Expr(ExprKind.OR, (to_expr(a), to_expr(b)))


@beartype
def not_(a: Any) -> Expr:
    """
    Logical NOT of a Boolean expression.

    Since Python's `not` keyword cannot be overloaded, use this function
    for Boolean negation in model equations.

    Parameters
    ----------
    a : Expr or SymbolicVar or bool
        Boolean operand

    Returns
    -------
    Expr
        Boolean expression representing `not a`
    """
    return Expr(ExprKind.NOT, (to_expr(a),))


# =============================================================================
# Conditional expression: if_then_else
# =============================================================================


@beartype
def if_then_else(condition: Any, then_expr: Any, else_expr: Any) -> Expr:
    """
    Conditional expression: if condition then then_expr else else_expr.

    This is the Modelica if-expression (MLS 3.6.5). Unlike if-statements,
    if-expressions always return a value and both branches must be provided.

    Parameters
    ----------
    condition : Expr or SymbolicVar or bool
        Boolean condition
    then_expr : Expr or SymbolicVar or numeric
        Value if condition is True
    else_expr : Expr or SymbolicVar or numeric
        Value if condition is False

    Returns
    -------
    Expr
        Conditional expression

    Notes
    -----
    For smooth simulation, consider using smooth conditional functions
    like `smooth_if` (not yet implemented) to avoid discontinuities.
    """
    return Expr(ExprKind.IF_THEN_ELSE, (to_expr(condition), to_expr(then_expr), to_expr(else_expr)))
