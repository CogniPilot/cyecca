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

from typing import TYPE_CHECKING, Union

from beartype import beartype

from cyecca.dsl.expr import Expr, ExprKind, ExprLike, to_expr
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
def and_(a: ExprLike, b: ExprLike) -> Expr:
    """
    Logical AND of two Boolean expressions.

    Since Python's `and` keyword cannot be overloaded, use this function
    for Boolean conjunction in model equations.

    Parameters
    ----------
    a : ExprLike
        First Boolean operand
    b : ExprLike
        Second Boolean operand

    Returns
    -------
    Expr
        Boolean expression representing `a and b`
    """
    from cyecca.dsl.context import get_current_equation_context

    ctx = get_current_equation_context()
    if ctx is not None:
        ctx.enter_expr()
    try:
        a_expr = to_expr(a)
        b_expr = to_expr(b)
    finally:
        if ctx is not None:
            ctx.exit_expr()

    return Expr(ExprKind.AND, (a_expr, b_expr))


@beartype
def or_(a: ExprLike, b: ExprLike) -> Expr:
    """
    Logical OR of two Boolean expressions.

    Since Python's `or` keyword cannot be overloaded, use this function
    for Boolean disjunction in model equations.

    Parameters
    ----------
    a : ExprLike
        First Boolean operand
    b : ExprLike
        Second Boolean operand

    Returns
    -------
    Expr
        Boolean expression representing `a or b`
    """
    from cyecca.dsl.context import get_current_equation_context

    ctx = get_current_equation_context()
    if ctx is not None:
        ctx.enter_expr()
    try:
        a_expr = to_expr(a)
        b_expr = to_expr(b)
    finally:
        if ctx is not None:
            ctx.exit_expr()

    return Expr(ExprKind.OR, (a_expr, b_expr))


@beartype
def not_(a: ExprLike) -> Expr:
    """
    Logical NOT of a Boolean expression.

    Since Python's `not` keyword cannot be overloaded, use this function
    for Boolean negation in model equations.

    Parameters
    ----------
    a : ExprLike
        Boolean operand

    Returns
    -------
    Expr
        Boolean expression representing `not a`
    """
    from cyecca.dsl.context import get_current_equation_context

    ctx = get_current_equation_context()
    if ctx is not None:
        ctx.enter_expr()
    try:
        a_expr = to_expr(a)
    finally:
        if ctx is not None:
            ctx.exit_expr()

    return Expr(ExprKind.NOT, (a_expr,))


# =============================================================================
# Conditional expression: if_then_else
# =============================================================================


@beartype
def if_then_else(condition: ExprLike, then_expr: ExprLike, else_expr: ExprLike) -> Expr:
    """
    Conditional expression: if condition then then_expr else else_expr.

    This is the Modelica if-expression (MLS 3.6.5). Unlike if-statements,
    if-expressions always return a value and both branches must be provided.

    Parameters
    ----------
    condition : ExprLike
        Boolean condition
    then_expr : ExprLike
        Value if condition is True
    else_expr : ExprLike
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
    from cyecca.dsl.context import get_current_equation_context

    # Enter expression-building mode so == returns Expr instead of registering equation
    ctx = get_current_equation_context()
    if ctx is not None:
        ctx.enter_expr()
    try:
        cond_expr = to_expr(condition)
        then_ex = to_expr(then_expr)
        else_ex = to_expr(else_expr)
    finally:
        if ctx is not None:
            ctx.exit_expr()

    return Expr(ExprKind.IF_THEN_ELSE, (cond_expr, then_ex, else_ex))


# =============================================================================
# Comparison operators: eq, ne
# =============================================================================
# Since == is overloaded for equation definition in @equations blocks,
# use these functions when you need equality/inequality comparisons
# inside expressions like if_then_else.


@beartype
def eq(a: ExprLike, b: ExprLike) -> Expr:
    """
    Equality comparison: a == b.

    Use this function instead of == when you need an equality comparison
    inside an expression (e.g., if_then_else condition) within an @equations
    block. The == operator is reserved for equation definition.

    Parameters
    ----------
    a : ExprLike
        First operand
    b : ExprLike
        Second operand

    Returns
    -------
    Expr
        Boolean expression representing `a == b`

    Example
    -------
    >>> from cyecca.dsl import model, var, der, equations, if_then_else, eq
    >>> @model
    ... class M:
    ...     x = var(start=1.0)
    ...     y = var(output=True)
    ...     @equations
    ...     def _(m):
    ...         der(m.x) == 0.0
    ...         m.y == if_then_else(eq(m.x, 1.0), 100.0, 0.0)
    """
    return Expr(ExprKind.EQ, (to_expr(a), to_expr(b)))


@beartype
def ne(a: ExprLike, b: ExprLike) -> Expr:
    """
    Inequality comparison: a != b.

    Use this function instead of != when you need an inequality comparison
    inside an expression within an @equations block.

    Parameters
    ----------
    a : ExprLike
        First operand
    b : ExprLike
        Second operand

    Returns
    -------
    Expr
        Boolean expression representing `a != b`
    """
    return Expr(ExprKind.NE, (to_expr(a), to_expr(b)))


# =============================================================================
# Special operators: initial(), terminal()
# =============================================================================


@beartype
def initial() -> Expr:
    """
    Return True only during initialization (t=0).

    In Modelica (MLS 3.7.4), initial() returns True during the initialization
    phase and False during simulation. This is typically used in when-clauses
    to execute one-time initialization logic:

        when(initial()):
            reinit(m.x, 0.0)

    Returns
    -------
    Expr
        Boolean expression that is True only at initialization

    Example
    -------
    .. code-block:: python

        from cyecca.dsl import equations, initial, model, reinit, var, when

        @model
        class Integrator:
            x = var(start=0.0)
            u = var(input=True)

            @equations
            def _(m):
                der(m.x) == m.u
                when(initial()):
                    reinit(m.x, 1.0)
    """
    return Expr(ExprKind.INITIAL)


@beartype
def terminal() -> Expr:
    """
    Return True only at the end of simulation.

    In Modelica (MLS 3.7.4), terminal() returns True at the final time
    of simulation and False otherwise. This can be used for end-of-simulation
    actions.

    Returns
    -------
    Expr
        Boolean expression that is True only at end of simulation

    Note
    ----
    This function is provided for Modelica conformance but has limited
    utility in the current implementation since the simulator doesn't
    distinguish a terminal phase.
    """
    return Expr(ExprKind.TERMINAL)


@beartype
def sample(start: Union[int, float], interval: Union[int, float]) -> Expr:
    """
    Generate periodic events at regular intervals.

    In Modelica (MLS 3.7.5), sample(start, interval) returns True at times
    start, start + interval, start + 2*interval, etc. This is used in
    when-clauses for sampled-data systems and digital controllers.

    Parameters
    ----------
    start : float
        Time of the first sample event (typically 0)
    interval : float
        Time interval between sample events (must be > 0)

    Returns
    -------
    Expr
        Boolean expression that is True at sample instants

    Example
    -------
    >>> from cyecca.dsl import model, var, der, equations, when, reinit, sample, pre
    >>> @model
    ... class SampledController:
    ...     x = var(start=0.0)        # Plant state
    ...     u = var(discrete=True)     # Control signal (discrete)
    ...     Kp = var(1.0, parameter=True)
    ...     ref = var(1.0, parameter=True)
    ...
    ...     @equations
    ...     def _(m):
    ...         der(m.x) == m.u        # Simple integrator plant
    ...         with when(sample(0, 0.1)):  # 10 Hz sampling
    ...             reinit(m.u, m.Kp * (m.ref - m.x))  # P controller update

    Note
    ----
    The sample() function is the standard way to implement discrete-time
    controllers in Modelica. The interval must be positive. At each sample
    instant, the when-clause body is executed, typically updating discrete
    control variables based on the current plant state.
    """
    if interval <= 0:
        raise ValueError(f"sample() interval must be positive, got {interval}")

    start_expr = to_expr(float(start))
    interval_expr = to_expr(float(interval))
    return Expr(ExprKind.SAMPLE, (start_expr, interval_expr))
