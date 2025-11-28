"""
Algorithm section support for the Cyecca DSL.

This module contains:
- AlgorithmVar: Local variable for algorithm sections
- local(): Create a local algorithm variable
- assign(): Create an assignment statement

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

from typing import Any, Union

from beartype import beartype

from cyecca.dsl.equations import Assignment
from cyecca.dsl.expr import Expr, ExprKind, to_expr
from cyecca.dsl.variables import SymbolicVar


class AlgorithmVar:
    """
    Local variable for use in algorithm sections.

    Algorithm sections can define local variables that exist only within
    the algorithm block. These are created using the `local()` function.

    Example
    -------
    >>> @algorithm  # doctest: +SKIP
    ... def _(m):
    ...     temp = local("temp")
    ...     temp @ (m.x * 2)
    ...     m.y @ (temp + 1)
    """

    def __init__(self, name: str):
        self._name = name
        self._expr = Expr(ExprKind.VARIABLE, name=name)

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"local({self._name})"

    # Assignment operator
    def __matmul__(self, other: Any) -> Assignment:
        """Create an assignment: local_var @ expr"""
        from cyecca.dsl.context import get_current_equation_context

        assign = Assignment(target=self._name, expr=to_expr(other), is_local=True)

        # If in an @algorithm context, register the assignment
        ctx = get_current_equation_context()
        if ctx is not None:
            ctx.add_assignment(assign)

        return assign

    # Arithmetic operators - return Expr
    def __add__(self, other: Any) -> Expr:
        return self._expr + other

    def __radd__(self, other: Any) -> Expr:
        return to_expr(other) + self._expr

    def __sub__(self, other: Any) -> Expr:
        return self._expr - other

    def __rsub__(self, other: Any) -> Expr:
        return to_expr(other) - self._expr

    def __mul__(self, other: Any) -> Expr:
        return self._expr * other

    def __rmul__(self, other: Any) -> Expr:
        return to_expr(other) * self._expr

    def __truediv__(self, other: Any) -> Expr:
        return self._expr / other

    def __rtruediv__(self, other: Any) -> Expr:
        return to_expr(other) / self._expr

    def __neg__(self) -> Expr:
        return -self._expr

    def __pow__(self, other: Any) -> Expr:
        return self._expr**other

    # Relational operators
    def __lt__(self, other: Any) -> Expr:
        return self._expr < other

    def __le__(self, other: Any) -> Expr:
        return self._expr <= other

    def __gt__(self, other: Any) -> Expr:
        return self._expr > other

    def __ge__(self, other: Any) -> Expr:
        return self._expr >= other


@beartype
def local(name: str) -> AlgorithmVar:
    """
    Create a local variable for use in algorithm sections.

    Local variables are temporary variables that exist only within
    an algorithm block. They are useful for storing intermediate
    calculations.

    Parameters
    ----------
    name : str
        Name of the local variable (for debugging/display)

    Returns
    -------
    AlgorithmVar
        A local variable that can be assigned and used in expressions
    """
    return AlgorithmVar(name)


@beartype
def assign(target: Union[SymbolicVar, AlgorithmVar, str], value: Any) -> Assignment:
    """
    Create an assignment for algorithm sections.

    This is an alternative to the @ operator for creating assignments.

    Parameters
    ----------
    target : SymbolicVar, AlgorithmVar, or str
        The variable to assign to
    value : Any
        The value to assign (will be converted to Expr)

    Returns
    -------
    Assignment
        An assignment for use in @algorithm blocks

    Example
    -------
    >>> @algorithm  # doctest: +SKIP
    ... def _(m):
    ...     assign(m.y, m.x * 2)
    ...     # Equivalent to: m.y @ (m.x * 2)
    """
    if isinstance(target, SymbolicVar):
        return Assignment(target=target._name, expr=to_expr(value), is_local=False)
    elif isinstance(target, AlgorithmVar):
        return Assignment(target=target._name, expr=to_expr(value), is_local=True)
    elif isinstance(target, str):
        return Assignment(target=target, expr=to_expr(value), is_local=True)
    else:
        raise TypeError(f"Cannot assign to {type(target)}")
