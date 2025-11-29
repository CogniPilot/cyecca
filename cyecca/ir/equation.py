"""
IREquation - Clean equation representation for the IR.

Equations are stored as expression trees with an equality relation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from beartype import beartype

from cyecca.ir.expr import Expr, ExprKind


@beartype
@dataclass(frozen=True)
class IREquation:
    """
    An equation in the IR: lhs == rhs.

    Equations represent mathematical relationships between variables.
    They are declarative (not assignments) and can be solved in either direction.

    Parameters
    ----------
    lhs : Expr
        Left-hand side expression
    rhs : Expr
        Right-hand side expression
    description : str
        Optional description of what this equation represents

    Example
    -------
    .. code-block:: python

        eq = IREquation(lhs=der_theta_expr, rhs=omega_expr)

        eq = IREquation(
            lhs=x_expr,
            rhs=mul_expr,
            description="Horizontal position from angle",
        )
    """

    lhs: Expr
    rhs: Expr
    description: str = ""

    @property
    def is_derivative(self) -> bool:
        """True if LHS is der(x) for some variable x."""
        return self.lhs.kind == ExprKind.DERIVATIVE

    @property
    def var_name(self) -> Optional[str]:
        """Return the variable name if LHS is der(x), else None."""
        if self.lhs.kind == ExprKind.DERIVATIVE and self.lhs.children:
            child = self.lhs.children[0]
            if child.kind == ExprKind.VARIABLE:
                return child.name
        return None

    def __repr__(self) -> str:
        return f"IREquation({self.lhs} == {self.rhs})"


@beartype
@dataclass(frozen=True)
class IRReinit:
    """
    A reinit statement: reinit(var, expr).

    Reinitializes a state variable to a new value at an event.

    Parameters
    ----------
    var_name : str
        Name of the variable to reinitialize
    expr : Expr
        New value expression
    """

    var_name: str
    expr: Expr

    def __repr__(self) -> str:
        return f"reinit({self.var_name}, {self.expr})"


@beartype
@dataclass(frozen=True)
class IRWhenClause:
    """
    A when-clause for event handling.

    When-clauses define event-driven behavior. The body is executed
    when the condition transitions from False to True (rising edge).

    Parameters
    ----------
    condition : Expr
        Boolean condition that triggers the when-clause
    reinits : list of IRReinit
        Reinit statements to execute when triggered
    description : str
        Optional description

    Example
    -------
    .. code-block:: python

        when_clause = IRWhenClause(
            condition=lt_expr,
            reinits=[IRReinit("v", bounce_expr)],
            description="Bounce on ground",
        )
    """

    condition: Expr
    reinits: List[IRReinit] = field(default_factory=list)
    description: str = ""

    def __repr__(self) -> str:
        return f"IRWhenClause({self.condition}, reinits={len(self.reinits)})"


@beartype
@dataclass(frozen=True)
class IRInitialEquation:
    """
    An initial equation: solved only at t=0.

    Parameters
    ----------
    lhs : Expr
        Left-hand side expression
    rhs : Expr
        Right-hand side expression
    """

    lhs: Expr
    rhs: Expr

    def __repr__(self) -> str:
        return f"IRInitialEquation({self.lhs} == {self.rhs})"


@beartype
@dataclass(frozen=True)
class IRAssignment:
    """
    An assignment in an algorithm section: var := expr.

    Unlike equations, assignments are directional and executed in order.

    Parameters
    ----------
    var_name : str
        Name of the variable being assigned
    expr : Expr
        Value expression
    """

    var_name: str
    expr: Expr

    def __repr__(self) -> str:
        return f"{self.var_name} := {self.expr}"
