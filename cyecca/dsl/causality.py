"""
Causality analysis and BLT decomposition for the Cyecca DSL.

This module provides equation sorting and causality analysis to convert
the flat equation system into a form suitable for simulation.

================================================================================
PROTOTYPE MODE - API IS IN FLUX
================================================================================

This DSL is in active prototype development. The API may change significantly
between versions. Do NOT maintain backward compatibility - iterate rapidly.

================================================================================
DESIGN PRINCIPLES
================================================================================

1. MODELICA CONFORMANCE: Based on Modelica Language Spec Appendix B
2. SELF-CONTAINED: Pure symbolic manipulation, no external solvers
3. INCREMENTAL: Can partially succeed (solve what we can, leave rest implicit)

================================================================================
USAGE
================================================================================

The flattener produces a FlatModel with all equations in implicit form.
Backends can either:

1. Use FlatModel directly (for IDAS - implicit DAE solver)
2. Call analyze_causality() first to get SortedSystem (for RK4, CVODES)

Example:
    flat = MyModel().flatten()

    # Option 1: Direct to IDAS (implicit)
    compiled = CasadiBackend.compile(flat, integrator=Integrator.IDAS)

    # Option 2: BLT first, then explicit solver
    sorted_sys = analyze_causality(flat)
    if sorted_sys.is_ode_explicit:
        compiled = CasadiBackend.compile(sorted_sys, integrator=Integrator.RK4)
    else:
        compiled = CasadiBackend.compile(sorted_sys, integrator=Integrator.IDAS)

================================================================================
ALGORITHM OVERVIEW (MLS Appendix B)
================================================================================

The goal is to transform the DAE system:
    0 = f(der(x), x, y, u, p, t)

Into sorted/solved form where possible:
    der(x) = f_x(x, y, u, p, t)   -- explicit ODE
    y = f_y(x, u, p, t)           -- solved algebraic

Steps:
1. Build incidence matrix: which variables appear in which equations
2. Match equations to unknowns (Dulmage-Mendelsohn or similar)
3. Sort into BLT form (strongly connected components)
4. For each block:
   - If scalar and linear in unknown: solve symbolically
   - Otherwise: leave as implicit (needs Newton iteration at runtime)

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from cyecca.dsl.equations import Equation
from cyecca.dsl.expr import Expr, ExprKind, find_derivatives
from cyecca.dsl.flat_model import FlatModel


@dataclass
class SolvedEquation:
    """An equation that has been solved for a specific variable.

    Represents: var = expr
    """

    var_name: str  # The variable being solved for (e.g., "v" or "der_x")
    expr: Expr  # The RHS expression
    original: Equation  # The original equation this came from
    is_derivative: bool = False  # True if solving for der(x)


@dataclass
class ImplicitBlock:
    """A block of equations that must be solved simultaneously.

    These equations couldn't be solved symbolically and require
    Newton iteration or similar at runtime.
    """

    equations: List[Equation]
    unknowns: List[str]  # Variables to solve for


@dataclass
class SortedSystem:
    """The result of causality analysis.

    Contains equations sorted into BLT order, with scalar equations
    solved where possible. Includes reference to original FlatModel
    for variable metadata.
    """

    # Reference to the original flat model (for variable info)
    model: FlatModel

    # Solved equations in evaluation order
    solved: List[SolvedEquation] = field(default_factory=list)

    # Implicit blocks that need runtime iteration
    implicit_blocks: List[ImplicitBlock] = field(default_factory=list)

    # Equations we couldn't handle (for debugging)
    unhandled: List[Equation] = field(default_factory=list)

    # Is the ODE part fully explicit? (all der(x) isolated on LHS)
    is_ode_explicit: bool = True

    # Does the system have algebraic equations?
    has_algebraic: bool = False


def find_variables(expr: Expr) -> Set[str]:
    """Find all variable names referenced in an expression."""
    result: Set[str] = set()

    if expr.kind == ExprKind.VARIABLE and expr.name:
        result.add(expr.name)
    elif expr.kind == ExprKind.DERIVATIVE and expr.name:
        # For derivatives, we track them as "der_varname"
        result.add(f"der_{expr.name}")

    for child in expr.children:
        result.update(find_variables(child))

    return result


def is_linear_in(expr: Expr, var_name: str) -> Tuple[bool, Optional[Expr], Optional[Expr]]:
    """Check if expression is linear in a variable.

    Returns (is_linear, coefficient, constant) where:
        expr = coefficient * var + constant

    If not linear, returns (False, None, None).

    This is a simplified check - handles common cases like:
    - x (coef=1, const=0)
    - a * x (coef=a, const=0)
    - x * a (coef=a, const=0)
    - x + b (coef=1, const=b)
    - a * x + b (coef=a, const=b)
    """
    # Check if var appears in expression
    vars_in_expr = find_variables(expr)

    # Handle derivative variables
    is_deriv = var_name.startswith("der_")
    actual_var = var_name[4:] if is_deriv else var_name
    var_to_find = f"der_{actual_var}" if is_deriv else var_name

    if var_to_find not in vars_in_expr:
        # Variable not in expression - it's constant w.r.t. this var
        return True, Expr(ExprKind.CONSTANT, value=0.0), expr

    # Simple case: expression is just the variable
    if is_deriv:
        if expr.kind == ExprKind.DERIVATIVE and expr.name == actual_var:
            return True, Expr(ExprKind.CONSTANT, value=1.0), Expr(ExprKind.CONSTANT, value=0.0)
    else:
        if expr.kind == ExprKind.VARIABLE and expr.name == var_name:
            return True, Expr(ExprKind.CONSTANT, value=1.0), Expr(ExprKind.CONSTANT, value=0.0)

    # Case: a * var or var * a
    if expr.kind == ExprKind.MUL and len(expr.children) == 2:
        left, right = expr.children

        # Check left * var
        if is_deriv:
            if right.kind == ExprKind.DERIVATIVE and right.name == actual_var:
                if var_to_find not in find_variables(left):
                    return True, left, Expr(ExprKind.CONSTANT, value=0.0)
        else:
            if right.kind == ExprKind.VARIABLE and right.name == var_name:
                if var_to_find not in find_variables(left):
                    return True, left, Expr(ExprKind.CONSTANT, value=0.0)

        # Check var * right
        if is_deriv:
            if left.kind == ExprKind.DERIVATIVE and left.name == actual_var:
                if var_to_find not in find_variables(right):
                    return True, right, Expr(ExprKind.CONSTANT, value=0.0)
        else:
            if left.kind == ExprKind.VARIABLE and left.name == var_name:
                if var_to_find not in find_variables(right):
                    return True, right, Expr(ExprKind.CONSTANT, value=0.0)

    # Case: expr + const or const + expr where expr is linear in var
    if expr.kind == ExprKind.ADD and len(expr.children) == 2:
        left, right = expr.children

        # Try left + right where left is linear and right is constant
        if var_to_find not in find_variables(right):
            is_lin, coef, const = is_linear_in(left, var_name)
            if is_lin and coef is not None and const is not None:
                new_const = Expr(ExprKind.ADD, children=(const, right))
                return True, coef, new_const

        # Try left + right where right is linear and left is constant
        if var_to_find not in find_variables(left):
            is_lin, coef, const = is_linear_in(right, var_name)
            if is_lin and coef is not None and const is not None:
                new_const = Expr(ExprKind.ADD, children=(left, const))
                return True, coef, new_const

    # Case: expr - const
    if expr.kind == ExprKind.SUB and len(expr.children) == 2:
        left, right = expr.children

        if var_to_find not in find_variables(right):
            is_lin, coef, const = is_linear_in(left, var_name)
            if is_lin and coef is not None and const is not None:
                new_const = Expr(ExprKind.SUB, children=(const, right))
                return True, coef, new_const

    # Not linear (or too complex for us to detect)
    return False, None, None


def solve_linear(eq: Equation, var_name: str) -> Optional[SolvedEquation]:
    """Try to solve equation for a variable.

    For equation: lhs == rhs
    Rearrange to: 0 == lhs - rhs
    Then solve for var_name if linear.

    Returns SolvedEquation if successful, None otherwise.
    """
    is_deriv = var_name.startswith("der_")
    actual_var = var_name[4:] if is_deriv else var_name

    # Build residual: lhs - rhs = 0
    # Check linearity in both lhs and rhs

    lhs_linear, lhs_coef, lhs_const = is_linear_in(eq.lhs, var_name)
    rhs_linear, rhs_coef, rhs_const = is_linear_in(eq.rhs, var_name)

    if not (lhs_linear and rhs_linear):
        return None

    if lhs_coef is None or rhs_coef is None:
        return None

    # Equation is: lhs_coef * var + lhs_const == rhs_coef * var + rhs_const
    # Rearrange: (lhs_coef - rhs_coef) * var == rhs_const - lhs_const
    # So: var == (rhs_const - lhs_const) / (lhs_coef - rhs_coef)

    # Numerator: rhs_const - lhs_const
    if lhs_const is not None and rhs_const is not None:
        numerator = Expr(ExprKind.SUB, children=(rhs_const, lhs_const))
    elif rhs_const is not None:
        numerator = rhs_const
    elif lhs_const is not None:
        numerator = Expr(ExprKind.NEG, children=(lhs_const,))
    else:
        numerator = Expr(ExprKind.CONSTANT, value=0.0)

    # Denominator: lhs_coef - rhs_coef
    denominator = Expr(ExprKind.SUB, children=(lhs_coef, rhs_coef))

    # Check if denominator is just a constant
    def is_zero_const(e: Expr) -> bool:
        if e.kind == ExprKind.CONSTANT and e.value == 0.0:
            return True
        if e.kind == ExprKind.SUB and len(e.children) == 2:
            l, r = e.children
            if l.kind == ExprKind.CONSTANT and r.kind == ExprKind.CONSTANT:
                return l.value == r.value
        return False

    def is_one_const(e: Expr) -> bool:
        if e.kind == ExprKind.CONSTANT and e.value == 1.0:
            return True
        if e.kind == ExprKind.SUB and len(e.children) == 2:
            l, r = e.children
            if l.kind == ExprKind.CONSTANT and r.kind == ExprKind.CONSTANT:
                return (l.value - r.value) == 1.0
        return False

    if is_zero_const(denominator):
        # Coefficient is zero - can't solve for this variable
        return None

    # Build the solution expression
    if is_one_const(denominator):
        solution = numerator
    else:
        solution = Expr(ExprKind.DIV, children=(numerator, denominator))

    return SolvedEquation(
        var_name=actual_var if is_deriv else var_name,
        expr=solution,
        original=eq,
        is_derivative=is_deriv,
    )


def analyze_causality(model: FlatModel) -> SortedSystem:
    """Perform causality analysis on a flattened model.

    This is a simplified version that:
    1. Tries to solve each equation for its "natural" unknown
    2. For derivative equations (der(x) == ...), tries to isolate der(x)
    3. Leaves unsolved equations as implicit blocks

    A full implementation would do proper BLT decomposition with
    Tarjan's algorithm for SCCs.
    """
    result = SortedSystem(model=model)

    # Classify equations
    for eq in model.equations:
        # Skip output equations (already extracted)
        if eq.lhs.kind == ExprKind.VARIABLE and eq.lhs.name in model.output_equations:
            continue

        # Check if this is a derivative equation
        lhs_derivs = find_derivatives(eq.lhs)
        rhs_derivs = find_derivatives(eq.rhs)

        if eq.is_derivative and eq.var_name:
            # Explicit form: der(x) == rhs
            # Check if rhs contains any derivatives (would make it implicit)
            if rhs_derivs:
                result.is_ode_explicit = False
                # Try to solve anyway
                solved = solve_linear(eq, f"der_{eq.var_name}")
                if solved:
                    result.solved.append(solved)
                else:
                    result.implicit_blocks.append(
                        ImplicitBlock(equations=[eq], unknowns=[f"der_{eq.var_name}"])
                    )
            else:
                # Pure explicit: der(x) = rhs (rhs has no derivatives)
                result.solved.append(
                    SolvedEquation(
                        var_name=eq.var_name,
                        expr=eq.rhs,
                        original=eq,
                        is_derivative=True,
                    )
                )
        elif lhs_derivs:
            # Implicit derivative equation: something with der() on LHS
            # e.g., m * der(v) == g
            result.is_ode_explicit = False

            # Try to solve for each derivative
            solved_any = False
            for der_var in lhs_derivs:
                solved = solve_linear(eq, f"der_{der_var}")
                if solved:
                    result.solved.append(solved)
                    solved_any = True
                    break

            if not solved_any:
                # Couldn't solve - add as implicit block
                unknowns = [f"der_{v}" for v in lhs_derivs]
                result.implicit_blocks.append(ImplicitBlock(equations=[eq], unknowns=unknowns))
        else:
            # Algebraic equation (no derivatives)
            # Try to identify what variable to solve for
            vars_in_eq = find_variables(eq.lhs) | find_variables(eq.rhs)

            # Filter to algebraic variables only
            algebraic_vars = [v for v in vars_in_eq if v in model.algebraic_names]

            if len(algebraic_vars) == 1:
                # Single algebraic unknown - try to solve
                solved = solve_linear(eq, algebraic_vars[0])
                if solved:
                    result.solved.append(solved)
                else:
                    result.implicit_blocks.append(
                        ImplicitBlock(equations=[eq], unknowns=algebraic_vars)
                    )
            elif algebraic_vars:
                # Multiple unknowns - need coupled solve
                result.implicit_blocks.append(
                    ImplicitBlock(equations=[eq], unknowns=algebraic_vars)
                )
                result.has_algebraic = True
            else:
                # No algebraic unknowns - might be a constraint or output
                result.unhandled.append(eq)

    # Set has_algebraic if any algebraic variables exist
    if model.algebraic_names:
        result.has_algebraic = True

    return result
