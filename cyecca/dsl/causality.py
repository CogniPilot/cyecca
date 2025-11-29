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
2. Match equations to unknowns (Hopcroft-Karp or greedy matching)
3. Build dependency graph from matching
4. Sort into BLT form using Tarjan's algorithm for SCCs
5. For each block:
   - If scalar and linear in unknown: solve symbolically
   - Otherwise: leave as implicit (needs Newton iteration at runtime)

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

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

        # Case: linear_expr - const (variable on left)
        if var_to_find not in find_variables(right):
            is_lin, coef, const = is_linear_in(left, var_name)
            if is_lin and coef is not None and const is not None:
                new_const = Expr(ExprKind.SUB, children=(const, right))
                return True, coef, new_const

        # Case: const - linear_expr (variable on right, negated coefficient)
        if var_to_find not in find_variables(left):
            is_lin, coef, const = is_linear_in(right, var_name)
            if is_lin and coef is not None and const is not None:
                # Negate the coefficient: const - (coef * var + c) = -coef * var + (const - c)
                neg_coef = Expr(ExprKind.NEG, children=(coef,))
                new_const = Expr(ExprKind.SUB, children=(left, const))
                return True, neg_coef, new_const

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
    """Perform causality analysis on a flattened model using BLT decomposition.

    This implements the standard Modelica causality analysis:
    1. Build incidence structure (bipartite graph: equations ↔ unknowns)
    2. Find maximum matching (which equation solves for which unknown)
    3. Build dependency graph based on matching
    4. Use Tarjan's algorithm to find SCCs (strongly connected components)
    5. Process blocks in topological order, solving scalar blocks symbolically

    The result is a SortedSystem with equations in evaluation order.
    """
    result = SortedSystem(model=model)

    # Collect equations to analyze (skip output equations)
    equations: List[Equation] = []
    for eq in model.equations:
        if eq.lhs.kind == ExprKind.VARIABLE and eq.lhs.name in model.output_equations:
            continue
        equations.append(eq)

    if not equations:
        return result

    # Determine unknowns: der(state) for each state, plus algebraic variables
    unknowns: List[str] = []
    for state in model.state_names:
        unknowns.append(f"der_{state}")
    unknowns.extend(model.algebraic_names)

    # Build incidence structure: which unknowns appear in which equations
    # incidence[eq_idx] = set of unknown indices that appear in equation
    incidence: List[Set[int]] = []
    unknown_to_idx = {u: i for i, u in enumerate(unknowns)}

    for eq in equations:
        vars_in_eq = find_variables(eq.lhs) | find_variables(eq.rhs)
        eq_unknowns: Set[int] = set()
        for v in vars_in_eq:
            if v in unknown_to_idx:
                eq_unknowns.add(unknown_to_idx[v])
        incidence.append(eq_unknowns)

    n_eq = len(equations)
    n_var = len(unknowns)

    # Step 2: Find maximum matching using augmenting paths (Hopcroft-Karp simplified)
    # matching[eq_idx] = var_idx that this equation is matched to (-1 if unmatched)
    # var_matched[var_idx] = eq_idx that this variable is matched to (-1 if unmatched)
    matching: List[int] = [-1] * n_eq
    var_matched: List[int] = [-1] * n_var

    def find_augmenting_path(eq: int, visited: Set[int]) -> bool:
        """Try to find an augmenting path starting from equation eq."""
        for var in incidence[eq]:
            if var in visited:
                continue
            visited.add(var)

            # If var is unmatched, or we can rematch its current equation
            if var_matched[var] == -1 or find_augmenting_path(var_matched[var], visited):
                matching[eq] = var
                var_matched[var] = eq
                return True
        return False

    # Greedy initial matching + augmenting paths
    for eq in range(n_eq):
        find_augmenting_path(eq, set())

    # Check for structurally singular system
    unmatched_eqs = [i for i in range(n_eq) if matching[i] == -1]
    if unmatched_eqs:
        # Some equations couldn't be matched - system may be over/under-determined
        for eq_idx in unmatched_eqs:
            result.unhandled.append(equations[eq_idx])

    # Step 3: Build dependency graph for matched equations
    # Edge from eq_i to eq_j if eq_i uses the variable that eq_j solves for
    # (and eq_i != eq_j)
    matched_eqs = [i for i in range(n_eq) if matching[i] != -1]

    # adj[eq_idx] = list of equation indices that this equation depends on
    adj: Dict[int, List[int]] = {eq: [] for eq in matched_eqs}

    for eq in matched_eqs:
        for var in incidence[eq]:
            if var != matching[eq]:  # Don't include the variable we solve for
                other_eq = var_matched[var]
                if other_eq != -1 and other_eq != eq:
                    adj[eq].append(other_eq)

    # Step 4: Tarjan's algorithm for SCCs
    sccs = _tarjan_scc(matched_eqs, adj)

    # Step 5: Process each SCC (block) in topological order
    # Tarjan returns SCCs in reverse topological order, so reverse
    sccs.reverse()

    for scc in sccs:
        block_eqs = [equations[i] for i in scc]
        block_unknowns = [unknowns[matching[i]] for i in scc]

        if len(scc) == 1:
            # Scalar block - try to solve symbolically
            eq = block_eqs[0]
            var_name = block_unknowns[0]

            solved = solve_linear(eq, var_name)
            if solved:
                result.solved.append(solved)
                # After solving, the derivative is isolated on LHS (der(x) = expr)
                # Check if the SOLVED expression still contains other derivatives
                if solved.is_derivative:
                    # Check if solved.expr contains any derivatives - that would be implicit
                    expr_derivs = find_derivatives(solved.expr)
                    if expr_derivs:
                        result.is_ode_explicit = False
            else:
                # Couldn't solve symbolically - implicit block
                result.implicit_blocks.append(
                    ImplicitBlock(equations=block_eqs, unknowns=block_unknowns)
                )
                if any(u.startswith("der_") for u in block_unknowns):
                    result.is_ode_explicit = False
        else:
            # Multi-equation block - coupled system, needs simultaneous solve
            result.implicit_blocks.append(
                ImplicitBlock(equations=block_eqs, unknowns=block_unknowns)
            )
            if any(u.startswith("der_") for u in block_unknowns):
                result.is_ode_explicit = False

    # Set has_algebraic flag
    if model.algebraic_names:
        result.has_algebraic = True

    return result


def _tarjan_scc(nodes: List[int], adj: Dict[int, List[int]]) -> List[List[int]]:
    """Find strongly connected components using Tarjan's algorithm.

    Args:
        nodes: List of node identifiers
        adj: Adjacency list (adj[node] = list of nodes this node points to)

    Returns:
        List of SCCs, each SCC is a list of nodes.
        SCCs are returned in reverse topological order.
    """
    index_counter = [0]
    stack: List[int] = []
    lowlink: Dict[int, int] = {}
    index: Dict[int, int] = {}
    on_stack: Dict[int, bool] = {}
    sccs: List[List[int]] = []

    def strongconnect(node: int) -> None:
        # Set the depth index for this node
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True

        # Consider successors
        for successor in adj.get(node, []):
            if successor not in index:
                # Successor has not been visited; recurse
                strongconnect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif on_stack.get(successor, False):
                # Successor is on stack, so in current SCC
                lowlink[node] = min(lowlink[node], index[successor])

        # If node is a root node, pop the stack and generate SCC
        if lowlink[node] == index[node]:
            scc: List[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == node:
                    break
            sccs.append(scc)

    for node in nodes:
        if node not in index:
            strongconnect(node)

    return sccs
