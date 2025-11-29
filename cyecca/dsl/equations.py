"""
Equation and statement representations for the Cyecca DSL.

This module contains dataclasses for representing:
- Equation: lhs == rhs
- ArrayEquation: array-form equations that expand to scalar equations
- Assignment: algorithm section assignments (target := expr)
- Reinit: state reinitialization at events
- WhenClause: event-driven equation groups

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from beartype import beartype

from cyecca.dsl.expr import Expr, ExprKind, format_indices, iter_indices, prefix_expr, to_expr

if TYPE_CHECKING:
    from cyecca.dsl.variables import SymbolicVar


@dataclass(frozen=True)
class Equation:
    """
    Represents an equation: lhs == rhs.

    Immutable representation of a model equation that can be processed
    by backends for compilation.
    """

    lhs: Expr
    rhs: Expr
    is_derivative: bool = False  # True if LHS is der(x)
    var_name: Optional[str] = None  # Name of variable if LHS is der(x)

    def __repr__(self) -> str:
        return f"Eq({self.lhs} == {self.rhs})"

    def _prefix_names(self, prefix: str) -> "Equation":
        """Create a new equation with all variable names prefixed."""
        new_lhs = prefix_expr(self.lhs, prefix)
        new_rhs = prefix_expr(self.rhs, prefix)
        new_var_name = f"{prefix}.{self.var_name}" if self.var_name else None
        return Equation(
            lhs=new_lhs,
            rhs=new_rhs,
            is_derivative=self.is_derivative,
            var_name=new_var_name,
        )


@dataclass(frozen=True)
class Assignment:
    """
    Represents an algorithm assignment: var := expr.

    Unlike equations (which are declarative and can be solved in any order),
    assignments are imperative and executed in sequence.

    In Modelica, assignments use := operator (vs == for equations).
    In this DSL, use the assign() function or @ operator in algorithm sections.

    Assignments can target:
    - Model variables (m.x := expr)
    - Local algorithm variables (temp := expr)
    """

    target: str  # Variable name being assigned
    expr: Expr  # Right-hand side expression
    is_local: bool = False  # True if target is a local algorithm variable

    def __repr__(self) -> str:
        return f"Assign({self.target} := {self.expr})"


@dataclass(frozen=True)
class Reinit:
    """
    Represents a reinit statement: reinit(x, expr).

    In Modelica, reinit(x, expr) is used within when-clauses to reinitialize
    a continuous-time state variable x to a new value when an event occurs.

    This is essential for hybrid systems like bouncing balls, where the
    velocity needs to be reset (with reversal) when the ball hits the ground.

    Example::

        with when(m.h < 0):
            reinit(m.v, -m.e * pre(m.v))

    Modelica Spec: Section 8.5 - When-Equations, reinit()
    """

    var_name: str  # Name of state variable to reinitialize
    expr: Expr  # New value expression

    def __repr__(self) -> str:
        return f"reinit({self.var_name}, {self.expr})"

    def _prefix_names(self, prefix: str) -> "Reinit":
        """Create a new Reinit with all variable names prefixed."""
        new_var_name = f"{prefix}.{self.var_name}"
        new_expr = prefix_expr(self.expr, prefix)
        return Reinit(var_name=new_var_name, expr=new_expr)


@dataclass
class WhenClause:
    """
    Represents a when-clause for event handling.

    In Modelica, when-clauses are used for event-driven behavior:

        when condition then
            // equations or reinit statements
        end when;

    The condition is a Boolean expression that triggers the when-clause
    when it becomes True (rising edge). The body contains equations or
    reinit statements that are executed at the event instant.

    Modelica Spec: Section 8.5 - When-Equations
    """

    condition: Expr  # Boolean condition that triggers the event
    body: List[Union[Equation, "Reinit"]]  # Equations or reinit statements

    def __repr__(self) -> str:
        return f"WhenClause({self.condition}, body={len(self.body)} items)"

    def _prefix_names(self, prefix: str) -> "WhenClause":
        """Create a new WhenClause with all variable names prefixed."""
        new_condition = prefix_expr(self.condition, prefix)
        new_body = []
        for item in self.body:
            if isinstance(item, Equation):
                new_body.append(item._prefix_names(prefix))
            elif isinstance(item, Reinit):
                new_body.append(item._prefix_names(prefix))
            else:
                new_body.append(item)
        return WhenClause(condition=new_condition, body=new_body)


@dataclass
class IfEquationBranch:
    """
    A single branch of an if-equation (then, elseif, or else).

    This holds the condition (for then/elseif branches) and the body equations.
    For else branches, condition is None.
    """

    condition: Optional[Expr]  # None for else branch
    body: List[Union["Equation", "WhenClause", "Reinit"]]

    def __repr__(self) -> str:
        if self.condition is None:
            return f"ElseBranch({len(self.body)} equations)"
        return f"IfBranch({self.condition}, {len(self.body)} equations)"

    def _prefix_names(self, prefix: str) -> "IfEquationBranch":
        """Create a new branch with all variable names prefixed."""
        new_condition = prefix_expr(self.condition, prefix) if self.condition else None
        new_body = []
        for item in self.body:
            if hasattr(item, "_prefix_names"):
                new_body.append(item._prefix_names(prefix))
            else:
                new_body.append(item)
        return IfEquationBranch(condition=new_condition, body=new_body)


@dataclass
class IfEquation:
    """
    Represents an if-equation for conditional equations.

    In Modelica, if-equations select which equations are active based on
    a condition:

        if condition then
            equations...
        elseif condition2 then
            equations...
        else
            equations...
        end if;

    Key Constraint (Modelica MLS 8.3.4):
    - For non-parameter conditions, all branches MUST have the same number
      of equations (the DAE structure cannot change dynamically).
    - For parameter conditions, branches can have different equation counts
      (structure is determined at compile time).

    Modelica Spec: Section 8.3.4 - If-Equations
    """

    branches: List[IfEquationBranch]  # First is "then", rest are "elseif"/"else"

    def __repr__(self) -> str:
        return f"IfEquation({len(self.branches)} branches)"

    def _prefix_names(self, prefix: str) -> "IfEquation":
        """Create a new IfEquation with all variable names prefixed."""
        new_branches = [b._prefix_names(prefix) for b in self.branches]
        return IfEquation(branches=new_branches)

    def is_parameter_condition(self) -> bool:
        """Check if all conditions are parameter expressions (constant at compile time)."""
        # TODO: Implement proper parameter detection
        # For now, check if conditions only reference parameters
        for branch in self.branches:
            if branch.condition is not None:
                # Need to check if condition only uses parameters
                # This requires analysis of the expression
                pass
        return False

    def validate_equation_counts(self) -> None:
        """Validate that non-parameter if-equations have equal branch sizes."""
        if self.is_parameter_condition():
            # Parameter conditions allow unequal branches
            return

        # Non-parameter conditions require equal equation counts
        if len(self.branches) < 2:
            return

        counts = [len(b.body) for b in self.branches]
        if len(set(counts)) > 1:
            raise ValueError(
                f"If-equation with non-parameter conditions must have equal "
                f"equation counts in all branches, got: {counts}"
            )

    def expand(self) -> List["Equation"]:
        """
        Expand if-equation into regular equations with conditional expressions.

        For non-parameter conditions, each equation in the branches is converted
        to an equation using if_then_else:

            if cond1 then
                y == expr1
            elseif cond2 then
                y == expr2
            else
                y == expr3
            end if;

        Becomes:
            y == if_then_else(cond1, expr1, if_then_else(cond2, expr2, expr3))

        This preserves the DAE structure while making the equations conditional.

        Returns
        -------
        List[Equation]
            The expanded equations with conditional expressions

        Raises
        ------
        ValueError
            If branches have unequal equation counts and not parameter conditions
        """
        # Validate equation counts
        self.validate_equation_counts()

        if len(self.branches) == 0:
            return []

        if len(self.branches) == 1:
            # Single branch (just if, no else) - return equations as-is
            return [eq for eq in self.branches[0].body if isinstance(eq, Equation)]

        # Multiple branches - need to merge corresponding equations
        num_equations = len(self.branches[0].body)
        expanded: List["Equation"] = []

        for i in range(num_equations):
            # Get the i-th equation from each branch
            branch_eqs = []
            for branch in self.branches:
                if i < len(branch.body):
                    eq = branch.body[i]
                    if isinstance(eq, Equation):
                        branch_eqs.append((branch.condition, eq))
                    else:
                        # Skip non-equation items (WhenClause, Reinit) for now
                        pass

            if not branch_eqs:
                continue

            # Build nested if_then_else expression for the RHS
            # Start from the last (else) branch and work backwards
            first_eq = branch_eqs[0][1]
            lhs = first_eq.lhs
            is_derivative = first_eq.is_derivative
            var_name = first_eq.var_name

            # Build the conditional RHS
            # For: if cond1 then e1 elseif cond2 then e2 else e3
            # Result: if_then_else(cond1, e1, if_then_else(cond2, e2, e3))
            rhs = self._build_conditional_rhs(branch_eqs)

            expanded.append(
                Equation(
                    lhs=lhs,
                    rhs=rhs,
                    is_derivative=is_derivative,
                    var_name=var_name,
                )
            )

        return expanded

    def _build_conditional_rhs(self, branch_eqs: List[Tuple[Optional[Expr], "Equation"]]) -> Expr:
        """Build nested if_then_else expression from branch equations."""
        if len(branch_eqs) == 1:
            # Single branch (else or final branch)
            return branch_eqs[0][1].rhs

        condition, eq = branch_eqs[0]
        if condition is None:
            # Else branch (shouldn't be first but handle it)
            return eq.rhs

        # Build: if_then_else(condition, eq.rhs, rest)
        rest = self._build_conditional_rhs(branch_eqs[1:])
        return Expr(
            ExprKind.IF_THEN_ELSE,
            children=[condition, eq.rhs, rest],
        )


@dataclass
class ArrayEquation:
    """
    Represents an array equation that expands to multiple scalar equations.

    For example: der(pos) == vel with shape=(3,) expands to:
        der(pos[0]) == vel[0]
        der(pos[1]) == vel[1]
        der(pos[2]) == vel[2]

    For 2D arrays with shape=(2,3):
        der(pos[0,0]) == vel[0,0]
        der(pos[0,1]) == vel[0,1]
        ...
    """

    lhs_var: "SymbolicVar"  # The LHS array variable
    rhs: Any  # The RHS (could be SymbolicVar, Expr, etc.)
    is_derivative: bool = False

    def expand(self) -> List[Equation]:
        """Expand array equation to scalar equations."""
        from cyecca.dsl.variables import SymbolicVar

        remaining_shape = self.lhs_var._remaining_shape
        base_indices = self.lhs_var._indices
        equations = []

        # Handle array literal RHS - flatten nested array literals for matrices
        rhs_elements = None
        if isinstance(self.rhs, Expr) and self.rhs.kind == ExprKind.ARRAY_LITERAL:
            # Flatten nested array literals (for matrices)
            # [[1,2], [3,4], [5,6]] -> [1, 2, 3, 4, 5, 6] in row-major order
            def flatten_array_literal(expr: Expr) -> list:
                """Recursively flatten nested ARRAY_LITERALs."""
                result = []
                for child in expr.children:
                    if isinstance(child, Expr) and child.kind == ExprKind.ARRAY_LITERAL:
                        result.extend(flatten_array_literal(child))
                    else:
                        result.append(child)
                return result

            rhs_elements = flatten_array_literal(self.rhs)

        for i, rel_indices in enumerate(iter_indices(remaining_shape)):
            full_indices = base_indices + rel_indices
            indexed_name = self.lhs_var._base_name + format_indices(full_indices)

            # Create LHS for this element
            if self.is_derivative:
                lhs = Expr(ExprKind.DERIVATIVE, name=indexed_name)
            else:
                lhs = Expr(ExprKind.VARIABLE, name=self.lhs_var._base_name, indices=full_indices)

            # Create RHS for this element
            if rhs_elements is not None:
                # Array literal - use the i-th element
                if i >= len(rhs_elements):
                    raise ValueError(
                        f"Array literal has {len(rhs_elements)} elements, " f"but LHS has shape {remaining_shape}"
                    )
                rhs = rhs_elements[i]
            elif isinstance(self.rhs, SymbolicVar):
                if self.rhs._remaining_shape != remaining_shape:
                    raise ValueError(
                        f"Shape mismatch in array equation: "
                        f"{self.lhs_var._name} has remaining shape {remaining_shape}, "
                        f"but {self.rhs._name} has remaining shape {self.rhs._remaining_shape}"
                    )
                rhs_indices = self.rhs._indices + rel_indices
                rhs = Expr(ExprKind.VARIABLE, name=self.rhs._base_name, indices=rhs_indices)
            elif isinstance(self.rhs, (int, float)):
                # Scalar broadcast
                rhs = Expr(ExprKind.CONSTANT, value=float(self.rhs))
            elif isinstance(self.rhs, Expr):
                # Expression - use as-is (assumed to be scalar broadcast)
                rhs = self.rhs
            else:
                raise TypeError(f"Cannot expand array equation with RHS type {type(self.rhs)}")

            eq = Equation(
                lhs=lhs,
                rhs=rhs,
                is_derivative=self.is_derivative,
                var_name=indexed_name if self.is_derivative else None,
            )
            equations.append(eq)

        return equations

    def __repr__(self) -> str:
        prefix = "der(" if self.is_derivative else ""
        suffix = ")" if self.is_derivative else ""
        return f"ArrayEq({prefix}{self.lhs_var._name}{suffix} == {self.rhs})"
