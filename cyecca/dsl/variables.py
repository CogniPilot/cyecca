"""
Symbolic variable wrappers for the Cyecca DSL.

This module contains user-facing symbolic variable types:
- SymbolicVar: Main variable proxy for building equations
- DerivativeExpr: Represents der(x) for scalar variables
- ArrayDerivativeExpr: Represents der(x) for array variables
- TimeVar: Represents the time variable t

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

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from cyecca.dsl.expr import Expr, ExprKind, format_indices, to_expr
from cyecca.ir.types import Indices, Shape, Var

if TYPE_CHECKING:
    from cyecca.dsl.equations import ArrayEquation, Assignment, Equation
    from cyecca.dsl.instance import ModelInstance


class SymbolicVar:
    """
    Symbolic variable proxy for building equations.

    Wraps an Expr and supports arithmetic operations.
    This is the user-facing object accessed via self.x in equations.

    Unified representation for both:
    - Base variables: shape=(), indices=() for scalar; shape=(3,), indices=() for vector
    - Indexed elements: shape=(3,), indices=(0,) for first element of vector

    For array variables, supports N-dimensional indexing:
        x[0]      - 1D indexing
        x[0, 1]   - 2D indexing
        x[0, 1, 2] - 3D indexing
    """

    def __init__(
        self,
        name: str,
        var: Var,
        model: "ModelInstance",
        indices: Indices = (),
    ):
        self._base_name = name
        self._var = var
        self._model = model
        self._shape = var.shape
        self._indices = indices

        # Compute effective shape after indexing
        # e.g., shape=(3,3), indices=(0,) -> remaining_shape=(3,)
        self._remaining_shape = self._shape[len(indices) :]

        # Build the expression
        self._expr = Expr(ExprKind.VARIABLE, name=name, indices=indices)

        # Full name including indices
        self._name = name + format_indices(indices)

        # Cache for indexed elements
        self._indexed_cache: Dict[Indices, "SymbolicVar"] = {}

    @property
    def name(self) -> str:
        """The variable name including indices (for use with SimulationResult)."""
        return self._name

    @property
    def base_name(self) -> str:
        """The base variable name without indices."""
        return self._base_name

    @property
    def shape(self) -> Shape:
        """The original shape of the variable."""
        return self._shape

    @property
    def remaining_shape(self) -> Shape:
        """The remaining shape after current indexing."""
        return self._remaining_shape

    @property
    def indices(self) -> Indices:
        """Current indices applied to this variable."""
        return self._indices

    @property
    def size(self) -> int:
        """Total number of scalar elements."""
        return self._var.size

    @property
    def ndim(self) -> int:
        """Number of remaining dimensions."""
        return len(self._remaining_shape)

    def is_scalar(self) -> bool:
        """Return True if fully indexed to a scalar."""
        return self._remaining_shape == ()

    def __repr__(self) -> str:
        return self._name

    def __getitem__(self, index: Union[int, Indices]) -> "SymbolicVar":
        """
        Index into an array variable (N-dimensional).

        Parameters
        ----------
        index : int or tuple of int
            Index or indices into the array.
            x[0] for 1D, x[0, 1] for 2D, etc.

        Returns
        -------
        SymbolicVar
            Symbolic representation of x[index] or x[i,j,...]

        Raises
        ------
        IndexError
            If index is out of bounds
        TypeError
            If variable has no remaining dimensions to index
        """
        # Normalize index to tuple
        if isinstance(index, int):
            new_indices = (index,)
        elif isinstance(index, tuple):
            new_indices = index
        else:
            raise TypeError(f"Index must be int or tuple, got {type(index).__name__}")

        # Check we have dimensions to index
        if len(new_indices) > len(self._remaining_shape):
            raise TypeError(
                f"Too many indices for '{self._name}': got {len(new_indices)}, "
                f"but remaining dimensions is {len(self._remaining_shape)}"
            )

        # Validate each index
        for i, (idx, dim) in enumerate(zip(new_indices, self._remaining_shape)):
            if not isinstance(idx, int):
                raise TypeError(f"Index {i} must be an integer, got {type(idx).__name__}")
            if idx < 0 or idx >= dim:
                raise IndexError(f"Index {idx} out of bounds for dimension {i} of '{self._name}' with size {dim}")

        # Combine with existing indices
        full_indices = self._indices + new_indices

        # Use cache
        if full_indices not in self._indexed_cache:
            self._indexed_cache[full_indices] = SymbolicVar(
                name=self._base_name,
                var=self._var,
                model=self._model,
                indices=full_indices,
            )

        return self._indexed_cache[full_indices]

    def __len__(self) -> int:
        """Return the size of the first remaining dimension."""
        if not self._remaining_shape:
            raise TypeError(f"Scalar variable '{self._name}' has no length")
        return self._remaining_shape[0]

    def __iter__(self):
        """Iterate over the first remaining dimension."""
        if not self._remaining_shape:
            raise TypeError(f"Cannot iterate over scalar variable '{self._name}'")
        for i in range(self._remaining_shape[0]):
            yield self[i]

    # Comparison operators for equations
    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        """Capture equation or return comparison expression.

        Equation definition: m.y == <expr>  (y is a simple variable)
        Comparison: m.x == 1.0 in if_then_else (returns Expr)

        We distinguish by checking if we're in expression-building mode.
        For array variables with remaining shape, creates ArrayEquation.
        """
        from cyecca.dsl.context import get_current_equation_context
        from cyecca.dsl.equations import ArrayEquation, Equation

        rhs = to_expr(other)

        # Check if in @equations context
        ctx = get_current_equation_context()
        if ctx is not None:
            # If building a subexpression, return comparison Expr
            if ctx.is_building_expr:
                return Expr(ExprKind.EQ, (self._expr, rhs))
            # Otherwise, register as equation
            # Use ArrayEquation if this is an array variable with remaining dimensions
            if self._remaining_shape != ():
                eq = ArrayEquation(lhs_var=self, rhs=rhs, is_derivative=False)
            else:
                eq = Equation(lhs=self._expr, rhs=rhs)
            ctx.add_equation(eq)
            return None

        # Outside @equations context, return comparison Expr
        return Expr(ExprKind.EQ, (self._expr, rhs))

    def __ne__(self, other: Any) -> Expr:  # type: ignore[override]
        """Return not-equal comparison expression."""
        rhs = to_expr(other)
        return Expr(ExprKind.NE, (self._expr, rhs))

    # Arithmetic operations - return Expr
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

    # Relational operators - return Boolean Expr
    def __lt__(self, other: Any) -> Expr:
        return self._expr < other

    def __le__(self, other: Any) -> Expr:
        return self._expr <= other

    def __gt__(self, other: Any) -> Expr:
        return self._expr > other

    def __ge__(self, other: Any) -> Expr:
        return self._expr >= other

    # Assignment operator for algorithm sections
    def __matmul__(self, other: Any) -> "Assignment":
        """
        Assignment operator for algorithm sections: m.x @ expr

        This creates an Assignment and registers it with the current context.
        The @ operator is used because := is not valid Python syntax for this,
        and @ is free since we use * for matrix multiplication (like Modelica).
        """
        from cyecca.dsl.context import get_current_equation_context
        from cyecca.dsl.equations import Assignment

        assign = Assignment(target=self._name, expr=to_expr(other), is_local=False)

        # If in an @algorithm context, register the assignment
        ctx = get_current_equation_context()
        if ctx is not None:
            ctx.add_assignment(assign)

        return assign


class DerivativeExpr:
    """Represents der(x) - the derivative of a state variable."""

    def __init__(self, var_name: str):
        self._var_name = var_name
        self._expr = Expr(ExprKind.DERIVATIVE, name=var_name)

    def __repr__(self) -> str:
        return f"der({self._var_name})"

    def __eq__(self, other: Any) -> Optional["Equation"]:  # type: ignore[override]
        """Capture equation: der(x) == expr.

        If inside an @equations block, auto-registers and returns None.
        Otherwise returns an Equation object.
        """
        from cyecca.dsl.context import get_current_equation_context
        from cyecca.dsl.equations import Equation

        rhs = to_expr(other)
        eq = Equation(
            lhs=self._expr,
            rhs=rhs,
            is_derivative=True,
            var_name=self._var_name,
        )

        # Auto-register if in @equations context
        ctx = get_current_equation_context()
        if ctx is not None:
            ctx.add_equation(eq)
            return None
        return eq

    # Arithmetic (for expressions like der(x) + y)
    def __add__(self, other: Any) -> Expr:
        return self._expr + other

    def __radd__(self, other: Any) -> Expr:
        return to_expr(other) + self._expr

    def __sub__(self, other: Any) -> Expr:
        return self._expr - other

    def __rsub__(self, other: Any) -> Expr:
        return to_expr(other) - self._expr

    def __neg__(self) -> Expr:
        return -self._expr


class TimeVar:
    """Represents the time variable t."""

    def __init__(self) -> None:
        self._expr = Expr(ExprKind.TIME)

    def __repr__(self) -> str:
        return "t"

    # Arithmetic operations
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


class ArrayDerivativeExpr:
    """
    Represents der(x) for an array variable with remaining dimensions.

    When used in an equation like `der(pos) == vel`, this expands to
    multiple scalar equations during flattening.
    """

    def __init__(self, var: SymbolicVar):
        self._var = var
        self._base_name = var._base_name
        self._indices = var._indices
        self._remaining_shape = var._remaining_shape

    def __repr__(self) -> str:
        return f"der({self._var._name})"

    def __eq__(self, other: Any) -> Optional["ArrayEquation"]:  # type: ignore[override]
        """Capture array equation: der(pos) == vel.

        If inside an @equations block, auto-registers and returns None.
        Otherwise returns an ArrayEquation object.
        """
        from cyecca.dsl.context import get_current_equation_context
        from cyecca.dsl.equations import ArrayEquation

        # Convert list to Expr (ARRAY_LITERAL)
        if isinstance(other, list):
            other = to_expr(other)

        eq = ArrayEquation(
            lhs_var=self._var,
            rhs=other,
            is_derivative=True,
        )

        # Auto-register if in @equations context
        ctx = get_current_equation_context()
        if ctx is not None:
            ctx.add_equation(eq)
            return None
        return eq

    def __getitem__(self, index: Union[int, Indices]) -> "DerivativeExpr":
        """Allow der(pos)[i] syntax as alternative to der(pos[i])."""
        indexed_var = self._var[index]
        if not indexed_var.is_scalar():
            # Return another ArrayDerivativeExpr for partial indexing
            return ArrayDerivativeExpr(indexed_var)
        return DerivativeExpr(indexed_var._name)
