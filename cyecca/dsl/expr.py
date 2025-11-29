"""
Expression tree representation for the Cyecca DSL.

This module contains the core Expr class and ExprKind enum that form
the abstract syntax tree for symbolic expressions.

The expression tree is backend-agnostic - it can be compiled to
CasADi, JAX, NumPy, or other backends.

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

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generator, Optional, Tuple

import numpy as np
from beartype import beartype

from cyecca.dsl.types import Indices, Shape

if TYPE_CHECKING:
    from cyecca.dsl.variables import DerivativeExpr, SymbolicVar, TimeVar


class ExprKind(Enum):
    """Kinds of expression nodes."""

    # Leaf nodes
    VARIABLE = auto()  # Named variable (state, param, input, etc.)
    DERIVATIVE = auto()  # der(x) - derivative of a variable
    CONSTANT = auto()  # Numeric constant
    TIME = auto()  # Time variable t

    # Unary operations
    NEG = auto()  # -x
    NOT = auto()  # not x (Boolean negation)

    # Binary arithmetic operations
    ADD = auto()  # x + y
    SUB = auto()  # x - y
    MUL = auto()  # x * y
    DIV = auto()  # x / y
    POW = auto()  # x ** y

    # Relational operations (Modelica MLS 3.5)
    LT = auto()  # x < y
    LE = auto()  # x <= y
    GT = auto()  # x > y
    GE = auto()  # x >= y
    EQ = auto()  # x == y (equality test, not equation)
    NE = auto()  # x != y (or x <> y in Modelica)

    # Boolean operations (Modelica MLS 3.5)
    AND = auto()  # x and y
    OR = auto()  # x or y

    # Conditional expression (Modelica MLS 3.6.5)
    IF_THEN_ELSE = auto()  # if cond then expr1 else expr2

    # Array operations
    INDEX = auto()  # x[i] - array indexing (stores index in 'value' field)

    # Discrete/event operators (Modelica MLS 3.8)
    PRE = auto()  # pre(x) - previous value of discrete variable
    EDGE = auto()  # edge(x) - True when x changes from False to True
    CHANGE = auto()  # change(x) - True when x changes value

    # Math functions
    SIN = auto()
    COS = auto()
    TAN = auto()
    ASIN = auto()
    ACOS = auto()
    ATAN = auto()
    ATAN2 = auto()
    SQRT = auto()
    EXP = auto()
    LOG = auto()
    LOG10 = auto()  # Base-10 logarithm
    ABS = auto()
    SIGN = auto()  # Sign function (-1, 0, or 1)
    FLOOR = auto()  # Floor function
    CEIL = auto()  # Ceiling function
    SINH = auto()  # Hyperbolic sine
    COSH = auto()  # Hyperbolic cosine
    TANH = auto()  # Hyperbolic tangent
    ASINH = auto()  # Inverse hyperbolic sine
    ACOSH = auto()  # Inverse hyperbolic cosine
    ATANH = auto()  # Inverse hyperbolic tangent
    MIN = auto()  # Minimum of two values
    MAX = auto()  # Maximum of two values
    MOD = auto()  # Modulo operation

    # Hybrid system operators (Modelica MLS 8.5)
    REINIT = auto()  # reinit(x, expr) - reinitialize state at event


@dataclass(frozen=True)
class Expr:
    """
    Immutable expression tree node.

    Represents symbolic mathematical expressions that can be compiled
    to different backends (CasADi, JAX, NumPy, etc.).

    This is the core abstraction that makes the DSL backend-agnostic.
    The DSL builds expression trees, and backends compile them to
    executable functions.

    For indexed variables, use VARIABLE kind with indices set.
    """

    kind: ExprKind
    children: Tuple["Expr", ...] = ()
    name: Optional[str] = None  # For VARIABLE, DERIVATIVE
    value: Optional[float] = None  # For CONSTANT
    indices: Indices = ()  # For indexed VARIABLE: (i,), (i,j), etc.

    def __repr__(self) -> str:
        if self.kind == ExprKind.VARIABLE:
            if self.indices:
                idx_str = ",".join(str(i) for i in self.indices)
                return f"{self.name}[{idx_str}]"
            return f"{self.name}"
        elif self.kind == ExprKind.DERIVATIVE:
            return f"der({self.name})"
        elif self.kind == ExprKind.CONSTANT:
            return f"{self.value}"
        elif self.kind == ExprKind.TIME:
            return "t"
        elif self.kind == ExprKind.NEG:
            return f"(-{self.children[0]})"
        elif self.kind == ExprKind.ADD:
            return f"({self.children[0]} + {self.children[1]})"
        elif self.kind == ExprKind.SUB:
            return f"({self.children[0]} - {self.children[1]})"
        elif self.kind == ExprKind.MUL:
            return f"({self.children[0]} * {self.children[1]})"
        elif self.kind == ExprKind.DIV:
            return f"({self.children[0]} / {self.children[1]})"
        elif self.kind == ExprKind.POW:
            return f"({self.children[0]} ** {self.children[1]})"
        elif self.kind == ExprKind.INDEX:
            # Legacy INDEX kind - prefer using VARIABLE with indices
            return f"{self.name}[{int(self.value)}]"
        elif self.kind in (
            ExprKind.SIN,
            ExprKind.COS,
            ExprKind.TAN,
            ExprKind.ASIN,
            ExprKind.ACOS,
            ExprKind.ATAN,
            ExprKind.SQRT,
            ExprKind.EXP,
            ExprKind.LOG,
            ExprKind.LOG10,
            ExprKind.ABS,
            ExprKind.SIGN,
            ExprKind.FLOOR,
            ExprKind.CEIL,
            ExprKind.SINH,
            ExprKind.COSH,
            ExprKind.TANH,
        ):
            return f"{self.kind.name.lower()}({self.children[0]})"
        elif self.kind == ExprKind.ATAN2:
            return f"atan2({self.children[0]}, {self.children[1]})"
        elif self.kind == ExprKind.MIN:
            return f"min({self.children[0]}, {self.children[1]})"
        elif self.kind == ExprKind.MAX:
            return f"max({self.children[0]}, {self.children[1]})"
        elif self.kind == ExprKind.PRE:
            return f"pre({self.name})"
        elif self.kind == ExprKind.EDGE:
            return f"edge({self.name})"
        elif self.kind == ExprKind.CHANGE:
            return f"change({self.name})"
        elif self.kind == ExprKind.LT:
            return f"({self.children[0]} < {self.children[1]})"
        elif self.kind == ExprKind.LE:
            return f"({self.children[0]} <= {self.children[1]})"
        elif self.kind == ExprKind.GT:
            return f"({self.children[0]} > {self.children[1]})"
        elif self.kind == ExprKind.GE:
            return f"({self.children[0]} >= {self.children[1]})"
        elif self.kind == ExprKind.EQ:
            return f"({self.children[0]} == {self.children[1]})"
        elif self.kind == ExprKind.NE:
            return f"({self.children[0]} != {self.children[1]})"
        elif self.kind == ExprKind.AND:
            return f"({self.children[0]} and {self.children[1]})"
        elif self.kind == ExprKind.OR:
            return f"({self.children[0]} or {self.children[1]})"
        elif self.kind == ExprKind.NOT:
            return f"(not {self.children[0]})"
        elif self.kind == ExprKind.IF_THEN_ELSE:
            return f"(if {self.children[0]} then {self.children[1]} else {self.children[2]})"
        elif self.kind == ExprKind.REINIT:
            return f"reinit({self.name}, {self.children[0]})"
        return f"Expr({self.kind})"

    @property
    def indexed_name(self) -> str:
        """Get the full name including indices: 'x' or 'x[0,1]'."""
        if self.indices:
            idx_str = ",".join(str(i) for i in self.indices)
            return f"{self.name}[{idx_str}]"
        return self.name or ""

    # Arithmetic operators - return new Expr nodes
    def __add__(self, other: Any) -> "Expr":
        return Expr(ExprKind.ADD, (self, to_expr(other)))

    def __radd__(self, other: Any) -> "Expr":
        return Expr(ExprKind.ADD, (to_expr(other), self))

    def __sub__(self, other: Any) -> "Expr":
        return Expr(ExprKind.SUB, (self, to_expr(other)))

    def __rsub__(self, other: Any) -> "Expr":
        return Expr(ExprKind.SUB, (to_expr(other), self))

    def __mul__(self, other: Any) -> "Expr":
        return Expr(ExprKind.MUL, (self, to_expr(other)))

    def __rmul__(self, other: Any) -> "Expr":
        return Expr(ExprKind.MUL, (to_expr(other), self))

    def __truediv__(self, other: Any) -> "Expr":
        return Expr(ExprKind.DIV, (self, to_expr(other)))

    def __rtruediv__(self, other: Any) -> "Expr":
        return Expr(ExprKind.DIV, (to_expr(other), self))

    def __pow__(self, other: Any) -> "Expr":
        return Expr(ExprKind.POW, (self, to_expr(other)))

    def __rpow__(self, other: Any) -> "Expr":
        return Expr(ExprKind.POW, (to_expr(other), self))

    def __neg__(self) -> "Expr":
        return Expr(ExprKind.NEG, (self,))

    def __pos__(self) -> "Expr":
        return self

    # Relational operators - return Boolean Expr
    def __lt__(self, other: Any) -> "Expr":
        return Expr(ExprKind.LT, (self, to_expr(other)))

    def __le__(self, other: Any) -> "Expr":
        return Expr(ExprKind.LE, (self, to_expr(other)))

    def __gt__(self, other: Any) -> "Expr":
        return Expr(ExprKind.GT, (self, to_expr(other)))

    def __ge__(self, other: Any) -> "Expr":
        return Expr(ExprKind.GE, (self, to_expr(other)))

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        """Equality comparison or equation registration.

        When inside an @equations context, this registers an equation.
        Otherwise, returns a comparison Expr for use in conditionals.
        """
        from cyecca.dsl.context import get_current_equation_context
        from cyecca.dsl.equations import Equation

        rhs = to_expr(other)

        # Check if in @equations context
        ctx = get_current_equation_context()
        if ctx is not None:
            # If building a subexpression (e.g., inside if_then_else), return comparison
            if ctx.is_building_expr:
                return Expr(ExprKind.EQ, (self, rhs))
            # Otherwise, register as equation
            eq = Equation(lhs=self, rhs=rhs)
            ctx.add_equation(eq)
            return None

        # Outside @equations context, return comparison Expr
        return Expr(ExprKind.EQ, (self, rhs))

    def __ne__(self, other: Any) -> "Expr":  # type: ignore[override]
        """Not-equal comparison expression."""
        return Expr(ExprKind.NE, (self, to_expr(other)))

    def __hash__(self) -> int:
        """Hash based on kind, children, name, value, and indices.

        Required because we override __eq__.
        """
        return hash((self.kind, self.children, self.name, self.value, self.indices))


@beartype
def to_expr(x: Any) -> Expr:
    """Convert various types to Expr."""
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
    raise TypeError(f"Cannot convert {type(x)} to Expr")


@beartype
def find_derivatives(expr: Expr) -> set[str]:
    """
    Find all variable names whose derivative (der) appears in an expression.

    This is used for automatic state detection: if der(x) appears anywhere
    in the equations, then x is a state variable.

    For indexed variables like der(pos[0]), returns "pos[0]".
    """
    result: set[str] = set()

    if expr.kind == ExprKind.DERIVATIVE and expr.name:
        result.add(expr.name)

    for child in expr.children:
        result.update(find_derivatives(child))

    return result


def prefix_expr(expr: Expr, prefix: str) -> Expr:
    """
    Create a new Expr with all variable names prefixed.

    This is used when flattening submodels to give all variables
    their fully qualified names (e.g., 'x' -> 'spring.x').
    """
    if expr.kind == ExprKind.VARIABLE:
        new_name = f"{prefix}.{expr.name}"
        return Expr(
            kind=ExprKind.VARIABLE,
            name=new_name,
            value=expr.value,
            children=tuple(prefix_expr(c, prefix) for c in expr.children),
        )
    elif expr.kind == ExprKind.DERIVATIVE:
        new_name = f"{prefix}.{expr.name}" if expr.name else None
        return Expr(
            kind=ExprKind.DERIVATIVE,
            name=new_name,
            value=expr.value,
            children=tuple(prefix_expr(c, prefix) for c in expr.children),
        )
    elif expr.kind == ExprKind.CONSTANT:
        return expr  # Constants don't need prefixing
    else:
        # Recursively prefix children for operators, functions, etc.
        return Expr(
            kind=expr.kind,
            name=expr.name,
            value=expr.value,
            children=tuple(prefix_expr(c, prefix) for c in expr.children),
        )


# =============================================================================
# Helper functions for variable names and indices
# =============================================================================


@beartype
def get_base_name(name: str) -> str:
    """Extract base name from indexed name: 'pos[0,1]' -> 'pos'."""
    if "[" in name:
        return name.split("[")[0]
    return name


@beartype
def parse_indices(name: str) -> Tuple[str, Indices]:
    """Parse indexed name: 'pos[0,1]' -> ('pos', (0, 1))."""
    if "[" not in name:
        return name, ()
    base = name.split("[")[0]
    idx_str = name.split("[")[1].rstrip("]")
    indices = tuple(int(i) for i in idx_str.split(","))
    return base, indices


@beartype
def format_indices(indices: Indices) -> str:
    """Format indices as string: (0, 1) -> '[0,1]'."""
    if not indices:
        return ""
    return "[" + ",".join(str(i) for i in indices) + "]"


@beartype
def iter_indices(shape: Shape) -> Generator[Indices, None, None]:
    """Iterate over all valid index tuples for a given shape."""
    if not shape:
        yield ()
        return
    import itertools

    for idx in itertools.product(*(range(dim) for dim in shape)):
        yield idx


@beartype
def is_array_state(name: str, shape: Shape, derivatives_used: set[str]) -> bool:
    """
    Check if an array variable is a state by checking if any element's
    derivative is used.

    For a variable 'pos' with shape=(3,), checks if 'pos[0]', 'pos[1]', or 'pos[2]'
    appear in derivatives_used.
    """
    if not shape:  # Scalar
        return False
    for indices in iter_indices(shape):
        indexed_name = f"{name}{format_indices(indices)}"
        if indexed_name in derivatives_used:
            return True
    return False
