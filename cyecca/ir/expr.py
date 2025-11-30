"""
Expression representation in the IR.

Expressions are symbolic mathematical expressions that can be:
- Evaluated by backends (CasADi, SymPy, JAX, etc.)
- Differentiated
- Simplified
- Code generated

This is a simple tree-based representation similar to an AST.
"""

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class Expr:
    """Base class for all expressions."""

    pass


@dataclass(frozen=True)
class Literal(Expr):
    """Literal constant value."""

    value: Union[float, int, bool, str]

    def __str__(self):
        return str(self.value)


@dataclass(frozen=True)
class ComponentRefPart:
    """
    One part of a hierarchical component reference.

    Examples:
        x         -> ComponentRefPart("x")
        arr[i]    -> ComponentRefPart("arr", (i,))
        pos[1,2]  -> ComponentRefPart("pos", (1, 2))
    """

    name: str
    subscripts: tuple[Expr, ...] = ()

    def __str__(self):
        if self.subscripts:
            subs = ",".join(str(s) for s in self.subscripts)
            return f"{self.name}[{subs}]"
        return self.name


@dataclass(frozen=True)
class ComponentRef(Expr):
    """
    Hierarchical component reference.

    This is the proper Modelica way to reference variables, supporting:
    - Simple variables: x
    - Hierarchical access: vehicle.engine.temp
    - Array indexing: positions[i]
    - Combined: vehicle.wheels[i].pressure

    Examples:
        ComponentRef((ComponentRefPart("x"),))
        ComponentRef((ComponentRefPart("a"), ComponentRefPart("b")))
        ComponentRef((ComponentRefPart("arr", (i,)),))
    """

    parts: tuple[ComponentRefPart, ...]

    def __str__(self):
        return ".".join(str(p) for p in self.parts)

    @property
    def is_simple(self) -> bool:
        """True if this is just a simple variable reference (one part, no subscripts)."""
        return len(self.parts) == 1 and len(self.parts[0].subscripts) == 0

    @property
    def simple_name(self) -> str:
        """Get name if simple reference, else raise."""
        if not self.is_simple:
            raise ValueError(f"Not a simple reference: {self}")
        return self.parts[0].name

    def to_varref(self) -> "VarRef":
        """Convert to legacy VarRef (only works for simple refs)."""
        if not self.is_simple:
            raise ValueError(f"Cannot convert hierarchical ref to VarRef: {self}")
        return VarRef(self.simple_name)


@dataclass(frozen=True)
class VarRef(Expr):
    """
    DEPRECATED: Simple variable reference.

    Use ComponentRef instead for new code. This is kept for backward compatibility.
    VarRef will be removed in a future version.
    """

    name: str

    def __str__(self):
        return self.name

    def to_component_ref(self) -> ComponentRef:
        """Convert to ComponentRef."""
        return ComponentRef((ComponentRefPart(self.name),))


@dataclass(frozen=True)
class ArrayRef(Expr):
    """
    DEPRECATED: Array element reference.

    Use ComponentRef with subscripts instead for new code.
    This is kept for backward compatibility only.
    """

    name: str
    indices: tuple[Expr, ...]

    def __str__(self):
        idx_str = ", ".join(str(i) for i in self.indices)
        return f"{self.name}[{idx_str}]"

    def to_component_ref(self) -> ComponentRef:
        """Convert to ComponentRef."""
        return ComponentRef((ComponentRefPart(self.name, self.indices),))


@dataclass(frozen=True)
class BinaryOp(Expr):
    """Binary operation: left op right."""

    op: str  # "+", "-", "*", "/", "^", "==", "<", ">", etc.
    left: Expr
    right: Expr

    def __str__(self):
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True)
class UnaryOp(Expr):
    """Unary operation: op operand."""

    op: str  # "-", "not"
    operand: Expr

    def __str__(self):
        return f"{self.op}({self.operand})"


@dataclass(frozen=True)
class FunctionCall(Expr):
    """Function call: func(args...)."""

    func: str
    args: tuple[Expr, ...]

    def __str__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.func}({args_str})"


@dataclass(frozen=True)
class IfExpr(Expr):
    """Conditional expression: if condition then true_expr else false_expr."""

    condition: Expr
    true_expr: Expr
    false_expr: Expr

    def __str__(self):
        return f"if {self.condition} then {self.true_expr} else {self.false_expr}"


@dataclass(frozen=True)
class ArrayLiteral(Expr):
    """Array literal: [elem1, elem2, ...]."""

    elements: tuple[Expr, ...]

    def __str__(self):
        elems_str = ", ".join(str(e) for e in self.elements)
        return f"[{elems_str}]"


@dataclass(frozen=True)
class Slice(Expr):
    """
    Array slice: start:stop or start:step:stop or :.

    In Modelica:
        x[:]        -> all elements (start=None, stop=None)
        x[1:3]      -> elements 1 to 3 (start=1, stop=3)
        x[1:2:10]   -> elements 1,3,5,7,9 (start=1, step=2, stop=10)

    Note: In Modelica, arrays are 1-indexed by default.
    """

    start: Expr | None = None
    stop: Expr | None = None
    step: Expr | None = None

    def __str__(self):
        if self.start is None and self.stop is None and self.step is None:
            return ":"
        elif self.step is None:
            start_str = str(self.start) if self.start is not None else ""
            stop_str = str(self.stop) if self.stop is not None else ""
            return f"{start_str}:{stop_str}"
        else:
            start_str = str(self.start) if self.start is not None else ""
            step_str = str(self.step) if self.step is not None else ""
            stop_str = str(self.stop) if self.stop is not None else ""
            return f"{start_str}:{step_str}:{stop_str}"


# Convenience constructors for common patterns
class ExprBuilder:
    """Helper class for building expressions with a fluent API."""

    @staticmethod
    def literal(value: Union[float, int, bool, str]) -> Literal:
        """Create a literal expression."""
        return Literal(value)

    @staticmethod
    def var_ref(name: str) -> ComponentRef:
        """
        Create a simple variable reference.

        Returns ComponentRef for forward compatibility.
        For legacy VarRef, use VarRef(name) directly.
        """
        return ComponentRef((ComponentRefPart(name),))

    @staticmethod
    def component_ref(*parts: Union[str, tuple[str, list[Expr]]]) -> ComponentRef:
        """
        Create hierarchical component reference.

        Examples:
            component_ref("x")                    # x
            component_ref("a", "b", "c")          # a.b.c
            component_ref(("arr", [i]), "field")  # arr[i].field
        """
        ref_parts = []
        for part in parts:
            if isinstance(part, str):
                ref_parts.append(ComponentRefPart(part))
            else:
                name, subs = part
                ref_parts.append(ComponentRefPart(name, tuple(subs)))
        return ComponentRef(tuple(ref_parts))

    @staticmethod
    def array_ref(name: str, *indices: Expr) -> ArrayRef:
        """
        DEPRECATED: Create an array reference.

        Use component_ref((name, [indices])) instead for new code.
        """
        return ArrayRef(name, indices)

    @staticmethod
    def binary_op(op: str, left: Expr, right: Expr) -> BinaryOp:
        """Create a binary operation."""
        return BinaryOp(op, left, right)

    @staticmethod
    def unary_op(op: str, operand: Expr) -> UnaryOp:
        """Create a unary operation."""
        return UnaryOp(op, operand)

    @staticmethod
    def call(func: str, *args: Expr) -> FunctionCall:
        """Create a function call."""
        return FunctionCall(func, args)

    @staticmethod
    def if_expr(condition: Expr, true_expr: Expr, false_expr: Expr) -> IfExpr:
        """Create a conditional expression."""
        return IfExpr(condition, true_expr, false_expr)

    @staticmethod
    def array_literal(*elements: Expr) -> ArrayLiteral:
        """Create an array literal."""
        return ArrayLiteral(elements)

    @staticmethod
    def slice(
        start: Expr | None = None, stop: Expr | None = None, step: Expr | None = None
    ) -> Slice:
        """
        Create a slice expression.

        Examples:
            slice()              # :         (all elements)
            slice(1, 3)          # 1:3       (elements 1 to 3)
            slice(1, 10, 2)      # 1:2:10    (elements 1,3,5,7,9)
            slice(None, 5)       # :5        (first 5 elements)
            slice(3, None)       # 3:        (from element 3 to end)
        """
        return Slice(start, stop, step)

    # Common operators
    @staticmethod
    def add(left: Expr, right: Expr) -> BinaryOp:
        """left + right"""
        return BinaryOp("+", left, right)

    @staticmethod
    def sub(left: Expr, right: Expr) -> BinaryOp:
        """left - right"""
        return BinaryOp("-", left, right)

    @staticmethod
    def mul(left: Expr, right: Expr) -> BinaryOp:
        """left * right"""
        return BinaryOp("*", left, right)

    @staticmethod
    def div(left: Expr, right: Expr) -> BinaryOp:
        """left / right"""
        return BinaryOp("/", left, right)

    @staticmethod
    def pow(left: Expr, right: Expr) -> BinaryOp:
        """left ^ right"""
        return BinaryOp("^", left, right)

    @staticmethod
    def neg(operand: Expr) -> UnaryOp:
        """-operand"""
        return UnaryOp("-", operand)

    # Common functions
    @staticmethod
    def sin(x: Expr) -> FunctionCall:
        """sin(x)"""
        return FunctionCall("sin", (x,))

    @staticmethod
    def cos(x: Expr) -> FunctionCall:
        """cos(x)"""
        return FunctionCall("cos", (x,))

    @staticmethod
    def exp(x: Expr) -> FunctionCall:
        """exp(x)"""
        return FunctionCall("exp", (x,))

    @staticmethod
    def log(x: Expr) -> FunctionCall:
        """log(x)"""
        return FunctionCall("log", (x,))

    @staticmethod
    def sqrt(x: Expr) -> FunctionCall:
        """sqrt(x)"""
        return FunctionCall("sqrt", (x,))

    @staticmethod
    def abs(x: Expr) -> FunctionCall:
        """abs(x)"""
        return FunctionCall("abs", (x,))

    # Modelica-specific operators
    @staticmethod
    def der(x: Expr) -> FunctionCall:
        """
        Derivative operator: der(x)

        In Modelica: der(x), der(vehicle.position), der(array[i])
        This is THE way to express derivatives - not a special equation type!
        """
        return FunctionCall("der", (x,))

    @staticmethod
    def pre(x: Expr) -> FunctionCall:
        """
        Previous value operator: pre(x)

        Returns the value of x at the previous event instant.
        Used in when clauses for discrete variables.
        """
        return FunctionCall("pre", (x,))

    @staticmethod
    def edge(x: Expr) -> FunctionCall:
        """
        Edge detector: edge(condition)

        Returns true when condition becomes true (rising edge).
        """
        return FunctionCall("edge", (x,))


# Make ExprBuilder available as Expr for convenience
# This allows: Expr.var_ref("x") instead of ExprBuilder.var_ref("x")
for name in dir(ExprBuilder):
    if not name.startswith("_"):
        setattr(Expr, name, getattr(ExprBuilder, name))


# Also export as top-level functions for convenience
def der(x: Expr) -> FunctionCall:
    """Derivative operator: der(x)"""
    return Expr.der(x)


def pre(x: Expr) -> FunctionCall:
    """Previous value operator: pre(x)"""
    return Expr.pre(x)


def edge(x: Expr) -> FunctionCall:
    """Edge detector: edge(condition)"""
    return Expr.edge(x)
