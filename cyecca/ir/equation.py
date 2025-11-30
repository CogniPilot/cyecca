"""
Equation representation in the IR.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from cyecca.ir.expr import Expr


class EquationType(Enum):
    """Type of equation."""

    SIMPLE = auto()  # lhs = rhs (unified: replaces DERIVATIVE/ALGEBRAIC/DISCRETE)
    WHEN = auto()  # when condition then ... end when
    INITIAL = auto()  # Equation that only holds at t=0
    FOR = auto()  # for index in range then equations end for
    IF = auto()  # if condition then ... elseif ... else ... end if (structural if)
    CONNECT = auto()  # connect(a, b) - physical connection


@dataclass
class Equation:
    """
    Represents an equation in the model.

    Examples:
        der(x) = v                      -> SIMPLE with der() in lhs
        y = sin(x)                      -> SIMPLE (algebraic)
        F = m * a                       -> SIMPLE (algebraic)
        pre(x) + 1 = x                  -> SIMPLE (discrete update with pre())
        when sample(0.1) then ...       -> WHEN
        x = 1.0 (at t=0)               -> INITIAL
        for i in 1:3 loop ...          -> FOR
    """

    eq_type: EquationType
    lhs: Optional[Expr] = None  # Left-hand side (can be None for implicit equations)
    rhs: Optional[Expr] = None  # Right-hand side

    # For when clauses: condition and list of equations
    condition: Optional[Expr] = None
    when_equations: Optional[list["Equation"]] = None

    # For for loops: index variable, range, and equations
    index_var: Optional[str] = None
    range_expr: Optional[Expr] = None  # Can be a Slice or a VarRef to an array
    for_equations: Optional[list["Equation"]] = None

    # For if equations: branches (condition, equations) and optional else branch
    if_branches: Optional[list[tuple[Expr, list["Equation"]]]] = None
    else_equations: Optional[list["Equation"]] = None

    # For connect equations: two component references
    connect_lhs: Optional["ComponentRef"] = None
    connect_rhs: Optional["ComponentRef"] = None

    # For initial equations
    is_initial: bool = False

    def __str__(self):
        if self.eq_type == EquationType.SIMPLE:
            if self.lhs is None:
                return f"0 = {self.rhs}"
            else:
                return f"{self.lhs} = {self.rhs}"
        elif self.eq_type == EquationType.WHEN:
            eqs_str = "; ".join(str(eq) for eq in (self.when_equations or []))
            return f"when {self.condition} then {eqs_str} end"
        elif self.eq_type == EquationType.INITIAL:
            return f"{self.lhs} = {self.rhs} (initial)"
        elif self.eq_type == EquationType.FOR:
            eqs_str = "; ".join(str(eq) for eq in (self.for_equations or []))
            return f"for {self.index_var} in {self.range_expr} loop {eqs_str} end for"
        elif self.eq_type == EquationType.IF:
            result = []
            for i, (cond, eqs) in enumerate(self.if_branches or []):
                if i == 0:
                    result.append(f"if {cond} then")
                else:
                    result.append(f"elseif {cond} then")
                for eq in eqs:
                    result.append(f"  {eq};")
            if self.else_equations:
                result.append("else")
                for eq in self.else_equations:
                    result.append(f"  {eq};")
            result.append("end if")
            return "\n".join(result)
        elif self.eq_type == EquationType.CONNECT:
            return f"connect({self.connect_lhs}, {self.connect_rhs})"
        else:
            return f"Unknown equation type: {self.eq_type}"

    @staticmethod
    def simple(lhs: Expr, rhs: Expr) -> "Equation":
        """
        Create a simple equation: lhs = rhs

        This is the unified equation type that replaces DERIVATIVE, ALGEBRAIC, and DISCRETE.

        Examples:
            Equation.simple(der(x), v)           # Derivative: der(x) = v
            Equation.simple(y, sin(x))           # Algebraic: y = sin(x)
            Equation.simple(x, pre(x) + 1)       # Discrete: x = pre(x) + 1
        """
        return Equation(eq_type=EquationType.SIMPLE, lhs=lhs, rhs=rhs)

    @staticmethod
    def when(condition: Expr, equations: list["Equation"]) -> "Equation":
        """Create a when clause: when condition then equations end."""
        return Equation(eq_type=EquationType.WHEN, condition=condition, when_equations=equations)

    @staticmethod
    def initial(lhs: Expr, rhs: Expr) -> "Equation":
        """Create an initial equation (only holds at t=0)."""
        return Equation(eq_type=EquationType.INITIAL, lhs=lhs, rhs=rhs, is_initial=True)

    @staticmethod
    def for_loop(index_var: str, range_expr: Expr, equations: list["Equation"]) -> "Equation":
        """
        Create a for loop equation.

        In Modelica:
            for i in 1:n loop
              der(x[i]) = v[i];
            end for

        Args:
            index_var: Loop index variable name (e.g., "i")
            range_expr: Range expression (Slice or array reference)
            equations: List of equations in the loop body

        Examples:
            # for i in 1:3 loop der(x[i]) = v[i]; end for
            Equation.for_loop("i", Expr.slice(Expr.literal(1), Expr.literal(3)), [eq])
        """
        return Equation(
            eq_type=EquationType.FOR,
            index_var=index_var,
            range_expr=range_expr,
            for_equations=equations,
        )

    @staticmethod
    def if_eq(
        *branches: tuple[Expr, list["Equation"]], else_eqs: Optional[list["Equation"]] = None
    ) -> "Equation":
        """
        Create a structural if-equation (not to be confused with if-expression).

        In Modelica:
            equation
              if x > 0 then
                y = 1;
              elseif x < 0 then
                y = -1;
              else
                y = 0;
              end if;

        Args:
            *branches: Tuples of (condition, equations) for if/elseif branches
            else_eqs: Optional list of equations for else branch

        Examples:
            # if x > 0 then y = 1; else y = 0; end if;
            Equation.if_eq(
                (x_gt_0, [eq_y_1]),
                else_eqs=[eq_y_0]
            )
        """
        return Equation(
            eq_type=EquationType.IF,
            if_branches=list(branches),
            else_equations=else_eqs or [],
        )

    @staticmethod
    def connect(lhs: "ComponentRef", rhs: "ComponentRef") -> "Equation":
        """
        Create a connect equation: connect(a, b)

        In Modelica:
            connect(resistor.p, capacitor.n);

        Args:
            lhs: Left connector reference
            rhs: Right connector reference

        Note: Connect equations are expanded by the compiler into:
        - Equality equations for potential variables (v1 = v2)
        - Zero-sum equations for flow variables (i1 + i2 = 0)
        """
        return Equation(
            eq_type=EquationType.CONNECT,
            connect_lhs=lhs,
            connect_rhs=rhs,
        )
