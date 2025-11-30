"""
Statement representation in the IR.

Statements are imperative constructs used in algorithm sections.
Unlike equations (declarative), statements execute sequentially with side effects.

In Modelica:
- Equations: x = y; (declarative, no order)
- Algorithms: x := y; (imperative, sequential)
"""

from dataclasses import dataclass
from typing import Optional

from cyecca.ir.expr import Expr, ComponentRef


@dataclass(frozen=True)
class Statement:
    """Base class for all statements in algorithm sections."""

    pass


@dataclass(frozen=True)
class Assignment(Statement):
    """
    Assignment statement: target := expr

    In Modelica:
        x := 5;
        positions[i] := 0.0;
        vehicle.velocity.x := v;

    Note: Uses := (assignment) not = (equation)
    """

    target: ComponentRef  # What variable to assign to
    expr: Expr  # Expression to evaluate

    def __str__(self):
        return f"{self.target} := {self.expr}"


@dataclass(frozen=True)
class IfStatement(Statement):
    """
    Conditional statement: if condition then stmts elseif ... else stmts end if

    In Modelica:
        if x > 0 then
          y := 1;
        elseif x < 0 then
          y := -1;
        else
          y := 0;
        end if;
    """

    branches: tuple[tuple[Expr, tuple[Statement, ...]], ...]  # (condition, statements) pairs
    else_statements: tuple[Statement, ...] = ()  # Else branch (optional)

    def __str__(self):
        result = []
        for i, (cond, stmts) in enumerate(self.branches):
            if i == 0:
                result.append(f"if {cond} then")
            else:
                result.append(f"elseif {cond} then")
            for stmt in stmts:
                result.append(f"  {stmt}")

        if self.else_statements:
            result.append("else")
            for stmt in self.else_statements:
                result.append(f"  {stmt}")

        result.append("end if")
        return "\n".join(result)


@dataclass(frozen=True)
class ForStatement(Statement):
    """
    For loop statement: for index in range loop stmts end for

    In Modelica:
        for i in 1:n loop
          sum := sum + x[i];
        end for;

        for i in {1, 3, 5, 7} loop
          process(i);
        end for;
    """

    indices: tuple[tuple[str, Expr], ...]  # (index_var, range_expr) pairs
    body: tuple[Statement, ...]  # Loop body statements

    def __str__(self):
        result = []
        index_strs = ", ".join(f"{idx} in {rng}" for idx, rng in self.indices)
        result.append(f"for {index_strs} loop")
        for stmt in self.body:
            result.append(f"  {stmt}")
        result.append("end for")
        return "\n".join(result)


@dataclass(frozen=True)
class WhileStatement(Statement):
    """
    While loop statement: while condition loop stmts end while

    In Modelica:
        while x > tol loop
          x := x / 2;
        end while;
    """

    condition: Expr  # Loop condition
    body: tuple[Statement, ...]  # Loop body statements

    def __str__(self):
        result = [f"while {self.condition} loop"]
        for stmt in self.body:
            result.append(f"  {stmt}")
        result.append("end while")
        return "\n".join(result)


@dataclass(frozen=True)
class WhenStatement(Statement):
    """
    When statement: when condition then stmts elsewhen ... end when

    In Modelica:
        when x > threshold then
          count := pre(count) + 1;
        elsewhen x < -threshold then
          count := pre(count) - 1;
        end when;

    Note: Similar to when-equations but used in algorithm sections
    """

    branches: tuple[tuple[Expr, tuple[Statement, ...]], ...]  # (condition, statements) pairs

    def __str__(self):
        result = []
        for i, (cond, stmts) in enumerate(self.branches):
            if i == 0:
                result.append(f"when {cond} then")
            else:
                result.append(f"elsewhen {cond} then")
            for stmt in stmts:
                result.append(f"  {stmt}")
        result.append("end when")
        return "\n".join(result)


@dataclass(frozen=True)
class ReinitStatement(Statement):
    """
    Reinitialize state variable: reinit(state, expr)

    In Modelica (within when clause):
        when h < 0 then
          reinit(h, 0);
          reinit(v, -e * pre(v));
        end when;

    Note: Only valid inside when clauses (events)
    Can only reinitialize continuous state variables
    """

    target: ComponentRef  # State variable to reinitialize
    expr: Expr  # New value

    def __str__(self):
        return f"reinit({self.target}, {self.expr})"


@dataclass(frozen=True)
class BreakStatement(Statement):
    """
    Break from loop: break

    In Modelica:
        for i in 1:n loop
          if found then
            break;
          end if;
        end for;

    Note: Only valid inside for/while loops
    """

    def __str__(self):
        return "break"


@dataclass(frozen=True)
class ReturnStatement(Statement):
    """
    Return from function: return

    In Modelica:
        function myFunction
          ...
          if error then
            return;
          end if;
        end myFunction;

    Note: Only valid inside functions
    """

    def __str__(self):
        return "return"


@dataclass(frozen=True)
class FunctionCallStatement(Statement):
    """
    Function call as statement (e.g., assert, terminate, print)

    In Modelica:
        assert(x > 0, "x must be positive");
        terminate("Simulation failed");
        print("Debug: x = " + String(x));

    Note: These are functions called for side effects, not return values
    """

    func: str  # Function name
    args: tuple[Expr, ...]  # Arguments

    def __str__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.func}({args_str})"


# Convenience constructors
class StatementBuilder:
    """Helper class for building statements with a fluent API."""

    @staticmethod
    def assign(target: ComponentRef, expr: Expr) -> Assignment:
        """Create assignment: target := expr"""
        return Assignment(target, expr)

    @staticmethod
    def if_stmt(
        *branches: tuple[Expr, list[Statement]], else_stmts: Optional[list[Statement]] = None
    ) -> IfStatement:
        """
        Create if statement with branches.

        Example:
            if_stmt(
                (condition1, [stmt1, stmt2]),
                (condition2, [stmt3]),
                else_stmts=[stmt4]
            )
        """
        branch_tuples = tuple((cond, tuple(stmts)) for cond, stmts in branches)
        else_tuple = tuple(else_stmts) if else_stmts else ()
        return IfStatement(branch_tuples, else_tuple)

    @staticmethod
    def for_loop(*indices: tuple[str, Expr], body: list[Statement]) -> ForStatement:
        """
        Create for loop.

        Example:
            for_loop(("i", range_expr), body=[stmt1, stmt2])
        """
        return ForStatement(tuple(indices), tuple(body))

    @staticmethod
    def while_loop(condition: Expr, body: list[Statement]) -> WhileStatement:
        """Create while loop."""
        return WhileStatement(condition, tuple(body))

    @staticmethod
    def when_stmt(*branches: tuple[Expr, list[Statement]]) -> WhenStatement:
        """
        Create when statement with branches.

        Example:
            when_stmt(
                (condition1, [stmt1, stmt2]),
                (condition2, [stmt3])
            )
        """
        branch_tuples = tuple((cond, tuple(stmts)) for cond, stmts in branches)
        return WhenStatement(branch_tuples)

    @staticmethod
    def reinit(target: ComponentRef, expr: Expr) -> ReinitStatement:
        """Create reinit statement."""
        return ReinitStatement(target, expr)

    @staticmethod
    def break_stmt() -> BreakStatement:
        """Create break statement."""
        return BreakStatement()

    @staticmethod
    def return_stmt() -> ReturnStatement:
        """Create return statement."""
        return ReturnStatement()

    @staticmethod
    def call(func: str, *args: Expr) -> FunctionCallStatement:
        """Create function call statement."""
        return FunctionCallStatement(func, args)


# Make StatementBuilder methods available as Statement class methods
for name in dir(StatementBuilder):
    if not name.startswith("_"):
        setattr(Statement, name, getattr(StatementBuilder, name))
