"""
Algorithm section executor.

Executes Modelica algorithm sections (imperative code blocks) in various contexts:
- Symbolic execution (CasADi/SymPy) for code generation
- Numeric execution (NumPy) for direct evaluation

Algorithm sections differ from equation sections:
- Sequential execution (order matters)
- Assignments use := (modify variable values)
- Control flow: if/for/while/when statements
"""

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Optional, Union
from abc import ABC, abstractmethod

import numpy as np

from cyecca.ir.statement import (
    Statement,
    Assignment,
    IfStatement,
    ForStatement,
    WhileStatement,
    WhenStatement,
    ReinitStatement,
    BreakStatement,
    ReturnStatement,
    FunctionCallStatement,
)
from cyecca.ir.expr import (
    Expr,
    Literal,
    VarRef,
    ComponentRef,
    ArrayRef,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    IfExpr,
    ArrayLiteral,
    Slice,
)
from cyecca.ir.algorithm import AlgorithmSection


class BreakException(Exception):
    """Raised when a break statement is executed."""

    pass


class ReturnException(Exception):
    """Raised when a return statement is executed."""

    pass


@dataclass
class ExecutionContext:
    """
    Context for algorithm execution.

    Holds variable values that can be read and modified during execution.
    """

    variables: dict[str, Any] = field(default_factory=dict)
    # For tracking which events have fired (for when statements)
    active_events: set[int] = field(default_factory=set)
    # Maximum iterations for while loops (safety limit)
    max_iterations: int = 10000
    # Event states for edge detection
    _prev_event_states: dict[int, bool] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        """Get variable value."""
        return self.variables.get(name, default)

    def set(self, name: str, value: Any) -> None:
        """Set variable value."""
        self.variables[name] = value

    def get_indexed(self, name: str, indices: tuple[int, ...]) -> Any:
        """Get array element value (0-based indices)."""
        val = self.variables.get(name)
        if val is None:
            raise KeyError(f"Variable '{name}' not found")
        for idx in indices:
            val = val[idx]
        return val

    def set_indexed(self, name: str, indices: tuple[int, ...], value: Any) -> None:
        """Set array element value (0-based indices)."""
        if name not in self.variables:
            raise KeyError(f"Variable '{name}' not found")

        if len(indices) == 1:
            self.variables[name][indices[0]] = value
        elif len(indices) == 2:
            self.variables[name][indices[0]][indices[1]] = value
        else:
            raise NotImplementedError("Arrays with >2 dimensions not supported")

    def copy(self) -> "ExecutionContext":
        """Create a shallow copy of the context."""
        return ExecutionContext(
            variables=dict(self.variables),
            active_events=set(self.active_events),
            max_iterations=self.max_iterations,
            _prev_event_states=dict(self._prev_event_states),
        )


class AlgorithmExecutor(ABC):
    """
    Abstract base class for algorithm executors.

    Subclasses implement execution for different backends (numeric, CasADi, SymPy).
    """

    @abstractmethod
    def execute(self, algo: AlgorithmSection, context: ExecutionContext) -> ExecutionContext:
        """Execute an algorithm section and return the modified context."""
        pass

    @abstractmethod
    def execute_statement(self, stmt: Statement, context: ExecutionContext) -> ExecutionContext:
        """Execute a single statement."""
        pass

    @abstractmethod
    def evaluate_expr(self, expr: Expr, context: ExecutionContext) -> Any:
        """Evaluate an expression in the given context."""
        pass


class NumericAlgorithmExecutor(AlgorithmExecutor):
    """
    Numeric algorithm executor using Python/NumPy.

    Executes algorithm sections with actual numeric values.
    Useful for:
    - Initial algorithm sections
    - Discrete updates
    - Testing and debugging
    """

    def __init__(self, functions: Optional[dict[str, Callable]] = None):
        """
        Initialize the executor.

        Args:
            functions: Optional dict of custom functions (name -> callable)
        """
        self.functions = functions or {}

        # Built-in math functions
        self._builtin_funcs = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "asin": np.arcsin,
            "acos": np.arccos,
            "atan": np.arctan,
            "atan2": np.arctan2,
            "exp": np.exp,
            "log": np.log,
            "ln": np.log,
            "log10": np.log10,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "sign": np.sign,
            "min": np.minimum,
            "max": np.maximum,
            "floor": np.floor,
            "ceil": np.ceil,
            "mod": np.mod,
            "rem": np.remainder,
            # Modelica-specific
            "pre": lambda x: x,  # pre() returns the value (handled specially for events)
            "edge": lambda x: False,  # edge() handled specially
            "change": lambda x: False,  # change() handled specially
        }

    def execute(self, algo: AlgorithmSection, context: ExecutionContext) -> ExecutionContext:
        """Execute an algorithm section."""
        for stmt in algo.statements:
            try:
                context = self.execute_statement(stmt, context)
            except (BreakException, ReturnException):
                break
        return context

    def execute_statement(self, stmt: Statement, context: ExecutionContext) -> ExecutionContext:
        """Execute a single statement."""
        if isinstance(stmt, Assignment):
            return self._execute_assignment(stmt, context)
        elif isinstance(stmt, IfStatement):
            return self._execute_if(stmt, context)
        elif isinstance(stmt, ForStatement):
            return self._execute_for(stmt, context)
        elif isinstance(stmt, WhileStatement):
            return self._execute_while(stmt, context)
        elif isinstance(stmt, WhenStatement):
            return self._execute_when(stmt, context)
        elif isinstance(stmt, ReinitStatement):
            return self._execute_reinit(stmt, context)
        elif isinstance(stmt, BreakStatement):
            raise BreakException()
        elif isinstance(stmt, ReturnStatement):
            raise ReturnException()
        elif isinstance(stmt, FunctionCallStatement):
            return self._execute_function_call(stmt, context)
        else:
            raise ValueError(f"Unknown statement type: {type(stmt)}")

    def _execute_assignment(self, stmt: Assignment, context: ExecutionContext) -> ExecutionContext:
        """Execute assignment: target := expr"""
        value = self.evaluate_expr(stmt.expr, context)
        target = stmt.target

        # Extract variable name and indices
        var_name = self._get_var_name(target)
        indices = self._get_indices(target, context)

        if indices:
            # Array element assignment
            # Convert from Modelica 1-based to Python 0-based
            indices_0based = tuple(i - 1 for i in indices)
            context.set_indexed(var_name, indices_0based, value)
        else:
            # Scalar assignment
            context.set(var_name, value)

        return context

    def _execute_if(self, stmt: IfStatement, context: ExecutionContext) -> ExecutionContext:
        """Execute if statement."""
        for condition, statements in stmt.branches:
            cond_value = self.evaluate_expr(condition, context)
            if cond_value:
                for s in statements:
                    context = self.execute_statement(s, context)
                return context

        # Execute else branch if no condition matched
        for s in stmt.else_statements:
            context = self.execute_statement(s, context)

        return context

    def _execute_for(self, stmt: ForStatement, context: ExecutionContext) -> ExecutionContext:
        """Execute for loop."""
        # Build iterator ranges for all indices
        ranges = []
        for idx_name, range_expr in stmt.indices:
            range_values = self._evaluate_range(range_expr, context)
            ranges.append((idx_name, range_values))

        # Execute loop body for each combination
        context = self._execute_for_nested(ranges, stmt.body, context)
        return context

    def _execute_for_nested(
        self,
        ranges: list[tuple[str, list]],
        body: tuple[Statement, ...],
        context: ExecutionContext,
    ) -> ExecutionContext:
        """Execute nested for loops recursively."""
        if not ranges:
            # Base case: execute body
            for stmt in body:
                try:
                    context = self.execute_statement(stmt, context)
                except BreakException:
                    raise  # Propagate break to outer loop
            return context

        idx_name, values = ranges[0]
        remaining_ranges = ranges[1:]

        for val in values:
            context.set(idx_name, val)
            try:
                context = self._execute_for_nested(remaining_ranges, body, context)
            except BreakException:
                break  # Exit this loop level

        return context

    def _execute_while(self, stmt: WhileStatement, context: ExecutionContext) -> ExecutionContext:
        """Execute while loop."""
        iterations = 0
        while self.evaluate_expr(stmt.condition, context):
            iterations += 1
            if iterations > context.max_iterations:
                raise RuntimeError(
                    f"While loop exceeded maximum iterations ({context.max_iterations})"
                )

            for s in stmt.body:
                try:
                    context = self.execute_statement(s, context)
                except BreakException:
                    return context  # Exit loop

        return context

    def _execute_when(self, stmt: WhenStatement, context: ExecutionContext) -> ExecutionContext:
        """Execute when statement (event-triggered)."""
        for branch_idx, (condition, statements) in enumerate(stmt.branches):
            cond_value = bool(self.evaluate_expr(condition, context))
            prev_value = context._prev_event_states.get(branch_idx, False)

            # Detect rising edge (false -> true)
            if cond_value and not prev_value:
                for s in statements:
                    context = self.execute_statement(s, context)

            # Update previous state
            context._prev_event_states[branch_idx] = cond_value

        return context

    def _execute_reinit(self, stmt: ReinitStatement, context: ExecutionContext) -> ExecutionContext:
        """Execute reinit statement."""
        value = self.evaluate_expr(stmt.expr, context)
        var_name = self._get_var_name(stmt.target)
        indices = self._get_indices(stmt.target, context)

        if indices:
            indices_0based = tuple(i - 1 for i in indices)
            context.set_indexed(var_name, indices_0based, value)
        else:
            context.set(var_name, value)

        return context

    def _execute_function_call(
        self, stmt: FunctionCallStatement, context: ExecutionContext
    ) -> ExecutionContext:
        """Execute function call statement."""
        args = [self.evaluate_expr(arg, context) for arg in stmt.args]

        if stmt.func == "assert":
            # assert(condition, message)
            if not args[0]:
                msg = args[1] if len(args) > 1 else "Assertion failed"
                raise AssertionError(msg)
        elif stmt.func == "terminate":
            # terminate(message)
            msg = args[0] if args else "Terminated"
            raise RuntimeError(f"Simulation terminated: {msg}")
        elif stmt.func == "print":
            # print(message)
            print(args[0] if args else "")
        elif stmt.func in self.functions:
            # Custom function
            self.functions[stmt.func](*args)
        else:
            # Try builtin
            if stmt.func in self._builtin_funcs:
                # Ignore return value for statement context
                self._builtin_funcs[stmt.func](*args)
            else:
                raise ValueError(f"Unknown function: {stmt.func}")

        return context

    def evaluate_expr(self, expr: Expr, context: ExecutionContext) -> Any:
        """Evaluate an expression."""
        if isinstance(expr, Literal):
            return expr.value

        elif isinstance(expr, (VarRef, ComponentRef)):
            var_name = self._get_var_name(expr)
            indices = self._get_indices(expr, context) if isinstance(expr, ComponentRef) else None

            if indices:
                indices_0based = tuple(i - 1 for i in indices)
                return context.get_indexed(var_name, indices_0based)
            else:
                val = context.get(var_name)
                if val is None:
                    raise KeyError(f"Variable '{var_name}' not found in context")
                return val

        elif isinstance(expr, ArrayRef):
            indices = tuple(
                int(self.evaluate_expr(idx, context)) - 1 for idx in expr.indices  # 0-based
            )
            return context.get_indexed(expr.name, indices)

        elif isinstance(expr, BinaryOp):
            left = self.evaluate_expr(expr.left, context)
            right = self.evaluate_expr(expr.right, context)
            return self._eval_binary_op(expr.op, left, right)

        elif isinstance(expr, UnaryOp):
            operand = self.evaluate_expr(expr.operand, context)
            return self._eval_unary_op(expr.op, operand)

        elif isinstance(expr, FunctionCall):
            return self._eval_function_call(expr, context)

        elif isinstance(expr, IfExpr):
            cond = self.evaluate_expr(expr.condition, context)
            if cond:
                return self.evaluate_expr(expr.true_expr, context)
            else:
                return self.evaluate_expr(expr.false_expr, context)

        elif isinstance(expr, ArrayLiteral):
            return np.array([self.evaluate_expr(e, context) for e in expr.elements])

        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")

    def _eval_binary_op(self, op: str, left: Any, right: Any) -> Any:
        """Evaluate binary operation."""
        ops = {
            "+": lambda l, r: l + r,
            "-": lambda l, r: l - r,
            "*": lambda l, r: l * r,
            "/": lambda l, r: l / r,
            "^": lambda l, r: l**r,
            "<": lambda l, r: l < r,
            "<=": lambda l, r: l <= r,
            ">": lambda l, r: l > r,
            ">=": lambda l, r: l >= r,
            "==": lambda l, r: l == r,
            "!=": lambda l, r: l != r,
            "<>": lambda l, r: l != r,
            "and": lambda l, r: l and r,
            "or": lambda l, r: l or r,
        }
        if op in ops:
            return ops[op](left, right)
        raise ValueError(f"Unknown binary operator: {op}")

    def _eval_unary_op(self, op: str, operand: Any) -> Any:
        """Evaluate unary operation."""
        if op == "-" or op == "neg":
            return -operand
        elif op == "+":
            return operand
        elif op == "not":
            return not operand
        raise ValueError(f"Unknown unary operator: {op}")

    def _eval_function_call(self, expr: FunctionCall, context: ExecutionContext) -> Any:
        """Evaluate function call."""
        args = [self.evaluate_expr(arg, context) for arg in expr.args]

        # Check custom functions first
        if expr.func in self.functions:
            return self.functions[expr.func](*args)

        # Check builtins
        if expr.func in self._builtin_funcs:
            return self._builtin_funcs[expr.func](*args)

        raise ValueError(f"Unknown function: {expr.func}")

    def _evaluate_range(self, range_expr: Expr, context: ExecutionContext) -> list:
        """Evaluate a range expression to a list of values."""
        if isinstance(range_expr, Slice):
            start = int(self.evaluate_expr(range_expr.start, context)) if range_expr.start else 1
            stop = int(self.evaluate_expr(range_expr.stop, context)) if range_expr.stop else start
            step = int(self.evaluate_expr(range_expr.step, context)) if range_expr.step else 1
            # Modelica ranges are inclusive
            return list(range(start, stop + 1, step))
        elif isinstance(range_expr, ArrayLiteral):
            return [self.evaluate_expr(e, context) for e in range_expr.elements]
        else:
            # Assume it's a single value or array reference
            val = self.evaluate_expr(range_expr, context)
            if hasattr(val, "__iter__") and not isinstance(val, str):
                return list(val)
            return [val]

    def _get_var_name(self, expr: Union[VarRef, ComponentRef]) -> str:
        """Extract variable name from reference."""
        if isinstance(expr, VarRef):
            return expr.name
        elif isinstance(expr, ComponentRef):
            # Build flattened name from parts (excluding subscripts from last part)
            parts = []
            for i, part in enumerate(expr.parts):
                if i < len(expr.parts) - 1:
                    parts.append(str(part))
                else:
                    parts.append(part.name)
            return ".".join(parts)
        raise ValueError(f"Cannot extract var name from: {expr}")

    def _get_indices(
        self, expr: ComponentRef, context: ExecutionContext
    ) -> Optional[tuple[int, ...]]:
        """Extract array indices from ComponentRef (1-based)."""
        if not isinstance(expr, ComponentRef):
            return None

        last_part = expr.parts[-1]
        if not last_part.subscripts:
            return None

        indices = []
        for sub in last_part.subscripts:
            idx = self.evaluate_expr(sub, context)
            indices.append(int(idx))
        return tuple(indices)


def execute_algorithm(
    algo: AlgorithmSection,
    initial_values: dict[str, Any],
    executor: Optional[AlgorithmExecutor] = None,
) -> dict[str, Any]:
    """
    Execute an algorithm section with given initial values.

    Args:
        algo: The algorithm section to execute
        initial_values: Dict of variable name -> initial value
        executor: Optional custom executor (defaults to NumericAlgorithmExecutor)

    Returns:
        Dict of variable name -> final value after execution

    Example:
        >>> algo = AlgorithmSection([
        ...     Assignment(Expr.var_ref("sum"), Expr.literal(0)),
        ...     ForStatement(
        ...         (("i", Expr.slice(Expr.literal(1), Expr.literal(3))),),
        ...         (Assignment(Expr.var_ref("sum"),
        ...                     Expr.add(Expr.var_ref("sum"), Expr.var_ref("i"))),)
        ...     )
        ... ])
        >>> result = execute_algorithm(algo, {"sum": 0})
        >>> result["sum"]
        6
    """
    if executor is None:
        executor = NumericAlgorithmExecutor()

    context = ExecutionContext(variables=dict(initial_values))
    context = executor.execute(algo, context)
    return dict(context.variables)
