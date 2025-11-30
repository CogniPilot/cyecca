"""
Event representation in the IR.

Events are discrete state changes triggered by conditions.
They contain imperative statements (similar to when-statements).
"""

from dataclasses import dataclass

from cyecca.ir.expr import Expr
from cyecca.ir.statement import Statement


@dataclass
class Event:
    """
    Discrete event (when clause with imperative statements).

    In Modelica:
        when h < 0 then
          reinit(h, 0);
          reinit(v, -e * pre(v));
        end when;

    Events are triggered when the condition becomes true (rising edge).
    They can contain:
    - reinit() statements to reset state variables
    - assignments to discrete variables
    - function calls (assert, terminate, etc.)

    The difference from when-equations:
    - Events contain statements (:=) not equations (=)
    - Can use reinit() to reset continuous states
    - Execute imperatively in order
    """

    condition: Expr  # Condition that triggers the event
    statements: list[Statement]  # Statements to execute when triggered
    is_initial: bool = False  # Whether this event can trigger at initialization

    def __str__(self):
        result = [f"when {self.condition} then"]
        for stmt in self.statements:
            # Indent each statement
            for line in str(stmt).split("\n"):
                result.append(f"  {line}")
        result.append("end when")
        if self.is_initial:
            result.append(" (initial)")
        return "\n".join(result)

    def __repr__(self):
        return f"Event(condition={self.condition}, statements={len(self.statements)}, is_initial={self.is_initial})"
