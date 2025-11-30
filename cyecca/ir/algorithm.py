"""
Algorithm section representation in the IR.

Algorithm sections contain imperative statements that execute sequentially.
They differ from equation sections which are declarative and unordered.
"""

from dataclasses import dataclass

from cyecca.ir.statement import Statement


@dataclass
class AlgorithmSection:
    """
    Algorithm section (imperative code block).

    In Modelica:
        algorithm
          sum := 0;
          for i in 1:n loop
            sum := sum + x[i];
          end for;
          average := sum / n;

    Algorithms execute statements in order, unlike equations which are unordered.
    """

    statements: list[Statement]
    is_initial: bool = False  # Whether this is an initial algorithm section

    def __str__(self):
        section_type = "initial algorithm" if self.is_initial else "algorithm"
        result = [section_type]
        for stmt in self.statements:
            # Indent each statement
            for line in str(stmt).split("\n"):
                result.append(f"  {line}")
        return "\n".join(result)

    def __repr__(self):
        return f"AlgorithmSection(statements={len(self.statements)}, is_initial={self.is_initial})"
