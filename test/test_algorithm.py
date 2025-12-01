"""
Tests for algorithm section execution.
"""

import pytest
import numpy as np
from cyecca.ir import (
    Expr,
    Statement,
    AlgorithmSection,
    Assignment,
    IfStatement,
    ForStatement,
    WhileStatement,
    WhenStatement,
    ReinitStatement,
    BreakStatement,
    FunctionCallStatement,
)
from cyecca.backends.algorithm import (
    NumericAlgorithmExecutor,
    ExecutionContext,
    execute_algorithm,
    BreakException,
)


class TestBasicAssignment:
    """Test basic assignment statements."""

    def test_scalar_assignment(self):
        """Test simple scalar assignment."""
        algo = AlgorithmSection(
            statements=[
                Assignment(Expr.var_ref("x"), Expr.literal(42.0)),
            ]
        )

        result = execute_algorithm(algo, {"x": 0.0})
        assert result["x"] == 42.0

    def test_expression_assignment(self):
        """Test assignment with expression."""
        algo = AlgorithmSection(
            statements=[
                Assignment(Expr.var_ref("y"), Expr.add(Expr.var_ref("a"), Expr.var_ref("b"))),
            ]
        )

        result = execute_algorithm(algo, {"y": 0.0, "a": 3.0, "b": 4.0})
        assert result["y"] == 7.0

    def test_chained_assignments(self):
        """Test multiple sequential assignments."""
        algo = AlgorithmSection(
            statements=[
                Assignment(Expr.var_ref("x"), Expr.literal(1.0)),
                Assignment(Expr.var_ref("y"), Expr.mul(Expr.var_ref("x"), Expr.literal(2.0))),
                Assignment(Expr.var_ref("z"), Expr.add(Expr.var_ref("x"), Expr.var_ref("y"))),
            ]
        )

        result = execute_algorithm(algo, {"x": 0.0, "y": 0.0, "z": 0.0})
        assert result["x"] == 1.0
        assert result["y"] == 2.0
        assert result["z"] == 3.0

    def test_self_update(self):
        """Test variable updating itself."""
        algo = AlgorithmSection(
            statements=[
                Assignment(Expr.var_ref("x"), Expr.add(Expr.var_ref("x"), Expr.literal(1.0))),
                Assignment(Expr.var_ref("x"), Expr.add(Expr.var_ref("x"), Expr.literal(1.0))),
                Assignment(Expr.var_ref("x"), Expr.add(Expr.var_ref("x"), Expr.literal(1.0))),
            ]
        )

        result = execute_algorithm(algo, {"x": 0.0})
        assert result["x"] == 3.0


class TestArrayAssignment:
    """Test array element assignments."""

    def test_array_element_assignment(self):
        """Test assignment to array element."""
        algo = AlgorithmSection(
            statements=[
                Assignment(
                    Expr.component_ref(("arr", [Expr.literal(2)])),  # arr[2]
                    Expr.literal(99.0),
                ),
            ]
        )

        result = execute_algorithm(algo, {"arr": np.array([1.0, 2.0, 3.0])})
        assert result["arr"][1] == 99.0  # 0-based: arr[2] -> index 1

    def test_array_loop_assignment(self):
        """Test assigning to array elements in a loop."""
        # for i in 1:3 loop arr[i] := i * 10; end for;
        algo = AlgorithmSection(
            statements=[
                ForStatement(
                    indices=(("i", Expr.slice(Expr.literal(1), Expr.literal(3))),),
                    body=(
                        Assignment(
                            Expr.component_ref(("arr", [Expr.var_ref("i")])),
                            Expr.mul(Expr.var_ref("i"), Expr.literal(10.0)),
                        ),
                    ),
                )
            ]
        )

        result = execute_algorithm(algo, {"arr": np.zeros(3), "i": 0})
        np.testing.assert_array_equal(result["arr"], [10.0, 20.0, 30.0])


class TestIfStatement:
    """Test if/elseif/else statements."""

    def test_simple_if(self):
        """Test simple if-then."""
        algo = AlgorithmSection(
            statements=[
                IfStatement(
                    branches=(
                        (
                            Expr.binary_op(">", Expr.var_ref("x"), Expr.literal(0.0)),
                            (Assignment(Expr.var_ref("sign"), Expr.literal(1.0)),),
                        ),
                    ),
                    else_statements=(),
                )
            ]
        )

        result = execute_algorithm(algo, {"x": 5.0, "sign": 0.0})
        assert result["sign"] == 1.0

        result = execute_algorithm(algo, {"x": -5.0, "sign": 0.0})
        assert result["sign"] == 0.0  # else branch not executed, stays 0

    def test_if_else(self):
        """Test if-then-else."""
        algo = AlgorithmSection(
            statements=[
                IfStatement(
                    branches=(
                        (
                            Expr.binary_op(">", Expr.var_ref("x"), Expr.literal(0.0)),
                            (Assignment(Expr.var_ref("sign"), Expr.literal(1.0)),),
                        ),
                    ),
                    else_statements=(Assignment(Expr.var_ref("sign"), Expr.literal(-1.0)),),
                )
            ]
        )

        result = execute_algorithm(algo, {"x": 5.0, "sign": 0.0})
        assert result["sign"] == 1.0

        result = execute_algorithm(algo, {"x": -5.0, "sign": 0.0})
        assert result["sign"] == -1.0

    def test_if_elseif_else(self):
        """Test if-elseif-else chain."""
        algo = AlgorithmSection(
            statements=[
                IfStatement(
                    branches=(
                        (
                            Expr.binary_op(">", Expr.var_ref("x"), Expr.literal(0.0)),
                            (Assignment(Expr.var_ref("sign"), Expr.literal(1.0)),),
                        ),
                        (
                            Expr.binary_op("<", Expr.var_ref("x"), Expr.literal(0.0)),
                            (Assignment(Expr.var_ref("sign"), Expr.literal(-1.0)),),
                        ),
                    ),
                    else_statements=(Assignment(Expr.var_ref("sign"), Expr.literal(0.0)),),
                )
            ]
        )

        result = execute_algorithm(algo, {"x": 5.0, "sign": 99.0})
        assert result["sign"] == 1.0

        result = execute_algorithm(algo, {"x": -5.0, "sign": 99.0})
        assert result["sign"] == -1.0

        result = execute_algorithm(algo, {"x": 0.0, "sign": 99.0})
        assert result["sign"] == 0.0


class TestForStatement:
    """Test for loop statements."""

    def test_simple_for_loop(self):
        """Test simple summation loop."""
        # sum := 0; for i in 1:5 loop sum := sum + i; end for;
        algo = AlgorithmSection(
            statements=[
                Assignment(Expr.var_ref("sum"), Expr.literal(0.0)),
                ForStatement(
                    indices=(("i", Expr.slice(Expr.literal(1), Expr.literal(5))),),
                    body=(
                        Assignment(
                            Expr.var_ref("sum"),
                            Expr.add(Expr.var_ref("sum"), Expr.var_ref("i")),
                        ),
                    ),
                ),
            ]
        )

        result = execute_algorithm(algo, {"sum": 0.0, "i": 0})
        assert result["sum"] == 15.0  # 1+2+3+4+5

    def test_for_loop_with_step(self):
        """Test for loop with step."""
        # sum := 0; for i in 1:2:9 loop sum := sum + 1; end for;
        algo = AlgorithmSection(
            statements=[
                Assignment(Expr.var_ref("count"), Expr.literal(0.0)),
                ForStatement(
                    indices=(("i", Expr.slice(Expr.literal(1), Expr.literal(9), Expr.literal(2))),),
                    body=(
                        Assignment(
                            Expr.var_ref("count"),
                            Expr.add(Expr.var_ref("count"), Expr.literal(1.0)),
                        ),
                    ),
                ),
            ]
        )

        result = execute_algorithm(algo, {"count": 0.0, "i": 0})
        assert result["count"] == 5.0  # i = 1, 3, 5, 7, 9

    def test_nested_for_loops(self):
        """Test nested for loops."""
        # for i in 1:3 loop for j in 1:2 loop count := count + 1; end for; end for;
        algo = AlgorithmSection(
            statements=[
                Assignment(Expr.var_ref("count"), Expr.literal(0.0)),
                ForStatement(
                    indices=(("i", Expr.slice(Expr.literal(1), Expr.literal(3))),),
                    body=(
                        ForStatement(
                            indices=(("j", Expr.slice(Expr.literal(1), Expr.literal(2))),),
                            body=(
                                Assignment(
                                    Expr.var_ref("count"),
                                    Expr.add(Expr.var_ref("count"), Expr.literal(1.0)),
                                ),
                            ),
                        ),
                    ),
                ),
            ]
        )

        result = execute_algorithm(algo, {"count": 0.0, "i": 0, "j": 0})
        assert result["count"] == 6.0  # 3 * 2 iterations


class TestWhileStatement:
    """Test while loop statements."""

    def test_simple_while(self):
        """Test simple while loop."""
        # x := 100; while x > 1 loop x := x / 2; end while;
        algo = AlgorithmSection(
            statements=[
                WhileStatement(
                    condition=Expr.binary_op(">", Expr.var_ref("x"), Expr.literal(1.0)),
                    body=(
                        Assignment(
                            Expr.var_ref("x"),
                            Expr.div(Expr.var_ref("x"), Expr.literal(2.0)),
                        ),
                    ),
                )
            ]
        )

        result = execute_algorithm(algo, {"x": 100.0})
        assert result["x"] < 1.0

    def test_while_with_counter(self):
        """Test while loop with iteration counter."""
        # n := 0; while n < 10 loop n := n + 1; end while;
        algo = AlgorithmSection(
            statements=[
                WhileStatement(
                    condition=Expr.binary_op("<", Expr.var_ref("n"), Expr.literal(10.0)),
                    body=(
                        Assignment(
                            Expr.var_ref("n"),
                            Expr.add(Expr.var_ref("n"), Expr.literal(1.0)),
                        ),
                    ),
                )
            ]
        )

        result = execute_algorithm(algo, {"n": 0.0})
        assert result["n"] == 10.0


class TestBreakStatement:
    """Test break statement in loops."""

    def test_break_in_for(self):
        """Test break in for loop."""
        # for i in 1:100 loop if i > 5 then break; end if; count := count + 1; end for;
        algo = AlgorithmSection(
            statements=[
                Assignment(Expr.var_ref("count"), Expr.literal(0.0)),
                ForStatement(
                    indices=(("i", Expr.slice(Expr.literal(1), Expr.literal(100))),),
                    body=(
                        IfStatement(
                            branches=(
                                (
                                    Expr.binary_op(">", Expr.var_ref("i"), Expr.literal(5.0)),
                                    (BreakStatement(),),
                                ),
                            ),
                            else_statements=(),
                        ),
                        Assignment(
                            Expr.var_ref("count"),
                            Expr.add(Expr.var_ref("count"), Expr.literal(1.0)),
                        ),
                    ),
                ),
            ]
        )

        result = execute_algorithm(algo, {"count": 0.0, "i": 0})
        assert result["count"] == 5.0  # Breaks when i > 5, so i=1,2,3,4,5

    def test_break_in_while(self):
        """Test break in while loop."""
        # while true loop n := n + 1; if n >= 10 then break; end if; end while;
        algo = AlgorithmSection(
            statements=[
                WhileStatement(
                    condition=Expr.literal(True),
                    body=(
                        Assignment(
                            Expr.var_ref("n"),
                            Expr.add(Expr.var_ref("n"), Expr.literal(1.0)),
                        ),
                        IfStatement(
                            branches=(
                                (
                                    Expr.binary_op(">=", Expr.var_ref("n"), Expr.literal(10.0)),
                                    (BreakStatement(),),
                                ),
                            ),
                            else_statements=(),
                        ),
                    ),
                )
            ]
        )

        result = execute_algorithm(algo, {"n": 0.0})
        assert result["n"] == 10.0


class TestMathFunctions:
    """Test math function evaluation."""

    def test_trig_functions(self):
        """Test trigonometric functions."""
        algo = AlgorithmSection(
            statements=[
                Assignment(Expr.var_ref("s"), Expr.sin(Expr.var_ref("x"))),
                Assignment(Expr.var_ref("c"), Expr.cos(Expr.var_ref("x"))),
            ]
        )

        result = execute_algorithm(algo, {"x": 0.0, "s": 0.0, "c": 0.0})
        assert np.isclose(result["s"], 0.0)
        assert np.isclose(result["c"], 1.0)

        result = execute_algorithm(algo, {"x": np.pi / 2, "s": 0.0, "c": 0.0})
        assert np.isclose(result["s"], 1.0)
        assert np.isclose(result["c"], 0.0, atol=1e-15)

    def test_exp_log_sqrt(self):
        """Test exp, log, sqrt functions."""
        algo = AlgorithmSection(
            statements=[
                Assignment(Expr.var_ref("e"), Expr.exp(Expr.literal(1.0))),
                Assignment(Expr.var_ref("l"), Expr.log(Expr.var_ref("e"))),
                Assignment(Expr.var_ref("r"), Expr.sqrt(Expr.literal(4.0))),
            ]
        )

        result = execute_algorithm(algo, {"e": 0.0, "l": 0.0, "r": 0.0})
        assert np.isclose(result["e"], np.e)
        assert np.isclose(result["l"], 1.0)
        assert np.isclose(result["r"], 2.0)


class TestIfExpression:
    """Test if expressions (not statements)."""

    def test_if_expression(self):
        """Test if expression in assignment."""
        # y := if x > 0 then 1 else -1;
        algo = AlgorithmSection(
            statements=[
                Assignment(
                    Expr.var_ref("y"),
                    Expr.if_expr(
                        Expr.binary_op(">", Expr.var_ref("x"), Expr.literal(0.0)),
                        Expr.literal(1.0),
                        Expr.literal(-1.0),
                    ),
                ),
            ]
        )

        result = execute_algorithm(algo, {"x": 5.0, "y": 0.0})
        assert result["y"] == 1.0

        result = execute_algorithm(algo, {"x": -5.0, "y": 0.0})
        assert result["y"] == -1.0


class TestFunctionCallStatement:
    """Test function call statements."""

    def test_assert_pass(self):
        """Test assert statement that passes."""
        algo = AlgorithmSection(
            statements=[
                FunctionCallStatement(
                    "assert",
                    (
                        Expr.binary_op(">", Expr.var_ref("x"), Expr.literal(0.0)),
                        Expr.literal("x must be positive"),
                    ),
                ),
            ]
        )

        # Should not raise
        result = execute_algorithm(algo, {"x": 5.0})
        assert result["x"] == 5.0

    def test_assert_fail(self):
        """Test assert statement that fails."""
        algo = AlgorithmSection(
            statements=[
                FunctionCallStatement(
                    "assert",
                    (
                        Expr.binary_op(">", Expr.var_ref("x"), Expr.literal(0.0)),
                        Expr.literal("x must be positive"),
                    ),
                ),
            ]
        )

        with pytest.raises(AssertionError, match="x must be positive"):
            execute_algorithm(algo, {"x": -5.0})


class TestExecutionContext:
    """Test ExecutionContext functionality."""

    def test_context_copy(self):
        """Test context copying."""
        ctx = ExecutionContext(variables={"x": 1.0, "y": 2.0})
        ctx_copy = ctx.copy()

        ctx_copy.set("x", 100.0)
        assert ctx.get("x") == 1.0  # Original unchanged
        assert ctx_copy.get("x") == 100.0

    def test_array_operations(self):
        """Test array get/set in context."""
        ctx = ExecutionContext(variables={"arr": np.array([1.0, 2.0, 3.0])})

        # Get element
        assert ctx.get_indexed("arr", (1,)) == 2.0  # 0-based

        # Set element
        ctx.set_indexed("arr", (1,), 99.0)
        assert ctx.get_indexed("arr", (1,)) == 99.0


class TestComplexAlgorithm:
    """Test more complex algorithm scenarios."""

    def test_bubble_sort(self):
        """Test a simple bubble sort algorithm."""
        # Simple bubble sort for small array
        # for i in 1:n-1 loop
        #   for j in 1:n-i loop
        #     if arr[j] > arr[j+1] then
        #       temp := arr[j];
        #       arr[j] := arr[j+1];
        #       arr[j+1] := temp;
        #     end if;
        #   end for;
        # end for;

        n = 4
        algo = AlgorithmSection(
            statements=[
                ForStatement(
                    indices=(("i", Expr.slice(Expr.literal(1), Expr.literal(n - 1))),),
                    body=(
                        ForStatement(
                            indices=(
                                (
                                    "j",
                                    Expr.slice(
                                        Expr.literal(1),
                                        Expr.sub(Expr.literal(n), Expr.var_ref("i")),
                                    ),
                                ),
                            ),
                            body=(
                                IfStatement(
                                    branches=(
                                        (
                                            Expr.binary_op(
                                                ">",
                                                Expr.component_ref(("arr", [Expr.var_ref("j")])),
                                                Expr.component_ref(
                                                    (
                                                        "arr",
                                                        [
                                                            Expr.add(
                                                                Expr.var_ref("j"), Expr.literal(1)
                                                            )
                                                        ],
                                                    )
                                                ),
                                            ),
                                            (
                                                # temp := arr[j]
                                                Assignment(
                                                    Expr.var_ref("temp"),
                                                    Expr.component_ref(
                                                        ("arr", [Expr.var_ref("j")])
                                                    ),
                                                ),
                                                # arr[j] := arr[j+1]
                                                Assignment(
                                                    Expr.component_ref(
                                                        ("arr", [Expr.var_ref("j")])
                                                    ),
                                                    Expr.component_ref(
                                                        (
                                                            "arr",
                                                            [
                                                                Expr.add(
                                                                    Expr.var_ref("j"),
                                                                    Expr.literal(1),
                                                                )
                                                            ],
                                                        )
                                                    ),
                                                ),
                                                # arr[j+1] := temp
                                                Assignment(
                                                    Expr.component_ref(
                                                        (
                                                            "arr",
                                                            [
                                                                Expr.add(
                                                                    Expr.var_ref("j"),
                                                                    Expr.literal(1),
                                                                )
                                                            ],
                                                        )
                                                    ),
                                                    Expr.var_ref("temp"),
                                                ),
                                            ),
                                        ),
                                    ),
                                    else_statements=(),
                                ),
                            ),
                        ),
                    ),
                ),
            ]
        )

        result = execute_algorithm(
            algo, {"arr": np.array([4.0, 2.0, 3.0, 1.0]), "i": 0, "j": 0, "temp": 0.0}
        )
        np.testing.assert_array_equal(result["arr"], [1.0, 2.0, 3.0, 4.0])

    def test_factorial(self):
        """Test factorial computation."""
        # result := 1; for i in 1:n loop result := result * i; end for;
        algo = AlgorithmSection(
            statements=[
                Assignment(Expr.var_ref("result"), Expr.literal(1.0)),
                ForStatement(
                    indices=(("i", Expr.slice(Expr.literal(1), Expr.var_ref("n"))),),
                    body=(
                        Assignment(
                            Expr.var_ref("result"),
                            Expr.mul(Expr.var_ref("result"), Expr.var_ref("i")),
                        ),
                    ),
                ),
            ]
        )

        result = execute_algorithm(algo, {"n": 5.0, "result": 0.0, "i": 0})
        assert result["result"] == 120.0  # 5! = 120


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
