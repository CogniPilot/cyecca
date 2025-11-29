"""Tests for IR causality analysis (cyecca.ir.causality).

Exercises find_variables, is_linear_in, solve_linear, analyze_causality,
and Tarjan SCC helpers using IR-only fixtures (no DSL).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from cyecca.dsl.equations import Equation  # DSL Equation used by FlatModel
from cyecca.ir.causality import (
    ImplicitBlock,
    SolvedEquation,
    SortedSystem,
    analyze_causality,
    find_variables,
    is_linear_in,
    solve_linear,
)
from cyecca.ir.expr import Expr, ExprKind
from cyecca.ir.flat_model import FlatModel
from cyecca.ir.types import Var

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _var(name: str) -> Expr:
    return Expr(ExprKind.VARIABLE, name=name)


def _const(value: float) -> Expr:
    return Expr(ExprKind.CONSTANT, value=value)


def _der(name: str) -> Expr:
    return Expr(ExprKind.DERIVATIVE, name=name)


def _eq(lhs: Expr, rhs: Expr) -> Equation:
    return Equation(lhs=lhs, rhs=rhs)


def _minimal_flat(
    state_names: List[str],
    algebraic_names: List[str],
    equations: List[Equation],
) -> FlatModel:
    """Build a minimal FlatModel for causality tests."""
    state_vars = {n: Var(name=n) for n in state_names}
    algebraic_vars = {n: Var(name=n) for n in algebraic_names}
    return FlatModel(
        name="TestModel",
        state_names=state_names,
        input_names=[],
        output_names=[],
        param_names=[],
        discrete_names=[],
        algebraic_names=algebraic_names,
        state_vars=state_vars,
        input_vars={},
        output_vars={},
        param_vars={},
        discrete_vars={},
        algebraic_vars=algebraic_vars,
        equations=equations,
    )


# ---------------------------------------------------------------------------
# find_variables
# ---------------------------------------------------------------------------


class TestFindVariables:
    def test_simple_variable(self) -> None:
        assert find_variables(_var("x")) == {"x"}

    def test_derivative(self) -> None:
        assert find_variables(_der("x")) == {"der_x"}

    def test_composite(self) -> None:
        e = _var("a") + _der("b") * _var("c")
        assert find_variables(e) == {"a", "der_b", "c"}


# ---------------------------------------------------------------------------
# is_linear_in
# ---------------------------------------------------------------------------


class TestIsLinearIn:
    def test_variable_alone(self) -> None:
        is_lin, coef, const = is_linear_in(_var("x"), "x")
        assert is_lin
        assert coef is not None and coef.value == 1.0
        assert const is not None and const.value == 0.0

    def test_constant_expr(self) -> None:
        is_lin, coef, const = is_linear_in(_const(5), "x")
        assert is_lin
        assert coef is not None and coef.value == 0.0
        assert const is not None and const.value == 5.0

    def test_mul_coef_left(self) -> None:
        # 2 * x
        e = _const(2) * _var("x")
        is_lin, coef, const = is_linear_in(e, "x")
        assert is_lin
        assert coef is not None and coef.value == 2.0

    def test_derivative_linear(self) -> None:
        # der(x) alone
        is_lin, coef, const = is_linear_in(_der("x"), "der_x")
        assert is_lin
        assert coef is not None and coef.value == 1.0


# ---------------------------------------------------------------------------
# solve_linear
# ---------------------------------------------------------------------------


class TestSolveLinear:
    def test_simple_deriv_eq(self) -> None:
        # der(x) == v  =>  der_x = v
        eq = _eq(_der("x"), _var("v"))
        solved = solve_linear(eq, "der_x")
        assert solved is not None
        assert solved.var_name == "x"
        assert solved.is_derivative

    def test_algebraic_eq(self) -> None:
        # y == 2 * x  =>  y = 2*x
        eq = _eq(_var("y"), _const(2) * _var("x"))
        solved = solve_linear(eq, "y")
        assert solved is not None
        assert solved.var_name == "y"
        assert not solved.is_derivative

    def test_unsolvable_returns_none(self) -> None:
        # nonlinear: x * x == 1
        eq = _eq(_var("x") * _var("x"), _const(1))
        solved = solve_linear(eq, "x")
        assert solved is None


# ---------------------------------------------------------------------------
# analyze_causality
# ---------------------------------------------------------------------------


class TestAnalyzeCausality:
    def test_explicit_ode(self) -> None:
        # Simple: der(x) == -x
        eqs = [_eq(_der("x"), -_var("x"))]
        fm = _minimal_flat(state_names=["x"], algebraic_names=[], equations=eqs)
        ss = analyze_causality(fm)
        assert ss.is_ode_explicit
        assert len(ss.solved) == 1
        assert ss.solved[0].is_derivative

    def test_algebraic_present(self) -> None:
        # der(x) == y, y == x
        eqs = [_eq(_der("x"), _var("y")), _eq(_var("y"), _var("x"))]
        fm = _minimal_flat(state_names=["x"], algebraic_names=["y"], equations=eqs)
        ss = analyze_causality(fm)
        assert ss.has_algebraic

    def test_empty_equations(self) -> None:
        fm = _minimal_flat(state_names=[], algebraic_names=[], equations=[])
        ss = analyze_causality(fm)
        assert ss.solved == []
        assert ss.implicit_blocks == []

    def test_implicit_block_for_coupled(self) -> None:
        # Coupled: der(x) == y, der(y) == x  (implicitly coupled)
        # After analysis, may fall back to implicit block if solve_linear can't isolate
        eqs = [_eq(_der("x"), _var("y")), _eq(_der("y"), _var("x"))]
        fm = _minimal_flat(state_names=["x", "y"], algebraic_names=[], equations=eqs)
        ss = analyze_causality(fm)
        # Both should be solvable as scalar blocks if linear
        assert ss.is_ode_explicit or len(ss.implicit_blocks) > 0


# ---------------------------------------------------------------------------
# SortedSystem / SolvedEquation / ImplicitBlock dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_solved_equation_fields(self) -> None:
        eq = _eq(_der("x"), _var("v"))
        se = SolvedEquation(var_name="x", expr=_var("v"), original=eq, is_derivative=True)
        assert se.var_name == "x"
        assert se.is_derivative

    def test_implicit_block_fields(self) -> None:
        eq = _eq(_var("a"), _var("b"))
        ib = ImplicitBlock(equations=[eq], unknowns=["a"])
        assert len(ib.equations) == 1
        assert ib.unknowns == ["a"]

    def test_sorted_system_defaults(self) -> None:
        fm = _minimal_flat([], [], [])
        ss = SortedSystem(model=fm)
        assert ss.solved == []
        assert ss.implicit_blocks == []
        assert ss.unhandled == []
        assert ss.is_ode_explicit
        assert not ss.has_algebraic
