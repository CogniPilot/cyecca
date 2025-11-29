"""Tests for FlatModel (cyecca.ir.flat_model).

Exercises FlatModel construction and helpers using IR-only fixtures.
"""

from __future__ import annotations

from typing import List

from cyecca.dsl.equations import Equation
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


# ---------------------------------------------------------------------------
# FlatModel tests
# ---------------------------------------------------------------------------


class TestFlatModel:
    def test_repr_minimal(self) -> None:
        fm = FlatModel(
            name="Empty",
            state_names=[],
            input_names=[],
            output_names=[],
            param_names=[],
            discrete_names=[],
            algebraic_names=[],
            state_vars={},
            input_vars={},
            output_vars={},
            param_vars={},
            discrete_vars={},
            algebraic_vars={},
            equations=[],
        )
        r = repr(fm)
        assert "Empty" in r

    def test_repr_with_states(self) -> None:
        fm = FlatModel(
            name="Ball",
            state_names=["h", "v"],
            input_names=[],
            output_names=[],
            param_names=["g"],
            discrete_names=[],
            algebraic_names=[],
            state_vars={"h": Var(name="h"), "v": Var(name="v")},
            input_vars={},
            output_vars={},
            param_vars={"g": Var(name="g", parameter=True)},
            discrete_vars={},
            algebraic_vars={},
            equations=[_eq(_der("h"), _var("v")), _eq(_der("v"), -_var("g"))],
        )
        r = repr(fm)
        assert "Ball" in r
        assert "states" in r
        assert "params" in r

    def test_defaults_stored(self) -> None:
        fm = FlatModel(
            name="Defaults",
            state_names=["x"],
            input_names=[],
            output_names=[],
            param_names=["p"],
            discrete_names=[],
            algebraic_names=[],
            state_vars={"x": Var(name="x")},
            input_vars={},
            output_vars={},
            param_vars={"p": Var(name="p", parameter=True)},
            discrete_vars={},
            algebraic_vars={},
            equations=[],
            state_defaults={"x": 1.0},
            param_defaults={"p": 9.81},
        )
        assert fm.state_defaults["x"] == 1.0
        assert fm.param_defaults["p"] == 9.81

    def test_when_clauses(self) -> None:
        """Test that FlatModel stores IRWhenClause objects."""
        from cyecca.ir import IRWhenClause

        wc = IRWhenClause(condition=_var("trigger"), reinits=[])

        fm = FlatModel(
            name="Hybrid",
            state_names=[],
            input_names=[],
            output_names=[],
            param_names=[],
            discrete_names=[],
            algebraic_names=[],
            state_vars={},
            input_vars={},
            output_vars={},
            param_vars={},
            discrete_vars={},
            algebraic_vars={},
            equations=[],
            when_clauses=[wc],
        )
        assert len(fm.when_clauses) == 1
        r = repr(fm)
        assert "when_clauses=1" in r

    def test_algorithm_assignments(self) -> None:
        from cyecca.dsl.equations import Assignment

        assign = Assignment(target="temp", expr=_var("x") + _const(1))

        fm = FlatModel(
            name="WithAlgo",
            state_names=[],
            input_names=[],
            output_names=[],
            param_names=[],
            discrete_names=[],
            algebraic_names=[],
            state_vars={},
            input_vars={},
            output_vars={},
            param_vars={},
            discrete_vars={},
            algebraic_vars={},
            equations=[],
            algorithm_assignments=[assign],
            algorithm_locals=["temp"],
        )
        assert fm.algorithm_locals == ["temp"]
        assert len(fm.algorithm_assignments) == 1

    def test_expand_arrays_flag(self) -> None:
        fm = FlatModel(
            name="Arrays",
            state_names=[],
            input_names=[],
            output_names=[],
            param_names=[],
            discrete_names=[],
            algebraic_names=[],
            state_vars={},
            input_vars={},
            output_vars={},
            param_vars={},
            discrete_vars={},
            algebraic_vars={},
            equations=[],
            expand_arrays=False,
        )
        assert fm.expand_arrays is False
