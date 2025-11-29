"""Tests for IR equation classes (cyecca.ir.equation).

Exercises IREquation, IRReinit, IRWhenClause, IRInitialEquation, and
IRAssignment dataclasses without using the DSL layer.
"""

from __future__ import annotations

from cyecca.ir.equation import IRAssignment, IREquation, IRInitialEquation, IRReinit, IRWhenClause
from cyecca.ir.expr import Expr, ExprKind

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _var(name: str) -> Expr:
    return Expr(ExprKind.VARIABLE, name=name)


def _const(value: float) -> Expr:
    return Expr(ExprKind.CONSTANT, value=value)


def _der(name: str) -> Expr:
    return Expr(ExprKind.DERIVATIVE, name=name)


# ---------------------------------------------------------------------------
# IREquation
# ---------------------------------------------------------------------------


class TestIREquation:
    def test_repr(self) -> None:
        eq = IREquation(lhs=_der("x"), rhs=_var("v"))
        assert "der(x)" in repr(eq) and "v" in repr(eq)

    def test_immutability(self) -> None:
        eq = IREquation(lhs=_var("y"), rhs=_const(0))
        # frozen dataclass should raise on assignment
        try:
            eq.lhs = _var("z")  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except AttributeError:
            pass

    def test_description_field(self) -> None:
        eq = IREquation(lhs=_var("a"), rhs=_var("b"), description="test equation")
        assert eq.description == "test equation"


# ---------------------------------------------------------------------------
# IRReinit
# ---------------------------------------------------------------------------


class TestIRReinit:
    def test_repr(self) -> None:
        r = IRReinit(var_name="v", expr=_const(-1))
        assert repr(r) == "reinit(v, -1)"

    def test_fields(self) -> None:
        r = IRReinit(var_name="x", expr=_var("y"))
        assert r.var_name == "x"
        assert r.expr.kind == ExprKind.VARIABLE


# ---------------------------------------------------------------------------
# IRWhenClause
# ---------------------------------------------------------------------------


class TestIRWhenClause:
    def test_repr(self) -> None:
        wc = IRWhenClause(
            condition=Expr(ExprKind.LT, children=(_var("h"), _const(0))),
            reinits=[IRReinit(var_name="v", expr=_const(0))],
        )
        r = repr(wc)
        assert "IRWhenClause" in r
        assert "reinits=1" in r

    def test_empty_reinits(self) -> None:
        wc = IRWhenClause(condition=_var("trigger"))
        assert wc.reinits == []

    def test_description(self) -> None:
        wc = IRWhenClause(condition=_var("c"), description="bounce event")
        assert wc.description == "bounce event"


# ---------------------------------------------------------------------------
# IRInitialEquation
# ---------------------------------------------------------------------------


class TestIRInitialEquation:
    def test_repr(self) -> None:
        ieq = IRInitialEquation(lhs=_var("x"), rhs=_const(1))
        r = repr(ieq)
        assert "IRInitialEquation" in r
        assert "x" in r and "1" in r


# ---------------------------------------------------------------------------
# IRAssignment
# ---------------------------------------------------------------------------


class TestIRAssignment:
    def test_repr(self) -> None:
        a = IRAssignment(var_name="temp", expr=_var("x") + _const(1))
        r = repr(a)
        assert "temp" in r
        assert ":=" in r
