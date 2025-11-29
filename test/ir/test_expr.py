"""Tests for IR expression trees (cyecca.ir.expr).

Exercises Expr operators, repr, helper functions, and edge cases
without relying on the DSL layer.
"""

from __future__ import annotations

import pytest

from cyecca.ir.expr import (
    Expr,
    ExprKind,
    find_derivatives,
    format_indices,
    get_base_name,
    is_array_state,
    iter_indices,
    parse_indices,
    prefix_expr,
)

# ---------------------------------------------------------------------------
# Factory helpers (no DSL)
# ---------------------------------------------------------------------------


def _var(name: str, indices: tuple[int, ...] = ()) -> Expr:
    return Expr(ExprKind.VARIABLE, name=name, indices=indices)


def _const(value: float) -> Expr:
    return Expr(ExprKind.CONSTANT, value=value)


def _der(name: str) -> Expr:
    return Expr(ExprKind.DERIVATIVE, name=name)


def _time() -> Expr:
    return Expr(ExprKind.TIME)


# ---------------------------------------------------------------------------
# Basic repr coverage
# ---------------------------------------------------------------------------


class TestExprRepr:
    """Verify __repr__ for various ExprKind nodes."""

    def test_variable_scalar(self) -> None:
        assert repr(_var("x")) == "x"

    def test_variable_indexed(self) -> None:
        assert repr(_var("pos", (0, 1))) == "pos[0,1]"

    def test_derivative(self) -> None:
        assert repr(_der("theta")) == "der(theta)"

    def test_constant(self) -> None:
        assert repr(_const(3.14)) == "3.14"

    def test_time(self) -> None:
        assert repr(_time()) == "t"

    def test_neg(self) -> None:
        e = Expr(ExprKind.NEG, children=(_var("x"),))
        assert repr(e) == "(-x)"

    def test_add(self) -> None:
        e = _var("a") + _var("b")
        assert repr(e) == "(a + b)"

    def test_sub(self) -> None:
        e = _var("a") - _var("b")
        assert repr(e) == "(a - b)"

    def test_mul(self) -> None:
        e = _var("a") * _var("b")
        assert repr(e) == "(a * b)"

    def test_div(self) -> None:
        e = _var("a") / _var("b")
        assert repr(e) == "(a / b)"

    def test_pow(self) -> None:
        e = _var("x") ** _const(2)
        assert repr(e) == "(x ** 2)"

    def test_index_kind(self) -> None:
        # ExprKind.INDEX is used for computed indexing (different from indexed VARIABLE)
        e = Expr(ExprKind.INDEX, name="arr", value=5)
        assert "arr" in repr(e) and "5" in repr(e)

    def test_array_literal(self) -> None:
        e = Expr(ExprKind.ARRAY_LITERAL, children=(_const(1), _const(2), _const(3)))
        assert repr(e) == "[1, 2, 3]"

    def test_pre(self) -> None:
        e = Expr(ExprKind.PRE, name="v")
        assert repr(e) == "pre(v)"

    def test_edge(self) -> None:
        e = Expr(ExprKind.EDGE, name="trigger")
        assert repr(e) == "edge(trigger)"

    def test_change(self) -> None:
        e = Expr(ExprKind.CHANGE, name="state")
        assert repr(e) == "change(state)"

    def test_relational_ops(self) -> None:
        x, y = _var("x"), _var("y")
        assert "<" in repr(x < y)
        assert "<=" in repr(x <= y)
        assert ">" in repr(x > y)
        assert ">=" in repr(x >= y)

    def test_boolean_ops(self) -> None:
        a = Expr(ExprKind.AND, children=(_var("p"), _var("q")))
        o = Expr(ExprKind.OR, children=(_var("p"), _var("q")))
        n = Expr(ExprKind.NOT, children=(_var("p"),))
        assert "and" in repr(a)
        assert "or" in repr(o)
        assert "not" in repr(n)

    def test_if_then_else(self) -> None:
        e = Expr(ExprKind.IF_THEN_ELSE, children=(_var("c"), _const(1), _const(0)))
        assert "if" in repr(e) and "then" in repr(e) and "else" in repr(e)

    def test_reinit(self) -> None:
        e = Expr(ExprKind.REINIT, name="x", children=(_const(0),))
        assert repr(e) == "reinit(x, 0)"

    def test_initial_terminal(self) -> None:
        assert repr(Expr(ExprKind.INITIAL)) == "initial()"
        assert repr(Expr(ExprKind.TERMINAL)) == "terminal()"

    def test_sample(self) -> None:
        e = Expr(ExprKind.SAMPLE, children=(_const(0), _const(0.1)))
        assert "sample" in repr(e)

    def test_math_funcs(self) -> None:
        for kind in (
            ExprKind.SIN,
            ExprKind.COS,
            ExprKind.TAN,
            ExprKind.ASIN,
            ExprKind.ACOS,
            ExprKind.ATAN,
            ExprKind.SQRT,
            ExprKind.EXP,
            ExprKind.LOG,
            ExprKind.LOG10,
            ExprKind.ABS,
            ExprKind.SIGN,
            ExprKind.FLOOR,
            ExprKind.CEIL,
            ExprKind.SINH,
            ExprKind.COSH,
            ExprKind.TANH,
        ):
            e = Expr(kind, children=(_var("x"),))
            assert kind.name.lower() in repr(e).lower()

    def test_atan2_min_max(self) -> None:
        assert "atan2" in repr(Expr(ExprKind.ATAN2, children=(_var("y"), _var("x"))))
        assert "min" in repr(Expr(ExprKind.MIN, children=(_var("a"), _var("b"))))
        assert "max" in repr(Expr(ExprKind.MAX, children=(_var("a"), _var("b"))))


# ---------------------------------------------------------------------------
# Arithmetic operator tests
# ---------------------------------------------------------------------------


class TestExprArithmetic:
    """Test Expr arithmetic dunder methods."""

    def test_radd(self) -> None:
        e = 1 + _var("x")
        assert e.kind == ExprKind.ADD

    def test_rsub(self) -> None:
        e = 1 - _var("x")
        assert e.kind == ExprKind.SUB

    def test_rmul(self) -> None:
        e = 2 * _var("x")
        assert e.kind == ExprKind.MUL

    def test_rtruediv(self) -> None:
        e = 1 / _var("x")
        assert e.kind == ExprKind.DIV

    def test_rpow(self) -> None:
        e = 2 ** _var("x")
        assert e.kind == ExprKind.POW

    def test_neg(self) -> None:
        e = -_var("x")
        assert e.kind == ExprKind.NEG

    def test_pos(self) -> None:
        x = _var("x")
        assert +x is x


# ---------------------------------------------------------------------------
# Equality / hashing
# ---------------------------------------------------------------------------


class TestExprEquality:
    """Test __eq__ and __hash__ for Expr when outside DSL context."""

    def test_eq_returns_expr_outside_context(self) -> None:
        # Outside @equations context, __eq__ should return an Expr (EQ kind)
        result = _var("x") == _const(1)
        assert isinstance(result, Expr)
        assert result.kind == ExprKind.EQ

    def test_ne_returns_expr(self) -> None:
        result = _var("x") != _const(1)
        assert isinstance(result, Expr)
        assert result.kind == ExprKind.NE

    def test_hash_consistency(self) -> None:
        a = _var("x")
        b = _var("x")
        assert hash(a) == hash(b)
        assert a == b  # still returns Expr, but hashes match


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Cover expr helper functions."""

    def test_find_derivatives_single(self) -> None:
        e = _der("theta")
        assert find_derivatives(e) == {"theta"}

    def test_find_derivatives_nested(self) -> None:
        e = _der("x") + _der("y") * _const(2)
        derivs = find_derivatives(e)
        assert derivs == {"x", "y"}

    def test_find_derivatives_none(self) -> None:
        e = _var("x") + _const(1)
        assert find_derivatives(e) == set()

    def test_prefix_expr_variable(self) -> None:
        e = _var("x")
        prefixed = prefix_expr(e, "sub")
        assert prefixed.name == "sub.x"

    def test_prefix_expr_derivative(self) -> None:
        e = _der("x")
        prefixed = prefix_expr(e, "sub")
        assert prefixed.name == "sub.x"

    def test_prefix_expr_pre(self) -> None:
        e = Expr(ExprKind.PRE, name="v")
        prefixed = prefix_expr(e, "sub")
        assert prefixed.name == "sub.v"

    def test_prefix_expr_constant_unchanged(self) -> None:
        e = _const(5)
        prefixed = prefix_expr(e, "sub")
        assert prefixed is e

    def test_prefix_expr_composite(self) -> None:
        e = _var("a") + _var("b")
        prefixed = prefix_expr(e, "m")
        # both children should be prefixed
        assert prefixed.children[0].name == "m.a"
        assert prefixed.children[1].name == "m.b"

    def test_get_base_name(self) -> None:
        assert get_base_name("pos[0,1]") == "pos"
        assert get_base_name("scalar") == "scalar"

    def test_parse_indices(self) -> None:
        assert parse_indices("pos[0,1]") == ("pos", (0, 1))
        assert parse_indices("scalar") == ("scalar", ())

    def test_format_indices(self) -> None:
        assert format_indices((0, 1)) == "[0,1]"
        assert format_indices(()) == ""

    def test_iter_indices_scalar(self) -> None:
        assert list(iter_indices(())) == [()]

    def test_iter_indices_vector(self) -> None:
        assert list(iter_indices((3,))) == [(0,), (1,), (2,)]

    def test_iter_indices_matrix(self) -> None:
        idxs = list(iter_indices((2, 2)))
        assert idxs == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_is_array_state_true(self) -> None:
        derivs = {"pos[0]", "pos[1]"}
        assert is_array_state("pos", (2,), derivs) is True

    def test_is_array_state_false(self) -> None:
        derivs = {"other"}
        assert is_array_state("pos", (2,), derivs) is False

    def test_is_array_state_scalar(self) -> None:
        # Scalars always return False from is_array_state
        assert is_array_state("x", (), {"x"}) is False


# ---------------------------------------------------------------------------
# indexed_name property
# ---------------------------------------------------------------------------


class TestIndexedName:
    def test_indexed_name_with_indices(self) -> None:
        e = _var("vec", (0,))
        assert e.indexed_name == "vec[0]"

    def test_indexed_name_without_indices(self) -> None:
        e = _var("scalar")
        assert e.indexed_name == "scalar"
