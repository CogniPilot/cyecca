"""Tests for the core IRModel helpers."""

from __future__ import annotations

import pytest

from cyecca.ir import (
    Expr,
    ExprKind,
    IRConnector,
    IREquation,
    IRInitialEquation,
    IRModel,
    IRReinit,
    IRVariable,
    IRWhenClause,
)


def _sym(name: str) -> Expr:
    return Expr(ExprKind.VARIABLE, name=name)


def _const(value: float) -> Expr:
    return Expr(ExprKind.CONSTANT, value=value)


class TestIRModel:
    """Unit tests that target IRModel-specific behavior."""

    def test_add_variable_prevents_duplicates(self) -> None:
        model = IRModel(name="TestModel")
        model.add_variable(IRVariable(name="x"))

        with pytest.raises(ValueError):
            model.add_variable(IRVariable(name="x"))

    def test_add_submodel_prevents_duplicates(self) -> None:
        parent = IRModel(name="Parent")
        parent.add_submodel("child", IRModel(name="Child"))

        with pytest.raises(ValueError):
            parent.add_submodel("child", IRModel(name="ChildDuplicate"))

    def test_connections_and_when_clause_storage(self) -> None:
        model = IRModel(name="Connections")

        reinit = IRReinit(var_name="x", expr=_const(0.0))
        when_clause = IRWhenClause(
            condition=Expr(ExprKind.GT, children=(_sym("x"), _const(1.0))),
            reinits=[reinit],
            description="reset",
        )

        model.add_when_clause(when_clause)
        model.add_connection("a.port", "b.port")

        assert model.when_clauses == [when_clause]
        assert model.connections == [("a.port", "b.port")]

    def test_summary_reflects_model_contents(self) -> None:
        model = IRModel(name="Summary", description="Example")
        model.add_variable(IRVariable(name="param", parameter=True))
        model.add_variable(IRVariable(name="input", input=True))
        model.add_variable(IRVariable(name="output", output=True))
        model.add_variable(IRVariable(name="disc", discrete=True))
        model.add_variable(IRVariable(name="alg"))

        model.add_equation(IREquation(lhs=_sym("alg"), rhs=_sym("param")))
        model.add_initial_equation(IRInitialEquation(lhs=_sym("alg"), rhs=_const(0.0)))
        model.add_when_clause(IRWhenClause(condition=Expr(ExprKind.GE, children=(_sym("disc"), _const(0.0)))))
        model.add_submodel("child", IRModel(name="Child"))
        model.add_connection("child.port", "root.port")

        summary = model.summary().splitlines()

        assert any(line == "Model: Summary" for line in summary)
        assert any("Parameters: 1" in line for line in summary)
        assert any("Inputs: 1" in line for line in summary)
        assert any("Outputs: 1" in line for line in summary)
        assert any("Discrete: 1" in line for line in summary)
        assert any("Algebraic: 1" in line for line in summary)
        assert any("Equations: 1" in line for line in summary)
        assert any("When-clauses: 1" in line for line in summary)
        assert any("Submodels: 1" in line for line in summary)


class TestIRConnector:
    """Minimal coverage of the connector helper."""

    def test_irconnector_repr_reports_counts(self) -> None:
        connector = IRConnector(
            name="Flange",
            potentials=[IRVariable(name="phi")],
            flows=[IRVariable(name="tau", flow=True)],
        )

        text = repr(connector)
        assert "Flange" in text
        assert "pot=1" in text
        assert "flow=1" in text
