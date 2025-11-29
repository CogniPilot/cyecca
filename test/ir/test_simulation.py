"""Tests for IR simulation utilities (cyecca.ir.simulation).

Exercises SimulationResult and the abstract Simulator interface
without relying on a real backend.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest

from cyecca.ir.simulation import SimulationResult, Simulator

# ---------------------------------------------------------------------------
# SimulationResult tests
# ---------------------------------------------------------------------------


class TestSimulationResult:
    """Unit tests for the SimulationResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> SimulationResult:
        t = np.linspace(0, 1, 11)
        return SimulationResult(
            t=t,
            _data={"x": np.sin(t), "y": np.cos(t), "u": np.ones_like(t)},
            model_name="TestModel",
            state_names=["x"],
            output_names=["y"],
            input_names=["u"],
            discrete_names=[],
        )

    def test_call_by_string(self, sample_result: SimulationResult) -> None:
        arr = sample_result("x")
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 11

    def test_call_by_object_with_name_attr(self, sample_result: SimulationResult) -> None:
        class FakeVar:
            name = "y"

        arr = sample_result(FakeVar())
        assert arr[0] == pytest.approx(1.0)

    def test_call_by_object_with_private_name(self, sample_result: SimulationResult) -> None:
        class SymVar:
            _name = "x"

        arr = sample_result(SymVar())
        assert len(arr) == 11

    def test_call_unknown_var_raises(self, sample_result: SimulationResult) -> None:
        with pytest.raises(KeyError):
            sample_result("nonexistent")

    def test_call_bad_type_raises(self, sample_result: SimulationResult) -> None:
        with pytest.raises(TypeError):
            sample_result(12345)  # type: ignore[arg-type]

    def test_getitem_time(self, sample_result: SimulationResult) -> None:
        t = sample_result["t"]
        assert t[0] == 0.0

    def test_getitem_variable(self, sample_result: SimulationResult) -> None:
        x = sample_result["x"]
        assert len(x) == 11

    def test_getitem_unknown_raises(self, sample_result: SimulationResult) -> None:
        with pytest.raises(KeyError):
            sample_result["nope"]

    def test_available_names(self, sample_result: SimulationResult) -> None:
        names = sample_result.available_names
        assert set(names) == {"x", "y", "u"}

    def test_states_property(self, sample_result: SimulationResult) -> None:
        states = sample_result.states
        assert "x" in states
        assert "y" not in states

    def test_outputs_property(self, sample_result: SimulationResult) -> None:
        outputs = sample_result.outputs
        assert "y" in outputs

    def test_inputs_property(self, sample_result: SimulationResult) -> None:
        inputs = sample_result.inputs
        assert "u" in inputs

    def test_discrete_property(self, sample_result: SimulationResult) -> None:
        disc = sample_result.discrete
        assert disc == {}

    def test_data_property(self, sample_result: SimulationResult) -> None:
        data = sample_result.data
        assert "t" in data
        assert "x" in data


# ---------------------------------------------------------------------------
# Simulator abstract interface test
# ---------------------------------------------------------------------------


class DummySimulator(Simulator):
    """Minimal concrete implementation of Simulator for testing."""

    def simulate(
        self,
        t0: float = 0.0,
        tf: float = 10.0,
        dt: float = 0.01,
        x0: Optional[Dict[str, Any]] = None,
        u: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        u_func: Optional[Callable[[float], Dict[str, Any]]] = None,
    ) -> SimulationResult:
        t = np.arange(t0, tf, dt)
        return SimulationResult(
            t=t,
            _data={"x": np.zeros_like(t)},
            model_name="Dummy",
            state_names=["x"],
        )

    @property
    def state_names(self) -> List[str]:
        return ["x"]

    @property
    def input_names(self) -> List[str]:
        return []

    @property
    def output_names(self) -> List[str]:
        return []

    @property
    def param_names(self) -> List[str]:
        return []


class TestSimulatorInterface:
    """Ensure abstract Simulator can be subclassed."""

    def test_dummy_simulate(self) -> None:
        sim = DummySimulator()
        result = sim.simulate(tf=1.0, dt=0.1)
        assert result.model_name == "Dummy"
        assert "x" in result.available_names

    def test_properties(self) -> None:
        sim = DummySimulator()
        assert sim.state_names == ["x"]
        assert sim.input_names == []
        assert sim.output_names == []
        assert sim.param_names == []
