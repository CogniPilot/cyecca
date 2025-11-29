"""Tests for the IR-level Var/SubmodelField helpers."""

from __future__ import annotations

from dataclasses import dataclass

from cyecca.ir.types import DType, SubmodelField, Var


class TestVar:
    """Tests covering the low-level IRVariable wrapper."""

    def test_var_default(self) -> None:
        v = Var()
        assert repr(v) == "var()"

    def test_var_with_dtype(self) -> None:
        v = Var(dtype=DType.INTEGER)
        assert "dtype=INTEGER" in repr(v)
        assert v.dtype == DType.INTEGER

    def test_var_with_default(self) -> None:
        v = Var(default=5.0)
        assert "default=5.0" in repr(v)
        assert v.default == 5.0

    def test_var_with_shape(self) -> None:
        v = Var(shape=(3, 3))
        assert "shape=(3, 3)" in repr(v)
        assert v.shape == (3, 3)

    def test_var_with_start_fixed(self) -> None:
        v = Var(start=1.0, fixed=True)
        assert "start=1.0" in repr(v)
        assert "fixed=True" in repr(v)
        assert v.start == 1.0
        assert v.fixed is True

    def test_var_with_min_max(self) -> None:
        v = Var(min=-10.0, max=10.0)
        assert "min=-10.0" in repr(v)
        assert "max=10.0" in repr(v)
        assert v.min == -10.0
        assert v.max == 10.0

    def test_var_flags(self) -> None:
        v_const = Var(constant=True)
        assert "constant=True" in repr(v_const)

        v_disc = Var(discrete=True)
        assert "discrete=True" in repr(v_disc)

        v_prot = Var(protected=True)
        assert "protected=True" in repr(v_prot)

        v_in = Var(input=True)
        assert "input=True" in repr(v_in)

        v_out = Var(output=True)
        assert "output=True" in repr(v_out)

        v_param = Var(parameter=True)
        assert "parameter=True" in repr(v_param)

    def test_var_get_initial_value(self) -> None:
        v1 = Var(start=5.0, default=1.0)
        assert v1.get_initial_value() == 5.0

        v2 = Var(default=1.0)
        assert v2.get_initial_value() == 1.0

        v3 = Var()
        assert v3.get_initial_value() is None

    def test_var_ndim(self) -> None:
        assert Var().ndim == 0
        assert Var(shape=(3,)).ndim == 1
        assert Var(shape=(2, 3)).ndim == 2

    def test_var_is_scalar(self) -> None:
        assert Var().is_scalar() is True
        assert Var(shape=(3,)).is_scalar() is False

    def test_var_size(self) -> None:
        assert Var().size == 1
        assert Var(shape=(3,)).size == 3
        assert Var(shape=(2, 3)).size == 6

    def test_var_unit_and_nominal(self) -> None:
        v = Var(unit="m/s", nominal=100.0)
        assert v.unit == "m/s"
        assert v.nominal == 100.0

    def test_var_desc(self) -> None:
        v = Var(desc="Test description")
        assert v.desc == "Test description"


class TestSubmodelField:
    """Minimal coverage of the SubmodelField helper."""

    def test_submodel_field_repr(self) -> None:
        @dataclass
        class DummyModel:
            pass

        field = SubmodelField(model_class=DummyModel)
        assert repr(field) == "submodel(DummyModel)"

        field_with_overrides = SubmodelField(model_class=DummyModel, overrides={"gain": 2})
        assert "gain=2" in repr(field_with_overrides)
