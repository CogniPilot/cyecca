"""
Tests for cyecca.dsl.types module.

Covers: Var, VarKind, DType, SubmodelField
"""

import pytest


class TestDType:
    """Test DType enum."""

    def test_dtype_values(self) -> None:
        """Test that DType has expected Modelica types."""
        from cyecca.dsl import DType

        assert hasattr(DType, "REAL")
        assert hasattr(DType, "INTEGER")
        assert hasattr(DType, "BOOLEAN")
        # STRING is not yet supported
        assert not hasattr(DType, "STRING")


class TestVarKind:
    """Test VarKind enum."""

    def test_varkind_values(self) -> None:
        """Test that VarKind has expected values."""
        from cyecca.dsl import VarKind

        assert hasattr(VarKind, "CONSTANT")
        assert hasattr(VarKind, "PARAMETER")
        assert hasattr(VarKind, "INPUT")
        assert hasattr(VarKind, "OUTPUT")
        assert hasattr(VarKind, "STATE")
        assert hasattr(VarKind, "ALGEBRAIC")


class TestVar:
    """Test Var declaration class."""

    def test_var_default(self) -> None:
        """Test var() defaults."""
        from cyecca.dsl.types import Var

        v = Var()
        assert repr(v) == "var()"

    def test_var_with_dtype(self) -> None:
        """Test var() with DType."""
        from cyecca.dsl.types import DType, Var

        v = Var(dtype=DType.INTEGER)
        assert "dtype=INTEGER" in repr(v)
        assert v.dtype == DType.INTEGER

    def test_var_with_default(self) -> None:
        """Test var() with default value."""
        from cyecca.dsl.types import Var

        v = Var(default=5.0)
        assert "default=5.0" in repr(v)
        assert v.default == 5.0

    def test_var_with_shape(self) -> None:
        """Test var() with shape."""
        from cyecca.dsl.types import Var

        v = Var(shape=(3, 3))
        assert "shape=(3, 3)" in repr(v)
        assert v.shape == (3, 3)

    def test_var_with_start_fixed(self) -> None:
        """Test var() with start and fixed."""
        from cyecca.dsl.types import Var

        v = Var(start=1.0, fixed=True)
        assert "start=1.0" in repr(v)
        assert "fixed=True" in repr(v)
        assert v.start == 1.0
        assert v.fixed is True

    def test_var_with_min_max(self) -> None:
        """Test var() with min/max bounds."""
        from cyecca.dsl.types import Var

        v = Var(min=-10.0, max=10.0)
        assert "min=-10.0" in repr(v)
        assert "max=10.0" in repr(v)
        assert v.min == -10.0
        assert v.max == 10.0

    def test_var_flags(self) -> None:
        """Test var() flag attributes."""
        from cyecca.dsl.types import Var

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
        """Test get_initial_value() method."""
        from cyecca.dsl.types import Var

        # start takes precedence over default
        v1 = Var(start=5.0, default=1.0)
        assert v1.get_initial_value() == 5.0

        # default used when no start
        v2 = Var(default=1.0)
        assert v2.get_initial_value() == 1.0

        # None when neither
        v3 = Var()
        assert v3.get_initial_value() is None

    def test_var_ndim(self) -> None:
        """Test ndim property."""
        from cyecca.dsl.types import Var

        assert Var().ndim == 0
        assert Var(shape=(3,)).ndim == 1
        assert Var(shape=(2, 3)).ndim == 2

    def test_var_is_scalar(self) -> None:
        """Test is_scalar() method."""
        from cyecca.dsl.types import Var

        assert Var().is_scalar() is True
        assert Var(shape=(3,)).is_scalar() is False

    def test_var_size(self) -> None:
        """Test size property."""
        from cyecca.dsl.types import Var

        assert Var().size == 1
        assert Var(shape=(3,)).size == 3
        assert Var(shape=(2, 3)).size == 6

    def test_var_unit_and_nominal(self) -> None:
        """Test unit and nominal attributes."""
        from cyecca.dsl.types import Var

        v = Var(unit="m/s", nominal=100.0)
        assert v.unit == "m/s"
        assert v.nominal == 100.0

    def test_var_desc(self) -> None:
        """Test description attribute."""
        from cyecca.dsl.types import Var

        v = Var(desc="Test description")
        assert v.desc == "Test description"


class TestSubmodelField:
    """Test SubmodelField class."""

    def test_submodel_field_repr(self) -> None:
        """Test SubmodelField __repr__."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.types import SubmodelField

        @model
        class Inner:
            x = var()

            def equations(m):
                yield der(m.x) == 0

        field = SubmodelField(model_class=Inner)
        assert "submodel(" in repr(field)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
