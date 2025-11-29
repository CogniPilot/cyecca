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
        from cyecca.dsl import Real, der, equations, model
        from cyecca.dsl.types import SubmodelField

        @model
        class Inner:
            x = Real()

            @equations
            def _(m):
                der(m.x) == 0

        field = SubmodelField(model_class=Inner)
        assert "submodel(" in repr(field)


class TestModelicaTypeConstructors:
    """Test Modelica-style type constructors: Real, Integer, Boolean, String."""

    def test_real_basic(self) -> None:
        """Test Real() creates a Var with dtype=REAL."""
        from cyecca.dsl import Real
        from cyecca.dsl.types import DType

        v = Real()
        assert v.dtype == DType.REAL

    def test_real_with_attributes(self) -> None:
        """Test Real() with common attributes."""
        from cyecca.dsl import Real

        v = Real(start=1.0, unit="m", parameter=True, desc="Position")
        assert v.start == 1.0
        assert v.unit == "m"
        assert v.parameter is True
        assert v.desc == "Position"

    def test_real_with_bounds(self) -> None:
        """Test Real() with min/max/nominal."""
        from cyecca.dsl import Real

        v = Real(min=0.0, max=100.0, nominal=50.0)
        assert v.min == 0.0
        assert v.max == 100.0
        assert v.nominal == 50.0

    def test_real_with_shape(self) -> None:
        """Test Real() with array shape."""
        from cyecca.dsl import Real

        v = Real(shape=(3, 3))
        assert v.shape == (3, 3)
        assert v.size == 9

    def test_integer_basic(self) -> None:
        """Test Integer() creates a Var with dtype=INTEGER."""
        from cyecca.dsl import Integer
        from cyecca.dsl.types import DType

        v = Integer()
        assert v.dtype == DType.INTEGER

    def test_integer_with_attributes(self) -> None:
        """Test Integer() with common attributes."""
        from cyecca.dsl import Integer

        v = Integer(5, parameter=True, desc="Count")
        assert v.default == 5
        assert v.parameter is True
        assert v.desc == "Count"
        # Integer should not have unit or nominal
        assert v.unit is None
        assert v.nominal is None

    def test_integer_with_bounds(self) -> None:
        """Test Integer() with min/max."""
        from cyecca.dsl import Integer

        v = Integer(min=0, max=100)
        assert v.min == 0
        assert v.max == 100

    def test_boolean_basic(self) -> None:
        """Test Boolean() creates a Var with dtype=BOOLEAN."""
        from cyecca.dsl import Boolean
        from cyecca.dsl.types import DType

        v = Boolean()
        assert v.dtype == DType.BOOLEAN

    def test_boolean_with_attributes(self) -> None:
        """Test Boolean() with common attributes."""
        from cyecca.dsl import Boolean

        v = Boolean(True, parameter=True, desc="Enable flag")
        assert v.default is True
        assert v.parameter is True
        assert v.desc == "Enable flag"
        # Boolean should not have min/max/unit/nominal
        assert v.min is None
        assert v.max is None
        assert v.unit is None
        assert v.nominal is None

    def test_boolean_discrete(self) -> None:
        """Test Boolean() as discrete variable."""
        from cyecca.dsl import Boolean

        v = Boolean(start=False, discrete=True)
        assert v.start is False
        assert v.discrete is True

    def test_string_basic(self) -> None:
        """Test String() constructor."""
        from cyecca.dsl import String

        v = String("default_name", parameter=True)
        assert v.default == "default_name"
        assert v.parameter is True
        # Strings are always discrete
        assert v.discrete is True

    def test_type_constructors_in_model(self) -> None:
        """Test type constructors work in model declarations."""
        from cyecca.dsl import Boolean, Integer, Real, der, equations, model
        from cyecca.dsl.backends import CasadiBackend

        @model
        class TestModel:
            # Real state
            x = Real(start=1.0, unit="m")
            v = Real(start=0.0, unit="m/s")

            # Real parameter
            k = Real(1.0, parameter=True)

            # Integer parameter
            n = Integer(5, parameter=True)

            # Boolean parameter
            enabled = Boolean(True, parameter=True)

            @equations
            def _(m):
                der(m.x) == m.v
                der(m.v) == -m.k * m.x

        m = TestModel()
        flat = m.flatten()

        assert "x" in flat.state_names
        assert "v" in flat.state_names
        assert "k" in flat.param_names
        assert "n" in flat.param_names
        assert "enabled" in flat.param_names

        # Should compile successfully
        compiled = CasadiBackend.compile(flat)
        assert compiled is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
