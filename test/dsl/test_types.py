"""Tests for DSL-facing type helpers (constructors, enums)."""

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


class TestModelicaTypeConstructors:
    """Test Modelica-style type constructors: Real, Integer, Boolean, String."""

    def test_real_basic(self) -> None:
        """Test Real() creates a Var with dtype=REAL."""
        from cyecca.dsl import Real
        from cyecca.ir.types import DType

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
        from cyecca.ir.types import DType

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
        from cyecca.ir.types import DType

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
        from cyecca.backends import CasadiBackend
        from cyecca.dsl import Boolean, Integer, Real, der, equations, model

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
