"""
Tests for cyecca.dsl.backends.casadi module.

Covers: CasadiBackend, CompiledModel, SimulationResult, SymbolicType
"""

import numpy as np
import pytest

# =============================================================================
# CasadiBackend Compile Tests
# =============================================================================


class TestCasadiBackendCompile:
    """Test CasadiBackend.compile()."""

    def test_compile_basic_model(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        compiled = CasadiBackend.compile(M().flatten())
        assert compiled.name == "M"
        assert compiled.state_names == ["x"]

    def test_compile_with_inputs(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)

            def equations(m):
                yield der(m.x) == m.u

        compiled = CasadiBackend.compile(M().flatten())
        assert "u" in compiled.input_names

    def test_compile_with_outputs(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == m.x * 2

        compiled = CasadiBackend.compile(M().flatten())
        assert "y" in compiled.output_names

    def test_compile_with_params(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            k = var(2.0, parameter=True)

            def equations(m):
                yield der(m.x) == m.k

        compiled = CasadiBackend.compile(M().flatten())
        assert "k" in compiled.param_names
        assert compiled.param_defaults["k"] == 2.0


class TestCasadiBackendMathOperators:
    """Test CasadiBackend compilation of math operators."""

    def test_trig_functions(self) -> None:
        from cyecca.dsl import acos, asin, atan, cos, der, model, sin, tan, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.5)
            y_sin = var(output=True)
            y_cos = var(output=True)
            y_tan = var(output=True)
            y_asin = var(output=True)
            y_acos = var(output=True)
            y_atan = var(output=True)

            def equations(m):
                yield der(m.x) == 0.0
                yield m.y_sin == sin(m.x)
                yield m.y_cos == cos(m.x)
                yield m.y_tan == tan(m.x)
                yield m.y_asin == asin(m.x)
                yield m.y_acos == acos(m.x)
                yield m.y_atan == atan(m.x)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        assert all(k in result.outputs for k in ["y_sin", "y_cos", "y_tan", "y_asin", "y_acos", "y_atan"])

    def test_atan2(self) -> None:
        from cyecca.dsl import atan2, der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=1.0)
            a = var(start=0.5)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 0.0
                yield der(m.a) == 0.0
                yield m.y == atan2(m.a, m.x)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs

    def test_exp_log_sqrt_abs(self) -> None:
        from cyecca.dsl import abs as dsl_abs
        from cyecca.dsl import der, exp, log, model, sqrt, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=1.0)
            y_exp = var(output=True)
            y_log = var(output=True)
            y_sqrt = var(output=True)
            y_abs = var(output=True)

            def equations(m):
                yield der(m.x) == 0.1
                yield m.y_exp == exp(m.x)
                yield m.y_log == log(m.x)
                yield m.y_sqrt == sqrt(m.x)
                yield m.y_abs == dsl_abs(m.x)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        assert all(k in result.outputs for k in ["y_exp", "y_log", "y_sqrt", "y_abs"])

    def test_log10(self) -> None:
        from cyecca.dsl import der, log10, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=100.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 0.0
                yield m.y == log10(m.x)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs
        # log10(100) = 2
        assert result.outputs["y"][0] == pytest.approx(2.0, abs=1e-10)

    def test_sign_floor_ceil(self) -> None:
        from cyecca.dsl import ceil, der, floor, model, sign, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=-2.5)
            y_sign = var(output=True)
            y_floor = var(output=True)
            y_ceil = var(output=True)

            def equations(m):
                yield der(m.x) == 0.0
                yield m.y_sign == sign(m.x)
                yield m.y_floor == floor(m.x)
                yield m.y_ceil == ceil(m.x)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y_sign" in result.outputs
        assert "y_floor" in result.outputs
        assert "y_ceil" in result.outputs
        # sign(-2.5) = -1, floor(-2.5) = -3, ceil(-2.5) = -2
        assert result.outputs["y_sign"][0] == pytest.approx(-1.0)
        assert result.outputs["y_floor"][0] == pytest.approx(-3.0)
        assert result.outputs["y_ceil"][0] == pytest.approx(-2.0)

    def test_hyperbolic_functions(self) -> None:
        from cyecca.dsl import cosh, der, model, sinh, tanh, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=1.0)
            y_sinh = var(output=True)
            y_cosh = var(output=True)
            y_tanh = var(output=True)

            def equations(m):
                yield der(m.x) == 0.0
                yield m.y_sinh == sinh(m.x)
                yield m.y_cosh == cosh(m.x)
                yield m.y_tanh == tanh(m.x)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        import math

        assert result.outputs["y_sinh"][0] == pytest.approx(math.sinh(1.0))
        assert result.outputs["y_cosh"][0] == pytest.approx(math.cosh(1.0))
        assert result.outputs["y_tanh"][0] == pytest.approx(math.tanh(1.0))

    def test_min_max(self) -> None:
        from cyecca.dsl import der
        from cyecca.dsl import max as dsl_max
        from cyecca.dsl import min as dsl_min
        from cyecca.dsl import model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            a = var(start=3.0)
            b = var(start=5.0)
            y_min = var(output=True)
            y_max = var(output=True)

            def equations(m):
                yield der(m.a) == 0.0
                yield der(m.b) == 0.0
                yield m.y_min == dsl_min(m.a, m.b)
                yield m.y_max == dsl_max(m.a, m.b)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        assert result.outputs["y_min"][0] == pytest.approx(3.0)
        assert result.outputs["y_max"][0] == pytest.approx(5.0)


class TestCasadiBackendConditionals:
    """Test CasadiBackend compilation of conditionals."""

    def test_if_then_else(self) -> None:
        from cyecca.dsl import der, if_then_else, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == if_then_else(m.x < 0.5, 0.0, 1.0)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0)
        assert "y" in result.outputs

    def test_comparison_operators(self) -> None:
        from cyecca.dsl import der, if_then_else, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y_lt = var(output=True)
            y_le = var(output=True)
            y_gt = var(output=True)
            y_ge = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y_lt == if_then_else(m.x < 0.5, 1.0, 0.0)
                yield m.y_le == if_then_else(m.x <= 0.5, 1.0, 0.0)
                yield m.y_gt == if_then_else(m.x > 0.5, 1.0, 0.0)
                yield m.y_ge == if_then_else(m.x >= 0.5, 1.0, 0.0)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0)
        assert all(k in result.outputs for k in ["y_lt", "y_le", "y_gt", "y_ge"])

    def test_boolean_operators(self) -> None:
        from cyecca.dsl import and_, der, if_then_else, model, not_, or_, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y_and = var(output=True)
            y_or = var(output=True)
            y_not = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y_and == if_then_else(and_(m.x > 0.25, m.x < 0.75), 1.0, 0.0)
                yield m.y_or == if_then_else(or_(m.x < 0.25, m.x > 0.75), 1.0, 0.0)
                yield m.y_not == if_then_else(not_(m.x > 0.5), 1.0, 0.0)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0)
        assert all(k in result.outputs for k in ["y_and", "y_or", "y_not"])


# =============================================================================
# CompiledModel Tests
# =============================================================================


class TestCompiledModel:
    """Test CompiledModel properties and methods."""

    def test_properties(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)
            k = var(1.0, parameter=True)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == m.k * m.u
                yield m.y == m.x

        compiled = CasadiBackend.compile(M().flatten())
        assert compiled.name == "M"
        assert compiled.state_names == ["x"]
        assert compiled.input_names == ["u"]
        assert compiled.output_names == ["y"]
        assert compiled.param_names == ["k"]
        assert compiled.param_defaults == {"k": 1.0}

    def test_repr(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        compiled = CasadiBackend.compile(M().flatten())
        repr_str = repr(compiled)
        assert "CompiledModel(" in repr_str
        assert "states=" in repr_str


# =============================================================================
# Simulation Tests
# =============================================================================


class TestSimulate:
    """Test CompiledModel.simulate()."""

    def test_basic_simulation(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0)

        assert len(result.t) > 10
        assert result["x"][-1] == pytest.approx(1.0, abs=0.1)

    def test_simulate_with_x0(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0, x0={"x": 5.0})

        assert result["x"][0] == pytest.approx(5.0, abs=0.01)
        assert result["x"][-1] == pytest.approx(6.0, abs=0.1)

    def test_simulate_with_constant_input(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)

            def equations(m):
                yield der(m.x) == m.u

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0, u={"u": 2.0})

        assert result["x"][-1] == pytest.approx(2.0, abs=0.1)

    def test_simulate_with_u_func(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)

            def equations(m):
                yield der(m.x) == m.u

        compiled = CasadiBackend.compile(M().flatten())

        def u_func(t):
            return {"u": 1.0 if t < 0.5 else -1.0}

        result = compiled.simulate(tf=1.0, u_func=u_func)

        # x goes up then down, should end near 0
        assert np.max(result["x"]) > 0.4
        assert abs(result["x"][-1]) < 0.1

    def test_simulate_with_params(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            k = var(1.0, parameter=True)

            def equations(m):
                yield der(m.x) == m.k

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0, params={"k": 3.0})

        assert result["x"][-1] == pytest.approx(3.0, abs=0.1)


# =============================================================================
# SimulationResult Tests
# =============================================================================


class TestSimulationResult:
    """Test SimulationResult properties and methods."""

    def test_t_property(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        assert result.t[0] == 0.0
        assert result.t[-1] == pytest.approx(1.0, abs=0.01)

    def test_states_property(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0
                yield der(m.y) == -m.y

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        states = result.states
        assert "x" in states
        assert "y" in states
        assert isinstance(states["x"], np.ndarray)

    def test_outputs_property(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == m.x * 2

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        outputs = result.outputs
        assert "y" in outputs
        assert isinstance(outputs["y"], np.ndarray)

    def test_inputs_property(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            u = var(input=True)
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == m.u

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0, u={"u": 1.0})

        inputs = result.inputs
        assert "u" in inputs

    def test_data_property(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        data = result.data
        assert "t" in data
        assert "x" in data

    def test_getitem_t(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        t = result["t"]
        assert t[0] == 0.0

    def test_getitem_keyerror(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        with pytest.raises(KeyError):
            _ = result["nonexistent"]

    def test_call_with_string(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        x_data = result("x")
        assert len(x_data) > 0

    def test_call_with_symbolic_var(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        m = M()
        result = CasadiBackend.compile(m.flatten()).simulate(tf=1.0)

        x_data = result(m.x)
        assert len(x_data) > 0

    def test_call_keyerror(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        with pytest.raises(KeyError):
            result("nonexistent")

    def test_call_typeerror(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        with pytest.raises(TypeError):
            result(123)  # type: ignore


# =============================================================================
# MX Backend Tests
# =============================================================================


class TestCasadiMXBackend:
    """Test CasADi MX backend (alternative symbolic type)."""

    def test_mx_backend_basic(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend, SymbolicType

        @model
        class M:
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == 1.0

        flat = M().flatten(expand_arrays=False)
        compiled = CasadiBackend.compile(flat, symbolic_type=SymbolicType.MX)
        result = compiled.simulate(tf=1.0)
        assert "x" in result.states

    def test_mx_backend_with_input(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend, SymbolicType

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)

            def equations(m):
                yield der(m.x) == m.u

        flat = M().flatten(expand_arrays=False)
        compiled = CasadiBackend.compile(flat, symbolic_type=SymbolicType.MX)
        result = compiled.simulate(tf=1.0, u={"u": 2.0})
        assert "x" in result.states

    def test_mx_backend_with_param(self) -> None:
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend, SymbolicType

        @model
        class M:
            x = var(start=0.0)
            k = var(1.0, parameter=True)

            def equations(m):
                yield der(m.x) == m.k

        flat = M().flatten(expand_arrays=False)
        compiled = CasadiBackend.compile(flat, symbolic_type=SymbolicType.MX)
        result = compiled.simulate(tf=1.0, params={"k": 2.0})
        assert "x" in result.states


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
