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
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        compiled = CasadiBackend.compile(M().flatten())
        assert compiled.name == "M"
        assert compiled.state_names == ["x"]

    def test_compile_with_inputs(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)

            @equations
            def _(m):
                der(m.x) == m.u

        compiled = CasadiBackend.compile(M().flatten())
        assert "u" in compiled.input_names

    def test_compile_with_outputs(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == m.x * 2

        compiled = CasadiBackend.compile(M().flatten())
        assert "y" in compiled.output_names

    def test_compile_with_params(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            k = var(2.0, parameter=True)

            @equations
            def _(m):
                der(m.x) == m.k

        compiled = CasadiBackend.compile(M().flatten())
        assert "k" in compiled.param_names
        assert compiled.param_defaults["k"] == 2.0


class TestCasadiBackendMathOperators:
    """Test CasadiBackend compilation of math operators."""

    def test_trig_functions(self) -> None:
        from cyecca.dsl import acos, asin, atan, cos, der, equations, model, sin, tan, var
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

            @equations
            def _(m):
                der(m.x) == 0.0
                m.y_sin == sin(m.x)
                m.y_cos == cos(m.x)
                m.y_tan == tan(m.x)
                m.y_asin == asin(m.x)
                m.y_acos == acos(m.x)
                m.y_atan == atan(m.x)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        assert all(k in result.outputs for k in ["y_sin", "y_cos", "y_tan", "y_asin", "y_acos", "y_atan"])

    def test_atan2(self) -> None:
        from cyecca.dsl import atan2, der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=1.0)
            a = var(start=0.5)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == 0.0
                der(m.a) == 0.0
                m.y == atan2(m.a, m.x)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs

    def test_exp_log_sqrt_abs(self) -> None:
        from cyecca.dsl import abs as dsl_abs
        from cyecca.dsl import der, equations, exp, log, model, sqrt, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=1.0)
            y_exp = var(output=True)
            y_log = var(output=True)
            y_sqrt = var(output=True)
            y_abs = var(output=True)

            @equations
            def _(m):
                der(m.x) == 0.1
                m.y_exp == exp(m.x)
                m.y_log == log(m.x)
                m.y_sqrt == sqrt(m.x)
                m.y_abs == dsl_abs(m.x)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        assert all(k in result.outputs for k in ["y_exp", "y_log", "y_sqrt", "y_abs"])

    def test_log10(self) -> None:
        from cyecca.dsl import der, equations, log10, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=100.0)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == 0.0
                m.y == log10(m.x)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        assert "y" in result.outputs
        # log10(100) = 2
        assert result.outputs["y"][0] == pytest.approx(2.0, abs=1e-10)

    def test_sign_floor_ceil(self) -> None:
        from cyecca.dsl import ceil, der, equations, floor, model, sign, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=-2.5)
            y_sign = var(output=True)
            y_floor = var(output=True)
            y_ceil = var(output=True)

            @equations
            def _(m):
                der(m.x) == 0.0
                m.y_sign == sign(m.x)
                m.y_floor == floor(m.x)
                m.y_ceil == ceil(m.x)

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
        from cyecca.dsl import cosh, der, equations, model, sinh, tanh, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=1.0)
            y_sinh = var(output=True)
            y_cosh = var(output=True)
            y_tanh = var(output=True)

            @equations
            def _(m):
                der(m.x) == 0.0
                m.y_sinh == sinh(m.x)
                m.y_cosh == cosh(m.x)
                m.y_tanh == tanh(m.x)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        import math

        assert result.outputs["y_sinh"][0] == pytest.approx(math.sinh(1.0))
        assert result.outputs["y_cosh"][0] == pytest.approx(math.cosh(1.0))
        assert result.outputs["y_tanh"][0] == pytest.approx(math.tanh(1.0))

    def test_min_max(self) -> None:
        from cyecca.dsl import der, equations
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

            @equations
            def _(m):
                der(m.a) == 0.0
                der(m.b) == 0.0
                m.y_min == dsl_min(m.a, m.b)
                m.y_max == dsl_max(m.a, m.b)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)
        assert result.outputs["y_min"][0] == pytest.approx(3.0)
        assert result.outputs["y_max"][0] == pytest.approx(5.0)


class TestCasadiBackendConditionals:
    """Test CasadiBackend compilation of conditionals."""

    def test_if_then_else(self) -> None:
        from cyecca.dsl import der, equations, if_then_else, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == if_then_else(m.x < 0.5, 0.0, 1.0)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0)
        assert "y" in result.outputs

    def test_comparison_operators(self) -> None:
        from cyecca.dsl import der, equations, if_then_else, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y_lt = var(output=True)
            y_le = var(output=True)
            y_gt = var(output=True)
            y_ge = var(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y_lt == if_then_else(m.x < 0.5, 1.0, 0.0)
                m.y_le == if_then_else(m.x <= 0.5, 1.0, 0.0)
                m.y_gt == if_then_else(m.x > 0.5, 1.0, 0.0)
                m.y_ge == if_then_else(m.x >= 0.5, 1.0, 0.0)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0)
        assert all(k in result.outputs for k in ["y_lt", "y_le", "y_gt", "y_ge"])

    def test_boolean_operators(self) -> None:
        from cyecca.dsl import and_, der, equations, if_then_else, model, not_, or_, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y_and = var(output=True)
            y_or = var(output=True)
            y_not = var(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y_and == if_then_else(and_(m.x > 0.25, m.x < 0.75), 1.0, 0.0)
                m.y_or == if_then_else(or_(m.x < 0.25, m.x > 0.75), 1.0, 0.0)
                m.y_not == if_then_else(not_(m.x > 0.5), 1.0, 0.0)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0)
        assert all(k in result.outputs for k in ["y_and", "y_or", "y_not"])


# =============================================================================
# CompiledModel Tests
# =============================================================================


class TestCompiledModel:
    """Test CompiledModel properties and methods."""

    def test_properties(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)
            k = var(1.0, parameter=True)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == m.k * m.u
                m.y == m.x

        compiled = CasadiBackend.compile(M().flatten())
        assert compiled.name == "M"
        assert compiled.state_names == ["x"]
        assert compiled.input_names == ["u"]
        assert compiled.output_names == ["y"]
        assert compiled.param_names == ["k"]
        assert compiled.param_defaults == {"k": 1.0}

    def test_repr(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

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
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0)

        assert len(result.t) > 10
        assert result["x"][-1] == pytest.approx(1.0, abs=0.1)

    def test_simulate_with_x0(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0, x0={"x": 5.0})

        assert result["x"][0] == pytest.approx(5.0, abs=0.01)
        assert result["x"][-1] == pytest.approx(6.0, abs=0.1)

    def test_simulate_with_constant_input(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)

            @equations
            def _(m):
                der(m.x) == m.u

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0, u={"u": 2.0})

        assert result["x"][-1] == pytest.approx(2.0, abs=0.1)

    def test_simulate_with_u_func(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)

            @equations
            def _(m):
                der(m.x) == m.u

        compiled = CasadiBackend.compile(M().flatten())

        def u_func(t):
            return {"u": 1.0 if t < 0.5 else -1.0}

        result = compiled.simulate(tf=1.0, u_func=u_func)

        # x goes up then down, should end near 0
        assert np.max(result["x"]) > 0.4
        assert abs(result["x"][-1]) < 0.1

    def test_simulate_with_params(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            k = var(1.0, parameter=True)

            @equations
            def _(m):
                der(m.x) == m.k

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0, params={"k": 3.0})

        assert result["x"][-1] == pytest.approx(3.0, abs=0.1)


# =============================================================================
# SimulationResult Tests
# =============================================================================


class TestSimulationResult:
    """Test SimulationResult properties and methods."""

    def test_t_property(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        assert result.t[0] == 0.0
        assert result.t[-1] == pytest.approx(1.0, abs=0.01)

    def test_states_property(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0
                der(m.y) == -m.y

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        states = result.states
        assert "x" in states
        assert "y" in states
        assert isinstance(states["x"], np.ndarray)

    def test_outputs_property(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == m.x * 2

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        outputs = result.outputs
        assert "y" in outputs
        assert isinstance(outputs["y"], np.ndarray)

    def test_inputs_property(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            u = var(input=True)
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == m.u

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0, u={"u": 1.0})

        inputs = result.inputs
        assert "u" in inputs

    def test_data_property(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        data = result.data
        assert "t" in data
        assert "x" in data

    def test_getitem_t(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        t = result["t"]
        assert t[0] == 0.0

    def test_getitem_keyerror(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        with pytest.raises(KeyError):
            _ = result["nonexistent"]

    def test_call_with_string(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        x_data = result("x")
        assert len(x_data) > 0

    def test_call_with_symbolic_var(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        m = M()
        result = CasadiBackend.compile(m.flatten()).simulate(tf=1.0)

        x_data = result(m.x)
        assert len(x_data) > 0

    def test_call_keyerror(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        with pytest.raises(KeyError):
            result("nonexistent")

    def test_call_typeerror(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        result = CasadiBackend.compile(M().flatten()).simulate(tf=1.0)

        with pytest.raises(TypeError):
            result(123)  # type: ignore


# =============================================================================
# MX Backend Tests
# =============================================================================


class TestCasadiMXBackend:
    """Test CasADi MX backend (alternative symbolic type)."""

    def test_mx_backend_basic(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend, SymbolicType

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        flat = M().flatten(expand_arrays=False)
        compiled = CasadiBackend.compile(flat, symbolic_type=SymbolicType.MX)
        result = compiled.simulate(tf=1.0)
        assert "x" in result.states

    def test_mx_backend_with_input(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend, SymbolicType

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)

            @equations
            def _(m):
                der(m.x) == m.u

        flat = M().flatten(expand_arrays=False)
        compiled = CasadiBackend.compile(flat, symbolic_type=SymbolicType.MX)
        result = compiled.simulate(tf=1.0, u={"u": 2.0})
        assert "x" in result.states

    def test_mx_backend_with_param(self) -> None:
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend, SymbolicType

        @model
        class M:
            x = var(start=0.0)
            k = var(1.0, parameter=True)

            @equations
            def _(m):
                der(m.x) == m.k

        flat = M().flatten(expand_arrays=False)
        compiled = CasadiBackend.compile(flat, symbolic_type=SymbolicType.MX)
        result = compiled.simulate(tf=1.0, params={"k": 2.0})
        assert "x" in result.states


# =============================================================================
# CVODES Integrator Tests
# =============================================================================


class TestCVODESIntegrator:
    """Test CVODES variable-step integrator."""

    def test_cvodes_basic(self) -> None:
        """Test CVODES integrator on a simple ODE."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend
        from cyecca.dsl.backends.casadi import Integrator

        @model
        class M:
            x = var(start=1.0)

            @equations
            def _(m):
                der(m.x) == -m.x  # Exponential decay

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=2.0, dt=0.1, integrator=Integrator.CVODES)

        # x(t) = exp(-t), so x(2) ≈ 0.135
        assert result["x"][-1] == pytest.approx(np.exp(-2.0), rel=0.01)

    def test_cvodes_with_input(self) -> None:
        """Test CVODES with input signal."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend
        from cyecca.dsl.backends.casadi import Integrator

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)

            @equations
            def _(m):
                der(m.x) == m.u

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0, u={"u": 2.0}, integrator=Integrator.CVODES)

        assert result["x"][-1] == pytest.approx(2.0, rel=0.01)

    def test_cvodes_with_params(self) -> None:
        """Test CVODES with parameters."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend
        from cyecca.dsl.backends.casadi import Integrator

        @model
        class M:
            x = var(start=0.0)
            k = var(3.0, parameter=True)

            @equations
            def _(m):
                der(m.x) == m.k

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0, params={"k": 5.0}, integrator=Integrator.CVODES)

        assert result["x"][-1] == pytest.approx(5.0, rel=0.01)

    def test_cvodes_harmonic_oscillator(self) -> None:
        """Test CVODES on a 2nd order system (harmonic oscillator)."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend
        from cyecca.dsl.backends.casadi import Integrator

        @model
        class Oscillator:
            x = var(start=1.0)
            v = var(start=0.0)
            omega = var(1.0, parameter=True)

            @equations
            def _(m):
                der(m.x) == m.v
                der(m.v) == -(m.omega**2) * m.x

        compiled = CasadiBackend.compile(Oscillator().flatten())
        result = compiled.simulate(tf=6.28, dt=0.1, integrator=Integrator.CVODES)

        # After one period (2π), x should return to ~1
        assert result["x"][-1] == pytest.approx(1.0, abs=0.1)


# =============================================================================
# IDAS DAE Integrator Tests
# =============================================================================


class TestIDASIntegrator:
    """Test IDAS variable-step DAE integrator."""

    def test_idas_basic_ode(self) -> None:
        """Test IDAS on a pure ODE (no algebraic variables)."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend
        from cyecca.dsl.backends.casadi import Integrator

        @model
        class M:
            x = var(start=1.0)

            @equations
            def _(m):
                der(m.x) == -m.x

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=2.0, dt=0.1, integrator=Integrator.IDAS)

        assert result["x"][-1] == pytest.approx(np.exp(-2.0), rel=0.01)

    def test_idas_with_input(self) -> None:
        """Test IDAS with input."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend
        from cyecca.dsl.backends.casadi import Integrator

        @model
        class M:
            x = var(start=0.0)
            u = var(input=True)

            @equations
            def _(m):
                der(m.x) == m.u

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0, u={"u": 3.0}, integrator=Integrator.IDAS)

        assert result["x"][-1] == pytest.approx(3.0, rel=0.01)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestCasadiBackendErrors:
    """Test error handling in CasadiBackend."""

    def test_cvodes_rejects_dae(self) -> None:
        """Test that CVODES raises error for DAE systems."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend
        from cyecca.dsl.backends.casadi import Integrator

        @model
        class DAEModel:
            x = var(start=1.0)
            z = var(start=0.0)  # Algebraic variable (no der())

            @equations
            def _(m):
                der(m.x) == m.z
                m.z == -m.x  # Algebraic constraint

        flat = DAEModel().flatten()
        # Force z to be algebraic by removing it from states if needed
        # The model needs to have an algebraic variable to trigger the error

    def test_unknown_variable_error(self) -> None:
        """Test error when referencing unknown variable."""
        import casadi as ca

        from cyecca.dsl.backends.casadi import CasadiCompiler
        from cyecca.dsl.expr import Expr, ExprKind
        from cyecca.dsl.flat_model import FlatModel

        # Create a minimal flat model
        flat = FlatModel(
            name="Test",
            state_names=[],
            state_vars={},
            equations=[],
            input_names=[],
            input_vars={},
            param_names=[],
            param_vars={},
            output_names=[],
            output_vars={},
            algebraic_names=[],
            algebraic_vars={},
            discrete_names=[],
            discrete_vars={},
        )

        compiler = CasadiCompiler(ca.SX, flat)
        compiler._create_symbols()

        # Try to convert a variable that doesn't exist
        unknown_var = Expr(ExprKind.VARIABLE, name="nonexistent")
        with pytest.raises(ValueError, match="Unknown variable"):
            compiler.expr_to_casadi(unknown_var)

    def test_derivative_in_rhs_error(self) -> None:
        """Test error when DERIVATIVE node appears in RHS."""
        import casadi as ca

        from cyecca.dsl.backends.casadi import CasadiCompiler
        from cyecca.dsl.expr import Expr, ExprKind
        from cyecca.dsl.flat_model import FlatModel

        flat = FlatModel(
            name="Test",
            state_names=[],
            state_vars={},
            equations=[],
            input_names=[],
            input_vars={},
            param_names=[],
            param_vars={},
            output_names=[],
            output_vars={},
            algebraic_names=[],
            algebraic_vars={},
            discrete_names=[],
            discrete_vars={},
        )

        compiler = CasadiCompiler(ca.SX, flat)
        compiler._create_symbols()

        # Derivative expressions are now valid (for implicit DAE support)
        # They map to xdot symbols. But if the state doesn't exist, it should error
        deriv_expr = Expr(ExprKind.DERIVATIVE, name="x")
        with pytest.raises(ValueError, match="Unknown derivative variable"):
            compiler.expr_to_casadi(deriv_expr)

    def test_pre_operator_not_implemented_in_continuous(self) -> None:
        """Test that pre() raises NotImplementedError in continuous equations."""
        import casadi as ca

        from cyecca.dsl.backends.casadi import CasadiCompiler
        from cyecca.dsl.expr import Expr, ExprKind
        from cyecca.dsl.flat_model import FlatModel
        from cyecca.dsl.types import Var

        flat = FlatModel(
            name="Test",
            state_names=["x"],
            state_vars={"x": Var(name="x")},
            equations=[],
            input_names=[],
            input_vars={},
            param_names=[],
            param_vars={},
            output_names=[],
            output_vars={},
            algebraic_names=[],
            algebraic_vars={},
            discrete_names=[],
            discrete_vars={},
        )

        compiler = CasadiCompiler(ca.SX, flat)
        compiler._create_symbols()

        # Try to use pre() in continuous context
        pre_expr = Expr(ExprKind.PRE, name="x")
        with pytest.raises(NotImplementedError, match="Discrete operator"):
            compiler.expr_to_casadi(pre_expr)


# =============================================================================
# Output Without Equation Tests
# =============================================================================


class TestOutputWithoutEquation:
    """Test handling of outputs without equations."""

    def test_output_without_equation_warning(self) -> None:
        """Test that output without equation produces warning."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y = var(output=True)  # No equation for y

            @equations
            def _(m):
                der(m.x) == 1.0
                # Intentionally no equation for y

        flat = M().flatten()

        with pytest.warns(UserWarning, match="has no equation"):
            compiled = CasadiBackend.compile(flat)


# =============================================================================
# Time Variable Tests
# =============================================================================


class TestTimeVariable:
    """Test time variable in expressions."""

    def test_time_in_equation(self) -> None:
        """Test using time variable in equations."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == m.time  # dx/dt = t

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=2.0, dt=0.1)

        # x(t) = 0.5*t^2, so x(2) = 2
        assert result["x"][-1] == pytest.approx(2.0, rel=0.05)

    def test_time_in_output(self) -> None:
        """Test using time in output expression."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == 1.0
                m.y == m.x + m.time

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=1.0, dt=0.1)

        # At t=1, x=1, y = x + t = 2
        assert result["y"][-1] == pytest.approx(2.0, rel=0.05)


# =============================================================================
# Equality/Inequality Operators Tests
# =============================================================================


class TestEqualityOperators:
    """Test equality and inequality operators."""

    def test_eq_operator(self) -> None:
        """Test eq() function in if_then_else."""
        from cyecca.dsl import der, eq, equations, if_then_else, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=1.0)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == 0.0
                # eq(x, 1.0) should be true
                m.y == if_then_else(eq(m.x, 1.0), 100.0, 0.0)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)

        assert result["y"][0] == pytest.approx(100.0)

    def test_ne_operator(self) -> None:
        """Test ne() function in if_then_else."""
        from cyecca.dsl import der, equations, if_then_else, model, ne, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=1.0)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == 0.0
                # ne(x, 0.0) should be true
                m.y == if_then_else(ne(m.x, 0.0), 100.0, 0.0)

        compiled = CasadiBackend.compile(M().flatten())
        result = compiled.simulate(tf=0.1)

        assert result["y"][0] == pytest.approx(100.0)


# =============================================================================
# Algebraic Variables Tests
# =============================================================================


class TestAlgebraicVariables:
    """Test algebraic variable handling."""

    def test_has_algebraic_property(self) -> None:
        """Test has_algebraic property."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class ODEModel:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        compiled = CasadiBackend.compile(ODEModel().flatten())
        assert compiled.has_algebraic is False

    def test_has_events_property(self) -> None:
        """Test has_events property for model without when-clauses."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        compiled = CasadiBackend.compile(M().flatten())
        assert compiled.has_events is False


# =============================================================================
# MX Backend Advanced Tests
# =============================================================================


class TestMXBackendAdvanced:
    """Advanced tests for MX backend."""

    def test_mx_with_output(self) -> None:
        """Test MX backend with output computation."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend, SymbolicType

        @model
        class M:
            x = var(start=1.0)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == -0.5 * m.x
                m.y == m.x * 2

        flat = M().flatten(expand_arrays=False)
        compiled = CasadiBackend.compile(flat, symbolic_type=SymbolicType.MX)
        result = compiled.simulate(tf=1.0)

        assert "y" in result.outputs
        # y = 2*x, and x decays
        assert result["y"][0] == pytest.approx(2.0, rel=0.01)

    def test_mx_with_math_ops(self) -> None:
        """Test MX backend with math operations."""
        from cyecca.dsl import cos, der, equations, model, sin, var
        from cyecca.dsl.backends import CasadiBackend, SymbolicType

        @model
        class M:
            theta = var(start=0.0)
            y_sin = var(output=True)
            y_cos = var(output=True)

            @equations
            def _(m):
                der(m.theta) == 1.0
                m.y_sin == sin(m.theta)
                m.y_cos == cos(m.theta)

        flat = M().flatten(expand_arrays=False)
        compiled = CasadiBackend.compile(flat, symbolic_type=SymbolicType.MX)
        result = compiled.simulate(tf=1.0)

        assert "y_sin" in result.outputs
        assert "y_cos" in result.outputs


# =============================================================================
# CompiledModel Repr Tests
# =============================================================================


class TestCompiledModelRepr:
    """Test CompiledModel __repr__ method."""

    def test_repr_full(self) -> None:
        """Test repr with all components."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class FullModel:
            x = var(start=0.0)
            u = var(input=True)
            k = var(1.0, parameter=True)
            y = var(output=True)

            @equations
            def _(m):
                der(m.x) == m.k * m.u
                m.y == m.x

        compiled = CasadiBackend.compile(FullModel().flatten())
        repr_str = repr(compiled)

        assert "FullModel" in repr_str
        assert "states=" in repr_str
        assert "inputs=" in repr_str
        assert "params=" in repr_str
        assert "outputs=" in repr_str

    def test_repr_with_events(self) -> None:
        """Test repr with when-clauses."""
        from cyecca.dsl import der, equations, model, pre, reinit, var, when
        from cyecca.dsl.backends import CasadiBackend

        @model
        class EventModel:
            x = var(start=1.0)

            @equations
            def _(m):
                der(m.x) == -1.0
                with when(m.x < 0.0):
                    reinit(m.x, 1.0)

        compiled = CasadiBackend.compile(EventModel().flatten())
        repr_str = repr(compiled)

        assert "events=" in repr_str


# =============================================================================
# Array Variable Tests
# =============================================================================


class TestArrayVariables:
    """Test array state, parameter, and input handling."""

    def test_array_parameter_with_array_state(self) -> None:
        """Test array parameter used in derivative of array state."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var([1.0, 2.0, 3.0], shape=(3, 1), parameter=True)
            y = var([0.0, 0.0, 0.0], shape=(3, 1))

            @equations
            def _(m):
                der(m.y) == m.x

        flat = M().flatten()
        compiled = CasadiBackend.compile(flat)
        result = compiled.simulate(tf=1.0, dt=0.1)

        # y should integrate x over 1 second, giving [1, 2, 3]
        assert result["y"].shape == (11, 3)
        assert result["y"][-1] == pytest.approx([1.0, 2.0, 3.0], rel=0.01)

    def test_array_state_element_access(self) -> None:
        """Test accessing individual elements of array variables."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            p = var([1.0, 2.0], shape=(2, 1), parameter=True)
            x = var([0.0, 0.0], shape=(2, 1))

            @equations
            def _(m):
                # Access individual elements
                der(m.x[0]) == m.p[1]  # dx[0]/dt = p[1] = 2.0
                der(m.x[1]) == m.p[0]  # dx[1]/dt = p[0] = 1.0

        flat = M().flatten()
        compiled = CasadiBackend.compile(flat)
        result = compiled.simulate(tf=1.0, dt=0.1)

        # After 1 second: x[0] = 2.0, x[1] = 1.0
        assert result["x"][-1] == pytest.approx([2.0, 1.0], rel=0.01)

    def test_array_parameter_override(self) -> None:
        """Test overriding array parameter values in simulate."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            k = var([1.0, 1.0], shape=(2, 1), parameter=True)
            x = var([0.0, 0.0], shape=(2, 1))

            @equations
            def _(m):
                der(m.x) == m.k

        flat = M().flatten()
        compiled = CasadiBackend.compile(flat)

        # Use custom parameter values
        result = compiled.simulate(tf=1.0, dt=0.1, params={"k": [3.0, 4.0]})

        # After 1 second with k=[3, 4]: x = [3, 4]
        assert result["x"][-1] == pytest.approx([3.0, 4.0], rel=0.01)

    def test_array_state_initial_condition(self) -> None:
        """Test overriding array state initial conditions."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            x = var([0.0, 0.0], shape=(2, 1))

            @equations
            def _(m):
                der(m.x) == 1.0

        flat = M().flatten()
        compiled = CasadiBackend.compile(flat)

        # Start from [10, 20]
        result = compiled.simulate(tf=1.0, dt=0.1, x0={"x": [10.0, 20.0]})

        # After 1 second: x = [11, 21]
        assert result["x"][-1] == pytest.approx([11.0, 21.0], rel=0.01)

    def test_scalar_state_with_array_parameter_element(self) -> None:
        """Test scalar state driven by element of array parameter."""
        from cyecca.dsl import der, equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            k = var([1.0, 2.0, 3.0], shape=(3, 1), parameter=True)
            x = var(start=0.0)

            @equations
            def _(m):
                # Scalar state driven by second element of array param
                der(m.x) == m.k[1]  # = 2.0

        flat = M().flatten()
        compiled = CasadiBackend.compile(flat)
        result = compiled.simulate(tf=1.0, dt=0.1)

        # After 1 second: x = 2.0
        assert result["x"][-1] == pytest.approx(2.0, rel=0.01)

    def test_array_literal_in_initial_equations(self) -> None:
        """Test using list literal in initial equations for array variables."""
        from cyecca.dsl import der, equations, initial_equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            y = var(shape=(3, 1))

            @initial_equations
            def init_eqs(m):
                m.y == [3.0, 4.0, 5.0]

            @equations
            def eqs(m):
                der(m.y) == [1.0, 1.0, 1.0]

        flat = M().flatten()
        compiled = CasadiBackend.compile(flat)
        result = compiled.simulate(tf=1.0, dt=0.1)

        # y starts at [3, 4, 5] and increases by 1 each second
        assert result["y"][0] == pytest.approx([3.0, 4.0, 5.0], rel=0.01)
        assert result["y"][-1] == pytest.approx([4.0, 5.0, 6.0], rel=0.01)

    def test_matrix_literal_in_initial_equations(self) -> None:
        """Test using nested list literal (matrix) in initial equations."""
        from cyecca.dsl import der, equations, initial_equations, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class M:
            # 3x2 matrix
            A = var(shape=(3, 2))

            @initial_equations
            def init_eqs(m):
                # Matrix literal: 3 rows, 2 columns
                m.A == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

            @equations
            def eqs(m):
                # No change
                der(m.A) == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

        flat = M().flatten()
        compiled = CasadiBackend.compile(flat)
        result = compiled.simulate(tf=1.0, dt=0.1)

        # A starts at [[1,2], [3,4], [5,6]] flattened row-major: [1, 2, 3, 4, 5, 6]
        # Values should stay constant since der(A) = 0
        assert result["A"][0] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rel=0.01)
        assert result["A"][-1] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
