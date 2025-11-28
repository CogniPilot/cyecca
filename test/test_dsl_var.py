"""
Tests for the Cyecca DSL unified var() API.

================================================================================
PROTOTYPE MODE - API IS IN FLUX
================================================================================

This test file covers the new unified var() function that replaces the
separate state(), param(), input(), output(), algebraic() functions.

Variable classification is automatic:
- var(parameter=True) → PARAMETER
- var(input=True) → INPUT
- var(output=True) → OUTPUT
- var(constant=True) → CONSTANT
- If der(var) appears in equations → STATE
- Otherwise → ALGEBRAIC
"""

import numpy as np
import pytest


class TestVarDeclaration:
    """Test var() function with different flags and dtypes."""

    def test_var_default_is_real(self) -> None:
        """Test that var() defaults to DType.REAL."""
        from cyecca.dsl import model, var, der, DType

        @model
        class TestModel:
            x = var(0.0)

            def equations(m):
                yield der(m.x) == 1.0

        flat = TestModel().flatten()
        assert flat.state_vars["x"].dtype == DType.REAL

    def test_var_integer_dtype(self) -> None:
        """Test var() with DType.INTEGER."""
        from cyecca.dsl import model, var, der, DType

        @model
        class TestModel:
            mode = var(1, dtype=DType.INTEGER, parameter=True)
            x = var(0.0)

            def equations(m):
                yield der(m.x) == 1.0

        flat = TestModel().flatten()
        assert flat.param_vars["mode"].dtype == DType.INTEGER
        assert flat.param_defaults["mode"] == 1

    def test_var_boolean_dtype(self) -> None:
        """Test var() with DType.BOOLEAN."""
        from cyecca.dsl import model, var, der, DType

        @model
        class TestModel:
            enabled = var(True, dtype=DType.BOOLEAN, parameter=True)
            x = var(0.0)

            def equations(m):
                yield der(m.x) == 1.0

        flat = TestModel().flatten()
        assert flat.param_vars["enabled"].dtype == DType.BOOLEAN
        assert flat.param_defaults["enabled"] == True

    def test_var_string_dtype(self) -> None:
        """Test that STRING dtype is not yet supported (commented out in types.py)."""
        from cyecca.dsl import DType

        # STRING is not yet supported in the DSL
        assert not hasattr(DType, "STRING")


class TestAutomaticClassification:
    """Test automatic variable classification based on der() usage."""

    def test_state_detected_from_der(self) -> None:
        """Test that variables with der() become states."""
        from cyecca.dsl import model, var, der, VarKind

        @model
        class TestModel:
            x = var(0.0)
            v = var(0.0)

            def equations(m):
                yield der(m.x) == m.v
                yield der(m.v) == -9.81

        flat = TestModel().flatten()
        
        assert "x" in flat.state_names
        assert "v" in flat.state_names
        assert flat.state_vars["x"].kind == VarKind.STATE
        assert flat.state_vars["v"].kind == VarKind.STATE

    def test_algebraic_detected_without_der(self) -> None:
        """Test that variables without der() become algebraic."""
        from cyecca.dsl import model, var, der, VarKind

        @model
        class TestModel:
            x = var(0.0)
            y = var()  # No der(y), so algebraic

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == m.x * 2

        flat = TestModel().flatten()
        
        assert "x" in flat.state_names
        assert "y" in flat.algebraic_names
        assert flat.algebraic_vars["y"].kind == VarKind.ALGEBRAIC

    def test_parameter_flag_overrides_classification(self) -> None:
        """Test that parameter=True overrides automatic classification."""
        from cyecca.dsl import model, var, der, VarKind

        @model
        class TestModel:
            g = var(9.81, parameter=True)
            x = var(0.0)

            def equations(m):
                yield der(m.x) == m.g

        flat = TestModel().flatten()
        
        assert "g" in flat.param_names
        assert "x" in flat.state_names
        assert flat.param_vars["g"].kind == VarKind.PARAMETER

    def test_input_flag_classification(self) -> None:
        """Test that input=True creates an input variable."""
        from cyecca.dsl import model, var, der, VarKind

        @model
        class TestModel:
            u = var(input=True)
            x = var(0.0)

            def equations(m):
                yield der(m.x) == m.u

        flat = TestModel().flatten()
        
        assert "u" in flat.input_names
        assert flat.input_vars["u"].kind == VarKind.INPUT

    def test_output_flag_classification(self) -> None:
        """Test that output=True creates an output variable."""
        from cyecca.dsl import model, var, der, VarKind, sin

        @model
        class TestModel:
            x = var(0.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0
                yield m.y == sin(m.x)

        flat = TestModel().flatten()
        
        assert "y" in flat.output_names
        assert flat.output_vars["y"].kind == VarKind.OUTPUT


class TestVarAttributes:
    """Test Modelica-style variable attributes."""

    def test_var_min_max(self) -> None:
        """Test min/max bounds."""
        from cyecca.dsl import model, var, der

        @model
        class TestModel:
            x = var(0.0, min=-10, max=10)

            def equations(m):
                yield der(m.x) == 1.0

        flat = TestModel().flatten()
        assert flat.state_vars["x"].min == -10
        assert flat.state_vars["x"].max == 10

    def test_var_unit(self) -> None:
        """Test unit attribute."""
        from cyecca.dsl import model, var, der

        @model
        class TestModel:
            velocity = var(0.0, unit="m/s")

            def equations(m):
                yield der(m.velocity) == 9.81

        flat = TestModel().flatten()
        assert flat.state_vars["velocity"].unit == "m/s"

    def test_var_start_and_fixed(self) -> None:
        """Test start value and fixed attribute."""
        from cyecca.dsl import model, var, der

        @model
        class TestModel:
            x = var(start=1.5, fixed=True)

            def equations(m):
                yield der(m.x) == 1.0

        flat = TestModel().flatten()
        assert flat.state_vars["x"].start == 1.5
        assert flat.state_vars["x"].fixed == True
        assert flat.state_defaults["x"] == 1.5

    def test_var_nominal(self) -> None:
        """Test nominal value for scaling."""
        from cyecca.dsl import model, var, der

        @model
        class TestModel:
            x = var(0.0, nominal=100.0)

            def equations(m):
                yield der(m.x) == 1.0

        flat = TestModel().flatten()
        assert flat.state_vars["x"].nominal == 100.0

    def test_var_desc(self) -> None:
        """Test description attribute."""
        from cyecca.dsl import model, var, der

        @model
        class TestModel:
            theta = var(0.0, desc="Pendulum angle in radians")

            def equations(m):
                yield der(m.theta) == 1.0

        flat = TestModel().flatten()
        assert flat.state_vars["theta"].desc == "Pendulum angle in radians"


class TestPendulumModel:
    """Test complete pendulum model with unified var() API."""

    def test_pendulum_definition(self) -> None:
        """Test pendulum model definition."""
        from cyecca.dsl import model, var, der, sin
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Pendulum:
            """Simple pendulum model."""
            g = var(9.81, parameter=True, unit="m/s^2")
            l = var(1.0, parameter=True, unit="m", min=0.01)
            theta = var(start=0.5, fixed=True, unit="rad")
            omega = var(start=0.0, unit="rad/s")

            def equations(m):
                yield der(m.theta) == m.omega
                yield der(m.omega) == -m.g / m.l * sin(m.theta)

        pend = Pendulum()
        flat = pend.flatten()
        compiled = CasadiBackend.compile(flat)

        assert compiled.name == "Pendulum"
        assert compiled.state_names == ["theta", "omega"]
        assert compiled.param_names == ["g", "l"]
        assert compiled.param_defaults == {"g": 9.81, "l": 1.0}

    def test_pendulum_simulation(self) -> None:
        """Test pendulum simulation."""
        from cyecca.dsl import model, var, der, sin
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Pendulum:
            g = var(9.81, parameter=True)
            l = var(1.0, parameter=True)
            theta = var()
            omega = var()

            def equations(m):
                yield der(m.theta) == m.omega
                yield der(m.omega) == -m.g / m.l * sin(m.theta)

        compiled = CasadiBackend.compile(Pendulum().flatten())

        result = compiled.simulate(
            t0=0.0, tf=5.0, dt=0.01, x0={"theta": 0.5, "omega": 0.0}
        )

        assert len(result.t) > 100
        assert "theta" in result._data
        assert "omega" in result._data
        assert np.max(result._data["theta"]) > 0.4
        assert np.min(result._data["theta"]) < -0.4

    def test_pendulum_with_outputs(self) -> None:
        """Test pendulum with output variables for cartesian position."""
        from cyecca.dsl import model, var, der, sin, cos
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Pendulum:
            g = var(9.81, parameter=True)
            l = var(1.0, parameter=True)
            theta = var()
            omega = var()
            x = var(output=True)
            y = var(output=True)

            def equations(m):
                yield der(m.theta) == m.omega
                yield der(m.omega) == -m.g / m.l * sin(m.theta)
                yield m.x == m.l * sin(m.theta)
                yield m.y == -m.l * cos(m.theta)

        flat = Pendulum().flatten()
        
        assert "x" in flat.output_names
        assert "y" in flat.output_names
        assert "x" in flat.output_equations
        assert "y" in flat.output_equations


class TestSubmodelComposition:
    """Test hierarchical model composition with submodels."""

    def test_submodel_with_var(self) -> None:
        """Test submodel using var() API."""
        from cyecca.dsl import model, var, der, submodel
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Controller:
            Kp = var(2.0, parameter=True)
            cmd = var(input=True)

        @model
        class Plant:
            ctrl = submodel(Controller)
            x = var(0.0)
            v = var(0.0)

            def equations(m):
                yield der(m.x) == m.v
                yield der(m.v) == m.ctrl.Kp * m.ctrl.cmd

        compiled = CasadiBackend.compile(Plant().flatten())

        assert "ctrl.cmd" in compiled.input_names
        assert "ctrl.Kp" in compiled.param_names
        assert compiled.param_defaults["ctrl.Kp"] == 2.0

    def test_submodel_simulation(self) -> None:
        """Test simulation with submodels."""
        from cyecca.dsl import model, var, der, submodel
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Gain:
            k = var(1.0, parameter=True)
            u = var(input=True)

        @model
        class Integrator:
            gain = submodel(Gain)
            x = var(0.0)

            def equations(m):
                yield der(m.x) == m.gain.k * m.gain.u

        compiled = CasadiBackend.compile(Integrator().flatten())

        result = compiled.simulate(
            t0=0.0, tf=1.0, dt=0.01,
            u={"gain.u": 2.0},
            params={"gain.k": 3.0}
        )

        # x(t) = 6*t for x(0)=0, so x(1) = 6
        assert abs(result._data["x"][-1] - 6.0) < 0.1


class TestExpressionTree:
    """Test expression tree representation."""

    def test_expr_tree_structure(self) -> None:
        """Test that equations are stored as expression trees."""
        from cyecca.dsl import model, var, der, sin, ExprKind

        @model
        class TestModel:
            g = var(9.81, parameter=True)
            theta = var()
            omega = var()

            def equations(m):
                yield der(m.theta) == m.omega
                yield der(m.omega) == -m.g * sin(m.theta)

        flat = TestModel().flatten()

        assert "theta" in flat.derivative_equations
        assert "omega" in flat.derivative_equations

        theta_deriv = flat.derivative_equations["theta"]
        assert theta_deriv.kind == ExprKind.VARIABLE
        assert theta_deriv.name == "omega"

        omega_deriv = flat.derivative_equations["omega"]
        assert omega_deriv.kind == ExprKind.MUL


class TestTimeVaryingInput:
    """Test time-varying inputs."""

    def test_time_varying_input(self) -> None:
        """Test with time-varying control function."""
        from cyecca.dsl import model, var, der
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Integrator:
            u = var(input=True)
            x = var()

            def equations(m):
                yield der(m.x) == m.u

        compiled = CasadiBackend.compile(Integrator().flatten())

        def u_func(t):
            return {"u": 1.0 if t < 0.5 else -1.0}

        result = compiled.simulate(t0=0.0, tf=1.0, dt=0.01, u_func=u_func)

        assert np.max(result._data["x"]) > 0.4
        assert abs(result._data["x"][-1]) < 0.1


class TestVarKindEnum:
    """Test VarKind enum values."""

    def test_varkind_values(self) -> None:
        """Test that VarKind has expected values."""
        from cyecca.dsl import VarKind

        assert hasattr(VarKind, "CONSTANT")
        assert hasattr(VarKind, "PARAMETER")
        assert hasattr(VarKind, "INPUT")
        assert hasattr(VarKind, "OUTPUT")
        assert hasattr(VarKind, "STATE")
        assert hasattr(VarKind, "ALGEBRAIC")


class TestDTypeEnum:
    """Test DType enum values."""

    def test_dtype_values(self) -> None:
        """Test that DType has expected Modelica types."""
        from cyecca.dsl import DType

        assert hasattr(DType, "REAL")
        assert hasattr(DType, "INTEGER")
        assert hasattr(DType, "BOOLEAN")
        # STRING is not yet supported
        assert not hasattr(DType, "STRING")


class TestNDimensionalArrays:
    """Test N-dimensional array support (MLS Chapter 10)."""

    def test_scalar_variable(self) -> None:
        """Test scalar variable (shape=())."""
        from cyecca.dsl import model, var, der

        @model
        class Scalar:
            x = var()

            def equations(m):
                yield der(m.x) == 1.0

        s = Scalar()
        assert s.x._shape == ()
        assert s.x.is_scalar()
        assert s.x.ndim == 0
        assert s.x.size == 1

    def test_vector_variable(self) -> None:
        """Test 1D vector variable (shape=(3,))."""
        from cyecca.dsl import model, var, der

        @model
        class Vector:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == m.vel

        v = Vector()
        assert v.pos._shape == (3,)
        assert not v.pos.is_scalar()
        assert v.pos.ndim == 1
        assert v.pos.size == 3
        assert len(v.pos) == 3

    def test_matrix_variable(self) -> None:
        """Test 2D matrix variable (shape=(3,3))."""
        from cyecca.dsl import model, var, der

        @model
        class Matrix:
            R = var(shape=(3, 3))

            def equations(m):
                yield m.R[0, 0] == 1.0  # Identity diagonal

        mat = Matrix()
        assert mat.R._shape == (3, 3)
        assert not mat.R.is_scalar()
        assert mat.R.ndim == 2
        assert mat.R.size == 9

    def test_3d_tensor_variable(self) -> None:
        """Test 3D tensor variable (shape=(2,3,4))."""
        from cyecca.dsl import model, var

        @model
        class Tensor:
            T = var(shape=(2, 3, 4))

            def equations(m):
                yield m.T[0, 0, 0] == 0.0

        t = Tensor()
        assert t.T._shape == (2, 3, 4)
        assert t.T.ndim == 3
        assert t.T.size == 24

    def test_vector_indexing(self) -> None:
        """Test indexing 1D vector."""
        from cyecca.dsl import model, var

        @model
        class Vector:
            pos = var(shape=(3,))

            def equations(m):
                yield m.pos[0] == 0.0

        v = Vector()
        elem = v.pos[0]
        assert elem._indices == (0,)
        assert elem._remaining_shape == ()
        assert elem.is_scalar()
        assert elem._name == "pos[0]"

    def test_matrix_indexing(self) -> None:
        """Test indexing 2D matrix."""
        from cyecca.dsl import model, var

        @model
        class Matrix:
            R = var(shape=(3, 3))

            def equations(m):
                yield m.R[0, 0] == 0.0

        mat = Matrix()
        
        # Single index gives a row (remaining 1D)
        row = mat.R[0]
        assert row._indices == (0,)
        assert row._remaining_shape == (3,)
        assert not row.is_scalar()
        
        # Double index gives a scalar
        elem = mat.R[0, 1]
        assert elem._indices == (0, 1)
        assert elem._remaining_shape == ()
        assert elem.is_scalar()
        assert elem._name == "R[0,1]"
        
        # Sequential indexing also works
        elem2 = mat.R[0][1]
        assert elem2._indices == (0, 1)
        assert elem2.is_scalar()

    def test_vector_der_expands(self) -> None:
        """Test that der(vector) expands to scalar equations."""
        from cyecca.dsl import model, var, der

        @model
        class Vector:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == m.vel

        flat = Vector().flatten()
        
        # Should have 3 derivative equations
        assert len(flat.derivative_equations) == 3
        assert "pos[0]" in flat.derivative_equations
        assert "pos[1]" in flat.derivative_equations
        assert "pos[2]" in flat.derivative_equations

    def test_matrix_der_expands(self) -> None:
        """Test that der(matrix) expands to 9 scalar equations."""
        from cyecca.dsl import model, var, der

        @model
        class Matrix:
            R = var(shape=(3, 3))
            R_dot = var(shape=(3, 3))

            def equations(m):
                yield der(m.R) == m.R_dot

        flat = Matrix().flatten()
        
        # Should have 9 derivative equations
        assert len(flat.derivative_equations) == 9
        assert "R[0,0]" in flat.derivative_equations
        assert "R[1,1]" in flat.derivative_equations
        assert "R[2,2]" in flat.derivative_equations

    def test_partial_row_indexing(self) -> None:
        """Test der() on matrix row expands correctly."""
        from cyecca.dsl import model, var, der

        @model
        class PartialMatrix:
            R = var(shape=(3, 3))
            R_dot = var(shape=(3, 3))

            def equations(m):
                # Only first row derivative
                yield der(m.R[0]) == m.R_dot[0]

        flat = PartialMatrix().flatten()
        
        # Should have 3 derivative equations for first row
        assert len(flat.derivative_equations) == 3
        assert "R[0,0]" in flat.derivative_equations
        assert "R[0,1]" in flat.derivative_equations
        assert "R[0,2]" in flat.derivative_equations
        assert "R[1,0]" not in flat.derivative_equations

    def test_scalar_element_der(self) -> None:
        """Test der() on single matrix element."""
        from cyecca.dsl import model, var, der

        @model
        class SingleElement:
            R = var(shape=(3, 3))
            R_dot = var(shape=(3, 3))

            def equations(m):
                yield der(m.R[1, 1]) == m.R_dot[1, 1]

        flat = SingleElement().flatten()
        
        assert len(flat.derivative_equations) == 1
        assert "R[1,1]" in flat.derivative_equations

    def test_vector_iteration(self) -> None:
        """Test iterating over vector elements."""
        from cyecca.dsl import model, var

        @model
        class Vector:
            pos = var(shape=(3,))

            def equations(m):
                yield m.pos[0] == 0.0

        v = Vector()
        elements = list(v.pos)
        assert len(elements) == 3
        assert elements[0]._indices == (0,)
        assert elements[1]._indices == (1,)
        assert elements[2]._indices == (2,)

    def test_index_bounds_checking(self) -> None:
        """Test that out-of-bounds indexing raises IndexError."""
        from cyecca.dsl import model, var

        @model
        class Vector:
            pos = var(shape=(3,))

            def equations(m):
                yield m.pos[0] == 0.0

        v = Vector()
        with pytest.raises(IndexError):
            _ = v.pos[5]
        with pytest.raises(IndexError):
            _ = v.pos[-1]

    def test_scalar_not_indexable(self) -> None:
        """Test that scalar variables cannot be indexed."""
        from cyecca.dsl import model, var, der

        @model
        class Scalar:
            x = var()

            def equations(m):
                yield der(m.x) == 1.0

        s = Scalar()
        with pytest.raises(TypeError):
            _ = s.x[0]

    def test_vector_state_detected(self) -> None:
        """Test that vector with der() is classified as state."""
        from cyecca.dsl import model, var, der, VarKind

        @model
        class Vector:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == m.vel

        flat = Vector().flatten()
        assert "pos" in flat.state_names
        assert flat.state_vars["pos"].kind == VarKind.STATE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
