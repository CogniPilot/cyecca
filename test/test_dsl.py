"""Tests for the Cyecca DSL.

NOTE: This file uses the new var() API which replaced the old state/param/input 
separate functions. The new API uses var() with keyword arguments to specify
the variable type:
- var() with der() usage -> state
- var(parameter=True) -> parameter  
- var(input=True) -> input
- var(output=True) -> output
"""

import numpy as np
import pytest


class TestDSLPendulum:
    """Test pendulum model using DSL."""

    def test_pendulum_definition(self) -> None:
        """Test that we can define a pendulum model."""
        from cyecca.dsl import der, model, sin, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Pendulum:
            """Simple pendulum model."""

            g = var(9.81, parameter=True)
            l = var(1.0, parameter=True)
            theta = var(start=0.0)
            omega = var(start=0.0)

            def equations(m):
                yield der(m.theta) == m.omega
                yield der(m.omega) == -m.g / m.l * sin(m.theta)

        # Flatten and compile the model
        pend = Pendulum()
        flat = pend.flatten()
        compiled = CasadiBackend.compile(flat)

        assert compiled.name == "Pendulum"
        assert compiled.state_names == ["theta", "omega"]
        assert compiled.param_names == ["g", "l"]
        assert compiled.param_defaults == {"g": 9.81, "l": 1.0}

    def test_pendulum_simulation(self) -> None:
        """Test pendulum simulation."""
        from cyecca.dsl import der, model, sin, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Pendulum:
            g = var(9.81, parameter=True)
            l = var(1.0, parameter=True)
            theta = var(start=0.0)
            omega = var(start=0.0)

            def equations(m):
                yield der(m.theta) == m.omega
                yield der(m.omega) == -m.g / m.l * sin(m.theta)

        compiled = CasadiBackend.compile(Pendulum().flatten())

        # Simulate from theta=0.5 rad
        result = compiled.simulate(tf=5.0, x0={"theta": 0.5, "omega": 0.0})

        # Check that we got reasonable results
        assert len(result.t) > 100
        assert "theta" in result._data
        assert "omega" in result._data

        # Pendulum should oscillate around zero
        assert np.max(result._data["theta"]) > 0.4
        assert np.min(result._data["theta"]) < -0.4


class TestDSLSubmodel:
    """Test submodel composition."""

    def test_submodel_composition(self) -> None:
        """Test hierarchical model composition."""
        from cyecca.dsl import der, model, submodel, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Autopilot:
            Kp = var(2.0, parameter=True)
            cmd = var(input=True)

        @model
        class Aircraft:
            ap = submodel(Autopilot)
            p = var(start=0.0)
            v = var(start=0.0)

            def equations(m):
                yield der(m.p) == m.v
                yield der(m.v) == m.ap.Kp * m.ap.cmd

        compiled = CasadiBackend.compile(Aircraft().flatten())

        assert "ap.cmd" in compiled.input_names
        assert "ap.Kp" in compiled.param_names
        assert compiled.param_defaults["ap.Kp"] == 2.0

    def test_submodel_simulation(self) -> None:
        """Test simulation with submodels."""
        from cyecca.dsl import der, model, submodel, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Controller:
            gain = var(1.0, parameter=True)
            cmd = var(input=True)

        @model
        class System:
            ctrl = submodel(Controller)
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == m.ctrl.gain * m.ctrl.cmd

        compiled = CasadiBackend.compile(System().flatten())

        # Constant input
        result = compiled.simulate(tf=1.0, u={"ctrl.cmd": 2.0}, params={"ctrl.gain": 3.0})

        # x(t) = 6*t for x(0)=0, so x(1) = 6
        assert abs(result._data["x"][-1] - 6.0) < 0.1


class TestDSLVectorStates:
    """Test vector-valued states."""

    def test_vector_state(self) -> None:
        """Test model with vector states."""
        from cyecca.dsl import model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Particle:
            g = var(9.81, parameter=True)
            pos = var(start=0.0, shape=(3,))
            vel = var(start=0.0, shape=(3,))

        compiled = CasadiBackend.compile(Particle().flatten())
        # Currently size>1 states are flattened - this is a limitation
        # For now just test it builds
        assert compiled.name == "Particle"


class TestDSLTimeVaryingInput:
    """Test time-varying inputs."""

    def test_time_varying_input(self) -> None:
        """Test with time-varying control function."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Integrator:
            u = var(input=True)
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == m.u

        compiled = CasadiBackend.compile(Integrator().flatten())

        def u_func(t):
            return {"u": 1.0 if t < 0.5 else -1.0}

        result = compiled.simulate(tf=1.0, u_func=u_func)

        # x goes up to 0.5, then back down to 0
        assert np.max(result._data["x"]) > 0.4
        assert abs(result._data["x"][-1]) < 0.1


class TestDSLFlatModel:
    """Test FlatModel representation."""

    def test_flat_model_expr_tree(self) -> None:
        """Test that equations are stored as expression trees."""
        from cyecca.dsl import ExprKind, der, model, sin, var

        @model
        class Pendulum:
            g = var(9.81, parameter=True)
            theta = var(start=0.0)
            omega = var(start=0.0)

            def equations(m):
                yield der(m.theta) == m.omega
                yield der(m.omega) == -m.g * sin(m.theta)

        flat = Pendulum().flatten()

        # Check derivative equations exist
        assert "theta" in flat.derivative_equations
        assert "omega" in flat.derivative_equations

        # Check expression tree structure
        theta_deriv = flat.derivative_equations["theta"]
        assert theta_deriv.kind == ExprKind.VARIABLE
        assert theta_deriv.name == "omega"

        omega_deriv = flat.derivative_equations["omega"]
        assert omega_deriv.kind == ExprKind.MUL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
