"""
Integration tests for cyecca.dsl.

Covers: Complete model workflows, real-world examples, complex scenarios
"""

import numpy as np
import pytest

# =============================================================================
# Pendulum Model Tests
# =============================================================================


class TestPendulumModel:
    """Test pendulum model as canonical DSL example."""

    def test_pendulum_definition(self) -> None:
        """Test pendulum model definition and flattening."""
        from cyecca.dsl import der, model, sin, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Pendulum:
            g = var(9.81, parameter=True, desc="Gravity [m/s^2]")
            L = var(1.0, parameter=True, desc="Length [m]")
            theta = var(start=0.5, desc="Angle [rad]")
            omega = var(start=0.0, desc="Angular velocity [rad/s]")

            def equations(m):
                yield der(m.theta) == m.omega
                yield der(m.omega) == -(m.g / m.L) * sin(m.theta)

        flat = Pendulum().flatten()
        assert flat.name == "Pendulum"
        assert "theta" in flat.state_names
        assert "omega" in flat.state_names
        assert "g" in flat.param_names
        assert "L" in flat.param_names

    def test_pendulum_simulation(self) -> None:
        """Test pendulum oscillation behavior."""
        from cyecca.dsl import der, model, sin, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Pendulum:
            g = var(9.81, parameter=True)
            L = var(1.0, parameter=True)
            theta = var(start=0.5)
            omega = var(start=0.0)

            def equations(m):
                yield der(m.theta) == m.omega
                yield der(m.omega) == -(m.g / m.L) * sin(m.theta)

        pend = Pendulum()
        compiled = CasadiBackend.compile(pend.flatten())
        result = compiled.simulate(tf=10.0)

        theta = result(pend.theta)
        assert np.max(theta) > 0.4
        assert np.min(theta) < -0.4

    def test_pendulum_with_outputs(self) -> None:
        """Test pendulum with cartesian output coordinates."""
        from cyecca.dsl import cos, der, model, sin, var

        @model
        class Pendulum:
            g = var(9.81, parameter=True)
            L = var(1.0, parameter=True)
            theta = var()
            omega = var()
            x = var(output=True)
            y = var(output=True)

            def equations(m):
                yield der(m.theta) == m.omega
                yield der(m.omega) == -m.g / m.L * sin(m.theta)
                yield m.x == m.L * sin(m.theta)
                yield m.y == -m.L * cos(m.theta)

        flat = Pendulum().flatten()
        assert "x" in flat.output_names
        assert "y" in flat.output_names


# =============================================================================
# Submodel Composition Tests
# =============================================================================


class TestSubmodelComposition:
    """Test hierarchical model composition."""

    def test_mass_spring_damper(self) -> None:
        """Test composed mass-spring-damper system."""
        from cyecca.dsl import der, model, submodel, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class MassSpring:
            m = var(1.0, parameter=True)
            k = var(10.0, parameter=True)
            x = var(start=0.0)
            v = var(start=0.0)
            F_ext = var(0.0, input=True)

            def equations(m):
                yield der(m.x) == m.v
                yield der(m.v) == (m.F_ext - m.k * m.x) / m.m

        @model
        class Damper:
            c = var(0.5, parameter=True)
            v = var(input=True)
            F = var(output=True)

            def equations(m):
                yield m.F == -m.c * m.v

        @model
        class MassSpringDamper:
            spring = submodel(MassSpring)
            damper = submodel(Damper)

            def equations(m):
                yield m.damper.v == m.spring.v
                yield m.spring.F_ext == m.damper.F

        msd = MassSpringDamper()
        flat = msd.flatten()

        assert "spring.x" in flat.state_names
        assert "spring.v" in flat.state_names
        assert "damper.c" in flat.param_names

        compiled = CasadiBackend.compile(flat)
        result = compiled.simulate(tf=10.0, x0={"spring.x": 1.0, "spring.v": 0.0})

        x = result(msd.spring.x)
        assert np.abs(x[-1]) < np.abs(x[0])  # Damping

    def test_double_pendulum(self) -> None:
        """Test coupled double pendulum."""
        from cyecca.dsl import der, model, sin, submodel, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class PendulumLink:
            L = var(1.0, parameter=True)
            m = var(1.0, parameter=True)
            theta = var(start=0.5)
            omega = var(start=0.0)
            tau_ext = var(0.0, input=True)
            g = var(9.81, parameter=True)

            def equations(m):
                yield der(m.theta) == m.omega
                I = m.m * m.L**2
                yield der(m.omega) == (m.tau_ext - m.m * m.g * m.L * sin(m.theta)) / I

        @model
        class DoublePendulum:
            link1 = submodel(PendulumLink)
            link2 = submodel(PendulumLink)
            coupling = var(5.0, parameter=True)

            def equations(m):
                yield m.link1.tau_ext == 0
                yield m.link2.tau_ext == -m.coupling * (m.link2.theta - m.link1.theta)

        dp = DoublePendulum()
        flat = dp.flatten()

        assert "link1.theta" in flat.state_names
        assert "link2.theta" in flat.state_names
        assert "coupling" in flat.param_names

        compiled = CasadiBackend.compile(flat)
        result = compiled.simulate(
            tf=10.0,
            x0={
                "link1.theta": 0.8,
                "link1.omega": 0.0,
                "link2.theta": 0.4,
                "link2.omega": 0.0,
            },
        )

        theta1 = result(dp.link1.theta)
        theta2 = result(dp.link2.theta)
        assert len(theta1) > 0
        assert len(theta2) > 0


# =============================================================================
# Conditional Logic Tests
# =============================================================================


class TestConditionalLogic:
    """Test conditional expressions in models."""

    def test_saturation(self) -> None:
        """Test saturated integrator."""
        from cyecca.dsl import der, if_then_else, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class SaturatedIntegrator:
            rate = var(1.0, parameter=True)
            limit = var(2.0, parameter=True)
            x = var(start=0.0)
            y = var(output=True)

            def equations(m):
                yield der(m.x) == m.rate
                yield m.y == if_then_else(m.x < -m.limit, -m.limit, if_then_else(m.x > m.limit, m.limit, m.x))

        sat = SaturatedIntegrator()
        compiled = CasadiBackend.compile(sat.flatten())
        result = compiled.simulate(tf=5.0)

        y = result(sat.y)
        assert np.all(y >= -2.01)
        assert np.all(y <= 2.01)

    def test_piecewise_function(self) -> None:
        """Test piecewise linear function."""
        from cyecca.dsl import der, if_then_else, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Piecewise:
            t_var = var(start=0.0)
            y = var(output=True)

            def equations(m):
                yield der(m.t_var) == 1.0
                yield m.y == if_then_else(
                    m.t_var < 1, 0.0, if_then_else(m.t_var < 3, m.t_var - 1, if_then_else(m.t_var < 5, 2.0, 0.0))
                )

        pw = Piecewise()
        compiled = CasadiBackend.compile(pw.flatten())
        result = compiled.simulate(tf=6.0)

        y = result(pw.y)
        assert y[0] == pytest.approx(0.0, abs=0.1)

    def test_boolean_alarm_logic(self) -> None:
        """Test boolean operators in alarm system."""
        from cyecca.dsl import and_, der, if_then_else, model, or_, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Alarm:
            temp = var(start=20.0)
            pressure = var(start=50.0)
            temp_alarm = var(output=True)
            any_alarm = var(output=True)

            def equations(m):
                yield der(m.temp) == 2.0
                yield der(m.pressure) == 5.0
                yield m.temp_alarm == if_then_else(or_(m.temp < 15, m.temp > 30), 1.0, 0.0)
                yield m.any_alarm == if_then_else(or_(m.temp > 30, m.pressure > 100), 1.0, 0.0)

        alarm = Alarm()
        compiled = CasadiBackend.compile(alarm.flatten())
        result = compiled.simulate(tf=10.0)

        any_alarm = result(alarm.any_alarm)
        assert np.any(any_alarm > 0.5)


# =============================================================================
# Algorithm Section Tests
# =============================================================================


class TestAlgorithmSections:
    """Test algorithm sections with local variables."""

    def test_algorithm_with_locals(self) -> None:
        """Test algorithm with local variables."""
        from cyecca.dsl import der, local, model, var

        @model
        class AlgModel:
            x = var(start=0.0)
            result = var(output=True)

            def equations(m):
                yield der(m.x) == 1.0

            def algorithm(m):
                temp1 = local("temp1")
                temp2 = local("temp2")
                yield temp1 @ (m.x * 2)
                yield temp2 @ (temp1 + 1)
                yield m.result @ (temp2 * temp2)

        flat = AlgModel().flatten()
        assert "temp1" in flat.algorithm_locals
        assert "temp2" in flat.algorithm_locals
        assert len(flat.algorithm_assignments) == 3

    def test_algorithm_basic_assignment(self) -> None:
        """Test basic @ assignment operator."""
        from cyecca.dsl import Assignment, model, var

        @model
        class AlgModel:
            u = var(input=True)
            y = var(output=True)

            def algorithm(m):
                yield m.y @ (m.u * 2)

        flat = AlgModel().flatten()
        assert len(flat.algorithm_assignments) == 1
        assert flat.algorithm_assignments[0].target == "y"


# =============================================================================
# Function and Block Tests
# =============================================================================


class TestFunctionsAndBlocks:
    """Test @function and @block decorators."""

    def test_saturation_function(self) -> None:
        """Test @function for saturation."""
        from cyecca.dsl import function, if_then_else, var

        @function
        class Saturate:
            x = var(input=True)
            lo = var(input=True)
            hi = var(input=True)
            y = var(output=True)

            def algorithm(f):
                yield f.y @ if_then_else(f.x < f.lo, f.lo, if_then_else(f.x > f.hi, f.hi, f.x))

        sat = Saturate()
        flat = sat.flatten()

        assert "x" in flat.input_names
        assert "lo" in flat.input_names
        assert "hi" in flat.input_names
        assert "y" in flat.output_names

    def test_pi_controller_block(self) -> None:
        """Test @block for PI controller."""
        from cyecca.dsl import block, der, var

        @block
        class PIController:
            Kp = var(1.0, parameter=True)
            Ki = var(0.5, parameter=True)
            error = var(input=True)
            control = var(output=True)
            integral = var(start=0.0, protected=True)

            def equations(m):
                yield der(m.integral) == m.error
                yield m.control == m.Kp * m.error + m.Ki * m.integral

        flat = PIController().flatten()
        assert "Kp" in flat.param_names
        assert "Ki" in flat.param_names
        assert "error" in flat.input_names
        assert "control" in flat.output_names
        assert "integral" in flat.state_names


# =============================================================================
# Array Variable Tests
# =============================================================================


class TestArrayVariables:
    """Test array (N-dimensional) variables."""

    def test_vector_state(self) -> None:
        """Test vector state variable."""
        from cyecca.dsl import der, model, var

        @model
        class Particle:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            def equations(m):
                yield der(m.pos) == m.vel

        flat = Particle().flatten()
        assert "pos[0]" in flat.derivative_equations
        assert "pos[1]" in flat.derivative_equations
        assert "pos[2]" in flat.derivative_equations

    def test_matrix_state(self) -> None:
        """Test matrix state variable."""
        from cyecca.dsl import der, model, var

        @model
        class Rotation:
            R = var(shape=(3, 3))
            R_dot = var(shape=(3, 3))

            def equations(m):
                yield der(m.R) == m.R_dot

        flat = Rotation().flatten()
        assert len(flat.derivative_equations) == 9
        assert "R[0,0]" in flat.derivative_equations
        assert "R[2,2]" in flat.derivative_equations

    def test_vector_indexing_in_equations(self) -> None:
        """Test using vector indexing in equations."""
        from cyecca.dsl import der, model, var

        @model
        class Indexed:
            pos = var(shape=(3,))
            vel = var(shape=(3,))

            def equations(m):
                yield der(m.pos)[0] == m.vel[0]
                yield der(m.pos)[1] == m.vel[1]
                yield der(m.pos)[2] == m.vel[2]

        flat = Indexed().flatten()
        assert "pos[0]" in flat.derivative_equations
        assert "pos[1]" in flat.derivative_equations
        assert "pos[2]" in flat.derivative_equations


# =============================================================================
# Time-Varying Input Tests
# =============================================================================


class TestTimeVaryingInputs:
    """Test time-varying inputs."""

    def test_u_func_simulation(self) -> None:
        """Test simulation with time-varying input function."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Integrator:
            u = var(input=True)
            x = var(start=0.0)

            def equations(m):
                yield der(m.x) == m.u

        integ = Integrator()
        compiled = CasadiBackend.compile(integ.flatten())

        def u_func(t):
            return {"u": 1.0 if t < 0.5 else -1.0}

        result = compiled.simulate(tf=1.0, u_func=u_func)

        x = result(integ.x)
        assert np.max(x) > 0.4
        assert abs(x[-1]) < 0.15


# =============================================================================
# Complete Workflow Tests
# =============================================================================


class TestCompleteWorkflows:
    """Test complete modeling workflows."""

    def test_controlled_system(self) -> None:
        """Test PI-controlled first-order system."""
        from cyecca.dsl import der, model, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class ControlledSystem:
            setpoint = var(1.0, parameter=True)
            tau = var(1.0, parameter=True)
            Kp = var(2.0, parameter=True)
            Ki = var(1.0, parameter=True)
            y = var(start=0.0)
            error = var(output=True)
            u = var(output=True)
            error_integral = var(start=0.0)

            def equations(m):
                yield m.error == m.setpoint - m.y
                yield der(m.error_integral) == m.error
                yield m.u == m.Kp * m.error + m.Ki * m.error_integral
                yield der(m.y) == (m.u - m.y) / m.tau

        sys = ControlledSystem()
        compiled = CasadiBackend.compile(sys.flatten())
        result = compiled.simulate(tf=10.0)

        y = result(sys.y)
        assert y[-1] > 0.9  # Close to setpoint

    def test_lissajous_trajectory(self) -> None:
        """Test trajectory generator producing Lissajous pattern."""
        from cyecca.dsl import cos, der, model, sin, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class Trajectory:
            time = var(start=0.0)
            period = var(5.0, parameter=True)
            amplitude = var(2.0, parameter=True)
            x = var(output=True)
            y = var(output=True)

            def equations(m):
                yield der(m.time) == 1.0
                omega = 2 * 3.14159 / m.period
                phase = omega * m.time
                yield m.x == m.amplitude * sin(phase)
                yield m.y == m.amplitude * sin(2 * phase)

        traj = Trajectory()
        compiled = CasadiBackend.compile(traj.flatten())
        result = compiled.simulate(tf=10.0)

        x = result(traj.x)
        y = result(traj.y)
        assert np.max(np.abs(x)) <= 2.1
        assert np.max(np.abs(y)) <= 2.1

    def test_submodel_with_conditional(self) -> None:
        """Test submodel containing conditional logic."""
        from cyecca.dsl import der, if_then_else, model, submodel, var
        from cyecca.dsl.backends import CasadiBackend

        @model
        class SaturatedComponent:
            x = var(start=0.0)
            y = var(output=True)
            limit = var(1.0, parameter=True)

            def equations(m):
                yield der(m.x) == 0.5
                yield m.y == if_then_else(m.x > m.limit, m.limit, if_then_else(m.x < -m.limit, -m.limit, m.x))

        @model
        class System:
            comp = submodel(SaturatedComponent)

            def equations(m):
                return
                yield

        sys = System()
        compiled = CasadiBackend.compile(sys.flatten())
        result = compiled.simulate(tf=5.0)

        y = result(sys.comp.y)
        assert np.all(y <= 1.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
