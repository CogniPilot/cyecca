"""Tests for the ModelSX and ModelMX API with @symbolic dataclass pattern."""

import casadi as ca
import numpy as np
import pytest
from cyecca.model import ModelMX, ModelSX, input_var, output_var, param, state, symbolic
from cyecca.model.composition import SubmodelProxy


class TestModelCreate:
    """Test that ModelSX.create() and ModelMX.create() work correctly."""

    def test_sx_model_basic(self):
        """Test ModelSX.create() sets all expected attributes."""

        @symbolic
        class States:
            x: ca.SX = state(1, 0.0)

        @symbolic
        class Inputs:
            pass

        @symbolic
        class Params:
            m: ca.SX = param(1.0)

        model = ModelSX.create(States, Inputs, Params)
        model.build(f_x=ca.SX.zeros(1))

        # Check types are created
        assert model.state_type == States
        assert model.input_type == Inputs
        assert model.param_type == Params

        # Check symbolic instances
        assert hasattr(model, "x")
        assert hasattr(model, "p")
        assert hasattr(model, "u")

        # Check default instances
        assert hasattr(model, "x0")
        assert hasattr(model, "p0")
        assert hasattr(model, "u0")

        # Verify types
        assert model.x.x.__class__.__name__ == "SX"
        assert model.x0.x == 0.0
        assert model.p0.m == 1.0

    def test_mx_model_basic(self):
        """Test ModelMX.create() sets all expected attributes."""

        @symbolic
        class States:
            x: ca.MX = state(1, 0.0)

        @symbolic
        class Inputs:
            pass

        @symbolic
        class Params:
            m: ca.MX = param(1.0)

        model = ModelMX.create(States, Inputs, Params)
        model.build(f_x=ca.MX.zeros(1))

        assert model.state_type == States
        assert hasattr(model, "x")
        assert hasattr(model, "x0")
        assert model.x.x.__class__.__name__ == "MX"
        assert model.x0.x == 0.0
        assert model.p0.m == 1.0

    def test_vector_states(self):
        """Test with vector-valued states."""

        @symbolic
        class States:
            position: ca.SX = state(3, [1.0, 2.0, 3.0])

        @symbolic
        class Inputs:
            pass

        @symbolic
        class Params:
            k: ca.SX = param(2.5)

        model = ModelSX.create(States, Inputs, Params)
        model.build(f_x=ca.SX.zeros(3))

        # Check vector field
        assert hasattr(model.x0, "position")
        np.testing.assert_array_equal(model.x0.position, [1.0, 2.0, 3.0])

        # Check symbolic vector
        assert model.x.position.shape == (3, 1)

    def test_with_inputs(self):
        """Test with input variables."""

        @symbolic
        class States:
            x: ca.SX = state(1, 0.0)

        @symbolic
        class Inputs:
            u: ca.SX = input_var(default=0.5)

        @symbolic
        class Params:
            pass

        model = ModelSX.create(States, Inputs, Params)
        model.build(f_x=ca.SX.zeros(1))

        assert hasattr(model, "u")
        assert hasattr(model, "u0")
        assert model.u0.u == 0.5

    def test_with_outputs(self):
        """Test with output variables."""

        @symbolic
        class States:
            x: ca.SX = state(1, 0.0)

        @symbolic
        class Inputs:
            pass

        @symbolic
        class Params:
            pass

        @symbolic
        class Outputs:
            energy: ca.SX = output_var(1, 0.0, "energy calculation")

        model = ModelSX.create(States, Inputs, Params, output_type=Outputs)
        model.build(f_x=ca.SX.zeros(1), f_y=model.x.x)

        assert hasattr(model, "output_type")
        assert model.output_type == Outputs
        assert "energy" in model.output_names


class TestSimulation:
    """Test model simulation capabilities."""

    def test_simple_integration(self):
        """Test basic forward integration."""

        @symbolic
        class States:
            x: ca.SX = state(1, 1.0)

        @symbolic
        class Inputs:
            u: ca.SX = input_var(1, 0.0)

        @symbolic
        class Params:
            k: ca.SX = param(1.0, "decay rate")

        model = ModelSX.create(States, Inputs, Params)
        # dx/dt = -k*x (exponential decay)
        model.build(f_x=-model.p.k * model.x.x)

        result = model.simulate(
            t0=0.0,
            tf=1.0,
            dt=0.1,
            u_func=lambda t, x, p: model.u0.as_vec(),
            p_vec=model.p0.as_vec(),
        )

        assert "t" in result
        assert "x" in result
        assert len(result["t"]) > 0

        # result["x"] has shape (n_states, n_timesteps), so transpose
        x_traj = result["x"].T  # Now shape is (n_timesteps, n_states)

        # Final value should be less than initial (decay)
        # For dx/dt = -k*x with x(0) = 1, solution is x(t) = e^(-t)
        # At t=1, x ≈ 0.368
        final_val = float(x_traj[-1, 0])
        assert final_val < 0.5, f"Expected decay but got {final_val}"


class TestHybridFeatures:
    """Test hybrid/discrete state features."""

    def test_simple_hybrid_model(self):
        """Test a simple hybrid model with discrete states."""

        @symbolic
        class States:
            x: ca.SX = state(1, 0.0, "continuous state")

        @symbolic
        class Inputs:
            pass

        @symbolic
        class Params:
            pass

        # For now, skip complex hybrid tests as they require more setup
        model = ModelSX.create(States, Inputs, Params)
        model.build(f_x=ca.SX.zeros(1))

        assert hasattr(model, "x")
        assert hasattr(model, "x0")


class TestModelComposition:
    """Test hierarchical model composition features."""

    def test_compose_basic(self):
        """Test basic model composition with two simple subsystems."""

        # Create first subsystem: integrator (dx/dt = u)
        @symbolic
        class IntegratorStates:
            x: ca.SX = state(1, 0.0, "position")

        @symbolic
        class IntegratorInputs:
            u: ca.SX = input_var(desc="velocity command")

        @symbolic
        class IntegratorParams:
            pass

        integrator = ModelSX.create(
            IntegratorStates, IntegratorInputs, IntegratorParams
        )
        x_int = integrator.x
        u_int = integrator.u
        integrator.build(f_x=u_int.u)

        # Create second subsystem: proportional controller (u = -k*x)
        @symbolic
        class ControllerStates:
            pass

        @symbolic
        class ControllerInputs:
            x_meas: ca.SX = input_var(desc="measured position")

        @symbolic
        class ControllerParams:
            k: ca.SX = param(2.0, "proportional gain")

        @symbolic
        class ControllerOutputs:
            u_cmd: ca.SX = output_var(desc="velocity command")

        controller = ModelSX.create(
            ControllerStates, ControllerInputs, ControllerParams, ControllerOutputs
        )
        u_ctrl = controller.u
        p_ctrl = controller.p
        controller.build(f_x=ca.SX.zeros(0), f_y=-p_ctrl.k * u_ctrl.x_meas)

        # Compose the two subsystems
        parent = ModelSX.compose({"integrator": integrator, "controller": controller})

        # Verify submodels are accessible
        assert hasattr(parent, "integrator")
        assert hasattr(parent, "controller")

        # Connect using string paths
        parent.connect("integrator.u.u", "controller.y.u_cmd")
        parent.connect("controller.u.x_meas", "integrator.x.x")

        # Build composed model
        parent.build_composed(integrator="rk4")

        # Verify composed model has combined states
        assert hasattr(parent, "x0_composed")
        assert parent.x0_composed.shape[0] == 1  # Only integrator has state

        # Verify can step the composed model
        x0 = parent.x0_composed
        x0[0] = 1.0  # Start with non-zero position

        result = parent.f_step(x=x0, u=parent.u0.as_vec(), p=parent.p0.as_vec(), dt=0.1)

        assert "x_next" in result
        # With k=2 and x0=1, u_cmd = -2, so dx/dt = -2
        # After dt=0.1, x ≈ 1 - 0.2 = 0.8
        x_next = float(result["x_next"][0])
        assert x_next < 1.0, "Position should decrease with negative feedback"
        assert x_next > 0.5, f"Expected x ≈ 0.8, got {x_next}"

    def test_compose_states_merge(self):
        """Test that compose_states properly merges multiple state types."""
        from cyecca.model import compose_states

        @symbolic
        class States1:
            x: ca.SX = state(1, 0.0, "state x")
            y: ca.SX = state(1, 0.0, "state y")

        @symbolic
        class States2:
            z: ca.SX = state(1, 0.0, "state z")

        # Compose the states
        Combined = compose_states(States1, States2)

        # Verify combined type has all fields
        combined_instance = Combined.numeric()
        assert hasattr(combined_instance, "x")
        assert hasattr(combined_instance, "y")
        assert hasattr(combined_instance, "z")

        # Verify defaults are preserved
        assert combined_instance.x == 0.0
        assert combined_instance.y == 0.0
        assert combined_instance.z == 0.0

        # Verify vector size
        assert combined_instance.as_vec().shape[0] == 3

    def test_add_submodel_creates_proxy(self):
        """Test that add_submodel creates accessible proxy for connections."""

        @symbolic
        class States:
            x: ca.SX = state(1, 0.0)

        @symbolic
        class Inputs:
            u: ca.SX = input_var()

        @symbolic
        class Params:
            pass

        parent = ModelSX.create(States, Inputs, Params)
        child = ModelSX.create(States, Inputs, Params)

        # Add submodel
        parent.add_submodel("child", child)

        # Verify child proxy is accessible as attribute
        assert hasattr(parent, "child")
        assert isinstance(parent.child, SubmodelProxy)

    def test_connect_signals(self):
        """Test signal connection API."""

        @symbolic
        class States:
            x: ca.SX = state(1, 0.0)

        @symbolic
        class Inputs:
            u: ca.SX = input_var()

        @symbolic
        class Params:
            pass

        @symbolic
        class Outputs:
            y: ca.SX = output_var(desc="output")

        parent = ModelSX.create(States, Inputs, Params)
        child1 = ModelSX.create(States, Inputs, Params, Outputs)
        child2 = ModelSX.create(States, Inputs, Params)

        # Build simple dynamics
        child1.build(f_x=ca.SX.zeros(1), f_y=child1.x.x)

        parent.add_submodel("child1", child1)
        parent.add_submodel("child2", child2)

        # Test connection API using string paths
        parent.connect("child2.u.u", "child1.y.y")

        # Verify connection was stored
        assert "child2" in parent._input_connections
        assert "child2.u.u" in parent._input_connections["child2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
