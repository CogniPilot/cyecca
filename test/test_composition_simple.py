"""Simple tests for model composition to debug signal routing."""

import casadi as ca
import numpy as np
import pytest

from cyecca.dynamics import ModelSX, input_var, output_var, param, state, symbolic


@symbolic
class SimpleStates:
    """Simple state: just position and velocity."""

    x: ca.SX = state(1, 0.0, "position")
    v: ca.SX = state(1, 0.0, "velocity")


@symbolic
class SimpleInputs:
    """Simple input: acceleration command."""

    a: ca.SX = input_var(desc="acceleration command")


@symbolic
class SimpleOutputs:
    """Simple output: expose state."""

    x: ca.SX = output_var(desc="position")
    v: ca.SX = output_var(desc="velocity")


@symbolic
class EmptyParams:
    """Empty parameters class."""

    pass


def create_simple_plant() -> ModelSX:
    """Create a simple plant: double integrator.

    Dynamics:
        dx/dt = v
        dv/dt = a

    Outputs:
        y.x = x
        y.v = v
    """
    model = ModelSX.create(SimpleStates, SimpleInputs, param_type=EmptyParams, output_type=SimpleOutputs)

    x, u, y = model.x, model.u, model.y

    # Dynamics: dx/dt = v, dv/dt = a
    f_x = ca.vertcat(x.v, u.a)

    # Outputs: expose states
    y.x = x.x
    y.v = x.v

    model.build(f_x=f_x, f_y=y.as_vec(), integrator="euler")
    return model


@symbolic
class ControllerStates:
    """Controller state: integral error."""

    i: ca.SX = state(1, 0.0, "integral error")


@symbolic
class ControllerInputs:
    """Controller inputs: reference and measurement."""

    ref: ca.SX = input_var(desc="reference")
    meas: ca.SX = input_var(desc="measurement")


@symbolic
class ControllerOutputs:
    """Controller output: control signal."""

    u: ca.SX = output_var(desc="control output")


def create_simple_controller(Kp: float = 1.0, Ki: float = 0.1) -> ModelSX:
    """Create a simple PI controller.

    Dynamics:
        di/dt = error

    Control law:
        u = Kp * error + Ki * i

    where error = ref - meas
    """
    model = ModelSX.create(ControllerStates, ControllerInputs, param_type=EmptyParams, output_type=ControllerOutputs)

    x, u, y = model.x, model.u, model.y

    # Error
    error = u.ref - u.meas

    # Dynamics: di/dt = error
    f_x = error

    # Control output
    y.u = Kp * error + Ki * x.i

    model.build(f_x=f_x, f_y=y.as_vec(), integrator="euler")
    return model


@symbolic
class ClosedLoopInputs:
    """Closed-loop system input: reference."""

    ref: ca.SX = input_var(desc="reference position")


@symbolic
class ClosedLoopOutputs:
    """Closed-loop system outputs."""

    x: ca.SX = output_var(desc="position")
    v: ca.SX = output_var(desc="velocity")
    u: ca.SX = output_var(desc="control signal")


def test_standalone_models():
    """Test that standalone models work correctly."""
    # Create models
    plant = create_simple_plant()
    controller = create_simple_controller()

    # Test plant
    x_plant = plant.x0.as_vec()
    u_plant = ca.DM([1.0])  # acceleration = 1.0
    p_plant = ca.DM([])

    dx_plant = plant.f_x(x_plant, u_plant, p_plant)
    y_plant = plant.f_y(x_plant, u_plant, p_plant)

    assert dx_plant.shape == (2, 1), "Plant dx should be 2x1"
    assert y_plant.shape == (2, 1), "Plant y should be 2x1"
    assert float(dx_plant[0]) == 0.0, "dx/dt should be 0 (v=0)"
    assert float(dx_plant[1]) == 1.0, "dv/dt should be 1.0 (a=1.0)"

    # Test controller
    x_ctrl = controller.x0.as_vec()
    u_ctrl = ca.vertcat(1.0, 0.0)  # ref=1.0, meas=0.0
    p_ctrl = ca.DM([])

    dx_ctrl = controller.f_x(x_ctrl, u_ctrl, p_ctrl)
    y_ctrl = controller.f_y(x_ctrl, u_ctrl, p_ctrl)

    assert dx_ctrl.shape == (1, 1), "Controller dx should be 1x1"
    assert y_ctrl.shape == (1, 1), "Controller y should be 1x1"
    assert float(dx_ctrl[0]) == 1.0, "di/dt should be 1.0 (error=1.0)"
    assert float(y_ctrl[0]) == 1.0, "u should be 1.0 (Kp*error + Ki*i = 1.0*1.0 + 0.1*0.0)"


def test_composition_basic():
    """Test basic composition creates correct structure."""
    plant = create_simple_plant()
    controller = create_simple_controller()

    # Compose
    parent = ModelSX.compose(
        {"plant": plant, "controller": controller},
        input_type=ClosedLoopInputs,
        output_type=ClosedLoopOutputs,
    )

    # Connect signals
    parent.connect("controller.u.ref", "u.ref")
    parent.connect("controller.u.meas", "plant.x.x")
    parent.connect("plant.u.a", "controller.y.u")

    parent.connect("y.x", "plant.y.x")
    parent.connect("y.v", "plant.y.v")
    parent.connect("y.u", "controller.y.u")

    # Build
    parent.build_composed(integrator="euler")

    # Check structure
    assert hasattr(parent, "f_x"), "Composed model should have f_x"
    assert hasattr(parent, "f_y"), "Composed model should have f_y"
    assert callable(parent.f_x), "f_x should be callable"
    assert callable(parent.f_y), "f_y should be callable"


def test_composition_output_routing():
    """Test that outputs are correctly routed in composed model."""
    plant = create_simple_plant()
    controller = create_simple_controller()

    # Compose
    parent = ModelSX.compose(
        {"plant": plant, "controller": controller},
        input_type=ClosedLoopInputs,
        output_type=ClosedLoopOutputs,
    )

    # Connect signals
    parent.connect("controller.u.ref", "u.ref")
    parent.connect("controller.u.meas", "plant.x.x")
    parent.connect("plant.u.a", "controller.y.u")

    parent.connect("y.x", "plant.y.x")
    parent.connect("y.v", "plant.y.v")
    parent.connect("y.u", "controller.y.u")

    # Build
    parent.build_composed(integrator="euler")

    # Test with initial conditions
    # Plant at x=0, v=0
    # Controller at i=0
    # Reference at ref=1.0
    x_composed = ca.vertcat(0.0, 0.0, 0.0)  # [plant.x, plant.v, controller.i]
    u_composed = ca.DM([1.0])  # ref=1.0
    p_composed = ca.DM([])

    # Evaluate outputs
    y_composed = parent.f_y(x_composed, u_composed, p_composed)

    assert y_composed.shape == (3, 1), "Composed y should be 3x1 [x, v, u]"

    # Check output values
    # Plant outputs x=0, v=0
    # Controller outputs u = Kp*(ref-meas) + Ki*i = 1.0*(1.0-0.0) + 0.1*0.0 = 1.0
    assert float(y_composed[0]) == 0.0, "y.x should be 0.0 (plant.x)"
    assert float(y_composed[1]) == 0.0, "y.v should be 0.0 (plant.v)"
    assert float(y_composed[2]) == 1.0, "y.u should be 1.0 (controller output)"


def test_composition_dynamics_consistency():
    """Test that composed dynamics match standalone when inputs are equivalent."""
    plant = create_simple_plant()
    controller = create_simple_controller()

    # Compose
    parent = ModelSX.compose(
        {"plant": plant, "controller": controller},
        input_type=ClosedLoopInputs,
        output_type=ClosedLoopOutputs,
    )

    # Connect signals
    parent.connect("controller.u.ref", "u.ref")
    parent.connect("controller.u.meas", "plant.x.x")
    parent.connect("plant.u.a", "controller.y.u")

    parent.connect("y.x", "plant.y.x")
    parent.connect("y.v", "plant.y.v")
    parent.connect("y.u", "controller.y.u")

    # Build
    parent.build_composed(integrator="euler")

    # Test state: plant at x=0.5, v=0.2; controller at i=0.1
    x_plant = ca.vertcat(0.5, 0.2)
    x_ctrl = ca.DM([0.1])
    x_composed = ca.vertcat(x_plant, x_ctrl)

    # Test with ref=1.0
    ref = 1.0
    u_composed = ca.DM([ref])

    # Evaluate composed dynamics
    dx_composed = parent.f_x(x_composed, u_composed, ca.DM([]))

    # Also evaluate the outputs to see what the controller is producing
    y_composed = parent.f_y(x_composed, u_composed, ca.DM([]))
    print(f"\nComposed outputs (f_y):")
    print(f"  y.x = {float(y_composed[0])}")
    print(f"  y.v = {float(y_composed[1])}")
    print(f"  y.u (controller output) = {float(y_composed[2])}")

    # Manually evaluate standalone dynamics
    # Controller: meas=0.5 (plant.x), ref=1.0
    error = ref - 0.5
    u_ctrl_standalone = ca.vertcat(ref, 0.5)
    y_ctrl_standalone = controller.f_y(x_ctrl, u_ctrl_standalone, ca.DM([]))
    control_output = float(y_ctrl_standalone[0])

    # Plant: a = control_output
    u_plant_standalone = ca.DM([control_output])
    dx_plant_standalone = plant.f_x(x_plant, u_plant_standalone, ca.DM([]))

    # Controller dynamics
    dx_ctrl_standalone = controller.f_x(x_ctrl, u_ctrl_standalone, ca.DM([]))

    # Compare
    print("\nComposed dynamics:")
    print(f"  dx_composed = {np.array(dx_composed).flatten()}")
    print("\nStandalone dynamics:")
    print(f"  dx_plant = {np.array(dx_plant_standalone).flatten()}")
    print(f"  dx_ctrl = {np.array(dx_ctrl_standalone).flatten()}")
    print(
        f"  Expected dx_composed = {np.concatenate([np.array(dx_plant_standalone).flatten(), np.array(dx_ctrl_standalone).flatten()])}"
    )

    # Plant portion should match
    np.testing.assert_allclose(
        np.array(dx_composed[:2]).flatten(),
        np.array(dx_plant_standalone).flatten(),
        rtol=1e-10,
        err_msg="Plant dynamics should match in composition",
    )

    # Controller portion should match
    np.testing.assert_allclose(
        np.array(dx_composed[2:]).flatten(),
        np.array(dx_ctrl_standalone).flatten(),
        rtol=1e-10,
        err_msg="Controller dynamics should match in composition",
    )


def test_passthrough_controller():
    """Test a simple passthrough controller to isolate composition issues."""

    @symbolic
    class PassthroughInputs:
        u_in: ca.SX = input_var(desc="input signal")

    @symbolic
    class PassthroughOutputs:
        u_out: ca.SX = output_var(desc="output signal")

    @symbolic
    class EmptyState:
        pass

    # Create passthrough controller (no states, just passes input to output)
    passthrough = ModelSX.create(
        state_type=EmptyState, input_type=PassthroughInputs, param_type=EmptyParams, output_type=PassthroughOutputs
    )
    passthrough.y.u_out = passthrough.u.u_in
    passthrough.build(f_x=ca.SX([]), f_y=passthrough.y.as_vec(), integrator="euler")

    # Create plant
    plant = create_simple_plant()

    # Compose with passthrough controller
    @symbolic
    class PassthroughClosedLoopInputs:
        a_cmd: ca.SX = input_var(desc="acceleration command")

    @symbolic
    class PassthroughClosedLoopOutputs:
        x: ca.SX = output_var(desc="position")
        a_actual: ca.SX = output_var(desc="actual acceleration command")

    parent = ModelSX.compose(
        {"plant": plant, "passthrough": passthrough},
        input_type=PassthroughClosedLoopInputs,
        output_type=PassthroughClosedLoopOutputs,
    )

    # Connect: parent input -> passthrough -> plant
    parent.connect("passthrough.u.u_in", "u.a_cmd")
    parent.connect("plant.u.a", "passthrough.y.u_out")

    # Connect outputs
    parent.connect("y.x", "plant.y.x")
    parent.connect("y.a_actual", "passthrough.y.u_out")

    parent.build_composed(integrator="euler")

    # Test: plant at x=0, v=0; input a_cmd=2.0
    x_composed = ca.vertcat(0.0, 0.0)  # only plant states
    u_composed = ca.DM([2.0])  # a_cmd

    # Evaluate standalone plant with a=2.0
    dx_plant_standalone = plant.f_x(x_composed, ca.DM([2.0]), ca.DM([]))

    # Evaluate composed
    dx_composed = parent.f_x(x_composed, u_composed, ca.DM([]))

    print("\nPassthrough test:")
    print(f"  Standalone plant dx: {np.array(dx_plant_standalone).flatten()}")
    print(f"  Composed dx: {np.array(dx_composed).flatten()}")

    # Should match exactly
    np.testing.assert_allclose(
        np.array(dx_composed).flatten(),
        np.array(dx_plant_standalone).flatten(),
        rtol=1e-12,
        err_msg="Passthrough composition should preserve plant dynamics exactly",
    )

    # Check outputs
    y_composed = parent.f_y(x_composed, u_composed, ca.DM([]))
    assert float(y_composed[0]) == 0.0, "y.x should be 0.0"
    assert float(y_composed[1]) == 2.0, "y.a_actual should be 2.0 (passthrough)"


if __name__ == "__main__":
    # Run tests manually
    print("Running standalone models test...")
    test_standalone_models()
    print("✓ Passed\n")

    print("Running composition basic test...")
    test_composition_basic()
    print("✓ Passed\n")

    print("Running composition output routing test...")
    test_composition_output_routing()
    print("✓ Passed\n")

    print("Running composition dynamics consistency test...")
    test_composition_dynamics_consistency()
    print("✓ Passed\n")

    print("Running passthrough controller test...")
    test_passthrough_controller()
    print("✓ Passed\n")

    print("All tests passed!")
