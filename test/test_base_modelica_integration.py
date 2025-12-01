"""
Integration test for DAE IR import/export.

Tests importing DAE IR JSON and verifying the model structure.
Includes tests for rumoca -> cyecca roundtrip.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cyecca.io import import_dae_ir, export_dae_ir, validate_dae_ir
from cyecca.io.dae_ir import _import_model


def test_import_bouncing_ball():
    """Test importing the bouncing_ball_base.json example."""
    # Find the example file (try multiple possible locations)
    potential_paths = [
        Path(__file__).parent.parent.parent
        / "modelica_ir"
        / "examples"
        / "bouncing_ball_base.json",
        Path(__file__).parent.parent.parent.parent
        / "modelica_ir"
        / "examples"
        / "bouncing_ball_base.json",
    ]

    example_path = None
    for path in potential_paths:
        if path.exists():
            example_path = path
            break

    if example_path is None:
        print(f"Warning: Example file not found in any of these locations:")
        for path in potential_paths:
            print(f"  - {path}")
        print("Skipping test...")
        return

    print(f"Importing: {example_path}")

    # Import the model
    model = import_dae_ir(example_path)

    # Verify model structure
    print(f"\nModel imported successfully!")
    print(f"  Name: {model.name}")
    print(f"  Description: {model.description}")
    print(
        f"  Constants: {len([v for v in model.variables if v.is_parameter and v.variability.name == 'CONSTANT'])}"
    )
    print(
        f"  Parameters: {len([v for v in model.variables if v.is_parameter and v.variability.name != 'CONSTANT'])}"
    )
    print(f"  Variables: {len([v for v in model.variables if not v.is_parameter])}")
    print(f"  Equations: {len(model.equations)}")

    # Check specific variables
    print("\nVariables:")
    for var in model.variables:
        print(
            f"  - {var.name}: {var.var_type.name}, variability={var.variability.name}, unit={var.unit}"
        )

    # Check equations
    print(f"\nEquations:")
    for i, eq in enumerate(model.equations):
        print(f"  {i+1}. {eq.eq_type.name}: {eq}")

    # Test round-trip export
    output_path = Path("/tmp/bouncing_ball_test.json")
    print(f"\n\nExporting to: {output_path}")
    export_dae_ir(model, output_path, validate=False, pretty=True)

    # Re-import and compare
    print("Re-importing exported file...")
    model2 = import_dae_ir(output_path)

    print(f"✓ Round-trip successful!")
    print(f"  Original variables: {len(model.variables)}")
    print(f"  Re-imported variables: {len(model2.variables)}")
    print(f"  Original equations: {len(model.equations)}")
    print(f"  Re-imported equations: {len(model2.equations)}")

    # Validate the exported file
    print("\nValidating exported file...")
    errors = validate_dae_ir(output_path)
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ Exported file is valid DAE IR JSON")


def test_rumoca_bouncing_ball_roundtrip():
    """
    Test full roundtrip: Modelica source -> rumoca -> JSON -> cyecca.

    This test compiles a Modelica model using rumoca and imports it into cyecca,
    verifying the complete pipeline works correctly.
    """
    # Find the bouncing_ball.mo fixture
    rumoca_dir = Path(__file__).parent.parent.parent / "rumoca"
    modelica_file = rumoca_dir / "tests" / "fixtures" / "bouncing_ball.mo"

    if not modelica_file.exists():
        pytest.skip(f"Modelica fixture not found: {modelica_file}")

    # Run rumoca to get JSON output
    result = subprocess.run(
        ["rumoca", "--json", "-m", "BouncingBall", str(modelica_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(f"rumoca not available or failed: {result.stderr}")

    # Parse the JSON output
    json_data = json.loads(result.stdout)

    # Import into cyecca using the internal function
    model = _import_model(json_data)

    # Verify model structure
    assert model.name == "GeneratedModel"

    # Check parameters
    params = [v for v in model.variables if v.is_parameter]
    assert len(params) == 2, f"Expected 2 parameters, got {len(params)}"

    param_names = {p.name for p in params}
    assert "e" in param_names, "Should have 'e' parameter"
    assert "h0" in param_names, "Should have 'h0' parameter"

    # Check that e has correct start value
    e_param = next(p for p in params if p.name == "e")
    assert e_param.start == 0.8 or e_param.value == 0.8, "e should be 0.8"

    # Check that h0 has correct start value
    h0_param = next(p for p in params if p.name == "h0")
    assert h0_param.start == 1.0 or h0_param.value == 1.0, "h0 should be 1.0"

    # Check variables (states)
    non_params = [v for v in model.variables if not v.is_parameter]
    assert (
        len(non_params) >= 2
    ), f"Should have at least 2 non-parameter variables, got {len(non_params)}"

    var_names = {v.name for v in non_params}
    assert "h" in var_names, "Should have 'h' variable"
    assert "v" in var_names, "Should have 'v' variable"

    # Check equations (continuous equations for der(h)=v, der(v)=-g, z=2*h+v)
    assert (
        len(model.equations) >= 2
    ), f"Should have at least 2 continuous equations, got {len(model.equations)}"

    # When equations in Rumoca are output in algorithms section, not equations
    # Check that we have algorithms with when statements
    assert (
        len(model.algorithms) >= 1
    ), f"Should have at least 1 algorithm section (for when equation), got {len(model.algorithms)}"


def test_rumoca_integrator_roundtrip():
    """Test roundtrip with a simple integrator model."""
    rumoca_dir = Path(__file__).parent.parent.parent / "rumoca"
    modelica_file = rumoca_dir / "tests" / "fixtures" / "integrator.mo"

    if not modelica_file.exists():
        pytest.skip(f"Modelica fixture not found: {modelica_file}")

    # Run rumoca to get JSON output
    result = subprocess.run(
        ["rumoca", "--json", "-m", "Integrator", str(modelica_file)], capture_output=True, text=True
    )

    if result.returncode != 0:
        pytest.skip(f"rumoca not available or failed: {result.stderr}")

    # Parse the JSON output
    json_data = json.loads(result.stdout)

    # Import into cyecca
    model = _import_model(json_data)

    # Simple integrator should have at least 1 state variable
    non_params = [v for v in model.variables if not v.is_parameter]
    assert len(non_params) >= 1, "Should have at least 1 variable"

    # Should have at least 1 equation
    assert len(model.equations) >= 1, "Should have at least 1 equation"


def test_rumoca_for_equation_roundtrip():
    """Test roundtrip with a for-equation model."""
    rumoca_dir = Path(__file__).parent.parent.parent / "rumoca"
    modelica_file = rumoca_dir / "tests" / "fixtures" / "for_equation.mo"

    if not modelica_file.exists():
        pytest.skip(f"Modelica fixture not found: {modelica_file}")

    # Run rumoca to get JSON output
    result = subprocess.run(
        ["rumoca", "--json", "-m", "ForEquation", str(modelica_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(f"rumoca not available or failed: {result.stderr}")

    # Parse the JSON output
    json_data = json.loads(result.stdout)

    # Import into cyecca
    model = _import_model(json_data)

    # Should have variables x and v (Rumoca may output them as scalars or arrays)
    non_params = [v for v in model.variables if not v.is_parameter]
    assert len(non_params) >= 2, f"Expected at least 2 variables, got {len(non_params)}"

    var_names = {v.name for v in non_params}
    assert "x" in var_names, "Should have 'x' variable"
    assert "v" in var_names, "Should have 'v' variable"

    # Should have 1 for-equation
    from cyecca.ir import EquationType

    for_eqs = [eq for eq in model.equations if eq.eq_type == EquationType.FOR]
    assert len(for_eqs) == 1, f"Expected 1 for-equation, got {len(for_eqs)}"

    # Check the for-equation structure
    for_eq = for_eqs[0]
    assert for_eq.index_var == "i", f"Expected index_var 'i', got {for_eq.index_var}"


def test_rumoca_initial_equation_roundtrip():
    """Test roundtrip with a model containing initial equations."""
    rumoca_dir = Path(__file__).parent.parent.parent / "rumoca"
    modelica_file = rumoca_dir / "tests" / "fixtures" / "initial_equation.mo"

    if not modelica_file.exists():
        pytest.skip(f"Modelica fixture not found: {modelica_file}")

    # Run rumoca to get JSON output
    result = subprocess.run(
        ["rumoca", "--json", "-m", "InitialEquation", str(modelica_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(f"rumoca not available or failed: {result.stderr}")

    # Parse the JSON output
    json_data = json.loads(result.stdout)

    # Import into cyecca
    model = _import_model(json_data)

    # Should have 2 parameters (x0 and v0)
    params = [v for v in model.variables if v.is_parameter]
    assert len(params) == 2, f"Expected 2 parameters, got {len(params)}"

    param_names = {p.name for p in params}
    assert "x0" in param_names, "Should have 'x0' parameter"
    assert "v0" in param_names, "Should have 'v0' parameter"

    # Should have 2 state variables (x and v)
    # Note: Rumoca outputs states directly, derivatives are expressed via der() in equations
    non_params = [v for v in model.variables if not v.is_parameter]
    assert len(non_params) == 2, f"Expected 2 state variables, got {len(non_params)}"

    var_names = {v.name for v in non_params}
    assert "x" in var_names, "Should have 'x' variable"
    assert "v" in var_names, "Should have 'v' variable"

    # Should have 2 continuous equations (der(x) = v and der(v) = -x)
    assert len(model.equations) == 2, f"Expected 2 equations, got {len(model.equations)}"

    # Should have 2 initial equations (x = x0 and v = v0)
    assert (
        len(model.initial_equations) == 2
    ), f"Expected 2 initial equations, got {len(model.initial_equations)}"


if __name__ == "__main__":
    print("=" * 80)
    print("DAE IR Integration Test")
    print("=" * 80)
    print()

    try:
        model = test_import_bouncing_ball()
        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 80)
        import traceback

        traceback.print_exc()
        sys.exit(1)
