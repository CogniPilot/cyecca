"""
Integration test for Base Modelica import/export.

Tests importing the bouncing_ball_base.json example and verifying the model structure.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cyecca.io import import_base_modelica, export_base_modelica, validate_base_modelica


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
    model = import_base_modelica(example_path)

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
    export_base_modelica(model, output_path, validate=False, pretty=True)

    # Re-import and compare
    print("Re-importing exported file...")
    model2 = import_base_modelica(output_path)

    print(f"✓ Round-trip successful!")
    print(f"  Original variables: {len(model.variables)}")
    print(f"  Re-imported variables: {len(model2.variables)}")
    print(f"  Original equations: {len(model.equations)}")
    print(f"  Re-imported equations: {len(model2.equations)}")

    # Validate the exported file
    print("\nValidating exported file...")
    errors = validate_base_modelica(output_path)
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ Exported file is valid Base Modelica JSON")

    return model


if __name__ == "__main__":
    print("=" * 80)
    print("Base Modelica Integration Test")
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
