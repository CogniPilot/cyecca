"""
Validation utilities for Base Modelica JSON files.

Provides schema validation using JSON Schema when jsonschema is available.
"""

import json
from pathlib import Path
from typing import Union, Optional, List

# Try to import jsonschema, but don't fail if not available
try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def validate_base_modelica(
    data: Union[dict, str, Path],
    schema_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Validate Base Modelica JSON data against the schema.

    Args:
        data: Either a dict with JSON data, or path to JSON file
        schema_path: Optional path to schema file (auto-detected if None)

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> errors = validate_base_modelica("model.json")
        >>> if errors:
        ...     print("Validation errors:")
        ...     for error in errors:
        ...         print(f"  - {error}")
        >>> else:
        ...     print("✓ Valid!")
    """
    if not HAS_JSONSCHEMA:
        return ["jsonschema package not available - install with: pip install jsonschema"]

    # Load data if path provided
    if isinstance(data, (str, Path)):
        with open(data, "r") as f:
            data = json.load(f)

    # Auto-detect schema path if not provided
    if schema_path is None:
        # Try to find schema in modelica_ir repository
        current_file = Path(__file__)
        potential_paths = [
            # Relative to cyecca package
            current_file.parent.parent.parent.parent
            / "modelica_ir"
            / "schemas"
            / "base_modelica_ir-0.1.0.schema.json",
            # Relative to current directory
            Path.cwd() / "modelica_ir" / "schemas" / "base_modelica_ir-0.1.0.schema.json",
            # Relative to workspace
            Path.cwd().parent / "modelica_ir" / "schemas" / "base_modelica_ir-0.1.0.schema.json",
        ]

        for path in potential_paths:
            if path.exists():
                schema_path = path
                break

        if schema_path is None:
            return ["Could not find Base Modelica schema file - please provide schema_path"]

    # Load schema
    with open(schema_path, "r") as f:
        schema = json.load(f)

    # Validate
    errors = []
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        errors.append(f"Validation error: {e.message}")
        if e.path:
            path_str = ".".join(str(p) for p in e.path)
            errors.append(f"  Location: {path_str}")
        if e.schema_path:
            schema_path_str = ".".join(str(p) for p in e.schema_path)
            errors.append(f"  Schema path: {schema_path_str}")
    except jsonschema.SchemaError as e:
        errors.append(f"Schema error: {e.message}")

    return errors


def validate_base_modelica_file(file_path: Union[str, Path]) -> bool:
    """
    Validate a Base Modelica JSON file and print results.

    Args:
        file_path: Path to JSON file

    Returns:
        True if valid, False otherwise

    Example:
        >>> if validate_base_modelica_file("model.json"):
        ...     print("Model is valid!")
    """
    errors = validate_base_modelica(file_path)

    if not errors:
        print(f"✓ {file_path} is valid Base Modelica JSON")
        return True
    else:
        print(f"✗ {file_path} has validation errors:")
        for error in errors:
            print(f"  {error}")
        return False


def get_schema_path() -> Optional[Path]:
    """
    Get the path to the Base Modelica schema file.

    Returns:
        Path to schema file, or None if not found
    """
    current_file = Path(__file__)
    potential_paths = [
        current_file.parent.parent.parent.parent
        / "modelica_ir"
        / "schemas"
        / "base_modelica_ir-0.1.0.schema.json",
        Path.cwd() / "modelica_ir" / "schemas" / "base_modelica_ir-0.1.0.schema.json",
        Path.cwd().parent / "modelica_ir" / "schemas" / "base_modelica_ir-0.1.0.schema.json",
    ]

    for path in potential_paths:
        if path.exists():
            return path

    return None
