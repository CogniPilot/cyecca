"""
IO module for importing and exporting models in DAE IR format.
"""

from typing import Union, Optional
from pathlib import Path
import re

from cyecca.io.dae_ir import (
    import_dae_ir,
    load_dae_ir_json,
    export_dae_ir,
)
from cyecca.io.validation import (
    validate_dae_ir,
    validate_dae_ir_file,
    get_schema_path,
)


def compile_modelica(
    source: str,
    model_name: Optional[str] = None,
    library_paths: Optional[list] = None,
    use_modelica_path: bool = True,
    threads: Optional[int] = None,
) -> "cyecca.ir.Model":
    """
    Compile Modelica source code directly to a cyecca Model.

    This is a convenience function that combines rumoca compilation
    and conversion to cyecca IR in one step.

    Args:
        source: Modelica source code as a string
        model_name: Name of the model to compile (auto-detected if not provided)
        library_paths: Optional list of library paths to include (e.g., ["/path/to/MSL"])
        use_modelica_path: If True, also search MODELICAPATH env var for libraries (default: True)
        threads: Number of threads for parallel parsing (default: num_cpus - 1)

    Returns:
        cyecca.ir.Model instance ready for use with backends

    Example:
        >>> from cyecca.io import compile_modelica
        >>> from cyecca.backends.casadi import CasadiBackend
        >>>
        >>> model = compile_modelica('''
        ...     model MyModel
        ...         Real x;
        ...     equation
        ...         der(x) = -x;
        ...     end MyModel;
        ... ''')
        >>> backend = CasadiBackend(model)

        >>> # With MSL library:
        >>> model = compile_modelica('''
        ...     model Test
        ...         import Modelica.Blocks.Continuous.PID;
        ...         PID pid;
        ...     end Test;
        ... ''', library_paths=["/path/to/MSL"])

        >>> # Use all CPU cores for parsing:
        >>> import os
        >>> model = compile_modelica(source, threads=os.cpu_count())
    """
    try:
        import rumoca
    except ImportError:
        raise ImportError(
            "rumoca is required for compile_modelica(). " "Install with: pip install rumoca"
        )

    # Auto-detect model name if not provided
    if model_name is None:
        match = re.search(
            r"\b(?:model|block|connector|record|type|package|function|class)\s+(\w+)", source
        )
        if match:
            model_name = match.group(1)
        else:
            raise ValueError(
                "Could not auto-detect model name from source. Please provide model_name explicitly."
            )

    result = rumoca.compile_source(
        source,
        model_name,
        library_paths=library_paths,
        use_modelica_path=use_modelica_path,
        threads=threads,
    )
    return load_dae_ir_json(result.to_base_modelica_json())


def from_rumoca(source: Union[str, Path, "rumoca.CompilationResult"]) -> "cyecca.ir.Model":
    """
    Load a model from rumoca compilation result, JSON string, or file path.

    This is a convenience function that accepts multiple input types:
    - rumoca.CompilationResult: Direct result from rumoca.compile()
    - str: Either a JSON string or file path
    - Path: File path to a DAE IR JSON file

    Args:
        source: CompilationResult, JSON string, or path to JSON file

    Returns:
        cyecca.ir.Model instance ready for use with backends

    Example:
        >>> import rumoca
        >>> from cyecca.io import from_rumoca
        >>> from cyecca.backends.casadi import CasadiBackend
        >>>
        >>> model = from_rumoca(rumoca.compile("model.mo"))
        >>> backend = CasadiBackend(model)
    """
    # Check if it's a rumoca CompilationResult
    if hasattr(source, "to_base_modelica_json"):
        return load_dae_ir_json(source.to_base_modelica_json())

    # Check if it's a Path or path-like string
    if isinstance(source, Path):
        return import_dae_ir(source)

    if isinstance(source, str):
        # Check if it looks like a file path
        if not source.strip().startswith("{"):
            path = Path(source)
            if path.exists():
                return import_dae_ir(path)

        # Otherwise treat as JSON string
        return load_dae_ir_json(source)

    raise TypeError(f"Expected CompilationResult, JSON string, or path, got {type(source)}")


__all__ = [
    "import_dae_ir",
    "load_dae_ir_json",
    "export_dae_ir",
    "validate_dae_ir",
    "validate_dae_ir_file",
    "get_schema_path",
    "from_rumoca",
    "compile_modelica",
]
