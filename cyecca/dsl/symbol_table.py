"""
Symbol table utilities for the Cyecca DSL.

Provides helpers for creating symbolic variables and flattening
submodel symbol namespaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from cyecca.dsl.variables import SymbolicVar, TimeVar

if TYPE_CHECKING:
    from cyecca.dsl.decorators import ModelMetadata
    from cyecca.dsl.instance import ModelInstance


def create_symbols(
    metadata: "ModelMetadata",
    instance: "ModelInstance",
) -> Dict[str, SymbolicVar]:
    """Create symbolic variables for all fields in model metadata.

    Parameters
    ----------
    metadata : ModelMetadata
        The model metadata containing variable definitions.
    instance : ModelInstance
        The model instance owning these symbols.

    Returns
    -------
    Dict[str, SymbolicVar]
        Mapping from variable name to its symbolic wrapper.
    """
    sym_vars: Dict[str, SymbolicVar] = {}
    for name, v in metadata.variables.items():
        sym_vars[name] = SymbolicVar(name, v, instance)
    return sym_vars


def flatten_submodel_symbols(
    submodels: Dict[str, "ModelInstance"],
    parent_sym_vars: Dict[str, SymbolicVar],
    parent_instance: "ModelInstance",
) -> None:
    """Flatten submodel symbols into parent namespace with prefixed names.

    Parameters
    ----------
    submodels : Dict[str, ModelInstance]
        Mapping of submodel instance name to the submodel instance.
    parent_sym_vars : Dict[str, SymbolicVar]
        The parent's symbol table to update in-place.
    parent_instance : ModelInstance
        The parent model instance (owner for new SymbolicVar objects).
    """
    for sub_name, sub_instance in submodels.items():
        for var_name, sym_var in sub_instance._sym_vars.items():
            full_name = f"{sub_name}.{var_name}"
            parent_sym_vars[full_name] = SymbolicVar(full_name, sym_var._var, parent_instance)


def create_time_var() -> TimeVar:
    """Return a fresh TimeVar instance."""
    return TimeVar()
