"""Typed aliases that bridge the DSL and IR layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type

from cyecca.ir.variable import DataType, IRVariable, NumericValue, VariableKind

Shape = Tuple[int, ...]
Indices = Tuple[int, ...]

# Canonical IR types (re-exported for backwards compatibility)
Var = IRVariable
VarKind = VariableKind
DType = DataType


@dataclass
class SubmodelField:
    """A submodel (nested model instance) with optional parameter overrides."""

    model_class: Type
    name: Optional[str] = None
    overrides: Dict[str, NumericValue] = field(default_factory=dict)

    def __repr__(self) -> str:
        if self.overrides:
            overrides_str = ", ".join(f"{k}={v}" for k, v in self.overrides.items())
            return f"submodel({self.model_class.__name__}, {overrides_str})"
        return f"submodel({self.model_class.__name__})"


__all__ = [
    "Var",
    "VarKind",
    "DType",
    "Shape",
    "Indices",
    "SubmodelField",
    "NumericValue",
]
