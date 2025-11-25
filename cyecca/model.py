"""Backwards compatibility shim for model.py -> model/ package migration.

This file maintains the original import path `from cyecca.model import ...`
while the implementation has moved to the cyecca.model/ package.
"""

# Re-export everything from the model package
from .model import *

__all__ = [
    # Field creators
    "state",
    "algebraic_var",
    "dependent_var",
    "quadrature_var",
    "discrete_state",
    "discrete_var",
    "event_indicator",
    "param",
    "input_var",
    "output_var",
    # Decorators
    "symbolic",
    "compose_states",
    # Model classes
    "ModelSX",
    "ModelMX",
]
