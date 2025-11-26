"""Type-safe CasADi modeling framework with full hybrid DAE support.

This package provides a declarative API for building hybrid dynamical systems
without dynamic class generation. Full autocomplete and type safety throughout.

Main components:
- Fields: Factory functions for creating typed state/input/parameter fields
- Decorators: @symbolic for adding CasADi methods to dataclasses
- Models: ModelSX/ModelMX for building and simulating systems

Quick Start:
    from cyecca.dynamics import ModelSX, state, input_var, param, symbolic

    @symbolic
    class States:
        x: ca.SX = state(1, 0.0, "position")
        v: ca.SX = state(1, 0.0, "velocity")

    model = ModelSX.create(States, Inputs, Params)
    model.build(f_x=f_x, integrator='rk4')
    result = model.simulate(0.0, 10.0, 0.01)

See README.md for detailed documentation and examples.
"""

# Model classes
from .core import ModelMX, ModelSX

# Decorators
from .decorators import compose_states, symbolic

# Field creators
from .fields import (
    algebraic_var,
    dependent_var,
    discrete_state,
    discrete_var,
    event_indicator,
    input_var,
    output_var,
    param,
    quadrature_var,
    state,
)

# Linearization and analysis tools
from .linearize import analyze_modes, find_trim, linearize_dynamics, print_trim_details

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
    # Linearization
    "find_trim",
    "linearize_dynamics",
    "analyze_modes",
    "print_trim_details",
]
