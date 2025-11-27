"""Unified namespace explicit ODE modeling framework.

This package provides a declarative API for building explicit ODE systems
with a single unified namespace (like Modelica).

Main components:
- Model: The main model class with unified namespace
- Fields: Factory functions for creating typed state/input/parameter fields
- Decorators: @explicit for adding CasADi methods to dataclasses

Quick Start:
    from cyecca.dynamics.explicit import Model, explicit, state, input_var, param, output_var

    @explicit
    class MassSpringDamper:
        # States
        x: float = state()
        v: float = state()
        # Inputs
        F: float = input_var()
        # Parameters
        m: float = param(default=1.0)
        k: float = param(default=1.0)
        c: float = param(default=0.1)
        # Outputs
        position: float = output_var()
    
    model = Model(MassSpringDamper)
    
    # Define dynamics in unified namespace
    model.ode(model.v.x, model.v.v)
    model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)
    model.output(model.v.position, model.v.x)
    
    model.build()
    
    # Simulate
    model.v0.x = 1.0
    t, data = model.simulate(0.0, 10.0, 0.01)
    
    import matplotlib.pyplot as plt
    plt.plot(t, data.x)
    
    # Linearize
    A, B, C, D = model.linearize()

See README.md for detailed documentation and examples.
"""

# Unified Model class
from .model import Model

# Decorators
from .decorators import explicit, symbolic

# Field creators (imported from shared module)
from ..fields import (
    VarDescriptor,
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
from .linearize import analyze_modes, find_trim, linearize_dynamics

__all__ = [
    # Unified Model
    "Model",
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
    "explicit",
    # Linearization
    "find_trim",
    "linearize_dynamics",
    "analyze_modes",
]
