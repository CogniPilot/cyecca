"""Type-safe CasADi modeling framework with full hybrid DAE support.

This package provides a declarative API for building hybrid dynamical systems
without dynamic class generation. Full autocomplete and type safety throughout.

Main components:
- Fields: Factory functions for creating typed state/input/parameter fields
- Decorators: @symbolic for adding CasADi methods to dataclasses
- Models: ModelSX/ModelMX for building and simulating systems

Quick Start:
    import casadi as ca
    from cyecca.dynamics import ModelSX, state, input_var, param, output_var, symbolic, EmptyOutputs

    @symbolic
    class States:
        x: ca.SX = state(1, 1.0, "position")
        v: ca.SX = state(1, 0.0, "velocity")

    @symbolic
    class Inputs:
        F: ca.SX = input_var(desc="force")

    @symbolic
    class Params:
        m: ca.SX = param(1.0, "mass")
        k: ca.SX = param(1.0, "spring constant")
        c: ca.SX = param(0.1, "damping")

    @symbolic
    class Outputs:
        position: ca.SX = output_var(desc="position")
        velocity: ca.SX = output_var(desc="velocity")

    # If you don't need outputs, use EmptyOutputs to avoid boilerplate:
    #   model = ModelSX.create(States, Inputs, Params, EmptyOutputs)
    
    model = ModelSX.create(States, Inputs, Params, Outputs)
    x, u, p, y = model.x, model.u, model.p, model.y
    
    # Mass-spring-damper: mx'' + cx' + kx = F
    f_x = ca.vertcat(x.v, (u.F - p.c * x.v - p.k * x.x) / p.m)
    f_y = ca.vertcat(x.x, x.v)
    
    model.build(f_x=f_x, f_y=f_y, integrator='rk4')
    model = model.simulate(0.0, 10.0, 0.01, compute_output=True)
    
    # Access trajectory data with full type safety:
    import matplotlib.pyplot as plt
    traj = model.trajectory
    plt.plot(traj.t, traj.x.x, label='position')  # plots all components
    plt.plot(traj.t, traj.x.v, label='velocity')
    plt.plot(traj.t, traj.y.position, '--', label='output')

See README.md for detailed documentation and examples.
"""

# Model classes
from .core import EmptyOutputs, ModelMX, ModelSX

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
    "EmptyOutputs",
    # Linearization
    "find_trim",
    "linearize_dynamics",
    "analyze_modes",
    "print_trim_details",
]
