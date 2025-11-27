"""Implicit DAE modeling API for acausal, equation-based modeling.

This module provides a Modelica-like interface for defining implicit DAEs
where equations are written as residuals F(ẋ,x,z,p,t) = 0.

Key features:
- Single namespace like Modelica (all vars in one class)
- Automatic state inference from .dot() usage (just like Modelica!)
- Time is built-in (model.t), just like Modelica
- Stateful model - internal state updated by simulate()
- Typed dataclasses for full autocomplete
- Equation-based modeling with .eq() method
- Automatic ODE/ALG residual splitting
- IDAS integration support

Example:
    >>> import casadi as ca
    >>> from cyecca.dynamics.implicit import Model, implicit, var, param
    >>> 
    >>> @implicit
    >>> class Pendulum:
    >>>     theta: float = var()   # Becomes state (has .dot() in equations)
    >>>     omega: float = var()   # Becomes state (has .dot() in equations)
    >>>     l: float = param(default=1.0)
    >>>     g: float = param(default=9.81)
    >>> 
    >>> model = Model(Pendulum)
    >>> model.eq(model.v.theta.dot() - model.v.omega)  # .dot() marks theta as state
    >>> model.eq(model.v.omega.dot() + model.v.g/model.v.l * ca.sin(model.v.theta))
    >>> 
    >>> model.build()  # Infers: theta, omega are states
    >>> 
    >>> # Set initial conditions on model's internal state
    >>> model.v0.theta = 0.5
    >>> 
    >>> t, data = model.simulate(0.0, 10.0, 0.01)
    >>> plt.plot(t, data.theta)  # Full autocomplete!
    >>> 
    >>> # model.v0 is now at final state - continue seamlessly
    >>> t2, data2 = model.simulate(10.0, 20.0, 0.01)
    >>> 
    >>> # Checkpoint restore
    >>> model.v0 = data[50]
"""

from .model import Model
from .decorators import implicit
from ..fields import var, param, VarDescriptor

__all__ = [
    # Unified Model API
    "Model",
    # Decorator
    "implicit",
    # Modelica-style fields
    "var",    # Variable (state/algebraic inferred from .dot() usage)
    "param",  # Parameter (constant during simulation)
    "VarDescriptor",
]
