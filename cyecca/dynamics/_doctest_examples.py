"""Pre-built example models for doctests.

This module provides partially-built models that doctests can use
to demonstrate API usage without all the boilerplate.
"""

import casadi as ca
import numpy as np

from .explicit import Model as ExplicitModel
from .explicit import explicit, input_var, output_var, param, state
from .implicit import Model as ImplicitModel
from .implicit import implicit
from .implicit import param as iparam
from .implicit import var

# ============================================================================
# Explicit model examples
# ============================================================================


@explicit
class MassSpringDamper:
    """Simple mass-spring-damper system for doctest examples."""

    # States
    x: float = state(desc="position")
    v: float = state(desc="velocity")
    # Inputs
    F: float = input_var(desc="force")
    # Parameters
    m: float = param(default=1.0, desc="mass")
    k: float = param(default=1.0, desc="spring constant")
    c: float = param(default=0.1, desc="damping")
    # Outputs
    position: float = output_var(desc="position output")
    energy: float = output_var(desc="kinetic energy")


def get_unbuilt_explicit_model():
    """Get an explicit model ready for ODE definitions.

    Returns a Model instance with no ODEs defined yet.
    Use model.ode() to add dynamics.

    Example:
        >>> from cyecca.dynamics._doctest_examples import get_unbuilt_explicit_model
        >>> model = get_unbuilt_explicit_model()
        >>> model.ode(model.v.x, model.v.v)  # dx/dt = v
        >>> model.ode(model.v.v, -model.v.k * model.v.x)  # dv/dt = -k*x
        >>> len(model._ode_defs)
        2
    """
    return ExplicitModel(MassSpringDamper)


def get_built_explicit_model():
    """Get a fully built explicit model ready for simulation.

    Example:
        >>> from cyecca.dynamics._doctest_examples import get_built_explicit_model
        >>> model = get_built_explicit_model()
        >>> model.v0.x = 1.0
        >>> t, data = model.simulate(0.0, 1.0, 0.1)
        >>> len(t) > 0
        True
    """
    model = ExplicitModel(MassSpringDamper)
    model.ode(model.v.x, model.v.v)
    model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)
    model.output(model.v.position, model.v.x)
    model.output(model.v.energy, 0.5 * model.v.m * model.v.v**2)
    model.build()
    return model


# ============================================================================
# Implicit model examples
# ============================================================================


@implicit
class Pendulum:
    """Simple pendulum for doctest examples."""

    theta: float = var(desc="angle")
    omega: float = var(desc="angular velocity")
    l: float = iparam(default=1.0, desc="length")
    g: float = iparam(default=9.81, desc="gravity")


def get_unbuilt_implicit_model():
    """Get an implicit model ready for equation definitions.

    Returns a Model instance with no equations defined yet.
    Use model.eq() to add equations.

    Example:
        >>> from cyecca.dynamics._doctest_examples import get_unbuilt_implicit_model
        >>> import casadi as ca
        >>> model = get_unbuilt_implicit_model()
        >>> model.eq(model.v.theta.dot() - model.v.omega)
        >>> model.eq(model.v.omega.dot() + model.v.g/model.v.l * ca.sin(model.v.theta))
        >>> len(model.equations)
        2
    """
    return ImplicitModel(Pendulum)


def get_built_implicit_model():
    """Get a fully built implicit model ready for simulation.

    Example:
        >>> from cyecca.dynamics._doctest_examples import get_built_implicit_model
        >>> model = get_built_implicit_model()
        >>> model.v0.theta = 0.5
        >>> t, data = model.simulate(0.0, 1.0, 0.1)
        >>> len(t) > 0
        True
    """
    model = ImplicitModel(Pendulum)
    model.eq(model.v.theta.dot() - model.v.omega)
    model.eq(model.v.omega.dot() + model.v.g / model.v.l * ca.sin(model.v.theta))
    model.build()
    return model
