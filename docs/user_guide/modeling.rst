Modeling Framework
==================

Type-safe declarative API for building dynamical systems with IDE autocomplete.

Key Features
------------

* **Type-Safe State Definitions**: Define states, inputs, parameters as typed dataclasses
* **Full Autocomplete**: IDE autocomplete for all signals
* **Hybrid Systems**: Continuous dynamics, discrete events, algebraic constraints
* **Multiple Integrators**: RK4, Euler, IDAS (DAE)
* **Hierarchical Composition**: Build complex systems from subsystems
* **Event Detection**: Zero-crossing detection for hybrid dynamics

Quick Start
-----------

.. doctest::

   >>> import casadi as ca
   >>> from cyecca.dynamics import ModelSX, state, input_var, param, symbolic
   >>> @symbolic
   ... class States:
   ...     x: ca.SX = state(1, 0.0, "position")
   ...     v: ca.SX = state(1, 0.0, "velocity")
   >>> @symbolic
   ... class Inputs:
   ...     F: ca.SX = input_var(desc="force")
   >>> @symbolic
   ... class Params:
   ...     m: float = param(1.0, "mass")
   ...     c: float = param(0.1, "damping")
   >>> model = ModelSX.create(States, Inputs, Params)
   >>> # Build dynamics: x' = v, v' = (F - c*v)/m
   >>> # model.build(f_x=ca.vertcat(model.x.v, (model.u.F - model.p.c * model.x.v) / model.p.m))

Supported Features
------------------

Continuous States (x)
~~~~~~~~~~~~~~~~~~~~~

Differential equations: ``dx/dt = f_x(x, u, p, ...)``

Use ``state()`` field creator and ``f_x`` in ``build()``

Discrete States (z)
~~~~~~~~~~~~~~~~~~~

Event-triggered updates: ``z⁺ = f_z(...)``

Use ``discrete_state()`` field creator with event detection

Event Indicators (c)
~~~~~~~~~~~~~~~~~~~~

Zero-crossing detection: ``c(x, u, p) = 0``

Use ``event_indicator()`` and ``f_c`` in ``build()``

Algebraic Constraints
~~~~~~~~~~~~~~~~~~~~~

DAE systems: ``0 = g(x, z_alg, u, p)``

Use ``algebraic_var()`` with IDAS integrator (⚠️ experimental)

Quadrature States (q)
~~~~~~~~~~~~~~~~~~~~~

Path integrals: ``dq/dt = f_q(x, u, p)``

Use ``quadrature_var()`` for cost accumulation

Hybrid Systems
--------------

Example: Bouncing ball with restitution coefficient

.. code-block:: python

   @symbolic
   class States:
       h: ca.SX = state(1, 10.0, "height (m)")
       v: ca.SX = state(1, 0.0, "velocity (m/s)")
   
   @symbolic
   class DiscreteVars:
       bounces: ca.SX = discrete_var(0, "bounce count")
   
   @symbolic
   class EventIndicators:
       ground: ca.SX = event_indicator("ground contact")
   
   model = ModelSX.create(States, ..., discrete_var_type=DiscreteVars, 
                          event_indicator_type=EventIndicators)
   
   # Continuous dynamics
   f_x = ca.vertcat(x.v, -9.81)
   
   # Event indicator
   f_c = x.h  # Triggers when height crosses zero
   
   # Reset map (reverse velocity with damping)
   f_m = ca.vertcat(x.h, -0.8 * x.v)
   
   model.build(f_x=f_x, f_c=f_c, f_m=f_m)
   result = model.simulate(0, 10, 0.01, detect_events=True)

Linearization & Analysis
------------------------

Find trim conditions and analyze stability:

.. code-block:: python

   from cyecca.dynamics import find_trim, linearize, analyze_modes
   
   # Find equilibrium
   x_trim, u_trim = find_trim(model, x_guess, u_guess, p_values)
   
   # Linearize about trim
   A, B = linearize(model, x_trim, u_trim, p_values)
   
   # Analyze eigenvalues
   modes = analyze_modes(A)
   for mode in modes:
       print(f"{mode['name']}: τ={mode['time_constant']:.2f}s")

See :doc:`../api/model` for complete API reference.
