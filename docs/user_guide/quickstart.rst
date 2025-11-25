Quick Start
===========

Basic Usage
-----------

Lie Groups
~~~~~~~~~~

Transform 3D positions and orientations:

.. doctest::

   >>> import cyecca.lie as lie
   >>> import casadi as ca
   >>> # SE(3) transformation
   >>> X = lie.SE3Quat.elem(ca.DM([1, 2, 3, 1, 0, 0, 0]))
   >>> X_inv = X.inverse()
   >>> X.p.param.shape
   (3, 1)

Modeling Framework
~~~~~~~~~~~~~~~~~~

Build a simple mass-spring-damper system:

.. code-block:: python

   from cyecca.model import ModelSX, state, input_var, param, symbolic
   import casadi as ca
   
   @symbolic
   class States:
       x: ca.SX = state(1, 0.0, "position")
       v: ca.SX = state(1, 0.0, "velocity")
   
   @symbolic
   class Inputs:
       F: ca.SX = input_var(desc="force")
   
   @symbolic
   class Params:
       m: float = param(1.0, "mass")
       c: float = param(0.1, "damping")
       k: float = param(1.0, "stiffness")
   
   model = ModelSX.create(States, Inputs, Params)
   x, u, p = model.x, model.u, model.p
   
   # Dynamics: x' = v, v' = (F - c*v - k*x)/m
   f_x = ca.vertcat(x.v, (u.F - p.c * x.v - p.k * x.x) / p.m)
   
   model.build(f_x=f_x, integrator='rk4')
   
   # Simulate
   result = model.simulate(
       t0=0.0, 
       tf=10.0, 
       dt=0.01,
       x0={'x': 1.0, 'v': 0.0},
       u={'F': 0.0},
       p={'m': 1.0, 'c': 0.1, 'k': 1.0}
   )
   
   # Plot results
   import matplotlib.pyplot as plt
   plt.plot(result['t'], result['x'][0, :], label='position')
   plt.plot(result['t'], result['x'][1, :], label='velocity')
   plt.legend()
   plt.show()

Next Steps
----------

* :doc:`lie_groups` - Learn about SO(3), SE(3), and other Lie groups
* :doc:`modeling` - Advanced modeling features (hybrid systems, DAE, composition)
* :doc:`../examples/index` - Jupyter notebook tutorials
* :doc:`../api/index` - Complete API reference
