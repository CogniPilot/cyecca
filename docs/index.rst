Cyecca Documentation
====================

Cyecca is a Python library for geometric control and robotics using Lie groups, differential-algebraic equations (DAEs), and CasADi symbolic computation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide/index
   api/index
   examples/index

Quick Start
-----------

Installation::

    pip install cyecca

Basic example:

.. doctest::

   >>> import cyecca.lie as lie
   >>> import casadi as ca
   >>> # Create SE(3) transformation
   >>> position = ca.DM([1, 2, 3])
   >>> quaternion = ca.DM([1, 0, 0, 0])  # Identity rotation
   >>> X = lie.SE3Quat.elem(ca.vertcat(position, quaternion))
   >>> X.p.param.shape
   (3, 1)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
