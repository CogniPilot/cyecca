Lie Groups
==========

Complete implementations of common Lie groups for robotics and control with multiple parameterizations.

Supported Groups
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 15 40

   * - Group
     - Description
     - Parameters
     - Common Use Cases
   * - **R^n**
     - Euclidean space
     - n
     - State vectors, positions, velocities
   * - **SO(2)**
     - 2D rotations
     - 1 (angle)
     - Planar robotics, heading estimation
   * - **SO(3)Quat**
     - 3D rotations (quaternion)
     - 4 (w,x,y,z)
     - Attitude estimation, spacecraft control
   * - **SO(3)EulerB321**
     - 3D rotations (Euler)
     - 3 (ψ,θ,φ)
     - Aircraft dynamics, human-readable angles
   * - **SO(3)Mrp**
     - 3D rotations (MRP)
     - 3
     - Singularity-free attitude control
   * - **SE(2)**
     - 2D rigid transformations
     - 3 (x,y,θ)
     - Mobile robots, 2D localization
   * - **SE(3)Quat**
     - 3D rigid transformations
     - 7 (pos + quat)
     - Manipulators, 3D SLAM
   * - **SE_2(3)**
     - Extended pose
     - 10 (p + v + quat)
     - IMU preintegration, VIO

Features
--------

All groups support:

* **Group operations**: Composition (``*``), inverse, identity
* **Group actions**: Transform vectors/points (``@``)
* **Lie algebra operations**: exp, log, adjoint
* **Jacobians**: Left and right Jacobians for integration
* **Conversions**: Between different parameterizations
* **Numerical stability**: Taylor series near singularities

Quick Examples
--------------

SE(3): 3D Rigid Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doctest::

   >>> import cyecca.lie as lie
   >>> import casadi as ca
   >>> # Create SE(3) transformation
   >>> position = ca.DM([1, 2, 3])
   >>> quaternion = ca.DM([1, 0, 0, 0])
   >>> X = lie.SE3Quat.elem(ca.vertcat(position, quaternion))
   >>> X.p.param.shape
   (3, 1)

SO(3): 3D Rotations
~~~~~~~~~~~~~~~~~~~

Quaternion to Euler conversion::

   >>> q = lie.SO3Quat.elem(ca.DM([1, 0, 0, 0]))
   >>> euler = lie.SO3EulerB321.from_Quat(q)
   >>> mrp = lie.SO3Mrp.from_Quat(q)

Integration Methods
-------------------

Lie groups provide methods for numerical integration:

* **Exponential map**: ``algebra_element.exp(group)``
* **Logarithmic map**: ``group_element.log()``
* **Left/Right Jacobians**: For manifold integration

See :doc:`../api/lie` for detailed API reference.
