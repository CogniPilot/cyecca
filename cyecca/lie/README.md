# Lie Groups Module

Complete implementations of common Lie groups for robotics and control with multiple parameterizations.

## Supported Groups

| Group | Description | Parameters | Common Use Cases |
|-------|-------------|------------|------------------|
| **R^n** | Euclidean space | n | State vectors, positions, velocities |
| **SO(2)** | 2D rotations | 1 (angle) | Planar robotics, heading estimation |
| **SO(3)Quat** | 3D rotations (quaternion) | 4 (w,x,y,z) | Attitude estimation, spacecraft control |
| **SO(3)EulerB321** | 3D rotations (Euler) | 3 (ψ,θ,φ) | Aircraft dynamics, human-readable angles |
| **SO(3)Mrp** | 3D rotations (MRP) | 3 | Singularity-free attitude control |
| **SO(3)Dcm** | 3D rotations (DCM) | 9 (3×3 matrix) | Direct rotation matrix operations |
| **SE(2)** | 2D rigid transformations | 3 (x,y,θ) | Mobile robots, 2D localization |
| **SE(3)Quat** | 3D rigid transformations | 7 (pos + quat) | Manipulators, 3D SLAM |
| **SE(3)Mrp** | 3D rigid transformations | 6 (pos + mrp) | Spacecraft pose estimation |
| **SE_2(3)** | Extended pose | 10 (p + v + quat) | IMU preintegration, VIO |

## Features

All groups support:
- **Group operations**: Composition (`@`), inverse, identity
- **Lie algebra operations**: exp, log, adjoint
- **Jacobians**: Left and right Jacobians for integration
- **Conversions**: Between different parameterizations (e.g., quaternion ↔ Euler)
- **Numerical stability**: Taylor series near singularities

## Quick Examples

### SE(3): 3D Rigid Transformations

```python
import cyecca.lie as lie
import casadi as ca

# Create SE(3) transformation (position + orientation)
position = ca.DM([1, 2, 3])
quaternion = ca.DM([1, 0, 0, 0])  # Identity rotation
X = lie.SE3Quat.elem(ca.vertcat(position, quaternion))

# Inverse transformation
X_inv = X.inverse()

# Compose transformations
X2 = lie.SE3Quat.from_components(
    p=ca.DM([0.5, 0, 0]),
    q=ca.DM([0.9239, 0, 0, 0.3827])  # 45° about z-axis
)
X_composed = X @ X2

# Extract components
print(f"Position: {X_composed.p.param.T}")
print(f"Quaternion: {X_composed.q.param.T}")
```

### SO(3): 3D Rotations

```python
import cyecca.lie as lie
import casadi as ca

# Create quaternion rotation
q = lie.SO3Quat.elem(ca.DM([1, 0, 0, 0]))  # Identity

# Convert to Euler angles (Body 3-2-1 convention)
euler = lie.SO3EulerB321.from_Quat(q)
print(f"Euler angles (ψ,θ,φ): {euler.param.T}")

# Convert to MRP (Modified Rodrigues Parameters)
mrp = lie.SO3Mrp.from_Quat(q)

# Compute Lie algebra element (angular velocity)
omega = ca.DM([0.1, 0.2, 0.3])  # rad/s
omega_lie = q.alg.elem(omega)

# Integrate rotation (exponential map)
q_updated = q @ omega_lie.exp()
```

### SO(2): 2D Rotations

```python
import cyecca.lie as lie
import casadi as ca

# Create 2D rotation
theta = ca.DM([ca.pi / 4])  # 45 degrees
R = lie.SO2.elem(theta)

# Compose rotations
R2 = lie.SO2.elem(ca.DM([ca.pi / 6]))
R_composed = R @ R2

# Extract angle
print(f"Angle: {R_composed.param}")  # pi/4 + pi/6 = 5*pi/12
```

### SE(2): 2D Rigid Transformations

```python
import cyecca.lie as lie
import casadi as ca

# Create SE(2) transformation for mobile robot
x = ca.DM([1.0])    # x position
y = ca.DM([2.0])    # y position
theta = ca.DM([0.5])  # heading
X = lie.SE2.elem(ca.vertcat(x, y, theta))

# Transform point from local to global frame
point_local = ca.DM([1, 0])  # 1m ahead of robot
point_global = X @ point_local

print(f"Global position: {point_global.T}")
```

### SE_2(3): Extended Pose (IMU Preintegration)

```python
import cyecca.lie as lie
import casadi as ca

# Extended pose with position, velocity, and orientation
p = ca.DM([0, 0, 0])           # position
v = ca.DM([1, 0, 0])           # velocity
q = ca.DM([1, 0, 0, 0])        # orientation (quaternion)
X = lie.SE23.elem(ca.vertcat(p, v, q))

# Update with IMU measurements
dt = 0.01
accel = ca.DM([0, 0, -9.81])  # acceleration
omega = ca.DM([0, 0, 0.1])    # angular velocity

# Propagate state (simplified)
dX = X.alg.elem(ca.vertcat(v, accel, omega))
X_updated = X @ (dX * dt).exp()
```

## Jacobians

Jacobians are critical for integration on manifolds:

```python
import cyecca.lie as lie
import casadi as ca

# SO(3) quaternion
q = lie.SO3Quat.elem(ca.DM([1, 0, 0, 0]))

# Angular velocity perturbation
omega = ca.DM([0.01, 0.02, 0.03])
omega_lie = q.alg.elem(omega)

# Right Jacobian (for integration)
Jr = q.alg.right_jacobian(omega)

# Integrate: q_new = q * exp(omega)
q_new = q @ omega_lie.exp()

# Left Jacobian (for derivatives)
Jl = q.alg.left_jacobian(omega)
```

## Conversions Between Representations

### SO(3) Conversions

```python
import cyecca.lie as lie
import casadi as ca

# Start with quaternion
q = lie.SO3Quat.elem(ca.DM([0.9239, 0.0, 0.0, 0.3827]))

# Convert to different representations
euler = lie.SO3EulerB321.from_Quat(q)
mrp = lie.SO3Mrp.from_Quat(q)
dcm = lie.SO3Dcm.from_Quat(q)

# Convert back to quaternion
q_roundtrip = lie.SO3Quat.from_EulerB321(euler)
```

### SE(3) Conversions

```python
import cyecca.lie as lie
import casadi as ca

# SE(3) with quaternion
X_quat = lie.SE3Quat.elem(ca.DM([1, 2, 3, 1, 0, 0, 0]))

# Convert to MRP representation
X_mrp = lie.SE3Mrp.from_Quat(X_quat)

# Convert back
X_quat_roundtrip = lie.SE3Quat.from_Mrp(X_mrp)
```

## Direct Products

Combine multiple Lie groups:

```python
from cyecca.lie.direct_product import LieGroupDirectProduct
import cyecca.lie as lie
import casadi as ca

# Create product group SE(3) × SO(3) (e.g., body pose + camera orientation)
ProductGroup = LieGroupDirectProduct([lie.SE3Quat, lie.SO3Quat])

# Create element
body_pose = lie.SE3Quat.elem(ca.DM([0, 0, 0, 1, 0, 0, 0]))
camera_orient = lie.SO3Quat.elem(ca.DM([1, 0, 0, 0]))
state = ProductGroup.elem(ca.vertcat(body_pose.param, camera_orient.param))

# Operations work on product
state_inv = state.inverse()
```

## Module Structure

```
cyecca/lie/
├── base.py               # Abstract base classes (Group, Algebra)
├── group_so2.py          # SO(2) - 2D rotations
├── group_so3.py          # SO(3) - 3D rotations (4 representations)
├── group_se2.py          # SE(2) - 2D rigid transformations
├── group_se3.py          # SE(3) - 3D rigid transformations (2 representations)
├── group_se23.py         # SE_2(3) - Extended pose for IMU
├── group_rn.py           # R^n - Euclidean vector spaces
└── direct_product.py     # Direct products of Lie groups
```

## See Also

- Main cyecca documentation: [../../README.md](../../README.md)
- Jupyter notebooks: `notebook/lie/` for detailed examples
- IMU preintegration: `notebook/ins/`
