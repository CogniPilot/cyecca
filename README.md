# cyecca

[![Build](https://github.com/CogniPilot/cyecca/actions/workflows/ci.yml/badge.svg)](https://github.com/CogniPilot/cyecca/actions/workflows/ci.yml)

**Cy**mbolic **E**stimation and **C**ontrol with **C**omputer **A**lgebra

A lightweight Python library for robotics and control systems using Lie groups and symbolic mathematics with [CasADi](https://web.casadi.org/).

## Features

- **Lie Group Library**: Complete implementations of SO(2), SO(3), SE(2), SE(3), SE_2(3), and R^n groups
- **Multiple SO(3) Representations**: Quaternions, Euler angles (B321), Modified Rodrigues Parameters (MRP), and Direction Cosine Matrices (DCM)
- **Symbolic Framework**: Built on CasADi for automatic differentiation and code generation
- **Dynamics Models**: Ready-to-use models for quadrotors, fixed-wing aircraft, and path planning (Bezier curves)
- **Jacobians for Euler Angles**: Efficient right and left Jacobians for attitude kinematics
- **Type Safety**: Uses beartype for runtime type checking

## Installation

```bash
git clone git@github.com:CogniPilot/cyecca.git
cd cyecca
poetry install
poetry shell
```

## Quick Start

```python
import cyecca.lie as lie
import casadi as ca

# Create a quaternion rotation
q = lie.SO3Quat.elem(ca.DM([1, 0, 0, 0]))  # Identity rotation

# Convert to Euler angles (B321 convention)
euler = lie.SO3EulerB321.from_Quat(q)
print(f"Euler angles: {euler.param.T}")  # [psi, theta, phi]

# Compute Euler angle Jacobians for kinematics
# euler_dot = Jr @ omega_body  (right Jacobian)
# euler_dot = Jl @ omega_spatial (left Jacobian)
Jr = euler.right_jacobian()
Jl = euler.left_jacobian()

# Verify: Jl = Jr @ R_eb
R_eb = euler.to_Matrix()
assert ca.norm_fro(Jl - Jr @ R_eb) < 1e-10
```

## Library Structure

```
cyecca/
├── lie/              # Lie group implementations
│   ├── group_so2.py  # SO(2) - 2D rotations
│   ├── group_so3.py  # SO(3) - 3D rotations (Quat, Euler, MRP, DCM)
│   ├── group_se2.py  # SE(2) - 2D rigid body motion
│   ├── group_se3.py  # SE(3) - 3D rigid body motion
│   ├── group_se23.py # SE_2(3) - Extended pose (position, velocity, attitude)
│   └── group_rn.py   # R^n - Euclidean spaces
├── models/           # Dynamics models
│   ├── quadrotor.py  # Quadrotor dynamics
│   ├── fixedwing.py  # Fixed-wing aircraft dynamics
│   └── bezier.py     # Bezier curve path planning
├── symbolic.py       # CasADi symbolic utilities
├── codegen.py        # Code generation tools
└── util.py           # Utility functions
```

## Supported Lie Groups

| Group | Description | Parameters | Use Case |
|-------|-------------|------------|----------|
| **SO(2)** | 2D rotations | 1 (angle) | Planar systems |
| **SO(3)Quat** | 3D rotations (quaternion) | 4 (w,x,y,z) | General 3D rotations |
| **SO(3)EulerB321** | 3D rotations (Euler) | 3 (ψ,θ,φ) | Aircraft dynamics |
| **SO(3)Mrp** | 3D rotations (MRP) | 3 | Singularity-free rotations |
| **SO(3)Dcm** | 3D rotations (DCM) | 9 (3×3 matrix) | Direct matrix operations |
| **SE(2)** | 2D rigid motion | 3 (x,y,θ) | Mobile robots |
| **SE(3)** | 3D rigid motion | 7 (pos + quat) | Manipulators, drones |
| **SE_2(3)** | Extended 3D pose | 10 (pos + vel + quat) | IMU preintegration |
| **R^n** | Euclidean space | n | State vectors |

## Euler Angle Conventions

The library uses **B321 (Body ZYX)** Euler angles with the following conventions:

- **State Order**: [ψ (yaw), θ (pitch), φ (roll)]
- **Rotation Sequence**: Intrinsic rotations about body-fixed axes Z→Y→X
- **Frame Conventions**: 
  - Earth frame: ENU (East-North-Up)
  - Body frame: FLU (Forward-Left-Up)
  - Aero frame: FRD (Forward-Right-Down)

### Euler Kinematics

```python
# Right Jacobian: relates body angular velocity to Euler rates
# euler_dot = Jr(euler) @ omega_body
Jr = euler.right_jacobian()

# Left Jacobian: relates spatial angular velocity to Euler rates  
# euler_dot = Jl(euler) @ omega_spatial
Jl = euler.left_jacobian()

# Relationship: Jl = Jr @ R_eb
```

## Examples

See the `notebook/` directory for Jupyter notebooks demonstrating:

- **lie/**: Lie group operations and conversions
- **path_planning/**: Bezier curve generation and optimization
- **ins/**: Inertial navigation examples

## Dependencies

Core dependencies (automatically installed):
- **casadi**: Symbolic framework and optimization
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **sympy**: Symbolic mathematics
- **beartype**: Runtime type checking

## Development

The library is actively used in CogniPilot projects for:
- Flight control system design
- Trajectory optimization
- State estimation
- Hardware-in-the-loop simulation

## License

See LICENSE file for details.

## Citation

If you use cyecca in your research, please cite:

```bibtex
@software{cyecca,
  title = {cyecca: Symbolic Estimation and Control with Computer Algebra},
  author = {Goppert, James and Lin, Li-Yu and Kmetko, Alex and others},
  organization = {CogniPilot},
  url = {https://github.com/CogniPilot/cyecca},
  year = {2025}
}
```
