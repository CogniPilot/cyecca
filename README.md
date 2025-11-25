# cyecca

[![Build](https://github.com/CogniPilot/cyecca/actions/workflows/ci.yml/badge.svg)](https://github.com/CogniPilot/cyecca/actions/workflows/ci.yml)

**Cy**mbolic **E**stimation and **C**ontrol with **C**omputer **A**lgebra

A lightweight Python library for robotics and control systems using Lie groups and symbolic mathematics with [CasADi](https://web.casadi.org/).

## Features

- **Lie Group Library**: Complete implementations of SO(2), SO(3), SE(2), SE(3), SE_2(3), and R^n groups with multiple parameterizations
- **Type-Safe Modeling Framework**: Declarative API for building dynamical systems with full IDE autocomplete support
- **Hybrid DAE Systems**: Support for continuous/discrete states, algebraic constraints, event detection, and quadrature integration
- **Symbolic Mathematics**: CasADi-SymPy interoperability with Taylor series approximations and expression simplification
- **Code Generation**: Automatic C code generation from symbolic expressions for embedded systems
- **Numerical Integration**: RK4, RK8 (DOP853), and Euler integrators with adaptive stepping
- **Trajectory Planning**: Bezier curve tools for path planning with derivatives up to snap
- **Visualization**: CasADi expression graph rendering for debugging and documentation

## Installation

```bash
git clone git@github.com:CogniPilot/cyecca.git
cd cyecca
poetry install
poetry shell
```

## Quick Start

### Working with Lie Groups

```python
import cyecca.lie as lie
import casadi as ca

# Create SE(3) transformation (3D rigid body motion)
position = ca.DM([1, 2, 3])
quaternion = ca.DM([1, 0, 0, 0])  # Identity rotation
X = lie.SE3Quat.elem(ca.vertcat(position, quaternion))

# Compute the group inverse (reverse transformation)
X_inv = X.inverse()

# Compose transformations
X2 = lie.SE3Quat.from_components(
    p=ca.DM([0.5, 0, 0]),
    q=ca.DM([0.9239, 0, 0, 0.3827])  # 45° about z
)
X_composed = X @ X2

# Extract components
print(f"Position: {X_composed.p.param.T}")
print(f"Quaternion: {X_composed.q.param.T}")

# Convert to different representation
euler = lie.SO3EulerB321.from_Quat(X_composed.q)
print(f"Euler angles (ψ,θ,φ): {euler.param.T}")
```

### Building Dynamical Systems

```python
from cyecca.model import ModelSX, state, input_var, param, symbolic
import casadi as ca

# Define system components with type-safe dataclasses
@symbolic
class States:
    x: ca.SX = state(1, 0.0, "position (m)")
    v: ca.SX = state(1, 0.0, "velocity (m/s)")

@symbolic
class Inputs:
    F: ca.SX = input_var(desc="force (N)")

@symbolic
class Params:
    m: ca.SX = param(1.0, "mass (kg)")
    c: ca.SX = param(0.1, "damping (Ns/m)")

# Create model with full IDE autocomplete
model = ModelSX.create(States, Inputs, Params)

# Access components with autocomplete
x, u, p = model.x, model.u, model.p

# Define continuous dynamics: dx/dt = f(x, u, p)
f_x = ca.vertcat(
    x.v,                        # dx/dt = v
    (u.F - p.c * x.v) / p.m    # dv/dt = (F - c*v) / m
)

# Build with RK4 integrator
model.build(f_x=f_x, integrator='rk4', integrator_options={'N': 4})

# Simulate
result = model.simulate(t0=0.0, tf=10.0, dt=0.01)
```

### Symbolic-Numeric Workflow

```python
from cyecca.symbolic import sympy_to_casadi, taylor_series_near_zero
import sympy as sp
import casadi as ca

# Define symbolic expression in SymPy
x = sp.Symbol('x')
f_sympy = sp.sin(x) / x  # sinc function

# Convert to CasADi with Taylor series for numerical stability
f_casadi = taylor_series_near_zero(x, f_sympy, order=6, eps=1e-4)

# Use in optimization or simulation
x_val = ca.DM(0.001)
print(f"sinc({x_val}) = {f_casadi(x_val)}")  # Numerically stable near zero
```

## Library Structure

```
cyecca/
├── lie/              # Lie group implementations → [Documentation](cyecca/lie/README.md)
│   ├── base.py       # Abstract base classes for groups and algebras
│   ├── group_so2.py  # SO(2) - 2D rotations
│   ├── group_so3.py  # SO(3) - 3D rotations (Quat, Euler, MRP, DCM)
│   ├── group_se2.py  # SE(2) - 2D rigid body transformations
│   ├── group_se3.py  # SE(3) - 3D rigid body transformations
│   ├── group_se23.py # SE_2(3) - Extended pose (IMU preintegration)
│   ├── group_rn.py   # R^n - Euclidean vector spaces
│   └── direct_product.py  # Direct products of Lie groups
├── model/            # Type-safe modeling framework → [Documentation](cyecca/model/README.md)
│   ├── fields.py     # Field creators (state, input, param, etc.)
│   ├── decorators.py # @symbolic decorator and compose_states
│   ├── composition.py # Hierarchical model composition
│   └── core.py       # ModelSX and ModelMX classes
├── models/           # Pre-built dynamics models
│   ├── quadrotor.py      # Quadrotor dynamics
│   ├── fixedwing.py      # Fixed-wing aircraft
│   ├── fixedwing_4ch.py  # 4-channel RC aircraft
│   ├── bezier.py         # Bezier trajectory generation
│   ├── rdd2.py           # Mellinger-style Quadrotor control
│   └── rdd2_loglinear.py # Log-linear Quadrotor control
├── integrators.py    # Numerical integration (RK4, RK8, Euler)
├── symbolic.py       # SymPy-CasADi interoperability
├── codegen.py        # C code generation
├── graph.py          # Expression graph visualization
└── util.py           # Utility functions
```

## Type-Safe Modeling Framework

The `cyecca.model` module provides a declarative API for building dynamical systems with full type safety and IDE autocomplete support. See [detailed documentation](cyecca/model/README.md).

### Quick Example

```python
from cyecca.model import ModelSX, state, input_var, param, symbolic
import casadi as ca

@symbolic
class States:
    x: ca.SX = state(1, 0.0, "position (m)")
    v: ca.SX = state(1, 0.0, "velocity (m/s)")

@symbolic
class Inputs:
    F: ca.SX = input_var(desc="force (N)")

@symbolic
class Params:
    m: ca.SX = param(1.0, "mass (kg)")
    c: ca.SX = param(0.1, "damping (Ns/m)")

model = ModelSX.create(States, Inputs, Params)
x, u, p = model.x, model.u, model.p

f_x = ca.vertcat(x.v, (u.F - p.c * x.v) / p.m)
model.build(f_x=f_x, integrator='rk4')

result = model.simulate(t0=0.0, tf=10.0, dt=0.01)
```

**Features:**
- Type-safe state definitions with full IDE autocomplete
- Hybrid systems: continuous/discrete states, events, algebraic constraints
- Hierarchical composition for building complex systems
- Multiple integrators (RK4, RK8, Euler, IDAS for DAE)

See [cyecca/model/README.md](cyecca/model/README.md) for:
- Hybrid systems (bouncing ball example)
- DAE systems with constraints
- Quadrature integration
- Hierarchical model composition
- Pre-built models in `cyecca/models/`

## Examples

See the `notebook/` directory for Jupyter notebooks demonstrating:

- **lie/**: Lie group operations, conversions, and applications
- **ins/**: Inertial navigation, IMU preintegration, invariant filtering
- **path_planning/**: Bezier curve generation and differential flatness
- **estimation/**: Attitude estimation and sensor fusion
- **sim/**: Quadrotor simulation with trajectory tracking

## Supported Lie Groups

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

All groups support:
- Group operations (composition `@`, inverse, identity)
- Lie algebra operations (exp, log, adjoint)
- Jacobians (left, right)
- Conversions between representations
- Numerical stability near singularities

## Dependencies

Core dependencies (automatically installed):
- **casadi**: Symbolic framework and optimization
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **sympy**: Symbolic mathematics
- **beartype**: Runtime type checking
- **pydot**: Graph visualization

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
