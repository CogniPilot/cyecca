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

```python
# Lie groups for 3D transformations
import cyecca.lie as lie
X = lie.SE3Quat.elem([1, 2, 3, 1, 0, 0, 0])  # position + quaternion
X_inv = X.inverse()

# Type-safe modeling with full autocomplete
from cyecca.model import ModelSX, state, input_var, param, symbolic
import casadi as ca

@symbolic
class States:
    x: ca.SX = state(1, 0.0, "position")
    v: ca.SX = state(1, 0.0, "velocity")

model = ModelSX.create(States, Inputs, Params)
x, u, p = model.x, model.u, model.p
f_x = ca.vertcat(x.v, (u.F - p.c * x.v) / p.m)
model.build(f_x=f_x, integrator='rk4')
result = model.simulate(0.0, 10.0, 0.01)
```

**Learn More:**
- **[Lie Groups →](cyecca/lie/README.md)** - SO(2), SO(3), SE(2), SE(3), SE_2(3) with examples
- **[Modeling Framework →](cyecca/model/README.md)** - Hybrid systems, DAE, composition, pre-built models
- **[Jupyter Notebooks →](notebook/)** - Detailed tutorials and applications

## Modules

### [Lie Groups](cyecca/lie/README.md)
Complete implementations of SO(2), SO(3), SE(2), SE(3), SE_2(3), and R^n with multiple parameterizations.

**Key features:** Group operations, Lie algebra, Jacobians, conversions between representations  
**Use cases:** Attitude estimation, 3D transformations, IMU preintegration, robot kinematics

### [Modeling Framework](cyecca/model/README.md)
Type-safe declarative API for building hybrid dynamical systems with IDE autocomplete.

**Key features:** Continuous/discrete states, events, DAE constraints, hierarchical composition, trim/linearization
**Use cases:** Flight dynamics, robotics, control systems, hybrid systems, stability analysis

### [Pre-Built Models](cyecca/models/README.md)
Ready-to-use dynamics models for robotics and aerospace systems.

**Available models:** Quadrotor, fixed-wing aircraft (3ch/4ch), RDD2 controller, Bezier trajectories  
**Use cases:** Flight simulation, control design, trajectory planning

### Other Modules
- **integrators.py** - RK4, RK8 (DOP853), Euler with adaptive stepping
- **symbolic.py** - SymPy ↔ CasADi conversion, Taylor series
- **codegen.py** - C code generation from symbolic expressions
- **graph.py** - CasADi expression graph visualization

## ROS 2 Integration

### [Launch Files & Simulation](launch/README.md)
ROS 2 launch configurations for running cyecca simulations with visualization.

**Available launches:** Quadrotor (RDD2), fixed-wing, RViz viewers  
**Features:** Joystick control, trajectory tracking, force visualization, IMU simulation

## Examples

See the `notebook/` directory for Jupyter notebooks:
- **lie/** - Lie group operations, conversions, and applications
- **ins/** - Inertial navigation, IMU preintegration, invariant filtering
- **path_planning/** - Bezier curve generation and differential flatness
- **estimation/** - Attitude estimation and sensor fusion
- **sim/** - Quadrotor simulation with trajectory tracking

## Supported Lie Groups

| Group | Parameters | Use Cases |
|-------|------------|-----------|
| **R^n** | n | State vectors, positions, velocities |
| **SO(2)** | 1 (angle) | Planar robotics, heading |
| **SO(3)Quat** | 4 (w,x,y,z) | Attitude estimation, spacecraft control |
| **SO(3)EulerB321** | 3 (ψ,θ,φ) | Aircraft dynamics, human-readable angles |
| **SO(3)Mrp** | 3 | Singularity-free attitude control |
| **SE(2)** | 3 (x,y,θ) | Mobile robots, 2D localization |
| **SE(3)Quat** | 7 (pos+quat) | Manipulators, 3D SLAM |
| **SE_2(3)** | 10 (p+v+quat) | IMU preintegration, VIO |

See [cyecca/lie/README.md](cyecca/lie/README.md) for detailed examples and conversions.

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
