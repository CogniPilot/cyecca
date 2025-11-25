# cyecca

[![Build](https://github.com/CogniPilot/cyecca/actions/workflows/ci.yml/badge.svg)](https://github.com/CogniPilot/cyecca/actions/workflows/ci.yml)

**Cy**mbolic **E**stimation and **C**ontrol with **C**omputer **A**lgebra

A lightweight Python library for robotics and control systems using Lie groups and symbolic mathematics with [CasADi](https://web.casadi.org/).

## Features

- **Lie Group Library**: Complete implementations of SO(2), SO(3), SE(2), SE(3), SE_2(3), and R^n groups
- **Multiple SO(3) Representations**: Quaternions, Euler angles (B321), Modified Rodrigues Parameters (MRP), and Direction Cosine Matrices (DCM)
- **Symbolic Framework**: Built on CasADi for automatic differentiation and code generation
- **Type-Safe Modeling Framework**: Declarative API for building hybrid dynamical systems with full autocomplete support
- **Hybrid DAE Systems**: Support for continuous states, algebraic constraints, discrete events, and quadrature states
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

### Lie Groups

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

### Type-Safe Dynamical Systems

```python
from cyecca.model import ModelSX, state, input_var, param, symbolic
import casadi as ca

# Define your system with type-safe dataclasses
@symbolic
class States:
    h: ca.SX = state(1, 10.0, "height (m)")
    v: ca.SX = state(1, 0.0, "velocity (m/s)")

@symbolic
class Inputs:
    thrust: ca.SX = input_var(desc="thrust (N)")

@symbolic
class Params:
    m: ca.SX = param(1.0, "mass (kg)")
    g: ca.SX = param(9.81, "gravity (m/s^2)")

# Create model with full IDE autocomplete
model = ModelSX.create(States, Inputs, Params)

# Access states, inputs, parameters with autocomplete
x = model.x  # x.h, x.v available with autocomplete!
u = model.u  # u.thrust
p = model.p  # p.m, p.g

# Define dynamics: dx/dt = f(x, u, p)
f_x = ca.vertcat(
    x.v,                    # dh/dt = v
    u.thrust/p.m - p.g      # dv/dt = thrust/m - g
)

# Build model with RK4 integrator
model.build(f_x=f_x, integrator='rk4')

# Simulate
result = model.simulate(t0=0.0, tf=5.0, dt=0.01)
print(f"Final height: {result['x'][0, -1]:.2f} m")
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
├── model.py          # Type-safe modeling framework for dynamical systems
├── integrators.py    # Numerical integration (RK4, RK8/DOP853, Euler)
├── symbolic.py       # CasADi symbolic utilities
├── codegen.py        # Code generation tools
└── util.py           # Utility functions
```

## Type-Safe Modeling Framework

The `cyecca.model` module provides a declarative API for building dynamical systems with full type safety and IDE autocomplete support.

### Key Features

- **Type-Safe State Definitions**: Define states, inputs, and parameters as typed dataclasses
- **Full Autocomplete**: IDE autocomplete for all signals (states, inputs, parameters, outputs)
- **Hybrid Systems Support**: Continuous dynamics, discrete events, algebraic constraints
- **Multiple Integrators**: RK4, Euler, IDAS (DAE), with configurable substeps
- **Hierarchical Composition**: Build complex systems from simpler subsystems
- **Event Detection**: Zero-crossing detection for hybrid dynamics

### System Components

```python
from cyecca.model import (
    ModelSX, ModelMX,           # Model classes (SX for small, MX for large systems)
    state,                      # Continuous state: dx/dt = f(x,u,p)
    algebraic_var,              # Algebraic variable: 0 = g(x,z_alg,u,p)
    dependent_var,              # Dependent variable: y = h(x,u,p)
    quadrature_var,             # Quadrature state: dq/dt = integrand(x,u,p)
    discrete_state,             # Discrete state: z⁺ = f_z(x,u,p) at events
    discrete_var,               # Discrete variable: m⁺ = f_m(x,u,p) at events
    event_indicator,            # Event indicator: event when c=0
    input_var,                  # Control input
    output_var,                 # Observable output
    param,                      # Time-independent parameter
    symbolic,                   # Decorator for dataclasses
    compose_states,             # Compose multiple state types
)
```

### Example: Bouncing Ball (Hybrid System)

```python
from cyecca.model import ModelSX, state, param, discrete_state, event_indicator, symbolic
import casadi as ca

@symbolic
class States:
    h: ca.SX = state(1, 10.0, "height (m)")
    v: ca.SX = state(1, 0.0, "velocity (m/s)")

@symbolic
class Inputs:
    pass  # No inputs

@symbolic
class Params:
    g: ca.SX = param(9.81, "gravity (m/s^2)")
    e: ca.SX = param(0.8, "coefficient of restitution")

@symbolic
class DiscreteStates:
    bounce_count: ca.SX = discrete_state(1, 0.0, "number of bounces")

@symbolic
class EventIndicators:
    ground_contact: ca.SX = event_indicator(1, "ground contact: h <= 0")

model = ModelSX.create(
    States, Inputs, Params,
    discrete_state_type=DiscreteStates,
    event_indicator_type=EventIndicators
)

x = model.x
p = model.p
z = model.z

# Continuous dynamics: free fall
f_x = ca.vertcat(x.v, -p.g)

# Event indicator: ground contact when h <= 0
f_c = x.h

# Discrete update: reverse velocity and increment counter
f_m = ca.vertcat(0, -p.e * x.v)  # Reset: h=0, v=-e*v
f_z = z.bounce_count + 1         # Increment bounce counter

model.build(f_x=f_x, f_c=f_c, f_m=f_m, f_z=f_z, integrator='rk4')

# Simulate with event detection
result = model.simulate(0.0, 10.0, 0.01, detect_events=True)
print(f"Total bounces: {result['z'][0, -1]}")
```

### Hierarchical Composition

Build complex systems by composing simpler subsystems:

```python
# Create subsystems
plant = create_aircraft_model()
controller = create_autopilot_model()

# Compose into closed-loop system
closed_loop = ModelSX.compose({
    "plant": plant,
    "controller": controller
})

# Connect signals with autocomplete
closed_loop.connect(controller.u.pitch_meas, plant.x.theta)
closed_loop.connect(plant.u.elevator, controller.y.elevator_cmd)

# Build and simulate
closed_loop.build_composed(integrator='rk4')
result = closed_loop.simulate(0.0, 10.0, 0.01)
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
