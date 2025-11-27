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
git clone https://github.com/CogniPilot/cyecca.git
cd cyecca
poetry install
poetry run ./tools/test.sh   # optional: run tests
```

## Quick Start

```python
# Lie groups for 3D transformations
import cyecca.lie as lie
import casadi as ca

X = lie.SE3Quat.elem(ca.DM([1, 2, 3, 1, 0, 0, 0]))  # position + quaternion
X_inv = X.inverse()

# Type-safe modeling with unified namespace (like Modelica)
from cyecca.dynamics.explicit import Model, explicit, state, input_var, param, output_var

@explicit
class MassSpringDamper:
    # States
    x: float = state(desc="position")
    v: float = state(desc="velocity")
    # Inputs
    F: float = input_var(desc="force")
    # Parameters
    m: float = param(default=1.0, desc="mass")
    c: float = param(default=0.1, desc="damping")
    k: float = param(default=1.0, desc="spring constant")
    # Outputs
    position: float = output_var(desc="position output")

# Create model - all variables in unified namespace (model.v)
model = Model(MassSpringDamper)

# Define dynamics using unified namespace
# model.v.x, model.v.v, model.v.F, model.v.m, etc. are symbolic variables
model.ode(model.v.x, model.v.v)  # dx/dt = v
model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)

# Define outputs
model.output(model.v.position, model.v.x)

model.build()

# Set initial conditions and simulate
model.v0.x = 1.0  # Start at x=1
t, data = model.simulate(0.0, 10.0, 0.01)

# Access trajectory data
# data.x is an ndarray of shape (n_steps,)
# data.v is an ndarray of shape (n_steps,)
print(f"Final position: {data.x[-1]:.6f}")
print(f"Final velocity: {data.v[-1]:.6f}")

# Linearize at operating point (uses model.v0)
A, B, C, D = model.linearize()
```

## Documentation

📚 **[Read the documentation on GitHub Pages](https://cognipilot.github.io/cyecca/)**

The full documentation includes:
- Installation guide and quick start tutorial
- Comprehensive Lie groups guide with examples
- Modeling framework guide (hybrid systems, DAE, composition)
- Complete API reference (auto-generated from docstrings)

**Interactive Examples:**
See [cyecca_notebooks](https://github.com/CogniPilot/cyecca_notebooks) for Jupyter notebooks with tutorials and demonstrations.

**Build locally:**
```bash
cd docs && poetry run make html
# Open docs/_build/html/index.html in your browser
```

## Modules

### Lie Groups (`cyecca.lie`)
Complete implementations of SO(2), SO(3), SE(2), SE(3), SE_2(3), and R^n with multiple parameterizations.

**Key features:** Group operations, Lie algebra, Jacobians, conversions
**Learn more:** [Lie Groups Guide](docs/user_guide/lie_groups.rst)

### Dynamics Framework (`cyecca.dynamics`)
Type-safe declarative API for building hybrid dynamical systems with IDE autocomplete.

**Key features:** Continuous/discrete states, events, DAE, composition, linearization
**Learn more:** [Modeling Guide](docs/user_guide/modeling.rst)

### Pre-Built Models (`cyecca.models`)
Ready-to-use dynamics models: quadrotor, fixed-wing aircraft, RDD2 controller, Bezier trajectories

### Path Planning (`cyecca.planning`)
Dubins path planner for fixed-wing aircraft with forward-only, fixed turn radius constraints.

**Key features:** CasADi-based symbolic planning, RSL/LSR/LSL/RSR path types, branch-free implementation
**Learn more:** [Planning Module](cyecca/planning/)

### Other Modules
- **integrators** - RK4, RK8 (DOP853), Euler with adaptive stepping
- **symbolic** - SymPy ↔ CasADi conversion, Taylor series
- **codegen** - C code generation for embedded systems
- **graph** - Expression graph visualization

## ROS 2 Integration

Launch files for quadrotor and fixed-wing simulation with RViz visualization, joystick control, and trajectory tracking.

```bash
ros2 launch cyecca rdd2_sim.xml        # Quadrotor simulation
ros2 launch cyecca fixedwing_sim.xml   # Fixed-wing simulation
```

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
