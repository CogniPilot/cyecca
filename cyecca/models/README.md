# Pre-Built Dynamics Models

This directory contains ready-to-use dynamics models for common robotics and aerospace systems, built using the cyecca modeling framework.

## Available Models

### Aircraft Dynamics

#### `fixedwing.py`
3-channel fixed-wing aircraft dynamics (E-Flite Night Vapor UAV)
- **Controls:** Throttle, Elevator, Rudder
- **Features:** Aerodynamic coefficients, longitudinal/lateral stability derivatives
- **Use cases:** Flight control design, trajectory optimization, simulation

#### `fixedwing_4ch.py`
4-channel fixed-wing aircraft dynamics (HH Sport Cub S2)
- **Controls:** Throttle, Aileron, Elevator, Rudder
- **Features:** Full 6-DOF dynamics with aerodynamic modeling, product of inertia
- **Use cases:** Advanced flight control, autonomous flight

### Rotorcraft Dynamics

#### `quadrotor.py`
Quadrotor multirotor dynamics
- **Features:** 4-motor configuration, aerodynamic drag, motor dynamics with time constants
- **Sensors:** IMU (accel, gyro), magnetometer, GPS
- **Use cases:** Attitude estimation, position control, sensor fusion

### Control Systems

#### `rdd2.py`
Mellinger-style quadrotor controller (RDD2 - Rotation Decoupled Dynamics on SE(2,3))
- **Features:** Cascaded position-velocity-attitude control loops
- **Implementation:** Integral action on position, rate limiting, angle wrapping
- **Parameters:** Configurable gains, integral limits, rate constraints
- **Use cases:** Trajectory tracking, waypoint navigation

#### `rdd2_loglinear.py`
Log-linear variant of RDD2 controller
- **Features:** Alternative control formulation with different stability characteristics
- **Use cases:** Comparative control studies, specific flight regimes

### Trajectory Planning

#### `bezier.py`
Bezier curve trajectory generation
- **Features:** De Casteljau's algorithm for evaluation, derivatives up to 4th order (snap)
- **Functions:** 
  - `Bezier.eval(t)` - Evaluate curve at time t
  - `Bezier.deriv(m)` - Compute m-th derivative
  - `derive_bezier7()` - 7th order Bezier with boundary conditions
- **Use cases:** Path planning, differential flatness, smooth trajectory generation

## Usage Pattern

Most models follow this structure:

```python
from cyecca.models import quadrotor

# Derive symbolic model
model_sx = quadrotor.derive_model()

# Set parameters
params = {...}

# Build and simulate
model_sx.build(f_x=f_x, integrator='rk4')
result = model_sx.simulate(t0, tf, dt, x0=x0, u=u, p=params)
```

## Model Integration

These models are designed to work with:
- **Lie Groups** - SO(3), SE(3), SE_2(3) for rotations and poses
- **Symbolic Framework** - CasADi for automatic differentiation
- **Composition** - Can be combined into hierarchical systems
- **Code Generation** - Export to C for embedded deployment

## Adding New Models

When adding new dynamics models:
1. Use `@symbolic` dataclasses for type-safe state/input/parameter definitions
2. Include docstrings explaining the system and control inputs
3. Return a built `ModelSX` or `ModelMX` instance
4. Document parameter units and expected ranges
5. Consider adding example usage in notebooks/
