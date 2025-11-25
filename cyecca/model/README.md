# Type-Safe Modeling Framework

The `cyecca.model` module provides a declarative API for building dynamical systems with full type safety and IDE autocomplete support.

## Key Features

- **Type-Safe State Definitions**: Define states, inputs, and parameters as typed dataclasses
- **Full Autocomplete**: IDE autocomplete for all signals (states, inputs, parameters, outputs)
- **Hybrid Systems Support**: Continuous dynamics, discrete events, algebraic constraints
- **Multiple Integrators**: RK4, Euler, IDAS (DAE), with configurable substeps
- **Hierarchical Composition**: Build complex systems from simpler subsystems
- **Event Detection**: Zero-crossing detection for hybrid dynamics

---

## Feature Implementation Status

This section tracks the implementation status of DAE (Differential-Algebraic Equation) features in the modeling framework.

### ✅ Fully Implemented

#### Continuous States (x)
- **Status:** ✅ Complete
- **Description:** `dx/dt = f_x(x, u, p, ...)`
- **API:** `state()` field creator, `f_x` in `build()`
- **Integrators:** RK4, Euler (working), IDAS (stub)
- **Usage:**
  ```python
  import casadi as ca
  from cyecca.model import state, symbolic
  
  @symbolic
  class States:
      x: ca.SX = state(1, 0.0, "position (m)")
      v: ca.SX = state(1, 0.0, "velocity (m/s)")
  
  # model.build(f_x=ca.vertcat(x.v, (u.F - p.c * x.v) / p.m))
  ```

#### Inputs (u)
- **Status:** ✅ Complete
- **Description:** Control signals, external forcing
- **API:** `input_var()` field creator
- **Usage:**
  ```python
  import casadi as ca
  from cyecca.model import input_var, symbolic
  
  @symbolic
  class Inputs:
      F: ca.SX = input_var(desc="force (N)")
  ```

#### Parameters (p)
- **Status:** ✅ Complete
- **Description:** Time-independent constants
- **API:** `param()` field creator
- **Usage:**
  ```python
  import casadi as ca
  from cyecca.model import param, symbolic
  
  @symbolic
  class Params:
      m: float = param(default=1.0, desc="mass (kg)")
      c: float = param(default=0.1, desc="damping (Ns/m)")
  ```

#### Outputs (y)
- **Status:** ✅ Complete
- **Description:** Observables/diagnostics `y = f_y(x, u, p)`
- **API:** `output_var()` field creator, `f_y` in `build()`
- **Usage:**
  ```python
  import casadi as ca
  from cyecca.model import output_var, symbolic
  
  @symbolic
  class Outputs:
      energy: ca.SX = output_var(desc="total energy (J)")
  
  # model.build(f_x=f_x, f_y=0.5 * p.m * x.v**2)
  ```

#### Quadrature States (q)
- **Status:** ✅ Complete
- **Description:** Path integrals `dq/dt = f_q(x, u, p)`
- **API:** `quadrature_var()` field creator, `f_q` in `build()`
- **Integration:** RK4/Euler during `simulate()`
- **Usage:**
  ```python
  import casadi as ca
  from cyecca.model import quadrature_var, symbolic
  
  @symbolic
  class Quadratures:
      cost: ca.SX = quadrature_var(desc="accumulated cost")
  
  # model.build(f_x=f_x, f_q=u.F**2)  # Minimize control effort
  ```

#### Discrete States (z)
- **Status:** ✅ Complete (event-triggered updates)
- **Description:** Discrete states updated at zero-crossings `z⁺ = f_z(...)`
- **API:** `discrete_state()` field creator, `f_z` in `build()`
- **Event Detection:** Via `detect_events=True` in `simulate()`
- **Usage:**
  ```python
  import casadi as ca
  from cyecca.model import discrete_state, symbolic
  
  @symbolic
  class DiscreteStates:
      mode: ca.SX = discrete_state(default=0, desc="flight mode")
  
  # model.build(f_x=f_x, f_z=new_mode, f_c=height_indicator)
  # result = model.simulate(t0, tf, dt, detect_events=True)
  ```

#### Event Indicators (c)
- **Status:** ✅ Complete
- **Description:** Event detection when `c(x, u, p)` crosses zero
- **API:** `event_indicator()` field creator, `f_c` in `build()`
- **Detection:** Zero-crossing during integration
- **Usage:**
  ```python
  import casadi as ca
  from cyecca.model import event_indicator, symbolic
  
  @symbolic
  class EventIndicators:
      ground_contact: ca.SX = event_indicator(desc="z=0 detector")
  
  # model.build(f_x=f_x, f_c=x.z)  # Trigger when z crosses 0
  ```

#### Discrete Variables (m)
- **Status:** ✅ Complete
- **Description:** Discrete variables (integers/booleans) or state resets `m⁺ = f_m(...)`
- **API:** `discrete_var()` field creator, `f_m` in `build()`
- **Dual Purpose:** Can reset discrete vars OR continuous states based on output dimension
- **Usage:**
  ```python
  import casadi as ca
  from cyecca.model import discrete_var, symbolic
  
  @symbolic
  class DiscreteVars:
      bounce_count: ca.SX = discrete_var(default=0, desc="bounces")
  
  # # Update discrete variable at event
  # model.build(f_x=f_x, f_m=m.bounce_count + 1, f_c=height)
  # 
  # # OR reset continuous states (bouncing ball)
  # model.build(f_x=f_x, f_m=ca.vertcat(x.h, -0.8*x.v), f_c=height)
  ```

#### Hierarchical Composition
- **Status:** ✅ Complete
- **Description:** Build systems from interconnected submodels
- **API:** `add_submodel()`, `connect()`, `build_composed()`
- **Usage:**
  ```python
  from cyecca.model import ModelSX
  
  # parent = ModelSX.compose({"plant": plant_model, "ctrl": controller})
  # parent.connect(parent.ctrl.outputs.u, parent.plant.inputs.thrust)
  # parent.build_composed(f_x_composed, integrator='rk4')
  ```

### ⚠️ Partially Implemented

#### Dependent Variables (dep)
- **Status:** ⚠️ Partial (API exists, integration incomplete)
- **Description:** Computed quantities `dep = f_dep(x, u, p)` that feed into dynamics
- **API:** `dependent_var()` field creator, `f_dep` in `build()`
- **Current State:**
  - API and function building works (`_build_f_dep`)
  - Passed to `f_x` in function signatures
  - **NOT** automatically evaluated in integrators (uses default values)
  - Would need evaluation before each `f_x` call for full support
- **Limitation:**
  ```python
  # This works for building f_x:
  dep.lift = 0.5 * p.rho * dep.airspeed**2 * p.S * p.CL
  f_x = ca.vertcat(..., dep.lift / p.m, ...)
  
  # But during integration, dep uses dep0 defaults, not f_dep evaluation
  # For dynamic dep, compute directly in f_x instead
  ```
- **Workaround:** Compute dependent values directly in `f_x` expression rather than as separate `dep` variables

#### Algebraic Variables (z_alg)
- **Status:** ⚠️ Stub only (API exists, solver not implemented)
- **Description:** Algebraic constraints `0 = g(x, z_alg, u, p)` for DAE systems
- **API:** `algebraic_var()` field creator, `f_alg` in `build()`
- **Current State:**
  - API exists and function builds
  - IDAS integrator is stubbed (falls back to RK4)
  - No algebraic equation solver integrated
- **Limitation:**
  ```python
  import casadi as ca
  from cyecca.model import algebraic_var, symbolic
  
  # This API works but doesn't solve the constraint:
  @symbolic
  class Algebraic:
      lambda_constraint: ca.SX = algebraic_var(desc="Lagrange multiplier")
  
  # model.build(f_x=f_x, f_alg=constraint_residual, integrator='idas')
  # Falls back to RK4, z_alg not solved!
  ```
- **Roadmap:** Needs IDAS/SUNDIALS integration for proper DAE solving

### ❌ Not Implemented

#### Advanced Event Handling
- **Status:** ❌ Not implemented
- **Missing Features:**
  - Multiple simultaneous events
  - Event priority/ordering
  - Event localization (bisection to find exact crossing time)
  - Chattering prevention
- **Current:** Simple sign-change detection between timesteps

#### Symbolic Differentiation for Jacobians
- **Status:** ❌ Not implemented
- **Missing:** Automatic Jacobian computation for stiff systems
- **Impact:** RK4/Euler may be inefficient for stiff ODEs
- **Workaround:** Use linearization tools for local analysis

#### Time-Varying Parameters
- **Status:** ❌ Not implemented
- **Current:** Parameters are constant throughout simulation
- **Workaround:** Model as inputs with `u_func(t, x, p)`

#### Delay Differential Equations
- **Status:** ❌ Not implemented
- **Missing:** `x(t-τ)` terms not supported

#### Partial Differential Equations
- **Status:** ❌ Not supported
- **Scope:** Framework is for ODEs/DAEs only

### Integration Methods

| Method | Status | Use Case |
|--------|--------|----------|
| **RK4** | ✅ Complete | General purpose, good accuracy |
| **Euler** | ✅ Complete | Simple systems, fast prototyping |
| **IDAS** | ✅ Complete (ODEs only) | Adaptive timestep ODEs with tight tolerances |
| **RK8** | ✅ Complete | High-precision trajectories |
| **Implicit** | ❌ Not implemented | Stiff systems (planned) |
| **IDAS (DAE)** | ⚠️ Partial | Full DAE support needs f_x refactoring |

### Recommended Usage Patterns

#### ✅ Well-Supported Workflows
- Continuous ODE systems with inputs and parameters
- Hybrid systems with discrete events (bouncing ball, mode switches)
- Hierarchical model composition (plant + controller)
- Quadrature integration (cost functionals, energy)
- Output observation and logging
- Trim finding and linearization

#### ⚠️ Limited Support
- DAE systems with algebraic constraints (use ODE reformulation instead)
- Systems with complex dependent variable chains (inline computations in `f_x`)

#### ❌ Not Recommended
- Stiff systems requiring implicit integration
- Systems with time delays
- PDE systems
- High-frequency discrete events (use continuous approximations)

### Future Roadmap

#### Short Term
1. Complete dependent variable evaluation in integrators
2. Add event localization (bisection)
3. Document composition patterns better

#### Medium Term
1. IDAS integration for DAE support
2. Implicit integrators (BDF, Radau)
3. Adaptive timestepping

#### Long Term
1. Automatic Jacobian computation
2. Symbolic sensitivity analysis
3. Code generation optimization

### Testing Coverage

| Feature | Unit Tests | Integration Tests | Examples |
|---------|------------|-------------------|----------|
| Continuous states | ✅ | ✅ | ✅ |
| Inputs/Outputs | ✅ | ✅ | ✅ |
| Parameters | ✅ | ✅ | ✅ |
| Quadratures | ✅ | ✅ | ⚠️ |
| Discrete states | ✅ | ✅ | ⚠️ |
| Event indicators | ✅ | ✅ | ✅ (bouncing ball) |
| Composition | ✅ | ✅ | ⚠️ |
| Dependent vars | ⚠️ | ❌ | ❌ |
| Algebraic vars | ⚠️ | ❌ | ❌ |

### Version History

- **v0.1.0** (Current): Core ODE, events, composition, linearization
- **Planned v0.2.0**: Full DAE support, implicit integrators
- **Planned v0.3.0**: Advanced event handling, sensitivity analysis

---

## System Components

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
    find_trim,                  # Find trim/equilibrium points
    linearize_dynamics,         # Linearize dynamics around operating point
    analyze_modes,              # Modal analysis (eigenvalues, stability)
)
```

## Quick Start: Mass-Spring-Damper

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
    k: ca.SX = param(10.0, "spring constant (N/m)")
    c: ca.SX = param(0.1, "damping (Ns/m)")

# Create model with full IDE autocomplete
model = ModelSX.create(States, Inputs, Params)

# Access components with autocomplete
x, u, p = model.x, model.u, model.p

# Define continuous dynamics: dx/dt = f(x, u, p)
f_x = ca.vertcat(
    x.v,                                    # dx/dt = v
    (u.F - p.k * x.x - p.c * x.v) / p.m    # dv/dt = (F - kx - cv) / m
)

# Build with RK4 integrator
model.build(f_x=f_x, integrator='rk4', integrator_options={'N': 4})

# Simulate
result = model.simulate(t0=0.0, tf=10.0, dt=0.01)

# Access results
import matplotlib.pyplot as plt
plt.plot(result['t'], result['x'][0, :], label='position')
plt.plot(result['t'], result['x'][1, :], label='velocity')
plt.legend()
plt.show()
```

## Hybrid Systems: Bouncing Ball

Systems with discrete events and state resets:

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

## DAE Systems with Algebraic Constraints

For systems with algebraic constraints (index-1 DAE):

```python
from cyecca.model import ModelSX, state, algebraic_var, param, symbolic
import casadi as ca

@symbolic
class States:
    x: ca.SX = state(1, 1.0, "x position")
    y: ca.SX = state(1, 0.0, "y position")
    vx: ca.SX = state(1, 0.0, "x velocity")
    vy: ca.SX = state(1, 1.0, "y velocity")

@symbolic
class AlgebraicVars:
    lambda_: ca.SX = algebraic_var(1, "constraint force")

@symbolic
class Params:
    L: ca.SX = param(1.0, "pendulum length")
    g: ca.SX = param(9.81, "gravity")

@symbolic
class Inputs:
    pass  # No inputs for this example

model = ModelSX.create(
    States, Inputs, Params,
    algebraic_type=AlgebraicVars
)

x = model.x
z_alg = model.z_alg
p = model.p

# Differential equations
f_x = ca.vertcat(
    x.vx,
    x.vy,
    -2 * x.x * z_alg.lambda_,
    -2 * x.y * z_alg.lambda_ - p.g
)

# Algebraic constraint: x^2 + y^2 = L^2
f_alg = x.x**2 + x.y**2 - p.L**2

model.build(f_x=f_x, f_alg=f_alg, integrator='idas')

result = model.simulate(0.0, 10.0, 0.01)
```

## Quadrature Integration

Track integrated quantities during simulation:

```python
from cyecca.model import ModelSX, state, quadrature_var, param, symbolic
import casadi as ca

@symbolic
class States:
    x: ca.SX = state(1, 0.0, "position")
    v: ca.SX = state(1, 1.0, "velocity")

@symbolic
class QuadratureVars:
    energy: ca.SX = quadrature_var(1, "total energy")

@symbolic
class Params:
    m: ca.SX = param(1.0, "mass")

model = ModelSX.create(
    States, None, Params,
    quadrature_var_type=QuadratureVars
)

x = model.x
p = model.p

# Dynamics
f_x = ca.vertcat(x.v, -x.x)  # Harmonic oscillator

# Quadrature: integrate kinetic energy
f_quad = 0.5 * p.m * x.v**2

model.build(f_x=f_x, f_quad=f_quad, integrator='rk4')

result = model.simulate(0.0, 10.0, 0.01)
print(f"Total energy dissipated: {result['quad'][0, -1]}")
```

## Hierarchical Composition

Build complex systems from simpler subsystems:

```python
from cyecca.model import ModelSX, state, input_var, output_var, param, symbolic
import casadi as ca

# Define plant
@symbolic
class PlantStates:
    theta: ca.SX = state(1, 0.0, "pitch angle")
    q: ca.SX = state(1, 0.0, "pitch rate")

@symbolic
class PlantInputs:
    elevator: ca.SX = input_var(desc="elevator deflection")

@symbolic
class PlantOutputs:
    theta_meas: ca.SX = output_var(desc="measured pitch")

@symbolic
class PlantParams:
    M_q: ca.SX = param(-5.0, "pitch damping")
    M_delta: ca.SX = param(2.0, "elevator effectiveness")

plant = ModelSX.create(PlantStates, PlantInputs, PlantParams, output_type=PlantOutputs)

# Plant dynamics
f_x_plant = ca.vertcat(
    plant.x.q,
    plant.p.M_q * plant.x.q + plant.p.M_delta * plant.u.elevator
)
f_y_plant = plant.x.theta

plant.build(f_x=f_x_plant, f_y=f_y_plant, integrator='rk4')

# Define controller
@symbolic
class CtrlStates:
    pass  # No states (proportional controller)

@symbolic
class CtrlInputs:
    theta_meas: ca.SX = input_var(desc="measured pitch")
    theta_cmd: ca.SX = input_var(desc="commanded pitch")

@symbolic
class CtrlOutputs:
    elevator_cmd: ca.SX = output_var(desc="elevator command")

@symbolic
class CtrlParams:
    K_p: ca.SX = param(1.0, "proportional gain")

controller = ModelSX.create(CtrlStates, CtrlInputs, CtrlParams, output_type=CtrlOutputs)

# Controller logic
f_y_ctrl = controller.p.K_p * (controller.u.theta_cmd - controller.u.theta_meas)

controller.build(f_y=f_y_ctrl)

# Compose into closed-loop system
closed_loop = ModelSX.compose({
    "plant": plant,
    "controller": controller
})

# Connect signals (with autocomplete!)
closed_loop.connect(controller.u.theta_meas, plant.y.theta_meas)
closed_loop.connect(plant.u.elevator, controller.y.elevator_cmd)

# Build composed system
closed_loop.build_composed(integrator='rk4')

# Simulate with commanded pitch
u_fn = lambda t: ca.DM([0.1])  # 0.1 rad command
result = closed_loop.simulate(0.0, 10.0, 0.01, u=u_fn)
```

## Model Types

### ModelSX vs ModelMX

- **ModelSX**: Uses CasADi SX (scalar expressions)
  - Faster for small systems (<100 states)
  - Better for symbolic simplification
  - Automatic differentiation via expression graph

- **ModelMX**: Uses CasADi MX (matrix expressions)
  - Better for large systems (>100 states)
  - More efficient memory usage
  - Supports sparse Jacobians

```python
from cyecca.model import ModelSX, ModelMX

# For small systems
small_model = ModelSX.create(States, Inputs, Params)

# For large systems
large_model = ModelMX.create(States, Inputs, Params)
```

## Integrator Options

### RK4 (Runge-Kutta 4th order)

```python
model.build(
    f_x=f_x,
    integrator='rk4',
    integrator_options={'N': 4}  # Number of substeps per dt
)
```

### Euler (Forward Euler)

```python
model.build(
    f_x=f_x,
    integrator='euler',
    integrator_options={'N': 10}  # More substeps for accuracy
)
```

### IDAS (DAE solver for algebraic constraints)

```python
model.build(
    f_x=f_x,
    f_alg=f_alg,  # Algebraic constraints
    integrator='idas'
)
```

## Linearization and Trim

Find equilibrium points and linearize dynamics for stability analysis:

```python
from cyecca.model import find_trim, linearize_dynamics, analyze_modes

# Find trim/equilibrium point
def cost_fn(model, x, u, p, x_dot):
    # Minimize state derivatives at desired operating point
    return ca.sumsqr(x_dot.as_vec())

def constraints_fn(model, x, u, p, x_dot):
    # Constrain specific variables (e.g., airspeed = 15 m/s)
    return [x.v == 15.0]

x_trim, u_trim, stats = find_trim(
    model,
    x_guess=model.x0,
    u_guess=model.u0,
    cost_fn=cost_fn,
    constraints_fn=constraints_fn,
    verbose=True
)

# Linearize around trim point
A, B = linearize_dynamics(model, x_trim, u_trim)

# Analyze eigenvalues and stability
state_names = ["x", "v", "theta", "omega"]
modes = analyze_modes(A, state_names=state_names)

for mode in modes:
    print(f"Eigenvalue: {mode['eigenvalue']:.4f}")
    print(f"Stable: {mode['stable']}")
    if mode['is_oscillatory']:
        print(f"Frequency: {mode['frequency_hz']:.2f} Hz")
        print(f"Period: {mode['period']:.2f} s")
        print(f"Damping ratio: {mode['damping_ratio']:.3f}")
    print(f"Time constant: {mode['time_constant']:.2f} s")
    if 'dominant_states' in mode:
        print(f"Dominant states: {mode['dominant_states']}")
```

## Pre-built Models

See `cyecca/models/` for complete examples:

- **quadrotor.py**: Quadrotor dynamics with full 6-DOF
- **fixedwing.py**: Fixed-wing aircraft model
- **fixedwing_4ch.py**: 4-channel RC aircraft
- **bezier.py**: Bezier trajectory generation
- **rdd2.py**: Differential drive robot
- **rdd2_loglinear.py**: Loglinear differential drive

## See Also

- Main cyecca documentation: [../README.md](../README.md)
- Lie groups for rigid body dynamics: [lie/README.md](lie/README.md)
- Jupyter notebooks: `notebook/sim/` for simulation examples
