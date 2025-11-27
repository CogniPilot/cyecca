# ROADMAP

**DESIGN PRINCIPLE**: Single-dataclass models with typed fields and unified `Model()` wrapper.
This ensures autocomplete works, type safety is maintained, and the API is clean.

---

## API Assessment (November 2025)

### Current Strengths

| Area | Rating | Notes |
|------|--------|-------|
| Type Safety | ⭐⭐⭐⭐⭐ | Explicit dataclasses with typed fields |
| Autocomplete | ⭐⭐⭐⭐⭐ | `model.v.fieldname` pattern works perfectly with IDEs |
| API Clarity | ⭐⭐⭐⭐⭐ | Unified `Model(MyClass)` pattern for both explicit and implicit |
| Explicit API | ⭐⭐⭐⭐⭐ | Clean single-dataclass with `@explicit` decorator |
| Implicit API | ⭐⭐⭐⭐⭐ | Modelica-style with `@implicit` decorator + auto state inference |
| Composition | ⚠️ **Removed** | Was removed during API migration - needs reimplementation |
| Documentation | ⭐⭐⭐⭐ | Updated docstrings + README, runnable doctests |

### Recent Improvements (November 2025)

| Improvement | Status | Notes |
|-------------|--------|-------|
| **Unified API** | ✅ Complete | Both explicit and implicit use `Model(MyClass)` pattern |
| **Single-dataclass pattern** | ✅ Complete | Replaced 4-class pattern with single `@explicit`/`@implicit` dataclass |
| **TypedViews** | ✅ Complete | `model.v`, `model.x`, `model.u`, `model.p`, `model.y` for filtered access |
| **Built-in time** | ✅ Complete | `model.t` like Modelica - no need for time field |
| **Split fields modules** | ✅ Complete | `explicit/fields.py` and `implicit/fields.py` with paradigm-specific fields |
| **Removed legacy time()** | ✅ Complete | Legacy `time()` field removed |
| **Doctest examples** | ✅ Complete | `_doctest_examples.py` with helper models for runnable doctests |
| **Mode classification** | ✅ Complete | Correct eigenvalue analysis (stable/unstable/oscillatory) |

### ⚠️ Removed Features (Need Reimplementation)

| Feature | Old Location | Lines | Notes |
|---------|--------------|-------|-------|
| **Model Composition** | `composition.py` | 664 | `CompositionMixin`, `connect()`, `build_composed()` |
| **Composition Tests** | `test_composition_simple.py` | 426 | All composition tests removed |
| **Old Core API** | `core.py` | 1508 | Old 4-class ModelSX/ModelMX pattern |

The composition system was removed during the API migration to the new unified `Model()` pattern. 
The old system provided:
- `CompositionMixin` - Hierarchical composition for ModelSX/ModelMX
- `compose()` - Create parent models from submodels  
- `add_submodel()` - Add subsystems to a parent
- `connect()` - Route signals between submodels (signal-flow)
- `build_composed()` - Build unified dynamics from composed system
- `SignalRef` / `SubmodelProxy` - Autocomplete-friendly connection helpers

### Remaining Bugs

| Bug | Location | Severity | Effort |
|-----|----------|----------|--------|
| `ModelMX.create()` signature mismatch | `core.py:1048` | Medium | 5 min |
| `from_vec()` poor dict error message | `decorators.py:170` | Low | 5 min |
| `_build_eval_args` returns wrong format | `core.py:900` | Medium | 15 min |

### API Examples

#### Explicit ODE API (New Unified Pattern)
```python
from cyecca.dynamics.explicit import Model, explicit, state, input_var, param, output_var

@explicit
class MassSpringDamper:
    # States
    x: float = state(desc="position")
    v: float = state(desc="velocity")
    # Inputs
    F: float = input_var(desc="force")
    # Parameters
    m: float = param(default=1.0)
    k: float = param(default=1.0)
    c: float = param(default=0.1)
    # Outputs
    position: float = output_var()

model = Model(MassSpringDamper)

# Define dynamics using model.v (unified namespace)
model.ode(model.v.x, model.v.v)
model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)
model.output(model.v.position, model.v.x)
model.build()

# Simulate
model.v0.x = 1.0
t, data = model.simulate(0.0, 10.0, 0.01)
```

#### Implicit DAE API (Modelica-style)
```python
from cyecca.dynamics.implicit import Model, implicit, var, param
import casadi as ca

@implicit
class Pendulum:
    theta: float = var()   # Becomes state (has .dot() in equations)
    omega: float = var()   # Becomes state (has .dot() in equations)
    l: float = param(default=1.0)
    g: float = param(default=9.81)

model = Model(Pendulum)

# Define equations - states auto-inferred from .dot() usage
model.eq(model.v.theta.dot() - model.v.omega)
model.eq(model.v.omega.dot() + model.v.g/model.v.l * ca.sin(model.v.theta))
model.build()

# Simulate
model.v0.theta = 0.5
t, data = model.simulate(0.0, 10.0, 0.01)
```

---

## Implementation Roadmap

### ✅ Completed Tasks

- [x] **Unified Model() API** ✅ - Both explicit and implicit use same pattern
- [x] **Single-dataclass pattern** ✅ - Replaced verbose 4-class pattern
- [x] **TypedViews (model.v, model.x, etc.)** ✅ - Filtered access with autocomplete
- [x] **Built-in time (model.t)** ✅ - Like Modelica, no time field needed
- [x] **Split fields modules** ✅ - Separate explicit/implicit field types
- [x] **Remove legacy time() field** ✅ - Cleaned up legacy code
- [x] **Doctest examples module** ✅ - Runnable documentation examples
- [x] **Mode classification fix** ✅ - Correct stable/unstable/oscillatory detection
- [x] **README updates** ✅ - Updated Quick Start with new API
- [x] **Implicit DAE API** ✅ - Modelica-style equation-based modeling

---

### Priority 1: Backend Abstraction Layer (High Priority, ~8-12 hours)

Isolate the symbolic math backend (currently CasADi) behind an abstract interface to enable
swapping in alternative backends like JAX, PyTorch, or SymPy.

- [ ] **Design backend interface** (2 hours)
  - Define abstract base class with core operations:
    - Symbol creation: `sym()`, `sym_struct()`
    - Matrix operations: `vertcat()`, `horzcat()`, `jacobian()`
    - Function creation: `function()`, `integrator()`
    - Differentiation: `jacobian()`, `gradient()`, `hessian()`
    - Optimization: `nlpsol()` (optional)
  - Design backend selection mechanism: `Model(MyClass, backend='jax')`

- [ ] **Audit current CasADi usage** (1 hour)
  - Catalog all CasADi imports and direct usage
  - Identify which features are essential vs optional
  - Group by: core symbolic, integration, optimization

- [ ] **Create backend module structure** (1 hour)
  - Location: `cyecca/backends/`
  - Files: `base.py` (ABC), `casadi_backend.py`, `jax_backend.py`
  - Registry pattern for backend discovery

- [ ] **Implement CasADi backend** (2 hours)
  - Wrap current CasADi calls in backend interface
  - This is the "default" backend, should be feature-complete
  - Zero behavior change - just indirection

- [ ] **Update Model classes to use backend** (2 hours)
  - Inject backend into `explicit/model.py`, `implicit/model.py`
  - Replace direct `ca.` calls with `backend.` calls
  - Update integrators to use backend

- [ ] **Implement JAX backend (MVP)** (2 hours)
  - Basic symbol creation and differentiation
  - jax.numpy for matrix operations
  - Integrate with existing models as proof-of-concept

- [ ] **Add backend tests** (2 hours)
  - Test same model with CasADi and JAX backends
  - Verify numerical equivalence

### Priority 2: Reimplement Composition (~8-12 hours)

The composition system was removed during the API migration and needs to be reimplemented
to work with the new `Model()` API.

- [ ] **Design new composition API** (2 hours)
  - How should `Model(MyClass)` instances compose?
  - Should we use `parent.add_submodel("name", model)` pattern?
  - How to expose submodel vars for connection: `parent.plant.v.x` or `parent["plant"].v.x`?

- [ ] **Implement CompositionMixin for new Model class** (4 hours)
  - Location: `explicit/composition.py` (new file)
  - Methods: `add_submodel()`, `connect()`, `build_composed()`
  - Support signal-flow connections between models
  
- [ ] **Add composition tests** (2 hours)
  - Port relevant tests from old `test_composition_simple.py`
  - Test simple 2-model plant+controller composition
  
- [ ] **Update documentation** (2 hours)
  - Add composition examples to README
  - Document new composition API

### Priority 3: Bug Fixes (Easy, ~25 min total)

- [ ] **Fix `ModelMX.create()` signature** (5 min)
  - Location: `core.py:1048-1065`
  - Issue: Doesn't accept `output_type` as positional argument like `ModelSX.create()`
  - Fix: Update signature to `create(state_type, input_type, param_type, output_type=None, **kwargs)`

- [ ] **Improve `from_vec()` error message** (5 min)
  - Location: `decorators.py:170`
  - Issue: `from_vec() received dict with multiple outputs` is unhelpful
  - Fix: Show available keys and suggest correct usage

- [ ] **Fix `_build_eval_args` return type** (15 min)
  - Location: `core.py:900`
  - Issue: Returns `(inputs, names)` when `symbolic=False` should return just `inputs`
  - Fix: Clean up the logic, add test

---

### Priority 4: API Ergonomics (Easy-Medium, ~1.5 hours)

- [ ] **Add shape inference from default** (45 min)
  - Location: `explicit/fields.py`, `implicit/fields.py`
  - Impact: Less redundancy in field definitions
  - Implementation:
    ```python
    def state(shape=None, default=0.0, desc=""):
        if shape is None:
            if isinstance(default, (list, tuple, np.ndarray)):
                shape = len(default)
            else:
                shape = 1
        return VarDescriptor(shape=shape, default=default, desc=desc, var_type="state")
    ```

- [ ] **Improve connection error messages** (45 min)
  - Location: `composition.py:_resolve_connection()`
  - Impact: Better debugging experience
  - Implementation: On connection failure, show available paths

---

### Priority 5: Testing & Documentation (Medium, ~2 hours)

- [ ] **Add composition tests for new API** (1 hour)
  - Location: `test/test_composition_new_api.py`
  - Test composing models using new `Model(MyClass)` pattern

- [ ] **Tutorial: Getting Started with Cyecca** (1 hour)
  - Location: `docs/tutorials/getting_started.md`
  - Step-by-step guide for new users

---

### Priority 5: Advanced Composition Features (Medium-Hard, ~8 hours)

- [ ] **Discrete State Composition** (3 hours)
  - Location: `composition.py::build_composed()`
  - Issue: Only `x` composed; `z`, `m`, `z_alg` ignored
  - Implementation: Add `_build_composed_discrete()` method

- [ ] **Dependent Variable Re-evaluation** (2 hours)
  - Location: `composition.py::build_composed()`
  - Issue: `f_dep` not re-evaluated after connection resolution
  - Implementation: Add dep evaluation step after connections

- [ ] **State Connection Implementation** (3 hours)
  - Location: `composition.py`
  - Issue: `_state_connections` stored but never used
  - Implementation: Add state aliasing or algebraic constraints

---

### Priority 6: Advanced DAE Features (Hard, ~20+ hours)

- [ ] **Acausal Connection Mode (Phase 1b)** (8 hours)
  - Add `mode='acausal'` to `connect()`
  - Generate algebraic constraints
  - Force IDAS integrator

- [ ] **Symbolic Elimination (Phase 2)** (12 hours)
  - Eliminate redundant variables from equality constraints
  - Reduce system dimension

- [ ] **Initialization Solver** (8 hours)
  - Solve for consistent DAE initial conditions
  - Use CasADi NLP solver

- [ ] **Index Reduction (Pantelides)** (20 hours)
  - Automatic constraint differentiation
  - Transform to index-1 DAE

---

### Backlog (Future Considerations)

- [ ] Algebraic loop resolution (Newton iteration)
- [ ] Event priority and coordination
- [ ] RK4 substep output evaluation
- [ ] Cross-connection algebraic constraints
- [ ] Automatic causalization (BLT decomposition)

---

## Technical Reference

### Current Capabilities

✅ **Working Features**:
- **Unified API**: `Model(MyClass)` pattern for both explicit and implicit
- **Single-dataclass models**: `@explicit` and `@implicit` decorators with typed fields
- **TypedViews**: `model.v`, `model.x`, `model.u`, `model.p`, `model.y` for filtered access
- **Built-in time**: `model.t` like Modelica
- **Auto state inference**: Implicit models infer states from `.dot()` usage
- **Separate field modules**: `explicit/fields.py` and `implicit/fields.py`
- RK4/Euler/IDAS integrators
- Linearization and mode analysis
- Clean typed representation of semi-explicit DAE models

### ⚠️ Removed (Needs Reimplementation)

❌ **Composition System Removed**:
- `CompositionMixin` - hierarchical model composition
- `add_submodel()` - adding subsystems
- `connect()` - signal routing between models
- `build_composed()` - building unified dynamics
- Block-concatenated state vector construction
- Connections from parent inputs and submodel states/outputs
- Quadrature composition
- Event function aggregation  
- Parameter connections
- Algebraic loop detection

### Missing for Full Modelica DAE Composition

❌ **Not Yet Implemented** (requires composition first):
- Equation-level DAE stacking (`f_alg_composed`) + IDAS at composed level
- State connections to create aliases or algebraic constraints
- Algebraic loop resolution
- Global handling of algebraic vars, discrete states across submodels

### Mathematical Summary

**Single Model (Current)**:
```
Explicit: ẋ = f(x, u, p, t),  y = g(x, u, p, t)
Implicit: F(ẋ, x, z, p, t) = 0  (states inferred from .dot() usage)
```

**Target (Causal ODE composition)** - needs reimplementation:
```
ẋ = [f_x1(x1, u1(x,y,u_parent), p1);
     f_x2(x2, u2(x,y,u_parent), p2);
     ...]
     
where u_i are computed by explicit assignment from connections
```

**Future Target (Modelica-style acausal DAE)**:
```
F_composed(ẋ, x, z, w, p, t) = [F1(ẋ1, x1, z1, w1, p1, t);
                                  F2(ẋ2, x2, z2, w2, p2, t);
                                  C_connections(w)] = 0
                                  
where:
- w are interface variables (no input/output distinction)
- C_connections are equality constraints from connect() statements
- Solver determines causality from full equation structure
```

---

### Testing Requirements

For each implemented feature:
- [x] Create unit tests with simple 2-model composition
- [x] Test against known analytical solutions
- [ ] Add integration tests for complex scenarios
- [ ] Benchmark performance impact
- [x] Update documentation and examples

### Documentation Updates

- [x] Update docstrings to reflect actual DAE support level
- [ ] Add composition tutorial with examples
- [x] Document limitations (algebraic loops, index, etc.)
- [ ] Create migration guide for new features
- [ ] Add troubleshooting section for common errors

---

## Current Test Status

| Test Suite | Status | Count |
|------------|--------|-------|
| Cyecca unit tests | ✅ Passing | 268 |
| Cyecca doctests | ✅ Passing | 20 (1 skipped) |
| Cubs2 tests | ✅ Passing | 7 (1 skipped) |

---

## Appendix: Acausal Modeling Deep Dive

### Why Acausal Modeling Is Hard

Acausal modeling creates equations like:
```
Model 1: 10 equations, 15 unknowns (under-determined)
Model 2: 8 equations, 12 unknowns (under-determined)  
Connections: 7 equations (constraints)
---
Total: 25 equations, 27 unknowns (still under-determined!)
```

The framework must:
1. **Identify which unknowns should be states (x)** vs algebraic (z) vs inputs (u)
2. **Structurally analyze** the equation system to determine causality
3. **Reduce** the system by eliminating algebraic variables where possible
4. **Index reduction** if the DAE is higher than index-1
5. **Partition** into semi-explicit form for the integrator

**Modelica compilers do this automatically** through:
- Symbolic manipulation (solve equations for derivatives)
- Matching algorithms (BLT decomposition, Tarjan's algorithm)
- Differentiation of algebraic constraints (index reduction)
- Tearing (select algebraic variables to iterate on)

### Proposed Hybrid Approach

The framework should support **three composition modes**:

1. **Explicit Mode** (current, fastest):
   ```python
   # New unified API
   model = Model(MassSpringDamper)
   model.ode(model.v.x, model.v.v)
   model.build()
   
   # Composition
   parent.connect("ctrl.u.pos", "plant.y.pos")  # Explicit signal flow
   parent.build_composed(mode='explicit', integrator='rk4')
   ```

2. **Implicit Mode - Simple** (Phase 1):
   ```python
   model = Model(Pendulum)  # @implicit decorated class
   model.eq(model.v.theta.dot() - model.v.omega)
   model.build()
   
   # Composition with acausal connections
   parent.connect("ctrl.pos", "plant.pos", mode='acausal')
   parent.build_composed(mode='implicit', integrator='idas')
   # Adds algebraic constraint: 0 = ctrl.pos - plant.pos
   ```

3. **Implicit Mode - Advanced** (Phase 2-3):
   ```python
   parent.connect("motor.shaft", "wheel.shaft", mode='acausal')
   parent.build_composed(mode='implicit_reduced', integrator='idas')
   # Performs symbolic manipulation to reduce system dimension
   ```

---

*Analysis last updated: November 27, 2025*
