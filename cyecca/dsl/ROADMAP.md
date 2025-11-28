# Cyecca DSL - Modelica Conformance Roadmap

## Executive Summary

This document provides an analysis of how the current Cyecca DSL implementation aligns with the Modelica Language Specification v3.7-dev and outlines a roadmap for future development.

**Current Status**: The DSL implements approximately **50-55%** of core Modelica specification features, covering:
- Basic ODE modeling with `der()` operator
- Discrete variability with `pre()`, `edge()`, `change()` operators
- Blocks with causality enforcement (`@block` decorator)
- User-defined functions (`@function` decorator)
- Relational operators (`<`, `<=`, `>`, `>=`, `==`, `!=`)
- Boolean operators (`and_()`, `or_()`, `not_()`)
- Conditional expressions (`if_then_else()`)
- Algorithm sections with local variables
- Protected visibility for internal variables
- Array indexing and derivatives

---

## Part 1: Current Implementation Status

### Implemented Features ✅

| Feature | MLS Chapter | Status | Notes |
|---------|-------------|--------|-------|
| Variable declaration | Ch. 4.4 | ✅ Complete | `var()` with all variability flags |
| Predefined types | Ch. 4.9 | ✅ Partial | `DType.REAL`, `DType.INTEGER`, `DType.BOOLEAN` |
| Variable attributes | Ch. 4.9 | ✅ Complete | `start`, `fixed`, `min`, `max`, `nominal`, `unit` |
| `der()` operator | Ch. 3.7.4 | ✅ Complete | Time derivative of variables and arrays |
| `pre()` operator | Ch. 3.7.5 | ✅ Complete | Previous value of discrete variable |
| `edge()` operator | Ch. 3.7.5 | ✅ Complete | Rising edge detection |
| `change()` operator | Ch. 3.7.5 | ✅ Complete | Value change detection |
| Relational operators | Ch. 3.5 | ✅ Complete | `<`, `<=`, `>`, `>=`, `==`, `!=` |
| Boolean operators | Ch. 3.5 | ✅ Complete | `and_()`, `or_()`, `not_()` functions |
| If-expressions | Ch. 3.6.5 | ✅ Complete | `if_then_else()` function |
| Math functions | Ch. 3.7.3 | ✅ Complete | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sqrt`, `exp`, `log`, `log10`, `abs`, `sign`, `floor`, `ceil`, `sinh`, `cosh`, `tanh`, `min`, `max` |
| Equations | Ch. 8.3 | ✅ Complete | ODE/DAE equations via `yield` syntax |
| Initial equations | Ch. 8.6 | ✅ Complete | `initial_equations()` method |
| Algorithm sections | Ch. 11 | ✅ Complete | `algorithm()` with `local()` and `@` operator |
| User-defined functions | Ch. 12 | ✅ Complete | `@function` decorator |
| Model flattening | Ch. 5.6 | ✅ Complete | Hierarchical flattening with prefixes |
| Submodels | Ch. 4.6 | ✅ Complete | Via `submodel()` function |
| Automatic classification | Ch. 4.5 | ✅ Complete | State/algebraic/input/output classification |
| Block class | Ch. 4.7 | ✅ Complete | `@block` decorator with causality enforcement |
| Protected visibility | Ch. 4.1 | ✅ Complete | `protected=True` for internal variables |
| Array indexing | Ch. 10 | ✅ Partial | Basic `x[i]` indexing, `der(x[i])` |

### Test Coverage

| Module | Coverage | Tests |
|--------|----------|-------|
| `__init__.py` | 100% | - |
| `backends/__init__.py` | 100% | - |
| `operators.py` | 100% | `test/dsl/test_operators.py` |
| `types.py` | 99% | `test/dsl/test_types.py` |
| `model.py` | 92% | `test/dsl/test_model.py` |
| `simulation.py` | 91% | `test/dsl/test_simulation.py` |
| `backends/casadi.py` | 64% | `test/dsl/test_backends_casadi.py` |
| **Total** | **85%** | `test/dsl/test_integration.py` |

---

## Part 2: Feature Examples

### Basic Model with Equations

```python
from cyecca.dsl import model, var, der

@model
class MassSpringDamper:
    """Mass-spring-damper system."""
    m = var(1.0, parameter=True, unit="kg")
    k = var(100.0, parameter=True, unit="N/m")
    c = var(10.0, parameter=True, unit="N*s/m")
    x = var(start=1.0, unit="m")
    v = var(start=0.0, unit="m/s")
    
    def equations(m):
        yield der(m.x) == m.v
        yield m.m * der(m.v) == -m.k * m.x - m.c * m.v
```

### Block with Input/Output Causality

```python
from cyecca.dsl import block, var, der, if_then_else

@block
class PIDController:
    """PID controller block with anti-windup."""
    Kp = var(1.0, parameter=True)
    Ki = var(0.1, parameter=True)
    Kd = var(0.01, parameter=True)
    u_max = var(10.0, parameter=True)
    
    error = var(input=True)
    output = var(output=True)
    integral = var(start=0.0, protected=True)
    
    def equations(m):
        yield der(m.integral) == m.error
        raw = m.Kp * m.error + m.Ki * m.integral + m.Kd * der(m.error)
        yield m.output == if_then_else(
            raw > m.u_max, m.u_max,
            if_then_else(raw < -m.u_max, -m.u_max, raw)
        )
```

### User-Defined Function

```python
from cyecca.dsl import function, var, sqrt

@function
class QuadraticSolver:
    """Solve ax² + bx + c = 0."""
    a = var(input=True)
    b = var(input=True)
    c = var(input=True)
    x1 = var(output=True)
    x2 = var(output=True)
    discriminant = var(protected=True)

    def algorithm(f):
        yield f.discriminant @ sqrt(f.b**2 - 4*f.a*f.c)
        yield f.x1 @ ((-f.b + f.discriminant) / (2*f.a))
        yield f.x2 @ ((-f.b - f.discriminant) / (2*f.a))
```

### Discrete Events

```python
from cyecca.dsl import model, var, der, pre, edge, change
from cyecca.dsl.types import DType

@model
class Counter:
    """Event counter using discrete operators."""
    trigger = var(dtype=DType.BOOLEAN)
    count = var(dtype=DType.INTEGER, discrete=True, start=0)
    prev_count = var(dtype=DType.INTEGER)
    
    def equations(m):
        yield m.prev_count == pre(m.count)
        # Note: actual increment would need when-clause (not yet implemented)
```

### Boolean and Conditional Logic

```python
from cyecca.dsl import model, var, and_, or_, not_, if_then_else
from cyecca.dsl.types import DType

@model
class Thermostat:
    """Thermostat with hysteresis."""
    T = var(input=True, unit="degC")
    T_set = var(20.0, parameter=True)
    hysteresis = var(2.0, parameter=True)
    heater_on = var(dtype=DType.BOOLEAN, output=True)
    in_range = var(dtype=DType.BOOLEAN)
    
    def equations(m):
        yield m.in_range == and_(m.T > m.T_set - m.hysteresis,
                                  m.T < m.T_set + m.hysteresis)
        yield m.heater_on == (m.T < m.T_set)
```

---

## Part 3: Planned Features (Not Yet Implemented)

### High Priority

| Feature | MLS Chapter | Complexity | Notes |
|---------|-------------|------------|-------|
| Connectors & Connections | Ch. 9 | High | Physical ports with flow/effort |
| Inheritance (extends) | Ch. 7.1 | Medium | Class inheritance |
| Modifications | Ch. 7.2 | Medium | Override inherited values |
| When-clauses | Ch. 8.5 | High | Event handling |
| reinit() | Ch. 8.5 | Medium | State reinitialization |

### Medium Priority

| Feature | MLS Chapter | Complexity | Notes |
|---------|-------------|------------|-------|
| Full array support | Ch. 10 | High | Slicing, matrix ops |
| Packages | Ch. 13 | Medium | Namespace organization |
| Balanced model checking | Ch. 4.8 | Medium | Equation counting |
| Unit checking | Ch. 4.9 | Medium | Dimensional analysis |
| If-equations | Ch. 8.3.4 | Medium | Conditional equations |

### Low Priority

| Feature | MLS Chapter | Complexity | Notes |
|---------|-------------|------------|-------|
| Stream connectors | Ch. 15 | High | Thermo-fluid modeling |
| Synchronous semantics | Ch. 16 | Very High | Clocked systems |
| State machines | Ch. 17 | High | Discrete state logic |
| Annotations | Ch. 18 | Medium | Metadata |
| Overloaded operators | Ch. 14 | Medium | Custom operator definitions |

---

## Part 4: Target Syntax for Planned Features

### Connectors (High Priority)

```python
@connector
class ElectricalPin:
    """Electrical connector with potential and flow."""
    v = var(unit="V")                    # Potential variable
    i = var(flow=True, unit="A")         # Flow variable

@model
class Resistor:
    R = var(100.0, parameter=True, unit="Ohm")
    p = submodel(ElectricalPin)
    n = submodel(ElectricalPin)
    
    def equations(m):
        yield m.p.v - m.n.v == m.R * m.p.i
    
    def connections(m):
        yield connect(m.p, m.n)  # Generates flow sum = 0
```

### Inheritance

```python
@model
class PartialMechanical:
    """Base class for mechanical components."""
    x = var(unit="m")
    v = var(unit="m/s")
    F = var(unit="N")
    
    def equations(m):
        yield der(m.x) == m.v

@model
class Mass(extends=PartialMechanical):
    """Point mass."""
    m = var(1.0, parameter=True, unit="kg")
    
    def equations(m):
        yield m.m * der(m.v) == m.F
```

### Initial Equations

```python
@model
class Pendulum:
    theta = var(unit="rad")
    omega = var(unit="rad/s")
    
    def equations(m):
        yield der(m.theta) == m.omega
        yield der(m.omega) == -9.81 * sin(m.theta)
    
    def initial_equations(m):
        yield m.theta == 0.5
        yield m.omega == 0.0
```

### When-Clauses (Hybrid Systems)

```python
@model
class BouncingBall:
    h = var(start=1.0, unit="m")
    v = var(start=0.0, unit="m/s")
    e = var(0.8, parameter=True)  # Restitution coefficient
    
    def equations(m):
        yield der(m.h) == m.v
        yield der(m.v) == -9.81
        
        when(m.h < 0):
            yield reinit(m.v, -m.e * pre(m.v))
```

---

## Part 5: Architecture Notes

### Expression Tree

The DSL uses an immutable expression tree (`Expr` dataclass) with operation kinds defined in `ExprKind` enum:

- **Leaf nodes**: `VARIABLE`, `DERIVATIVE`, `CONSTANT`, `TIME`
- **Arithmetic**: `NEG`, `ADD`, `SUB`, `MUL`, `DIV`, `POW`
- **Relational**: `LT`, `LE`, `GT`, `GE`, `EQ`, `NE`
- **Boolean**: `AND`, `OR`, `NOT`
- **Control flow**: `IF_THEN_ELSE`
- **Discrete**: `PRE`, `EDGE`, `CHANGE`
- **Array**: `INDEX`
- **Functions**: `SIN`, `COS`, `TAN`, `ASIN`, `ACOS`, `ATAN`, `ATAN2`, `SQRT`, `EXP`, `LOG`, `ABS`

### Backend System

- `Backend` abstract base class in `simulation.py`
- `CasADiBackend` implementation using CasADi for symbolic math
- Future: JAX backend for GPU acceleration

### Model Processing Pipeline

1. **Model definition** via `@model`, `@block`, or `@function` decorator
2. **Variable collection** from class attributes
3. **Equation generation** from `equations()` and `algorithm()` methods
4. **Flattening** via `flatten()` to produce `FlatModel`
5. **Backend compilation** to produce executable simulation code

---

## Changelog

- **2024-11**: Initial DSL implementation with basic ODE support
- **2024-12**: Added discrete operators (`pre`, `edge`, `change`)
- **2025-01**: Added `@block` decorator, boolean/relational operators
- **2025-02**: Added `@function` decorator, algorithm sections
- **2025-11**: Streamlined test suite, 85% coverage achieved
- **2025-11**: Added math functions: `log10`, `sign`, `floor`, `ceil`, `sinh`, `cosh`, `tanh`, `min`, `max`
- **2025-11**: Implemented initial equations (`initial_equations()` method per Modelica Spec Section 8.6)
