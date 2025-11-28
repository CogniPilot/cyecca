# Cyecca DSL - Modelica Conformance Roadmap

## Executive Summary

This document provides an analysis of how the current Cyecca DSL implementation aligns with the Modelica Language Specification v3.7-dev and outlines a roadmap for future development.

**Current Status**: The DSL implements approximately **55-60%** of core Modelica specification features, covering:
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
- **When-clauses for hybrid systems** (`when()`, `reinit()`)
- **@equations decorator** for clean, yield-free equation syntax

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
| Equations | Ch. 8.3 | ✅ Complete | `@equations` decorator with auto-capture |
| Initial equations | Ch. 8.6 | ✅ Complete | `initial_equations()` method |
| **When-clauses** | **Ch. 8.5** | **✅ Complete** | **`when()` context manager, `reinit()` for hybrid systems** |
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
| `when_clauses` | - | `test/dsl/test_when_clauses.py` |
| **Total** | **85%** | `test/dsl/test_integration.py` |

---

## Part 2: Feature Examples

### Basic Model with @equations Decorator

```python
from cyecca.dsl import model, var, der, equations

@model
class MassSpringDamper:
    """Mass-spring-damper system."""
    m = var(1.0, parameter=True, unit="kg")
    k = var(100.0, parameter=True, unit="N/m")
    c = var(10.0, parameter=True, unit="N*s/m")
    x = var(start=1.0, unit="m")
    v = var(start=0.0, unit="m/s")
    
    @equations
    def _(m):
        der(m.x) == m.v
        m.m * der(m.v) == -m.k * m.x - m.c * m.v
```

### Block with Input/Output Causality

```python
from cyecca.dsl import block, var, der, if_then_else, equations

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
    
    @equations
    def _(m):
        der(m.integral) == m.error
        raw = m.Kp * m.error + m.Ki * m.integral + m.Kd * der(m.error)
        m.output == if_then_else(
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

### Boolean and Conditional Logic

```python
from cyecca.dsl import model, var, and_, or_, not_, if_then_else, equations
from cyecca.dsl.types import DType

@model
class Thermostat:
    """Thermostat with hysteresis."""
    T = var(input=True, unit="degC")
    T_set = var(20.0, parameter=True)
    hysteresis = var(2.0, parameter=True)
    heater_on = var(dtype=DType.BOOLEAN, output=True)
    in_range = var(dtype=DType.BOOLEAN)
    
    @equations
    def _(m):
        m.in_range == and_(m.T > m.T_set - m.hysteresis,
                           m.T < m.T_set + m.hysteresis)
        m.heater_on == (m.T < m.T_set)
```

### When-Clauses (Hybrid Systems) ✅

```python
from cyecca.dsl import model, var, der, when, reinit, pre, equations

@model
class BouncingBall:
    h = var(start=1.0, unit="m")
    v = var(start=0.0, unit="m/s")
    e = var(0.8, parameter=True)  # Restitution coefficient
    
    @equations
    def _(m):
        # Continuous dynamics
        der(m.h) == m.v
        der(m.v) == -9.81
        
        # When-clause for bounce - auto-registered!
        with when(m.h < 0):
            reinit(m.v, -m.e * pre(m.v))
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
- **2025-01**: **Implemented when-clauses for hybrid systems** (`when()` context manager, `reinit()` operator, event-detecting simulation with zero-crossing detection)
- **2025-01**: **Migrated to `@equations` decorator syntax** - Replaced yield-based equation definitions with side-effect based `@equations` decorator. Equations are now defined using `==` operator within `@equations`-decorated methods, which auto-registers them via thread-local context. Cleaner, more Pythonic syntax inspired by Modelica.
