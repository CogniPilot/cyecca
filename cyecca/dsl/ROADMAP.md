# Cyecca DSL - Modelica Conformance Roadmap

## Executive Summary

This document provides a thorough analysis of how the current Cyecca DSL implementation deviates from the Modelica Language Specification v3.7-dev and outlines a detailed roadmap to achieve full Modelica-like functionality in Python.

**Current Status**: The DSL implements approximately **40-45%** of the Modelica specification, covering basic ODE modeling, discrete variability, event operators (pre/edge/change), blocks with causality enforcement, protected visibility, relational/boolean operators, conditional expressions (if_then_else), algorithm sections with local variables, and user-defined functions. Major features like connectors, inheritance, hybrid systems, and DAE support are still planned.

---

## Part 1: Current Implementation Analysis

### What's Implemented ✅

| Feature | MLS Chapter | Status | Notes |
|---------|-------------|--------|-------|
| Variable declaration | Ch. 4.4 | ✅ Partial | `var()` with `parameter`, `input`, `output`, `constant`, `discrete`, `protected` flags |
| Predefined types | Ch. 4.9 | ✅ Partial | `DType.REAL`, `DType.INTEGER`, `DType.BOOLEAN` |
| Variable attributes | Ch. 4.9 | ✅ Partial | `start`, `fixed`, `min`, `max`, `nominal`, `unit` |
| `der()` operator | Ch. 3.7.4 | ✅ Basic | Time derivative of variables |
| `pre()` operator | Ch. 3.7.5 | ✅ Basic | Previous value of discrete variable |
| `edge()` operator | Ch. 3.7.5 | ✅ Basic | Rising edge detection |
| `change()` operator | Ch. 3.7.5 | ✅ Basic | Value change detection |
| Relational operators | Ch. 3.5 | ✅ Complete | `<`, `<=`, `>`, `>=` via Python operators |
| Boolean operators | Ch. 3.5 | ✅ Complete | `and_()`, `or_()`, `not_()` functions |
| If-expressions | Ch. 3.6.5 | ✅ Complete | `if_then_else()` function |
| Math functions | Ch. 3.7.3 | ✅ Partial | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sqrt`, `exp`, `log`, `abs` |
| Equations | Ch. 8.3 | ✅ Basic | ODE equations via `yield` syntax |
| Algorithm sections | Ch. 11 | ✅ Basic | `algorithm()` method with `local()` and `@` assignment operator |
| User-defined functions | Ch. 12 | ✅ Basic | `@function` decorator (restricted model with algorithm only) |
| Model flattening | Ch. 5.6 | ✅ Basic | Simple hierarchical flattening |
| Submodels | Ch. 4.6 | ✅ Basic | Via `submodel()` function |
| Automatic classification | Ch. 4.5 | ✅ Basic | State vs algebraic based on `der()` usage |
| Block class | Ch. 4.7 | ✅ Complete | `@block` decorator with causality enforcement |
| Protected visibility | Ch. 4.1 | ✅ Complete | `protected=True` for internal variables |

### What's Missing ❌

| Feature | MLS Chapter | Priority | Complexity |
|---------|-------------|----------|------------|
| Connectors & Connections | Ch. 9 | **Critical** | High |
| Inheritance (extends) | Ch. 7.1 | **High** | Medium |
| Modifications | Ch. 7.2 | **High** | Medium |
| Initial equations | Ch. 8.6 | **High** | Medium |
| Conditional equations | Ch. 8.3.4 | **High** | Medium |
| Arrays | Ch. 10 | **High** | High |
| Events (`when`) | Ch. 8.5 | **Medium** | High |
| Hybrid systems | Ch. 8.5 | **Medium** | High |
| Redeclaration | Ch. 7.3 | **Medium** | High |
| Packages | Ch. 13 | **Medium** | Medium |
| Stream connectors | Ch. 15 | **Medium** | High |
| Synchronous semantics | Ch. 16 | **Low** | Very High |
| State machines | Ch. 17 | **Low** | High |
| Annotations | Ch. 18 | **Low** | Medium |
| Overloaded operators | Ch. 14 | **Low** | Medium |

---

## Part 2: Detailed Gap Analysis

### Chapter 3: Operators and Expressions

#### 3.5 Relational and Boolean Operators ✅ Complete
**Current**: Fully implemented  
**Operators**: `<`, `<=`, `>`, `>=` (via Python operators), `and_()`, `or_()`, `not_()` (functions)

```python
# Implemented syntax
@model
class ThermostatModel:
    T = var()
    heater_on = var(dtype=DType.BOOLEAN)
    in_range = var(dtype=DType.BOOLEAN)
    
    def equations(m):
        yield m.heater_on == (m.T < 20.0)  # Relational → Boolean
        yield m.in_range == and_(m.T > 10, m.T < 30)  # Boolean AND
```

**Note**: Python's `and`, `or`, `not` are keywords and cannot be overloaded.
We use `and_()`, `or_()`, `not_()` functions following PEP 8 convention.

#### 3.6.5 If-Expressions ✅ Complete
**Current**: Implemented via `if_then_else()` function  

```python
# Implemented syntax
def equations(m):
    yield der(m.x) == if_then_else(m.x > 0, m.a, m.b)
    
    # Nested for saturation
    yield m.y == if_then_else(
        m.u > m.limit,
        m.limit,
        if_then_else(m.u < -m.limit, -m.limit, m.u)
    )
```

#### 3.7 Built-in Functions - Missing Items
| Function | Status | Notes |
|----------|--------|-------|
| `sign(x)` | ❌ | Sign function |
| `floor(x)` | ❌ | Floor function |
| `ceil(x)` | ❌ | Ceiling function |
| `mod(x, y)` | ❌ | Modulo |
| `rem(x, y)` | ❌ | Remainder |
| `div(x, y)` | ❌ | Integer division |
| `sinh`, `cosh`, `tanh` | ❌ | Hyperbolic functions |
| `log10(x)` | ❌ | Base-10 logarithm |
| `cross(x, y)` | ❌ | Cross product (arrays) |
| `transpose(A)` | ❌ | Matrix transpose |
| `min(x, y)`, `max(x, y)` | ❌ | Element-wise min/max |
| `noEvent(expr)` | ❌ | Event suppression |
| `smooth(p, expr)` | ❌ | Smooth approximation |
| `sample(start, interval)` | ❌ | Sampling function |
| `pre(x)` | ✅ | Previous value (discrete) |
| `edge(b)` | ✅ | Rising edge |
| `change(v)` | ✅ | Value change detection |
| `reinit(x, expr)` | ❌ | Reinitialize state |

### Chapter 4: Classes and Declarations

#### 4.5 Component Variability ✅ Complete
**Current**: `constant`, `parameter`, `discrete`, `input`, `output` flags  
**Implemented**:
- **Discrete variability**: Variables that only change at events via `discrete=True`
- **Protected visibility**: Internal variables via `protected=True`

```python
# Discrete variable
mode = var(dtype=DType.INTEGER, discrete=True)  # Changes only at events

# Protected variable (internal implementation)
integral = var(start=0.0, protected=True)  # Not part of public interface
```

#### 4.6 Class Declarations ⚠️ Partial
**Current**: `@model` and `@block` decorators  
**Implemented**:
- `@model` (general dynamic system)
- `@block` (signal-flow with input/output causality enforcement)

**Missing**:
- `record` (data-only class)
- `type` (alias)
- `connector` (interface definition)
- `function` (algorithmic)
- `package` (namespace)

```python
# Implemented
@block
class PIDController:
    u = var(input=True)
    y = var(output=True)
    Kp = var(1.0, parameter=True)
    integral = var(start=0.0, protected=True)  # Internal state
    # ... block enforces causality

# Target syntax (future)
@record
class Position:
    x = var()
    y = var()
    z = var()

@connector
class FluidPort:
    p = var()           # Pressure (effort/potential)
    m_flow = var(flow=True)  # Mass flow rate (flow variable)
```

#### 4.8 Balanced Models ❌
**Current**: No equation counting  
**Required**: Check that number of equations equals number of unknowns

### Chapter 5: Scoping, Name Lookup, and Flattening

#### 5.3 Static Name Lookup ⚠️ Basic
**Current**: Simple dot-notation for submodels  
**Missing**:
- Qualified names (`Modelica.SIunits.Velocity`)
- Import statements
- Enclosing class access

#### 5.6 Flattening Process ⚠️ Basic
**Current**: Simple hierarchical flattening  
**Missing**:
- Array expansion
- Conditional component handling
- Proper modification merging

### Chapter 7: Inheritance, Modification, Redeclaration

#### 7.1 Inheritance (extends) ❌ **CRITICAL**
**Current**: Not implemented  
**Required**: Class inheritance with `extends`

```python
# Target syntax
@model
class PartialMass:
    m = var(parameter=True)
    a = var()
    F = var()
    
    def equations(m):
        yield m.F == m.m * m.a

@model
class FreeFallingBody(extends=PartialMass):
    g = var(9.81, parameter=True)
    v = var()
    s = var()
    
    def equations(m):
        yield m.a == -m.g
        yield der(m.v) == m.a
        yield der(m.s) == m.v
```

#### 7.2 Modifications ❌ **CRITICAL**
**Current**: Not implemented  
**Required**: Modify inherited/composed components

```python
# Target syntax
@model
class Aircraft:
    engine = submodel(Engine, P_max=500e3)  # Modification
    wing = submodel(Wing(span=10.0, chord=1.5))  # Modification
```

#### 7.3 Redeclaration ❌
**Current**: Not implemented  
**Required**: Replace inherited components with different types

### Chapter 8: Equations

#### 8.1 Equation Categories ⚠️ Partial
**Current**: Only continuous-time equations  
**Missing**:
- Initial equations (`initial equation`)
- Discrete equations (in `when` clauses)
- Conditional equations (`if` equations)

#### 8.3.4 If-Equations ❌
```python
# Target syntax
def equations(m):
    if m.use_linear:
        yield m.y == m.k * m.x
    else:
        yield m.y == m.k * sin(m.x)
```

#### 8.5 Events and Synchronization ❌ **CRITICAL for hybrid systems**
**Current**: Not implemented  
**Required**: `when` clauses for discrete events

```python
# Target syntax
def equations(m):
    yield der(m.h) == m.v
    yield der(m.v) == -m.g
    
    # Event: bouncing ball
    when(m.h < 0):
        yield reinit(m.v, -m.e * pre(m.v))
```

#### 8.6 Initialization ❌ **HIGH PRIORITY**
**Current**: Only `start`/`fixed` attributes, no initial equations  
**Required**: Full initialization support

```python
# Target syntax
@model
class Pendulum:
    theta = var(start=0.5, fixed=True)  # OK - current
    omega = var()
    
    # Missing: initial equations section
    def initial_equations(m):
        yield m.omega == 0  # Initial condition
```

### Chapter 9: Connectors and Connections ❌ **CRITICAL**

This is the **most significant gap** for physical system modeling.

#### 9.1 Connector Definition
```python
# Target syntax
@connector
class Pin:
    """Electrical pin connector."""
    v = var(unit="V")        # Potential variable (effort)
    i = var(flow=True, unit="A")  # Flow variable
    
@connector  
class Flange:
    """Mechanical rotational flange."""
    phi = var(unit="rad")    # Angle (potential)
    tau = var(flow=True, unit="N.m")  # Torque (flow)

@connector
class FluidPort:
    """Thermo-fluid port."""
    p = var(unit="Pa")                    # Pressure (potential)
    m_flow = var(flow=True, unit="kg/s")  # Mass flow (flow)
    h_outflow = var(stream=True, unit="J/kg")  # Specific enthalpy (stream)
```

#### 9.2 Connection Equations
```python
# Target syntax
@model
class Circuit:
    R1 = submodel(Resistor(R=100))
    R2 = submodel(Resistor(R=200))
    V = submodel(VoltageSource(V=12))
    G = submodel(Ground)
    
    def connections(m):
        yield connect(m.V.p, m.R1.p)
        yield connect(m.R1.n, m.R2.p)
        yield connect(m.R2.n, m.G.p)
        yield connect(m.G.p, m.V.n)
```

**Connection semantics**:
- Effort variables: All connected = (equality)
- Flow variables: Sum = 0 (Kirchhoff's law)

### Chapter 10: Arrays ❌ **HIGH PRIORITY**

#### Current Limitation
Only scalar variables with `size` attribute (not properly handled)

#### Required Features
```python
# Target syntax
@model
class MultiBody:
    n = var(3, dtype=DType.INTEGER, parameter=True)  # Number of bodies
    
    # Array declarations
    pos = var(size=(n, 3), unit="m")      # n×3 position matrix
    vel = var(size=(n, 3), unit="m/s")    # n×3 velocity matrix
    mass = var(size=n, parameter=True)    # n masses
    
    def equations(m):
        for i in range(m.n):
            yield der(m.pos[i, :]) == m.vel[i, :]
            yield m.mass[i] * der(m.vel[i, :]) == m.F[i, :]
```

**Required array operations**:
- Indexing: `x[i]`, `x[i, j]`, `x[i, :]`
- Slicing: `x[1:5]`, `x[:, 2]`
- Element-wise operations: `+`, `-`, `*`, `/`, `^`
- Matrix operations: `*` (matrix multiply), `transpose`, `inv`
- Reduction: `sum`, `product`, `min`, `max`
- Construction: `zeros`, `ones`, `fill`, `linspace`

### Chapter 11: Statements and Algorithm Sections ✅ Basic

Algorithm sections provide imperative-style assignments for procedural computations.

```python
# Implemented syntax
@model
class Saturation:
    u = var(input=True)
    y = var(output=True)
    limit = var(5.0, parameter=True)
    
    def algorithm(m):
        temp = local("temp")          # Local variable
        yield temp @ (m.u * 2)        # Assignment via @ operator
        
        # Conditional assignment using if_then_else
        yield m.y @ if_then_else(
            temp > m.limit,
            m.limit,
            if_then_else(temp < -m.limit, -m.limit, temp)
        )
```

**Features**:
- `local(name)`: Create local algorithm variables
- `@` operator: Assignment operator (`target @ expr`)
- `assign(target, value)`: Explicit assignment function
- Integration with `if_then_else()` for conditional logic

**Limitations**:
- No `if`/`for`/`while` statement syntax (Python limitations)
- Conditional logic via `if_then_else()` expressions only

### Chapter 12: Functions ✅ Basic

Functions are restricted models that use only algorithm sections.

```python
# Implemented syntax
@function
class Quadratic:
    '''Solve ax^2 + bx + c = 0.'''
    a = var(input=True)
    b = var(input=True)
    c = var(input=True)
    x1 = var(output=True)
    x2 = var(output=True)
    d = var(protected=True)  # Intermediate variable

    def algorithm(f):
        yield f.d @ sqrt(f.b**2 - 4*f.a*f.c)
        yield f.x1 @ ((-f.b + f.d) / (2*f.a))
        yield f.x2 @ ((-f.b - f.d) / (2*f.a))

# Usage (get function metadata)
quad = Quadratic()
meta = quad.get_function_metadata()
```

**Features**:
- `@function` decorator (restricted `@model`)
- All non-parameter public variables must be `input` or `output`
- Uses `algorithm()` only (no `equations()`)
- Protected variables for intermediate calculations
- `get_function_metadata()` for introspection

**Limitations**:
- No inline function calls in model equations yet
- No automatic differentiation for functions yet

### Chapter 15: Stream Connectors ❌

For thermo-fluid modeling with reversible flow.

```python
@connector
class FluidPort:
    p = var()
    m_flow = var(flow=True)
    h_outflow = var(stream=True)  # Stream variable

# Built-in operators
def equations(m):
    # inStream(port.h_outflow) - upstream enthalpy
    yield m.Q == m.port.m_flow * inStream(m.port.h_outflow)
```

### Chapter 16: Synchronous Language Elements ❌

For sampled-data systems and embedded code generation.

```python
# Target syntax (future)
@model
class DiscretePID:
    clock = Clock(0.01)  # 100 Hz sampling
    
    u = var(input=True)
    y = var(output=True)
    
    def equations(m):
        # Clocked equation
        when(m.clock):
            e = m.u - m.y_meas
            y = m.Kp * e + m.Ki * previous(m.integral) + e * m.dt
            yield m.integral == previous(m.integral) + e * m.dt
```

### Chapter 17: State Machines ❌

```python
# Target syntax (future)
@model
class TrafficLight:
    state = var(dtype=DType.INTEGER)
    timer = var()
    
    def state_machine(m):
        with state("red"):
            entry: m.timer = 0
            transition(m.timer > 30, to="green")
            
        with state("green"):
            entry: m.timer = 0
            transition(m.timer > 25, to="yellow")
            
        with state("yellow"):
            entry: m.timer = 0
            transition(m.timer > 5, to="red")
```

---

## Part 3: Implementation Roadmap

### Phase 1: Core Language Completion (Months 1-3) 🎯

#### 1.1 Expression System Enhancement
**Effort**: 2 weeks

- [ ] Boolean operators: `and`, `or`, `not`
- [ ] Relational operators: `<`, `<=`, `>`, `>=`, `==`, `!=`
- [ ] If-then-else expressions
- [ ] Missing math functions: `sign`, `floor`, `ceil`, `mod`, `sinh`, `cosh`, `tanh`

```python
# Add to ExprKind enum
class ExprKind(Enum):
    # ... existing ...
    # Relational
    LT = auto()   # <
    LE = auto()   # <=
    GT = auto()   # >
    GE = auto()   # >=
    EQ = auto()   # ==
    NE = auto()   # !=
    # Boolean
    AND = auto()
    OR = auto()
    NOT = auto()
    # Control flow
    IF_THEN_ELSE = auto()
```

#### 1.2 Initial Equations
**Effort**: 1 week

- [ ] Add `initial_equations()` method to models
- [ ] Separate initial equation handling in FlatModel
- [ ] Backend support for initialization

```python
@model
class Pendulum:
    theta = var()
    omega = var()
    
    def equations(m):
        yield der(m.theta) == m.omega
        yield der(m.omega) == -9.81 * sin(m.theta)
    
    def initial_equations(m):
        yield m.theta == 0.5
        yield m.omega == 0.0
```

#### 1.3 Inheritance (extends)
**Effort**: 3 weeks

- [ ] `extends` parameter for `@model` decorator
- [ ] Variable/equation merging from parent
- [ ] Modification support for inherited components

```python
@model
class PartialTwoPort:
    p = var(input=True)
    n = var(output=True)
    v = var()
    i = var()
    
    def equations(m):
        yield m.v == m.p - m.n

@model
class Resistor(extends=PartialTwoPort):
    R = var(parameter=True)
    
    def equations(m):
        yield m.v == m.R * m.i
```

#### 1.4 Modifications
**Effort**: 2 weeks

- [ ] Value modifications: `submodel(Motor, J=0.5)`
- [ ] Nested modifications: `submodel(Motor, shaft(d=0.01))`
- [ ] Final modifications (non-overridable)

### Phase 2: Connectors & Physical Modeling (Months 4-6) 🔌

#### 2.1 Connector Definition
**Effort**: 3 weeks

- [ ] `@connector` decorator
- [ ] `flow=True` attribute for flow variables
- [ ] Connector type checking

```python
@connector
class Pin:
    v = var(unit="V")
    i = var(flow=True, unit="A")
```

#### 2.2 Connection Equations
**Effort**: 4 weeks

- [ ] `connect()` function
- [ ] Automatic equality equations for potential variables
- [ ] Automatic sum=0 equations for flow variables
- [ ] Connection validation

```python
def connections(m):
    yield connect(m.R1.p, m.R2.p)  # Generates: R1.p.v = R2.p.v, R1.p.i + R2.p.i = 0
```

#### 2.3 Basic Component Library
**Effort**: 2 weeks

- [ ] Electrical: `Resistor`, `Capacitor`, `Inductor`, `Ground`, `VoltageSource`
- [ ] Mechanical rotational: `Inertia`, `Spring`, `Damper`, `Fixed`
- [ ] Mechanical translational: `Mass`, `Spring1D`, `Damper1D`

### Phase 3: Arrays & Functions (Months 7-9) 📊

#### 3.1 Array Support
**Effort**: 6 weeks

- [ ] Array variable declarations
- [ ] Indexing expressions
- [ ] Slicing
- [ ] For-equation loops
- [ ] Array construction functions
- [ ] Element-wise operations
- [ ] Matrix operations

```python
@model
class NMass:
    n = var(5, dtype=DType.INTEGER, parameter=True)
    x = var(size=n)
    v = var(size=n)
    
    def equations(m):
        for i in range(m.n):
            yield der(m.x[i]) == m.v[i]
```

#### 3.2 User-Defined Functions
**Effort**: 4 weeks

- [ ] `@function` decorator
- [ ] Input/output declarations
- [ ] Algorithm section execution
- [ ] Automatic differentiation support
- [ ] Inline functions

```python
@function
def saturate(x: Real, lo: Real, hi: Real) -> Real:
    if x < lo:
        return lo
    elif x > hi:
        return hi
    else:
        return x
```

### Phase 4: Hybrid Systems (Months 10-12) ⚡

#### 4.1 When-Clauses
**Effort**: 4 weeks

- [ ] `when()` syntax for event detection
- [ ] `reinit()` for state reinitialization
- [ ] `pre()` for previous values
- [ ] `edge()` and `change()` detection

```python
def equations(m):
    yield der(m.h) == m.v
    yield der(m.v) == -m.g
    
    when(m.h < 0):
        yield reinit(m.v, -0.8 * pre(m.v))
```

#### 4.2 Event Handling
**Effort**: 4 weeks

- [ ] Zero-crossing detection in backends
- [ ] Event iteration
- [ ] Proper event semantics

#### 4.3 Discrete Variables
**Effort**: 2 weeks

- [ ] Discrete variability prefix
- [ ] Pre/post event values
- [ ] Integration with when-clauses

### Phase 5: Advanced Features (Months 13-18) 🚀

#### 5.1 Function Enhancements ✅ Partial
**Effort**: 2 weeks (remaining)

- [x] `@function` decorator
- [x] Input/output declarations
- [x] Algorithm section execution in functions
- [x] Protected variables for intermediates
- [ ] Inline function calls in model equations
- [ ] Automatic differentiation support for functions

#### 5.2 Packages & Organization
**Effort**: 3 weeks

- [ ] Package support
- [ ] Import statements
- [ ] Library structure

#### 5.3 Stream Connectors
**Effort**: 4 weeks

- [ ] `stream=True` variable attribute
- [ ] `inStream()` operator
- [ ] `actualStream()` operator
- [ ] Stream connection semantics

#### 5.4 Balanced Model Checking
**Effort**: 2 weeks

- [ ] Equation counting
- [ ] Variable counting
- [ ] Under/over-determined warnings

#### 5.5 Unit Checking
**Effort**: 3 weeks

- [ ] Unit parsing
- [ ] Unit propagation
- [ ] Dimensional analysis

### Phase 6: Production Features (Months 19-24) 🏭

#### 6.1 Synchronous Elements
**Effort**: 8 weeks

- [ ] Clock types
- [ ] Clocked equations
- [ ] Sample/hold operators
- [ ] Clock partitioning

#### 6.2 State Machines
**Effort**: 6 weeks

- [ ] State/transition syntax
- [ ] Entry/exit actions
- [ ] Hierarchical state machines

#### 6.3 Backend Enhancements
**Effort**: 8 weeks

- [ ] DAE solver support (implicit systems)
- [ ] Index reduction
- [ ] Sparse matrix support
- [ ] Code generation for embedded systems
- [ ] JAX backend

#### 6.4 Tooling
**Effort**: 4 weeks

- [ ] Model validation
- [ ] Debugging tools
- [ ] Documentation generation
- [ ] LSP support for IDE integration

---

## Part 4: Recommended Priority Order

### Immediate (Next 1-3 months)
1. **Boolean/relational operators** - Required for any conditional logic
2. **Initial equations** - Required for proper simulation initialization
3. **Missing math functions** - Low-hanging fruit, high utility
4. **If-then-else expressions** - Required for conditional models

### Short-term (3-6 months)
5. **Inheritance (extends)** - Critical for code reuse
6. **Modifications** - Required for parameterized components
7. **Connectors** - Critical for physical modeling
8. **Connection equations** - Enables multi-domain modeling

### Medium-term (6-12 months)
9. **Arrays** - Required for multi-body, FEM, etc.
10. **Functions** - Required for complex models
11. **When-clauses** - Required for hybrid systems

### Long-term (12-24 months)
13. **Stream connectors** - For thermo-fluid modeling
14. **Synchronous elements** - For embedded systems
15. **State machines** - For complex logic
16. **Advanced tooling** - For production use

---

## Part 5: Architecture Recommendations

### Expression Tree Enhancements

```python
class ExprKind(Enum):
    # Current
    VARIABLE = auto()
    DERIVATIVE = auto()
    CONSTANT = auto()
    TIME = auto()
    NEG = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    SIN = auto()
    # ... etc
    
    # Phase 1 additions
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    EQ = auto()
    NE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IF_THEN_ELSE = auto()
    SIGN = auto()
    FLOOR = auto()
    CEIL = auto()
    MOD = auto()
    
    # Phase 4 additions
    PRE = auto()
    EDGE = auto()
    CHANGE = auto()
    REINIT = auto()
    
    # Phase 5 additions
    INSTREAM = auto()
    ACTUALSTREAM = auto()
    
    # Array operations
    INDEX = auto()
    SLICE = auto()
    ARRAY_CONSTRUCT = auto()
```

### Connector System

```python
@dataclass
class ConnectorVar:
    """Variable in a connector."""
    var: Var
    is_flow: bool = False
    is_stream: bool = False

@dataclass
class Connector:
    """Connector type definition."""
    name: str
    variables: Dict[str, ConnectorVar]
    
@dataclass
class Connection:
    """A connection between two connectors."""
    from_path: str
    to_path: str
```

### Flat Model Extensions

```python
@dataclass
class FlatModel:
    # Current fields
    name: str
    state_names: List[str]
    # ... etc
    
    # New fields for full Modelica support
    initial_equations: List[Equation]  # Initial equations
    when_clauses: List[WhenClause]     # Event equations
    connections: List[Connection]       # Connector connections
    discrete_vars: Dict[str, Var]      # Discrete-time variables
    algorithms: List[Algorithm]        # Algorithm sections
```

---

## Conclusion

The current Cyecca DSL provides a solid foundation with:
- Clean Python-native syntax via decorators
- Unified `var()` API with Modelica-style flags
- Automatic state/algebraic classification
- Expression tree architecture enabling multiple backends
- Basic submodel composition

To become a full Modelica-like DSL, the highest-priority additions are:
1. **Boolean/relational operators** and **if-then-else** for conditional logic
2. **Initial equations** for proper initialization
3. **Inheritance** for code reuse
4. **Connectors** for physical modeling

These four features would cover approximately 60-70% of typical Modelica use cases and enable realistic multi-domain physical system modeling.
