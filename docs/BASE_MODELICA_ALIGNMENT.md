# Cyecca IR Alignment with Base Modelica

## Overview

Cyecca's internal representation (IR) is aligned with **Base Modelica** ([MCP-0031](https://github.com/modelica/ModelicaSpecification/blob/MCP/0031/RationaleMCP/0031/)), a simplified intermediate representation designed for tool interoperability.

This document explains:
1. Why we chose Base Modelica
2. How Cyecca IR maps to Base Modelica
3. Lie group extensions
4. Import/export workflow

## Why Base Modelica?

Base Modelica is designed as a **compilation target** and **interchange format** with:

- **Simplification**: Flattened, no hierarchical components, no connect equations
- **DAE-Ready**: Clear separation of constants, parameters, and variables
- **Standardization**: Tool-agnostic format for the Modelica ecosystem
- **Extensibility**: Supports tool-specific annotations via metadata

This matches Cyecca's needs perfectly:
- Cyecca is a **backend**, not a frontend compiler
- We need a flattened, solver-ready representation
- We want to interoperate with other Modelica tools
- We need to add Lie group annotations for manifold-aware code generation

## Architecture

```
┌─────────────────────────┐
│  Modelica Source (.mo)  │
└───────────┬─────────────┘
            │
            ↓
    ┌───────────────┐
    │     Rumoca    │  Parsing, flattening,
    │   Compiler    │  connect expansion
    └───────┬───────┘
            │
            ↓
┌─────────────────────────────────┐
│  Base Modelica JSON             │
│  (MCP-0031 + Lie annotations)   │  ← Standard interchange format
└───────────┬─────────────────────┘
            │
            ↓  import_base_modelica()
    ┌───────────────┐
    │  Cyecca IR    │  Python dataclasses
    │  (Python)     │  Variable, Equation, Model
    └───────┬───────┘
            │
            ↓  CasadiBackend, SympyBackend, etc.
┌─────────────────────────┐
│  Code Generation &      │
│  Analysis               │
└─────────────────────────┘
```

## Cyecca IR to Base Modelica Mapping

### Variables

**Base Modelica JSON:**
```json
{
  "constants": [
    {"name": "g", "type": "Real", "value": 9.81, "unit": "m/s^2"}
  ],
  "parameters": [
    {"name": "m", "type": "Real", "value": 1.0, "unit": "kg"}
  ],
  "variables": [
    {"name": "x", "type": "Real", "variability": "continuous", "unit": "m"},
    {"name": "v", "type": "Real", "variability": "continuous", "unit": "m/s"}
  ]
}
```

**Cyecca IR (Python):**
```python
from cyecca.ir import Variable, VariableType, Variability, PrimitiveType

# Constant
g = Variable(
    name="g",
    var_type=VariableType.CONSTANT,
    primitive_type=PrimitiveType.REAL,
    value=9.81,
    unit="m/s^2"
)

# Parameter
m = Variable(
    name="m",
    var_type=VariableType.PARAMETER,
    primitive_type=PrimitiveType.REAL,
    value=1.0,
    unit="kg"
)

# State variable
x = Variable(
    name="x",
    var_type=VariableType.STATE,
    primitive_type=PrimitiveType.REAL,
    variability=Variability.CONTINUOUS,
    unit="m"
)
```

**Mapping:**
| Base Modelica | Cyecca VariableType | Cyecca Variability |
|---------------|---------------------|-------------------|
| constant      | CONSTANT            | CONSTANT          |
| parameter     | PARAMETER           | FIXED             |
| variable (continuous) | STATE or ALGEBRAIC | CONTINUOUS |
| variable (discrete) | DISCRETE_STATE | DISCRETE |

### Equations

**Base Modelica JSON:**
```json
{
  "equations": [
    {
      "eq_type": "simple",
      "lhs": {"op": "der", "args": [{"op": "var", "name": "x"}]},
      "rhs": {"op": "var", "name": "v"}
    },
    {
      "eq_type": "when",
      "condition": {"op": "<", "args": [{"op": "var", "name": "h"}, {"op": "literal", "value": 0}]},
      "statements": [
        {
          "stmt": "reinit",
          "target": "v",
          "expr": {"op": "*", "args": [{"op": "var", "name": "v"}, {"op": "literal", "value": -0.7}]}
        }
      ]
    }
  ]
}
```

**Cyecca IR (Python):**
```python
from cyecca.ir import Equation, EquationType, Expr, Statement

# Simple equation: der(x) = v
eq1 = Equation(
    eq_type=EquationType.SIMPLE,
    lhs=Expr.der(Expr.var_ref("x")),
    rhs=Expr.var_ref("v")
)

# When equation: when h < 0 then reinit(v, v * -0.7) end
from cyecca.ir import ReinitStatement
eq2 = Equation(
    eq_type=EquationType.WHEN,
    condition=Expr.binary_op("<", Expr.var_ref("h"), Expr.literal(0)),
    when_equations=[
        # Note: In Base Modelica, when-equations contain statements
        # In Cyecca, we may need to adapt this representation
    ]
)
```

**Equation Types Mapping:**
| Base Modelica | Cyecca EquationType |
|---------------|---------------------|
| simple        | SIMPLE              |
| for           | FOR                 |
| if (balanced) | IF                  |
| when          | WHEN                |

**Note:** Base Modelica does NOT support:
- `connect` equations (must be expanded before export)
- Unbalanced if-equations (else branch required)
- Separate `Event` objects (uses when-equations instead)

### Expressions

**Base Modelica JSON:**
```json
{
  "op": "+",
  "args": [
    {"op": "var", "name": "x"},
    {"op": "literal", "value": 1.0}
  ]
}
```

**Cyecca IR (Python):**
```python
from cyecca.ir import Expr

expr = Expr.binary_op("+", Expr.var_ref("x"), Expr.literal(1.0))
```

**Operator Mapping:**
| Base Modelica | Cyecca Expr Method |
|---------------|-------------------|
| `literal`     | `Expr.literal(value)` |
| `var`         | `Expr.var_ref(name)` |
| `+`, `-`, `*`, `/` | `Expr.binary_op(op, lhs, rhs)` |
| `der`         | `Expr.der(arg)` or `der(var)` helper |
| `pre`         | `Expr.pre(arg)` or `pre(var)` helper |
| `sin`, `cos`, etc. | `Expr.function_call(func, args)` |

## Lie Group Extensions

Base Modelica supports tool-specific extensions via annotations. Cyecca adds **Lie group metadata** for manifold-aware code generation.

### Variable-Level Annotations

**Base Modelica JSON:**
```json
{
  "name": "q",
  "type": "Real",
  "dimensions": [4],
  "annotations": {
    "lie_group": "SO3",
    "manifold_chart": "quaternion",
    "nominal": 1.0
  }
}
```

**Cyecca IR (Python):**
```python
q = Variable(
    name="q",
    var_type=VariableType.STATE,
    primitive_type=PrimitiveType.REAL,
    shape=[4],
    lie_group_type="SO3",
    manifold_chart="quaternion",
    nominal=1.0
)
```

### Model-Level Metadata

**Base Modelica JSON:**
```json
{
  "metadata": {
    "lie_groups": {
      "position": {
        "type": "R3",
        "variables": ["x", "y", "z"]
      },
      "orientation": {
        "type": "SO3",
        "chart": "quaternion",
        "variables": ["q[1]", "q[2]", "q[3]", "q[4]"]
      },
      "velocity": {
        "type": "se3_algebra",
        "variables": ["v", "omega"]
      }
    }
  }
}
```

**Cyecca IR (Python):**
```python
model = Model(
    name="RigidBody",
    metadata={
        "lie_groups": {
            "position": {
                "type": "R3",
                "variables": ["x", "y", "z"]
            },
            "orientation": {
                "type": "SO3",
                "chart": "quaternion",
                "variables": ["q[1]", "q[2]", "q[3]", "q[4]"]
            },
            "velocity": {
                "type": "se3_algebra",
                "variables": ["v", "omega"]
            }
        }
    }
)
```

### Supported Lie Groups

| Group | Description | Dimension | Chart Options |
|-------|-------------|-----------|---------------|
| `R3` | Euclidean 3-space | 3 | `standard` |
| `SO3` | 3D rotations | 3 (tangent) | `quaternion`, `exp`, `cayley` |
| `SE3` | Rigid body transformations | 6 | `exp`, `cayley` |
| `SE23` | Extended poses (position, velocity, orientation) | 9 | `left_invariant`, `right_invariant` |

## Import/Export Workflow

### Importing from Base Modelica JSON

```python
from cyecca.io.base_modelica import import_base_modelica

# Import model from JSON file
model = import_base_modelica("model.json")

# Access variables
print(f"States: {[v.name for v in model.states]}")
print(f"Parameters: {[v.name for v in model.parameters]}")

# Access Lie group metadata
if "lie_groups" in model.metadata:
    for name, group_info in model.metadata["lie_groups"].items():
        print(f"{name}: {group_info['type']}")
```

### Exporting to Base Modelica JSON

```python
from cyecca.io.base_modelica import export_base_modelica

# Export model to JSON file
export_base_modelica(model, "output.json", validate=True)
```

### Validation

```python
from cyecca.io.base_modelica import validate_base_modelica
import json

# Load and validate JSON against Base Modelica schema
with open("model.json") as f:
    data = json.load(f)

errors = validate_base_modelica(data)
if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✓ Valid Base Modelica JSON")
```

## Differences from Full Modelica IR

Cyecca's IR is simpler than the full `modelica_ir-0.2.0` schema because:

| Feature | Base Modelica | Full Modelica IR |
|---------|---------------|------------------|
| Connect equations | ❌ (must be expanded) | ✅ Supported |
| Unbalanced if-equations | ❌ (else required) | ✅ Supported |
| Event objects | ❌ (use when-equations) | ✅ Separate Event type |
| Algorithm sections | ✅ | ✅ |
| Hierarchical components | ❌ (flat names) | ✅ ComponentRef |
| Variable separation | ✅ (constants/params/vars) | ❌ (unified list) |

## Benefits of Base Modelica Alignment

1. **Tool Interoperability**
   - Any compiler that exports Base Modelica can work with Cyecca
   - Cyecca models can be imported by other Base Modelica tools

2. **Standardization**
   - Aligned with MCP-0031 emerging standard
   - Reduces fragmentation in Modelica ecosystem

3. **Simplicity**
   - Flattened representation ideal for code generation
   - Clear variable categorization (constants/parameters/variables)

4. **Future-Proof**
   - As Base Modelica becomes standard, Cyecca is ready
   - Easy to extend with domain-specific annotations (Lie groups)

## References

- [MCP-0031: Base Modelica Specification](https://github.com/modelica/ModelicaSpecification/blob/MCP/0031/RationaleMCP/0031/)
- [Base Modelica JSON Schema](../modelica_ir/schemas/base_modelica_ir-0.1.0.schema.json)
- [Cyecca IR Documentation](ir/README.md)
- [Rumoca Compiler](https://github.com/CogniPilot/rumoca)
