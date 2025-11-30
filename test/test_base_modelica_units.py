"""
Unit tests for Base Modelica importer/exporter.

Tests individual functions for importing and exporting expressions, equations, variables, etc.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cyecca.ir import (
    Model,
    Variable,
    VariableType,
    Variability,
    PrimitiveType,
    Equation,
    EquationType,
    Expr,
    ComponentRef,
    ComponentRefPart,
)
from cyecca.io.base_modelica import (
    _import_expr,
    _export_expr,
    _import_variable,
    _export_variable,
    _import_constant,
    _export_constant,
    _import_parameter,
    _export_parameter,
    _import_equation,
    _export_equation,
    _import_primitive_type,
    _export_primitive_type,
    _import_variability,
    _export_variability,
)


# ==================== Expression Tests ====================


def test_literal_import_export():
    """Test importing and exporting literal expressions."""
    # Integer literal
    data = {"op": "literal", "value": 42}
    expr = _import_expr(data)
    assert isinstance(expr, Expr)
    exported = _export_expr(expr)
    assert exported == data

    # Float literal
    data = {"op": "literal", "value": 3.14}
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported == data

    # Boolean literal
    data = {"op": "literal", "value": True}
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported == data


def test_var_ref_import_export():
    """Test importing and exporting variable references."""
    data = {"op": "var", "name": "x"}
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported == data


def test_binary_op_import_export():
    """Test importing and exporting binary operations."""
    # Addition
    data = {"op": "+", "args": [{"op": "var", "name": "x"}, {"op": "literal", "value": 1}]}
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported == data

    # Multiplication
    data = {"op": "*", "args": [{"op": "var", "name": "m"}, {"op": "var", "name": "a"}]}
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported == data


def test_unary_op_import_export():
    """Test importing and exporting unary operations."""
    # Negation
    data = {"op": "neg", "args": [{"op": "var", "name": "x"}]}
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported == data

    # Logical not
    data = {"op": "not", "args": [{"op": "var", "name": "flag"}]}
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported == data


def test_der_import_export():
    """Test importing and exporting der() operator."""
    data = {
        "op": "der",
        "args": [{"op": "component_ref", "parts": [{"name": "x", "subscripts": []}]}],
    }
    expr = _import_expr(data)
    exported = _export_expr(expr)
    # Simple component refs inside der() export as "var"
    expected = {"op": "der", "args": [{"op": "var", "name": "x"}]}
    assert exported == expected


def test_pre_import_export():
    """Test importing and exporting pre() operator."""
    data = {
        "op": "pre",
        "args": [{"op": "component_ref", "parts": [{"name": "v", "subscripts": []}]}],
    }
    expr = _import_expr(data)
    exported = _export_expr(expr)
    # Simple component refs inside pre() export as "var"
    expected = {"op": "pre", "args": [{"op": "var", "name": "v"}]}
    assert exported == expected


def test_component_ref_import_export():
    """Test importing and exporting component references."""
    # Simple component reference (exports as "var")
    data = {"op": "component_ref", "parts": [{"name": "x", "subscripts": []}]}
    expr = _import_expr(data)
    exported = _export_expr(expr)
    # Simple component refs export as "var" for compatibility
    assert exported == {"op": "var", "name": "x"}

    # Hierarchical component reference
    data = {
        "op": "component_ref",
        "parts": [
            {"name": "vehicle", "subscripts": []},
            {"name": "wheels", "subscripts": [{"op": "literal", "value": 1}]},
            {"name": "pressure", "subscripts": []},
        ],
    }
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported == data


def test_function_call_import_export():
    """Test importing and exporting function calls."""
    data = {"op": "call", "func": "sin", "args": [{"op": "var", "name": "theta"}]}
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported == data


def test_if_expr_import_export():
    """Test importing and exporting if-expressions (ternary)."""
    data = {
        "op": "if",
        "condition": {
            "op": ">",
            "args": [{"op": "var", "name": "x"}, {"op": "literal", "value": 0}],
        },
        "then": {"op": "literal", "value": 1},
        "else": {"op": "literal", "value": -1},
    }
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported == data


# ==================== Variable Tests ====================


def test_constant_import_export():
    """Test importing and exporting constants."""
    data = {
        "name": "pi",
        "type": "Real",
        "value": 3.14159,
        "unit": "rad",
        "description": "Pi constant",
    }
    var = _import_constant(data)
    assert var.name == "pi"
    assert var.var_type == VariableType.CONSTANT
    assert var.value == 3.14159
    assert var.unit == "rad"

    exported = _export_constant(var)
    assert exported["name"] == "pi"
    assert exported["value"] == 3.14159


def test_parameter_import_export():
    """Test importing and exporting parameters."""
    data = {
        "name": "m",
        "type": "Real",
        "value": 1.0,
        "unit": "kg",
        "min": 0.0,
        "max": 100.0,
        "nominal": 1.0,
        "annotations": {"description": "Mass"},
    }
    var = _import_parameter(data)
    assert var.name == "m"
    assert var.var_type == VariableType.PARAMETER
    assert var.value == 1.0
    assert var.unit == "kg"
    assert var.min_value == 0.0
    assert var.max_value == 100.0
    assert var.nominal == 1.0

    exported = _export_parameter(var)
    assert exported["name"] == "m"
    assert exported["value"] == 1.0
    assert exported["unit"] == "kg"


def test_variable_import_export():
    """Test importing and exporting variables (states/algebraic)."""
    data = {
        "name": "x",
        "type": "Real",
        "variability": "continuous",
        "start": 1.0,
        "unit": "m",
        "min": -10.0,
        "max": 10.0,
    }
    var = _import_variable(data)
    assert var.name == "x"
    assert var.variability == Variability.CONTINUOUS
    assert var.start == 1.0
    assert var.unit == "m"

    exported = _export_variable(var)
    assert exported["name"] == "x"
    assert exported["variability"] == "continuous"
    assert exported["start"] == 1.0


def test_variable_with_lie_groups():
    """Test importing and exporting variables with Lie group annotations."""
    data = {
        "name": "q",
        "type": "Real",
        "variability": "continuous",
        "dimensions": [4],
        "annotations": {"lie_group": "SO3", "manifold_chart": "quaternion"},
    }
    var = _import_variable(data)
    assert var.name == "q"
    assert var.shape == [4]
    assert var.lie_group_type == "SO3"
    assert var.manifold_chart == "quaternion"

    exported = _export_variable(var)
    assert exported["dimensions"] == [4]
    assert exported["annotations"]["lie_group"] == "SO3"
    assert exported["annotations"]["manifold_chart"] == "quaternion"


# ==================== Type Conversion Tests ====================


def test_primitive_type_conversion():
    """Test PrimitiveType to/from Base Modelica string conversion."""
    assert _import_primitive_type("Real") == PrimitiveType.REAL
    assert _import_primitive_type("Integer") == PrimitiveType.INTEGER
    assert _import_primitive_type("Boolean") == PrimitiveType.BOOLEAN
    assert _import_primitive_type("String") == PrimitiveType.STRING

    assert _export_primitive_type(PrimitiveType.REAL) == "Real"
    assert _export_primitive_type(PrimitiveType.INTEGER) == "Integer"
    assert _export_primitive_type(PrimitiveType.BOOLEAN) == "Boolean"
    assert _export_primitive_type(PrimitiveType.STRING) == "String"


def test_variability_conversion():
    """Test Variability to/from Base Modelica string conversion."""
    assert _import_variability("constant") == Variability.CONSTANT
    assert _import_variability("fixed") == Variability.FIXED
    assert _import_variability("tunable") == Variability.TUNABLE
    assert _import_variability("discrete") == Variability.DISCRETE
    assert _import_variability("continuous") == Variability.CONTINUOUS

    assert _export_variability(Variability.CONSTANT) == "constant"
    assert _export_variability(Variability.FIXED) == "fixed"
    assert _export_variability(Variability.TUNABLE) == "tunable"
    assert _export_variability(Variability.DISCRETE) == "discrete"
    assert _export_variability(Variability.CONTINUOUS) == "continuous"


# ==================== Equation Tests ====================


def test_simple_equation_import_export():
    """Test importing and exporting simple equations."""
    # der(x) = v
    data = {
        "eq_type": "simple",
        "lhs": {
            "op": "der",
            "args": [{"op": "component_ref", "parts": [{"name": "x", "subscripts": []}]}],
        },
        "rhs": {"op": "component_ref", "parts": [{"name": "v", "subscripts": []}]},
    }
    eq = _import_equation(data)
    assert eq.eq_type == EquationType.SIMPLE

    exported = _export_equation(eq)
    assert exported["eq_type"] == "simple"


def test_for_equation_import_export():
    """Test importing and exporting for-equations."""
    data = {
        "eq_type": "for",
        "indices": [
            {
                "index": "i",
                "range": {
                    "op": "call",
                    "func": "range",
                    "args": [{"op": "literal", "value": 1}, {"op": "literal", "value": 3}],
                },
            }
        ],
        "equations": [
            {
                "eq_type": "simple",
                "lhs": {
                    "op": "component_ref",
                    "parts": [{"name": "x", "subscripts": [{"op": "var", "name": "i"}]}],
                },
                "rhs": {"op": "literal", "value": 0},
            }
        ],
    }
    eq = _import_equation(data)
    assert eq.eq_type == EquationType.FOR
    assert eq.index_var == "i"

    exported = _export_equation(eq)
    assert exported["eq_type"] == "for"
    assert exported["indices"][0]["index"] == "i"


def test_if_equation_import_export():
    """Test importing and exporting if-equations."""
    data = {
        "eq_type": "if",
        "branches": [
            {
                "condition": {
                    "op": ">",
                    "args": [{"op": "var", "name": "x"}, {"op": "literal", "value": 0}],
                },
                "equations": [
                    {
                        "eq_type": "simple",
                        "lhs": {"op": "component_ref", "parts": [{"name": "y", "subscripts": []}]},
                        "rhs": {"op": "literal", "value": 1},
                    }
                ],
            }
        ],
        "else_equations": [
            {
                "eq_type": "simple",
                "lhs": {"op": "component_ref", "parts": [{"name": "y", "subscripts": []}]},
                "rhs": {"op": "literal", "value": 0},
            }
        ],
    }
    eq = _import_equation(data)
    assert eq.eq_type == EquationType.IF
    assert len(eq.if_branches) == 1
    assert len(eq.else_equations) == 1

    exported = _export_equation(eq)
    assert exported["eq_type"] == "if"
    assert len(exported["branches"]) == 1
    assert len(exported["else_equations"]) == 1


def test_when_equation_import_export():
    """Test importing and exporting when-equations."""
    data = {
        "eq_type": "when",
        "branches": [
            {
                "condition": {
                    "op": "<",
                    "args": [{"op": "var", "name": "h"}, {"op": "literal", "value": 0}],
                },
                "equations": [
                    {
                        "eq_type": "simple",
                        "lhs": {"op": "component_ref", "parts": [{"name": "v", "subscripts": []}]},
                        "rhs": {
                            "op": "*",
                            "args": [
                                {"op": "literal", "value": -0.7},
                                {
                                    "op": "pre",
                                    "args": [
                                        {
                                            "op": "component_ref",
                                            "parts": [{"name": "v", "subscripts": []}],
                                        }
                                    ],
                                },
                            ],
                        },
                    }
                ],
            }
        ],
    }
    eq = _import_equation(data)
    assert eq.eq_type == EquationType.WHEN
    assert eq.condition is not None
    assert len(eq.when_equations) == 1

    exported = _export_equation(eq)
    assert exported["eq_type"] == "when"


# ==================== Complex Expression Tests ====================


def test_nested_expression_import_export():
    """Test importing and exporting nested expressions."""
    # (a + b) * (c - d)
    data = {
        "op": "*",
        "args": [
            {"op": "+", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]},
            {"op": "-", "args": [{"op": "var", "name": "c"}, {"op": "var", "name": "d"}]},
        ],
    }
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported == data


def test_comparison_operators():
    """Test all comparison operators."""
    for op in ["==", "!=", "<", "<=", ">", ">="]:
        data = {"op": op, "args": [{"op": "var", "name": "x"}, {"op": "literal", "value": 0}]}
        expr = _import_expr(data)
        exported = _export_expr(expr)
        assert exported["op"] == op


def test_logical_operators():
    """Test logical operators."""
    # and
    data = {"op": "and", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]}
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported["op"] == "and"

    # or
    data["op"] = "or"
    expr = _import_expr(data)
    exported = _export_expr(expr)
    assert exported["op"] == "or"


# ==================== Test Runner ====================


def run_all_tests():
    """Run all unit tests."""
    tests = [
        # Expression tests
        ("Literal import/export", test_literal_import_export),
        ("VarRef import/export", test_var_ref_import_export),
        ("BinaryOp import/export", test_binary_op_import_export),
        ("UnaryOp import/export", test_unary_op_import_export),
        ("der() import/export", test_der_import_export),
        ("pre() import/export", test_pre_import_export),
        ("ComponentRef import/export", test_component_ref_import_export),
        ("FunctionCall import/export", test_function_call_import_export),
        ("IfExpr import/export", test_if_expr_import_export),
        ("Nested expressions", test_nested_expression_import_export),
        ("Comparison operators", test_comparison_operators),
        ("Logical operators", test_logical_operators),
        # Variable tests
        ("Constant import/export", test_constant_import_export),
        ("Parameter import/export", test_parameter_import_export),
        ("Variable import/export", test_variable_import_export),
        ("Variable with Lie groups", test_variable_with_lie_groups),
        # Type conversion tests
        ("PrimitiveType conversion", test_primitive_type_conversion),
        ("Variability conversion", test_variability_conversion),
        # Equation tests
        ("Simple equation import/export", test_simple_equation_import_export),
        ("For equation import/export", test_for_equation_import_export),
        ("If equation import/export", test_if_equation_import_export),
        ("When equation import/export", test_when_equation_import_export),
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 80)
    print("Base Modelica Unit Tests")
    print("=" * 80)
    print()

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {name}")
            failed += 1
            errors.append((name, str(e)))
        except Exception as e:
            print(f"✗ {name} (error)")
            failed += 1
            errors.append((name, f"{type(e).__name__}: {e}"))

    print()
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    if errors:
        print("\nFailures:")
        for name, error in errors:
            print(f"\n{name}:")
            print(f"  {error}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
