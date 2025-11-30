"""
Import and export Base Modelica IR (MCP-0031) JSON format.

This module provides functions to convert between Base Modelica JSON
and Cyecca's internal IR representation.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

from cyecca.ir import (
    Model,
    Variable,
    VariableType,
    Variability,
    PrimitiveType,
    Causality,
    Equation,
    EquationType,
    Expr,
    Statement,
    Assignment,
    IfStatement,
    ForStatement,
    WhileStatement,
    WhenStatement,
    ReinitStatement,
    BreakStatement,
    ReturnStatement,
    FunctionCallStatement,
    AlgorithmSection,
    ComponentRef,
    ComponentRefPart,
)


def import_base_modelica(path: Union[str, Path]) -> Model:
    """
    Import a Base Modelica JSON file into a Cyecca Model.

    Args:
        path: Path to the Base Modelica JSON file

    Returns:
        Cyecca Model instance

    Example:
        >>> model = import_base_modelica("bouncing_ball.json")
        >>> print(model.name)
        >>> print(f"States: {len(model.states)}")
    """
    with open(path, "r") as f:
        data = json.load(f)

    return _import_model(data)


def export_base_modelica(
    model: Model,
    path: Union[str, Path],
    validate: bool = True,
    pretty: bool = True,
) -> None:
    """
    Export a Cyecca Model to Base Modelica JSON format.

    Args:
        model: Cyecca Model instance
        path: Output file path
        validate: Validate against Base Modelica schema (if jsonschema available)
        pretty: Pretty-print the JSON output

    Example:
        >>> export_base_modelica(model, "output.json")
    """
    data = _export_model(model)

    if validate:
        try:
            import jsonschema

            schema_path = (
                Path(__file__).parent.parent.parent.parent
                / "modelica_ir"
                / "schemas"
                / "base_modelica_ir-0.1.0.schema.json"
            )
            if schema_path.exists():
                with open(schema_path) as f:
                    schema = json.load(f)
                jsonschema.validate(data, schema)
        except ImportError:
            pass  # jsonschema not available, skip validation
        except Exception as e:
            print(f"Warning: Schema validation failed: {e}")

    with open(path, "w") as f:
        if pretty:
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f)


# ==================== Import Functions ====================


def _import_model(data: dict) -> Model:
    """Import model from Base Modelica JSON dict."""
    model = Model(
        name=data["model_name"],
        description=data.get("metadata", {}).get("description", ""),
        metadata=data.get("metadata", {}),
    )

    # Import constants
    for const_data in data.get("constants", []):
        var = _import_constant(const_data)
        model.add_variable(var)

    # Import parameters
    for param_data in data.get("parameters", []):
        var = _import_parameter(param_data)
        model.add_variable(var)

    # Import variables (states and algebraic)
    for var_data in data.get("variables", []):
        var = _import_variable(var_data)
        model.add_variable(var)

    # Import equations
    for eq_data in data.get("equations", []):
        eq = _import_equation(eq_data)
        model.add_equation(eq)

    # Import initial equations
    for eq_data in data.get("initial_equations", []):
        eq = _import_equation(eq_data)
        model.initial_equations.append(eq)

    # Import algorithms
    for algo_data in data.get("algorithms", []):
        algo = _import_algorithm(algo_data)
        model.add_algorithm(algo)

    # Import initial algorithms
    for algo_data in data.get("initial_algorithms", []):
        algo = _import_algorithm(algo_data)
        model.initial_algorithms.append(algo)

    # Post-process: Identify state variables from der() equations
    _refine_state_variables(model)

    return model


def _refine_state_variables(model: Model) -> None:
    """
    Post-process model to identify state variables.

    This function handles two cases:
    1. Models with der() function calls - extracts state names from der(x)
    2. Models with flattened derivatives - finds variables with der_X naming pattern

    Changes identified variables from ALGEBRAIC to STATE and creates corresponding
    DER_STATE variables if they don't exist.
    """
    from cyecca.ir.expr import FunctionCall, VarRef

    state_names = set()
    derivative_var_names = set()

    # Collect all variable names that start with "der_"
    for var in model.variables:
        if var.name.startswith("der_"):
            derivative_var_names.add(var.name)
            # Extract the base state name (remove "der_" prefix)
            base_name = var.name[4:]  # Remove "der_" prefix
            state_names.add(base_name)

    # Also scan equations for der() function calls (for non-flattened models)
    def find_der_calls(expr: Optional[Expr]) -> None:
        """Recursively find all der() calls in an expression."""
        if expr is None:
            return

        if isinstance(expr, FunctionCall) and expr.func == "der":
            # Extract the argument to der() - should be a variable name
            if len(expr.args) > 0:
                arg = expr.args[0]
                if isinstance(arg, VarRef):
                    state_names.add(arg.name)
                # Handle component references (simple cases)
                elif hasattr(arg, "parts") and len(arg.parts) == 1:
                    state_names.add(arg.parts[0].name)

        # Recursively check sub-expressions
        if hasattr(expr, "left"):
            find_der_calls(expr.left)
        if hasattr(expr, "right"):
            find_der_calls(expr.right)
        if hasattr(expr, "operand"):
            find_der_calls(expr.operand)
        if hasattr(expr, "args"):
            for arg in expr.args:
                find_der_calls(arg)
        if hasattr(expr, "condition"):
            find_der_calls(expr.condition)
        if hasattr(expr, "true_expr"):
            find_der_calls(expr.true_expr)
        if hasattr(expr, "false_expr"):
            find_der_calls(expr.false_expr)

    # Scan all equations for der() calls
    for eq in model.equations:
        find_der_calls(eq.lhs)
        find_der_calls(eq.rhs)

    # Also check initial equations
    for eq in model.initial_equations:
        find_der_calls(eq.lhs)
        find_der_calls(eq.rhs)

    # Update variable types for identified states
    for var_name in state_names:
        var = model.get_variable(var_name)
        if var and var.var_type == VariableType.ALGEBRAIC:
            # Change from ALGEBRAIC to STATE
            var.var_type = VariableType.STATE

            # Check if derivative variable already exists (from flattened model)
            der_var_name = f"der_{var_name}"
            der_var = model.get_variable(der_var_name)

            if der_var:
                # Update existing derivative variable to DER_STATE type
                der_var.var_type = VariableType.DER_STATE
            else:
                # Create new DER_STATE variable (for non-flattened models)
                der_var = Variable(
                    name=der_var_name,
                    var_type=VariableType.DER_STATE,
                    primitive_type=var.primitive_type,
                    variability=var.variability,
                    shape=var.shape,
                    unit=f"{var.unit}/s" if var.unit else "",
                    description=f"Derivative of {var_name}",
                )
                model.add_variable(der_var)


def _import_constant(data: dict) -> Variable:
    """Import a constant variable."""
    return Variable(
        name=data["name"],
        var_type=VariableType.CONSTANT,
        primitive_type=_import_primitive_type(data.get("type", "Real")),
        value=data.get("value"),
        shape=data.get("dimensions"),
        unit=data.get("unit", ""),
        description=data.get("description", ""),
        comment=data.get("comment", ""),
        metadata=data.get("annotations", {}),
    )


def _import_parameter(data: dict) -> Variable:
    """Import a parameter variable."""
    return Variable(
        name=data["name"],
        var_type=VariableType.PARAMETER,
        primitive_type=_import_primitive_type(data.get("type", "Real")),
        value=data.get("value"),
        start=data.get("start"),
        shape=data.get("dimensions"),
        unit=data.get("unit", ""),
        description=data.get("description", ""),
        comment=data.get("comment", ""),
        min_value=data.get("min"),
        max_value=data.get("max"),
        nominal=data.get("nominal"),
        metadata=data.get("annotations", {}),
    )


def _import_variable(data: dict) -> Variable:
    """Import a variable (state or algebraic)."""
    variability_str = data.get("variability", "continuous")
    variability = _import_variability(variability_str)

    # Determine variable type based on variability
    if variability == Variability.DISCRETE:
        var_type = VariableType.DISCRETE_STATE
    elif data.get("causality") == "input":
        var_type = VariableType.INPUT
    elif data.get("causality") == "output":
        var_type = VariableType.OUTPUT
    else:
        # Will be refined based on equations (state vs algebraic)
        var_type = VariableType.ALGEBRAIC

    var = Variable(
        name=data["name"],
        var_type=var_type,
        primitive_type=_import_primitive_type(data.get("type", "Real")),
        variability=variability,
        start=data.get("start"),
        shape=data.get("dimensions"),
        unit=data.get("unit", ""),
        description=data.get("description", ""),
        comment=data.get("comment", ""),
        min_value=data.get("min"),
        max_value=data.get("max"),
        nominal=data.get("nominal"),
        metadata=data.get("annotations", {}),
    )

    # Extract Lie group annotations if present
    annotations = data.get("annotations", {})
    if "lie_group" in annotations:
        var.lie_group_type = annotations["lie_group"]
    if "manifold_chart" in annotations:
        var.manifold_chart = annotations["manifold_chart"]

    return var


def _import_primitive_type(type_str: str) -> PrimitiveType:
    """Convert Base Modelica type string to PrimitiveType."""
    mapping = {
        "Real": PrimitiveType.REAL,
        "Integer": PrimitiveType.INTEGER,
        "Boolean": PrimitiveType.BOOLEAN,
        "String": PrimitiveType.STRING,
    }
    return mapping.get(type_str, PrimitiveType.REAL)


def _import_variability(var_str: str) -> Variability:
    """Convert Base Modelica variability string to Variability."""
    mapping = {
        "constant": Variability.CONSTANT,
        "fixed": Variability.FIXED,
        "tunable": Variability.TUNABLE,
        "discrete": Variability.DISCRETE,
        "continuous": Variability.CONTINUOUS,
    }
    return mapping.get(var_str, Variability.CONTINUOUS)


def _import_equation(data: dict) -> Equation:
    """Import an equation."""
    eq_type = data["eq_type"]

    if eq_type == "simple":
        lhs = _import_expr(data["lhs"]) if data.get("lhs") else None
        rhs = _import_expr(data["rhs"])
        return (
            Equation.simple(lhs, rhs)
            if lhs
            else Equation(eq_type=EquationType.SIMPLE, lhs=None, rhs=rhs)
        )

    elif eq_type == "for":
        indices = data["indices"]
        index_var = indices[0]["index"]  # Base Modelica uses single index
        range_expr = _import_expr(indices[0]["range"])
        equations = [_import_equation(eq) for eq in data["equations"]]
        return Equation.for_loop(index_var, range_expr, equations)

    elif eq_type == "if":
        branches = []
        for branch in data["branches"]:
            condition = _import_expr(branch["condition"])
            equations = [_import_equation(eq) for eq in branch["equations"]]
            branches.append((condition, equations))

        else_eqs = [_import_equation(eq) for eq in data.get("else_equations", [])]
        return Equation.if_eq(*branches, else_eqs=else_eqs if else_eqs else None)

    elif eq_type == "when":
        # Base Modelica when-equations use branches format (like if-equations)
        # Each branch has a condition and equations
        branches = data.get("branches", [])
        if branches:
            # Use the first branch's condition and equations
            # (Base Modelica typically has single-branch when-equations)
            first_branch = branches[0]
            condition = _import_expr(first_branch["condition"])
            equations = [_import_equation(eq) for eq in first_branch["equations"]]

            return Equation(
                eq_type=EquationType.WHEN,
                condition=condition,
                when_equations=equations,
            )
        else:
            # Fallback for old format with direct condition field
            condition = _import_expr(data["condition"])
            statements = [_import_statement(stmt) for stmt in data.get("statements", [])]
            return Equation(
                eq_type=EquationType.WHEN,
                condition=condition,
                when_equations=[],
            )

    else:
        raise ValueError(f"Unknown equation type: {eq_type}")


def _import_expr(data: dict) -> Expr:
    """Import an expression."""
    op = data["op"]

    if op == "literal":
        return Expr.literal(data["value"])

    elif op == "var":
        return Expr.var_ref(data["name"])

    elif op in ["+", "-", "*", "/", "^", "==", "!=", "<", "<=", ">", ">=", "and", "or"]:
        lhs = _import_expr(data["args"][0])
        rhs = _import_expr(data["args"][1])
        return Expr.binary_op(op, lhs, rhs)

    elif op in ["neg", "not"]:
        arg = _import_expr(data["args"][0])
        return Expr.unary_op(op, arg)

    elif op == "der":
        arg = _import_expr(data["args"][0])
        from cyecca.ir import der

        return der(arg)

    elif op == "pre":
        arg = _import_expr(data["args"][0])
        from cyecca.ir import pre

        return pre(arg)

    elif op == "if":
        # Ternary if-expression
        condition = _import_expr(data["condition"])
        then_expr = _import_expr(data["then"])
        else_expr = _import_expr(data["else"])
        return Expr.if_expr(condition, then_expr, else_expr)

    elif op == "call":
        func = data["func"]
        args = [_import_expr(arg) for arg in data.get("args", [])]
        return Expr.call(func, *args)

    elif op == "component_ref":
        # Component reference like vehicle.wheels[1].pressure
        parts = []
        for part_data in data["parts"]:
            subscripts = [_import_expr(sub) for sub in part_data.get("subscripts", [])]
            parts.append(ComponentRefPart(name=part_data["name"], subscripts=subscripts))
        return ComponentRef(parts=tuple(parts))

    else:
        # Try as function call
        args = [_import_expr(arg) for arg in data.get("args", [])]
        return Expr.call(op, *args)


def _import_statement(data: dict) -> Statement:
    """Import a statement."""
    stmt_type = data["stmt"]

    if stmt_type == "assign":
        target = _import_component_ref(data["target"])
        expr = _import_expr(data["expr"])
        return Assignment(target=target, expr=expr)

    elif stmt_type == "if":
        branches = []
        for branch in data["branches"]:
            condition = _import_expr(branch["condition"])
            statements = tuple(_import_statement(s) for s in branch["statements"])
            branches.append((condition, statements))

        else_stmts = tuple(_import_statement(s) for s in data.get("else_statements", []))
        return IfStatement(branches=tuple(branches), else_statements=else_stmts)

    elif stmt_type == "for":
        indices = data["indices"]
        index_var = indices[0]["index"]
        range_expr = _import_expr(indices[0]["range"])
        body = tuple(_import_statement(s) for s in data["body"])
        return ForStatement(index_var=index_var, range_expr=range_expr, body=body)

    elif stmt_type == "while":
        condition = _import_expr(data["condition"])
        body = tuple(_import_statement(s) for s in data["body"])
        return WhileStatement(condition=condition, body=body)

    elif stmt_type == "when":
        condition = _import_expr(data["condition"])
        body = tuple(_import_statement(s) for s in data["body"])
        return WhenStatement(condition=condition, body=body)

    elif stmt_type == "reinit":
        target = data["target"]
        expr = _import_expr(data["expr"])
        return ReinitStatement(target=target, expr=expr)

    elif stmt_type == "break":
        return BreakStatement()

    elif stmt_type == "return":
        expr = _import_expr(data["expr"]) if "expr" in data else None
        return ReturnStatement(expr=expr)

    elif stmt_type == "call":
        func = data["func"]
        args = [_import_expr(arg) for arg in data.get("args", [])]
        return FunctionCallStatement(func=func, args=tuple(args))

    else:
        raise ValueError(f"Unknown statement type: {stmt_type}")


def _import_component_ref(data: Union[str, list]) -> ComponentRef:
    """Import a component reference from target field."""
    if isinstance(data, str):
        # Simple variable name
        return ComponentRef(parts=(ComponentRefPart(name=data, subscripts=[]),))
    elif isinstance(data, list):
        # Component reference path
        parts = []
        for part in data:
            if isinstance(part, str):
                parts.append(ComponentRefPart(name=part, subscripts=[]))
            elif isinstance(part, dict):
                subscripts = [_import_expr(sub) for sub in part.get("subscripts", [])]
                parts.append(ComponentRefPart(name=part["name"], subscripts=subscripts))
        return ComponentRef(parts=tuple(parts))
    else:
        raise ValueError(f"Invalid component reference format: {data}")


def _import_algorithm(data: dict) -> AlgorithmSection:
    """Import an algorithm section."""
    statements = [_import_statement(stmt) for stmt in data["statements"]]
    return AlgorithmSection(
        statements=statements,
        is_initial=data.get("is_initial", False),
    )


# ==================== Export Functions ====================


def _export_model(model: Model) -> dict:
    """Export model to Base Modelica JSON dict."""
    data = {
        "ir_version": "base-0.1.0",
        "base_modelica_version": "0.1",
        "model_name": model.name,
        "constants": [],
        "parameters": [],
        "variables": [],
        "equations": [],
    }

    # Export constants
    for var in model.variables:
        if var.var_type == VariableType.CONSTANT:
            data["constants"].append(_export_constant(var))

    # Export parameters
    for var in model.variables:
        if var.var_type == VariableType.PARAMETER:
            data["parameters"].append(_export_parameter(var))

    # Export variables (states, algebraic, inputs, outputs)
    for var in model.variables:
        if var.var_type in [
            VariableType.STATE,
            VariableType.ALGEBRAIC,
            VariableType.DISCRETE_STATE,
            VariableType.INPUT,
            VariableType.OUTPUT,
        ]:
            data["variables"].append(_export_variable(var))

    # Export equations
    for eq in model.equations:
        data["equations"].append(_export_equation(eq))

    # Export initial equations
    if model.initial_equations:
        data["initial_equations"] = [_export_equation(eq) for eq in model.initial_equations]

    # Export algorithms
    if model.algorithms:
        data["algorithms"] = [_export_algorithm(algo) for algo in model.algorithms]

    # Export initial algorithms
    if model.initial_algorithms:
        data["initial_algorithms"] = [_export_algorithm(algo) for algo in model.initial_algorithms]

    # Export metadata
    if model.metadata:
        data["metadata"] = model.metadata

    # Add source info
    data["source_info"] = {
        "generated_by": "Cyecca",
        "base_modelica_version": "0.1",
    }

    return data


def _export_constant(var: Variable) -> dict:
    """Export a constant variable."""
    data = {
        "name": var.name,
        "type": _export_primitive_type(var.primitive_type),
        "value": var.value,
    }

    if var.shape:
        data["dimensions"] = var.shape
    if var.unit:
        data["unit"] = var.unit
    if var.description:
        data["description"] = var.description
    if var.comment:
        data["comment"] = var.comment
    if var.metadata:
        data["annotations"] = var.metadata

    return data


def _export_parameter(var: Variable) -> dict:
    """Export a parameter variable."""
    data = {
        "name": var.name,
        "type": _export_primitive_type(var.primitive_type),
    }

    if var.value is not None:
        data["value"] = var.value
    if var.start is not None:
        data["start"] = var.start
    if var.shape:
        data["dimensions"] = var.shape
    if var.unit:
        data["unit"] = var.unit
    if var.description:
        data["description"] = var.description
    if var.comment:
        data["comment"] = var.comment
    if var.min_value is not None:
        data["min"] = var.min_value
    if var.max_value is not None:
        data["max"] = var.max_value
    if var.nominal is not None:
        data["nominal"] = var.nominal

    # Include Lie group annotations if present
    annotations = dict(var.metadata) if var.metadata else {}
    if var.lie_group_type:
        annotations["lie_group"] = var.lie_group_type
    if var.manifold_chart:
        annotations["manifold_chart"] = var.manifold_chart
    if annotations:
        data["annotations"] = annotations

    return data


def _export_variable(var: Variable) -> dict:
    """Export a variable (state, algebraic, input, output)."""
    data = {
        "name": var.name,
        "type": _export_primitive_type(var.primitive_type),
        "variability": _export_variability(var.variability),
    }

    if var.var_type == VariableType.INPUT:
        data["causality"] = "input"
    elif var.var_type == VariableType.OUTPUT:
        data["causality"] = "output"

    if var.start is not None:
        data["start"] = var.start
    if var.shape:
        data["dimensions"] = var.shape
    if var.unit:
        data["unit"] = var.unit
    if var.description:
        data["description"] = var.description
    if var.comment:
        data["comment"] = var.comment
    if var.min_value is not None:
        data["min"] = var.min_value
    if var.max_value is not None:
        data["max"] = var.max_value
    if var.nominal is not None:
        data["nominal"] = var.nominal

    # Include Lie group annotations if present
    annotations = dict(var.metadata) if var.metadata else {}
    if var.lie_group_type:
        annotations["lie_group"] = var.lie_group_type
    if var.manifold_chart:
        annotations["manifold_chart"] = var.manifold_chart
    if annotations:
        data["annotations"] = annotations

    return data


def _export_primitive_type(ptype: PrimitiveType) -> str:
    """Convert PrimitiveType to Base Modelica type string."""
    mapping = {
        PrimitiveType.REAL: "Real",
        PrimitiveType.INTEGER: "Integer",
        PrimitiveType.BOOLEAN: "Boolean",
        PrimitiveType.STRING: "String",
    }
    return mapping[ptype]


def _export_variability(var: Variability) -> str:
    """Convert Variability to Base Modelica variability string."""
    mapping = {
        Variability.CONSTANT: "constant",
        Variability.FIXED: "fixed",
        Variability.TUNABLE: "tunable",
        Variability.DISCRETE: "discrete",
        Variability.CONTINUOUS: "continuous",
    }
    return mapping[var]


def _export_equation(eq: Equation) -> dict:
    """Export an equation."""
    if eq.eq_type == EquationType.SIMPLE:
        data = {"eq_type": "simple"}
        if eq.lhs:
            data["lhs"] = _export_expr(eq.lhs)
        data["rhs"] = _export_expr(eq.rhs)
        return data

    elif eq.eq_type == EquationType.FOR:
        return {
            "eq_type": "for",
            "indices": [
                {
                    "index": eq.index_var,
                    "range": _export_expr(eq.range_expr),
                }
            ],
            "equations": [_export_equation(e) for e in eq.for_equations],
        }

    elif eq.eq_type == EquationType.IF:
        branches = []
        for cond, eqs in eq.if_branches:
            branches.append(
                {
                    "condition": _export_expr(cond),
                    "equations": [_export_equation(e) for e in eqs],
                }
            )

        data = {"eq_type": "if", "branches": branches}
        if eq.else_equations:
            data["else_equations"] = [_export_equation(e) for e in eq.else_equations]
        return data

    elif eq.eq_type == EquationType.WHEN:
        return {
            "eq_type": "when",
            "condition": _export_expr(eq.condition),
            "statements": [],  # TODO: Convert when_equations to statements
        }

    else:
        raise ValueError(f"Cannot export equation type {eq.eq_type} to Base Modelica")


def _export_expr(expr: Expr) -> dict:
    """Export an expression."""
    from cyecca.ir.expr import Literal, BinaryOp, UnaryOp, FunctionCall, VarRef, IfExpr

    if isinstance(expr, Literal):
        return {"op": "literal", "value": expr.value}

    elif isinstance(expr, VarRef):
        return {"op": "var", "name": expr.name}

    elif isinstance(expr, ComponentRef):
        # Simple variable reference: export as "var" for compatibility
        if len(expr.parts) == 1 and not expr.parts[0].subscripts:
            return {"op": "var", "name": expr.parts[0].name}
        # Complex component reference: export full structure
        else:
            parts = []
            for part in expr.parts:
                subscripts = [_export_expr(sub) for sub in part.subscripts]
                parts.append({"name": part.name, "subscripts": subscripts})
            return {"op": "component_ref", "parts": parts}

    elif isinstance(expr, BinaryOp):
        return {
            "op": expr.op,
            "args": [_export_expr(expr.left), _export_expr(expr.right)],
        }

    elif isinstance(expr, UnaryOp):
        return {"op": expr.op, "args": [_export_expr(expr.operand)]}

    elif isinstance(expr, FunctionCall):
        if expr.func == "der":
            return {"op": "der", "args": [_export_expr(expr.args[0])]}
        elif expr.func == "pre":
            return {"op": "pre", "args": [_export_expr(expr.args[0])]}
        else:
            return {
                "op": "call",
                "func": expr.func,
                "args": [_export_expr(arg) for arg in expr.args],
            }

    elif isinstance(expr, IfExpr):
        return {
            "op": "if",
            "condition": _export_expr(expr.condition),
            "then": _export_expr(expr.true_expr),
            "else": _export_expr(expr.false_expr),
        }

    else:
        raise ValueError(f"Cannot export expression type: {type(expr)}")


def _export_statement(stmt: Statement) -> dict:
    """Export a statement."""
    if isinstance(stmt, Assignment):
        return {
            "stmt": "assign",
            "target": _export_component_ref(stmt.target),
            "expr": _export_expr(stmt.expr),
        }

    elif isinstance(stmt, IfStatement):
        branches = []
        for cond, stmts in stmt.branches:
            branches.append(
                {
                    "condition": _export_expr(cond),
                    "statements": [_export_statement(s) for s in stmts],
                }
            )

        data = {"stmt": "if", "branches": branches}
        if stmt.else_statements:
            data["else_statements"] = [_export_statement(s) for s in stmt.else_statements]
        return data

    elif isinstance(stmt, ForStatement):
        return {
            "stmt": "for",
            "indices": [
                {
                    "index": stmt.index_var,
                    "range": _export_expr(stmt.range_expr),
                }
            ],
            "body": [_export_statement(s) for s in stmt.body],
        }

    elif isinstance(stmt, WhileStatement):
        return {
            "stmt": "while",
            "condition": _export_expr(stmt.condition),
            "body": [_export_statement(s) for s in stmt.body],
        }

    elif isinstance(stmt, WhenStatement):
        return {
            "stmt": "when",
            "condition": _export_expr(stmt.condition),
            "body": [_export_statement(s) for s in stmt.body],
        }

    elif isinstance(stmt, ReinitStatement):
        return {
            "stmt": "reinit",
            "target": stmt.target,
            "expr": _export_expr(stmt.expr),
        }

    elif isinstance(stmt, BreakStatement):
        return {"stmt": "break"}

    elif isinstance(stmt, ReturnStatement):
        data = {"stmt": "return"}
        if stmt.expr:
            data["expr"] = _export_expr(stmt.expr)
        return data

    elif isinstance(stmt, FunctionCallStatement):
        return {
            "stmt": "call",
            "func": stmt.func,
            "args": [_export_expr(arg) for arg in stmt.args],
        }

    else:
        raise ValueError(f"Cannot export statement type: {type(stmt)}")


def _export_component_ref(comp_ref: ComponentRef) -> Union[str, list]:
    """Export a component reference."""
    if len(comp_ref.parts) == 1 and not comp_ref.parts[0].subscripts:
        # Simple variable name
        return comp_ref.parts[0].name
    else:
        # Component reference path
        parts = []
        for part in comp_ref.parts:
            if part.subscripts:
                parts.append(
                    {
                        "name": part.name,
                        "subscripts": [_export_expr(sub) for sub in part.subscripts],
                    }
                )
            else:
                parts.append(part.name)
        return parts


def _export_algorithm(algo: AlgorithmSection) -> dict:
    """Export an algorithm section."""
    return {
        "statements": [_export_statement(stmt) for stmt in algo.statements],
        "is_initial": algo.is_initial,
    }
