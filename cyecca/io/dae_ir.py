"""
Import and export DAE IR (Differential-Algebraic Equation Intermediate Representation) JSON format.

This module provides functions to convert between DAE IR JSON (dae-0.1.0)
and Cyecca's internal IR representation.

DAE IR is a superset of Base Modelica IR (MCP-0031) that adds explicit variable
classification matching the Modelica specification's DAE formalism (Appendix B):
- States (x): Continuous-time variables appearing differentiated
- Algebraic (y): Continuous-time variables not differentiated
- Discrete Real (z): Discrete-time Real variables
- Discrete Valued (m): Boolean/Integer discrete variables
- Parameters (p): Fixed after initialization
- Constants: Compile-time values
- Inputs/Outputs: External interface

Derivatives are represented as der(x) function calls in equations, not as
separate variables.
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


def _extract_start_value(start_data: Any) -> Optional[float]:
    """
    Extract a float value from a start expression.

    The start value in Base Modelica JSON can be:
    - None
    - A float/int directly
    - A dict like {'op': 'literal', 'value': 0.8}
    - A dict like {'op': 'neg', 'args': [{'op': 'literal', 'value': 5.0}]}

    Returns:
        The extracted float value, or None if not extractable
    """
    if start_data is None:
        return None
    if isinstance(start_data, (int, float)):
        return float(start_data)
    if isinstance(start_data, dict):
        if start_data.get("op") == "literal":
            return float(start_data["value"])
        elif start_data.get("op") == "neg" and "args" in start_data:
            inner = _extract_start_value(start_data["args"][0])
            return -inner if inner is not None else None
        elif start_data.get("op") == "pos" and "args" in start_data:
            return _extract_start_value(start_data["args"][0])
    return None


def import_dae_ir(path: Union[str, Path]) -> Model:
    """
    Import a DAE IR JSON file into a Cyecca Model.

    Args:
        path: Path to the DAE IR JSON file

    Returns:
        Cyecca Model instance

    Example:
        >>> model = import_dae_ir("bouncing_ball.json")
        >>> print(model.name)
        >>> print(f"States: {len(model.states)}")
    """
    with open(path, "r") as f:
        data = json.load(f)

    return _import_model(data)


def load_dae_ir_json(json_str: str) -> Model:
    """
    Load a DAE IR JSON string into a Cyecca Model.

    Args:
        json_str: JSON string in DAE IR format

    Returns:
        Cyecca Model instance

    Example:
        >>> json_str = '{"model_name": "Test", ...}'
        >>> model = load_dae_ir_json(json_str)
        >>> print(model.name)
    """
    data = json.loads(json_str)
    return _import_model(data)


def export_dae_ir(
    model: Model,
    path: Union[str, Path],
    validate: bool = True,
    pretty: bool = True,
) -> None:
    """
    Export a Cyecca Model to DAE IR JSON format.

    Args:
        model: Cyecca Model instance
        path: Output file path
        validate: Validate against DAE IR schema (if jsonschema available)
        pretty: Pretty-print the JSON output

    Example:
        >>> export_dae_ir(model, "output.json")
    """
    data = _export_model(model)

    if validate:
        try:
            import jsonschema

            schema_path = (
                Path(__file__).parent.parent.parent.parent
                / "modelica_ir"
                / "schemas"
                / "dae_ir-0.1.0.schema.json"
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
    """Import model from DAE IR JSON dict."""
    model = Model(
        name=data["model_name"],
        description=data.get("metadata", {}).get("description", ""),
        metadata=data.get("metadata", {}),
    )

    return _import_dae_ir_model(data, model)


def _import_dae_ir_model(data: dict, model: Model) -> Model:
    """Import model from DAE IR format with classified variables."""
    variables = data.get("variables", {})

    # Import states (x) - explicitly classified
    for state_data in variables.get("states", []):
        var = _import_state_variable(state_data)
        model.add_variable(var)

    # Import algebraic (y)
    for alg_data in variables.get("algebraic", []):
        var = _import_algebraic_variable(alg_data)
        model.add_variable(var)

    # Import discrete real (z)
    for disc_data in variables.get("discrete_real", []):
        var = _import_discrete_real_variable(disc_data)
        model.add_variable(var)

    # Import discrete valued (m) - Boolean, Integer
    for disc_data in variables.get("discrete_valued", []):
        var = _import_discrete_valued_variable(disc_data)
        model.add_variable(var)

    # Import parameters (p)
    for param_data in variables.get("parameters", []):
        var = _import_parameter(param_data)
        model.add_variable(var)

    # Import constants
    for const_data in variables.get("constants", []):
        var = _import_constant(const_data)
        model.add_variable(var)

    # Import inputs (u)
    for input_data in variables.get("inputs", []):
        var = _import_input_variable(input_data)
        model.add_variable(var)

    # Import outputs
    for output_data in variables.get("outputs", []):
        var = _import_output_variable(output_data)
        model.add_variable(var)

    # Import equations from classified structure
    equations = data.get("equations", {})

    # Import continuous equations
    for eq_data in equations.get("continuous", []):
        eq = _import_equation(eq_data)
        model.add_equation(eq)

    # Import event equations
    for eq_data in equations.get("event", []):
        eq = _import_equation(eq_data)
        model.add_equation(eq)

    # Import discrete real equations
    for eq_data in equations.get("discrete_real", []):
        eq = _import_equation(eq_data)
        model.add_equation(eq)

    # Import discrete valued equations
    for eq_data in equations.get("discrete_valued", []):
        eq = _import_equation(eq_data)
        model.add_equation(eq)

    # Import initial equations
    for eq_data in equations.get("initial", []):
        eq = _import_equation(eq_data)
        model.initial_equations.append(eq)

    # Import event indicators and link with algorithms to create WHEN equations
    # Rumoca exports events as:
    # - event_indicators: [{name, expression, direction}, ...]
    # - algorithms: [{statements: [{stmt: "reinit", source_ref: "c0", ...}]}]
    # We need to combine these into WHEN equations
    event_indicators = data.get("event_indicators", [])
    algorithms = data.get("algorithms", [])

    if event_indicators:
        # Build map from event indicator name to condition expression
        indicator_map: dict[str, Expr] = {}
        for indicator in event_indicators:
            name = indicator.get("name", "")
            expr_data = indicator.get("expression")
            if name and expr_data:
                indicator_map[name] = _import_expr(expr_data)

        # Find algorithms with reinit statements that reference event indicators
        # and create WHEN equations from them
        for algo_data in algorithms:
            statements = algo_data.get("statements", [])
            # Group statements by their source_ref (event indicator)
            stmts_by_indicator: dict[str, list[Equation]] = {}
            non_event_stmts = []

            for stmt_data in statements:
                source_ref = stmt_data.get("source_ref", "")
                if source_ref in indicator_map:
                    # This statement is triggered by an event indicator
                    if source_ref not in stmts_by_indicator:
                        stmts_by_indicator[source_ref] = []
                    # Convert reinit statement to equation for WHEN clause
                    eq = _convert_reinit_to_equation(stmt_data)
                    if eq:
                        stmts_by_indicator[source_ref].append(eq)
                else:
                    non_event_stmts.append(stmt_data)

            # Create WHEN equations for each event indicator
            for indicator_name, when_eqs in stmts_by_indicator.items():
                condition = indicator_map[indicator_name]
                when_eq = Equation(
                    eq_type=EquationType.WHEN,
                    condition=condition,
                    when_equations=when_eqs,
                )
                model.add_equation(when_eq)

            # Import remaining non-event statements as regular algorithm
            if non_event_stmts:
                algo = _import_algorithm({"statements": non_event_stmts})
                model.add_algorithm(algo)
    else:
        # No event indicators - import algorithms normally
        for algo_data in algorithms:
            algo = _import_algorithm(algo_data)
            model.add_algorithm(algo)

    # Import initial algorithms
    for algo_data in data.get("initial_algorithms", []):
        algo = _import_algorithm(algo_data)
        model.initial_algorithms.append(algo)

    return model


def _convert_reinit_to_equation(stmt_data: dict) -> Optional[Equation]:
    """Convert a reinit statement to an equation for use in WHEN clauses."""
    stmt_type = stmt_data.get("stmt", "")
    if stmt_type != "reinit":
        return None

    # reinit(target, expr) -> target = expr
    target_data = stmt_data.get("target")
    expr_data = stmt_data.get("expr")

    if not target_data or not expr_data:
        return None

    # Target is a component ref (list of parts like [{"name": "v", "subscripts": []}])
    lhs = _import_component_ref(target_data)
    rhs = _import_expr(expr_data)

    return Equation(eq_type=EquationType.SIMPLE, lhs=lhs, rhs=rhs)


def _import_state_variable(data: dict) -> Variable:
    """Import a state variable from DAE IR format."""
    shape = data.get("shape")
    if shape:
        shape = list(shape)

    var = Variable(
        name=data["name"],
        var_type=VariableType.STATE,
        primitive_type=_import_primitive_type(data.get("vartype", "Real")),
        variability=Variability.CONTINUOUS,
        start=_extract_start_value(data.get("start")),
        shape=shape,
        unit=data.get("unit", ""),
        description=data.get("description", ""),
        comment=data.get("comment", ""),
        min_value=data.get("min"),
        max_value=data.get("max"),
        nominal=data.get("nominal"),
        metadata=data.get("annotations", {}),
    )

    # Store state_index if present
    if "state_index" in data:
        var.metadata = {**(var.metadata or {}), "state_index": data["state_index"]}

    return var


def _import_algebraic_variable(data: dict) -> Variable:
    """Import an algebraic variable from DAE IR format."""
    shape = data.get("shape")
    if shape:
        shape = list(shape)

    return Variable(
        name=data["name"],
        var_type=VariableType.ALGEBRAIC,
        primitive_type=_import_primitive_type(data.get("vartype", "Real")),
        variability=Variability.CONTINUOUS,
        start=_extract_start_value(data.get("start")),
        shape=shape,
        unit=data.get("unit", ""),
        description=data.get("description", ""),
        comment=data.get("comment", ""),
        min_value=data.get("min"),
        max_value=data.get("max"),
        nominal=data.get("nominal"),
        metadata=data.get("annotations", {}),
    )


def _import_discrete_real_variable(data: dict) -> Variable:
    """Import a discrete Real variable from DAE IR format."""
    shape = data.get("shape")
    if shape:
        shape = list(shape)

    return Variable(
        name=data["name"],
        var_type=VariableType.DISCRETE_STATE,
        primitive_type=PrimitiveType.REAL,
        variability=Variability.DISCRETE,
        start=_extract_start_value(data.get("start")),
        shape=shape,
        unit=data.get("unit", ""),
        description=data.get("description", ""),
        comment=data.get("comment", ""),
        min_value=data.get("min"),
        max_value=data.get("max"),
        nominal=data.get("nominal"),
        metadata=data.get("annotations", {}),
    )


def _import_discrete_valued_variable(data: dict) -> Variable:
    """Import a discrete-valued (Boolean/Integer) variable from DAE IR format."""
    shape = data.get("shape")
    if shape:
        shape = list(shape)

    return Variable(
        name=data["name"],
        var_type=VariableType.DISCRETE_STATE,
        primitive_type=_import_primitive_type(data.get("vartype", "Boolean")),
        variability=Variability.DISCRETE,
        start=_extract_start_value(data.get("start")),
        shape=shape,
        unit=data.get("unit", ""),
        description=data.get("description", ""),
        comment=data.get("comment", ""),
        metadata=data.get("annotations", {}),
    )


def _import_input_variable(data: dict) -> Variable:
    """Import an input variable from DAE IR format."""
    shape = data.get("shape")
    if shape:
        shape = list(shape)

    return Variable(
        name=data["name"],
        var_type=VariableType.INPUT,
        primitive_type=_import_primitive_type(data.get("vartype", "Real")),
        variability=Variability.CONTINUOUS,
        start=_extract_start_value(data.get("start")),
        shape=shape,
        unit=data.get("unit", ""),
        description=data.get("description", ""),
        comment=data.get("comment", ""),
        metadata=data.get("annotations", {}),
    )


def _import_output_variable(data: dict) -> Variable:
    """Import an output variable from DAE IR format."""
    shape = data.get("shape")
    if shape:
        shape = list(shape)

    return Variable(
        name=data["name"],
        var_type=VariableType.OUTPUT,
        primitive_type=_import_primitive_type(data.get("vartype", "Real")),
        variability=Variability.CONTINUOUS,
        start=_extract_start_value(data.get("start")),
        shape=shape,
        unit=data.get("unit", ""),
        description=data.get("description", ""),
        comment=data.get("comment", ""),
        metadata=data.get("annotations", {}),
    )


def _import_constant(data: dict) -> Variable:
    """Import a constant variable."""
    shape = data.get("shape")
    if shape:
        shape = list(shape)
    const_value = _extract_start_value(data.get("start"))
    return Variable(
        name=data["name"],
        var_type=VariableType.CONSTANT,
        primitive_type=_import_primitive_type(data.get("vartype", "Real")),
        value=const_value,
        start=const_value,
        shape=shape,
        unit=data.get("unit", ""),
        description=data.get("description", ""),
        comment=data.get("comment", ""),
        metadata=data.get("annotations", {}),
    )


def _import_parameter(data: dict) -> Variable:
    """Import a parameter variable."""
    shape = data.get("shape")
    if shape:
        shape = list(shape)
    return Variable(
        name=data["name"],
        var_type=VariableType.PARAMETER,
        primitive_type=_import_primitive_type(data.get("vartype", "Real")),
        value=_extract_start_value(data.get("value")),
        start=_extract_start_value(data.get("start")),
        shape=shape,
        unit=data.get("unit", ""),
        description=data.get("description", ""),
        comment=data.get("comment", ""),
        min_value=data.get("min"),
        max_value=data.get("max"),
        nominal=data.get("nominal"),
        metadata=data.get("annotations", {}),
    )


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
        # Handle both formats:
        # 1. Rumoca format: {"op": "if", "branches": [[cond, expr], ...], "else": ...}
        # 2. Legacy format: {"op": "if", "condition": ..., "then": ..., "else": ...}
        if "branches" in data:
            # Rumoca format with branches array
            branches = data["branches"]
            else_expr = _import_expr(data["else"])
            # Build nested if-expressions from branches (right-to-left)
            result = else_expr
            for cond_data, then_data in reversed(branches):
                condition = _import_expr(cond_data)
                then_expr = _import_expr(then_data)
                result = Expr.if_expr(condition, then_expr, result)
            return result
        else:
            # Legacy format with condition/then/else
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
            subscripts = tuple(_import_expr(sub) for sub in part_data.get("subscripts", []))
            parts.append(ComponentRefPart(name=part_data["name"], subscripts=subscripts))
        return ComponentRef(parts=tuple(parts))

    elif op == "array":
        # Array literal construction
        values = [_import_expr(val) for val in data.get("values", [])]
        from cyecca.ir import ArrayLiteral

        return ArrayLiteral(elements=tuple(values))

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
        # WhenStatement uses branches format: ((condition, statements), ...)
        return WhenStatement(branches=((condition, body),))

    elif stmt_type == "reinit":
        target = _import_component_ref(data["target"])
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
        return ComponentRef(parts=(ComponentRefPart(name=data, subscripts=()),))
    elif isinstance(data, list):
        # Component reference path
        parts = []
        for part in data:
            if isinstance(part, str):
                parts.append(ComponentRefPart(name=part, subscripts=()))
            elif isinstance(part, dict):
                subscripts = tuple(_import_expr(sub) for sub in part.get("subscripts", []))
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
    """Export model to DAE IR JSON dict."""
    # Build classified variables
    states = []
    algebraic = []
    discrete_real = []
    discrete_valued = []
    parameters = []
    constants = []
    inputs = []
    outputs = []

    state_index = 0
    for var in model.variables:
        if var.var_type == VariableType.STATE:
            states.append(_export_state_variable(var, state_index))
            state_index += 1
        elif var.var_type == VariableType.ALGEBRAIC:
            algebraic.append(_export_algebraic_variable(var))
        elif var.var_type == VariableType.DISCRETE_STATE:
            if var.primitive_type == PrimitiveType.REAL:
                discrete_real.append(_export_basic_variable(var))
            else:
                discrete_valued.append(_export_basic_variable(var))
        elif var.var_type == VariableType.PARAMETER:
            parameters.append(_export_parameter(var))
        elif var.var_type == VariableType.CONSTANT:
            constants.append(_export_constant(var))
        elif var.var_type == VariableType.INPUT:
            inputs.append(_export_basic_variable(var))
        elif var.var_type == VariableType.OUTPUT:
            outputs.append(_export_basic_variable(var))

    # Build classified equations
    continuous_eqs = []
    initial_eqs = []

    for eq in model.equations:
        continuous_eqs.append(_export_equation(eq))

    for eq in model.initial_equations:
        initial_eqs.append(_export_equation(eq))

    data = {
        "ir_version": "dae-0.1.0",
        "base_modelica_version": "0.1",
        "model_name": model.name,
        "variables": {
            "states": states,
            "algebraic": algebraic,
            "discrete_real": discrete_real,
            "discrete_valued": discrete_valued,
            "parameters": parameters,
            "constants": constants,
            "inputs": inputs,
            "outputs": outputs,
        },
        "equations": {
            "continuous": continuous_eqs,
            "event": [],
            "discrete_real": [],
            "discrete_valued": [],
            "initial": initial_eqs,
        },
        "event_indicators": [],
        "algorithms": [],
        "initial_algorithms": [],
        "functions": [],
        "structure": {
            "n_states": len(states),
            "n_algebraic": len(algebraic),
            "n_equations": len(continuous_eqs),
            "dae_index": 0,
            "is_ode": len(algebraic) == 0,
        },
        "source_info": {},
        "metadata": model.metadata or {},
    }

    # Export algorithms
    if model.algorithms:
        data["algorithms"] = [_export_algorithm(algo) for algo in model.algorithms]

    # Export initial algorithms
    if model.initial_algorithms:
        data["initial_algorithms"] = [_export_algorithm(algo) for algo in model.initial_algorithms]

    return data


def _export_state_variable(var: Variable, state_index: int) -> dict:
    """Export a state variable to DAE IR format."""
    data = {
        "name": var.name,
        "vartype": _export_primitive_type(var.primitive_type),
        "state_index": state_index,
    }

    if var.start is not None:
        data["start"] = var.start
    if var.shape:
        data["shape"] = var.shape
    if var.unit:
        data["unit"] = var.unit
    if var.description:
        data["comment"] = var.description
    if var.min_value is not None:
        data["min"] = var.min_value
    if var.max_value is not None:
        data["max"] = var.max_value
    if var.nominal is not None:
        data["nominal"] = var.nominal

    return data


def _export_algebraic_variable(var: Variable) -> dict:
    """Export an algebraic variable to DAE IR format."""
    data = {
        "name": var.name,
        "vartype": _export_primitive_type(var.primitive_type),
    }

    if var.start is not None:
        data["start"] = var.start
    if var.shape:
        data["shape"] = var.shape
    if var.unit:
        data["unit"] = var.unit
    if var.description:
        data["comment"] = var.description
    if var.min_value is not None:
        data["min"] = var.min_value
    if var.max_value is not None:
        data["max"] = var.max_value
    if var.nominal is not None:
        data["nominal"] = var.nominal

    return data


def _export_basic_variable(var: Variable) -> dict:
    """Export a basic variable to DAE IR format."""
    data = {
        "name": var.name,
        "vartype": _export_primitive_type(var.primitive_type),
    }

    if var.start is not None:
        data["start"] = var.start
    if var.shape:
        data["shape"] = var.shape
    if var.unit:
        data["unit"] = var.unit
    if var.description:
        data["comment"] = var.description

    return data


def _export_constant(var: Variable) -> dict:
    """Export a constant variable."""
    data = {
        "name": var.name,
        "type": _export_primitive_type(var.primitive_type),
        "value": var.value,
    }

    if var.shape:
        data["shape"] = var.shape
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
        data["shape"] = var.shape
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
        data["shape"] = var.shape
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


def _equation_to_statement(eq: Equation) -> dict:
    """
    Convert an equation to an assignment statement for use in when clauses.

    In Modelica, when clauses contain equations like "x = expr", but in
    Base Modelica JSON these are represented as assignment statements.
    """
    from cyecca.ir.expr import FunctionCall

    # Check if this is a reinit call: reinit(x, expr)
    if isinstance(eq.rhs, FunctionCall) and eq.rhs.name == "reinit" and len(eq.rhs.args) == 2:
        target_expr, value_expr = eq.rhs.args
        # Extract target name from the first argument
        if isinstance(target_expr, ComponentRef):
            target = _export_component_ref(target_expr)
        else:
            target = str(target_expr)
        return {
            "stmt": "reinit",
            "target": target,
            "expr": _export_expr(value_expr),
        }

    # Regular equation: lhs = rhs becomes assignment lhs := rhs
    if eq.lhs is not None:
        # Convert lhs expression to component reference for target
        if isinstance(eq.lhs, ComponentRef):
            target = _export_component_ref(eq.lhs)
        else:
            # For other expressions like der(x), we need to handle specially
            # For now, convert to string representation
            target = str(eq.lhs)
        return {
            "stmt": "assign",
            "target": target,
            "expr": _export_expr(eq.rhs),
        }
    else:
        # Implicit equation without lhs - this shouldn't happen in when clauses
        raise ValueError("Cannot convert implicit equation (no lhs) to statement")


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
        # Convert when_equations to statements
        # In when clauses, equations like "x = expr" become assignments "x := expr"
        statements = []
        for when_eq in eq.when_equations or []:
            if when_eq.eq_type == EquationType.SIMPLE and when_eq.lhs is not None:
                # Convert equation to assignment statement
                statements.append(_equation_to_statement(when_eq))
            # Nested when equations could be handled here if needed
        return {
            "eq_type": "when",
            "condition": _export_expr(eq.condition),
            "statements": statements,
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
        # WhenStatement uses branches format, export first branch for Base Modelica
        # (elsewhen branches would need multiple when statements in Base Modelica)
        if stmt.branches:
            condition, body = stmt.branches[0]
            return {
                "stmt": "when",
                "condition": _export_expr(condition),
                "body": [_export_statement(s) for s in body],
            }
        else:
            return {"stmt": "when", "condition": {"op": "literal", "value": False}, "body": []}

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
