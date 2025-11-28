"""
CasADi backend for the Cyecca DSL.

Compiles FlatModel representations into CasADi functions for
simulation and optimization.

================================================================================
PROTOTYPE MODE - API IS IN FLUX
================================================================================

This DSL is in active prototype development. The API may change significantly
between versions. Do NOT maintain backward compatibility - iterate rapidly.

================================================================================
CasADi SX vs MX
================================================================================

CasADi offers two symbolic types:

- **SX**: Scalar symbolic expressions. Each element is a separate symbol.
  Good for small models, exact Jacobian sparsity patterns.
  Use with `flatten(expand_arrays=True)` (default).

- **MX**: Matrix symbolic expressions. Keeps matrix structure.
  Better for large-scale problems, matrix operations (A @ x), efficient graphs.
  Use with `flatten(expand_arrays=False)` to preserve array structure.

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import casadi as ca
import numpy as np
from beartype import beartype

from cyecca.dsl.model import Expr, ExprKind, FlatModel, Reinit, SymbolicVar, WhenClause
from cyecca.dsl.simulation import SimulationResult, Simulator


class SymbolicType(Enum):
    """CasADi symbolic type selection."""

    SX = auto()  # Scalar symbolic - expands arrays, good for small models
    MX = auto()  # Matrix symbolic - keeps array structure, good for large models


class CasadiBackend:
    """
    Backend that compiles FlatModel to CasADi functions.

    Supports two symbolic types:
    - SX (default): Scalar expressions, uses expanded arrays
    - MX: Matrix expressions, keeps array structure for efficiency

    Example
    -------
    >>> model_instance = MyModel()  # doctest: +SKIP
    >>> flat = model_instance.flatten()  # doctest: +SKIP
    >>> compiled = CasadiBackend.compile(flat)  # doctest: +SKIP
    >>> result = compiled.simulate(tf=10.0)  # doctest: +SKIP

    For large models with arrays, use MX:

    >>> flat_mx = model_instance.flatten(expand_arrays=False)  # doctest: +SKIP
    >>> compiled_mx = CasadiBackend.compile(flat_mx, symbolic_type=SymbolicType.MX)  # doctest: +SKIP
    """

    @staticmethod
    @beartype
    def compile(model: FlatModel, symbolic_type: SymbolicType = SymbolicType.SX) -> "CompiledModel":
        """
        Compile a FlatModel into a CompiledModel with CasADi functions.

        Parameters
        ----------
        model : FlatModel
            The flattened model representation
        symbolic_type : SymbolicType, default=SymbolicType.SX
            Which CasADi symbolic type to use:
            - SX: Scalar expressions (use with expand_arrays=True)
            - MX: Matrix expressions (use with expand_arrays=False)

        Returns
        -------
        CompiledModel
            A compiled model ready for simulation
        """
        if symbolic_type == SymbolicType.MX:
            return CasadiBackend._compile_mx(model)
        else:
            return CasadiBackend._compile_sx(model)

    @staticmethod
    def _compile_sx(model: FlatModel) -> "CompiledModel":
        """Compile using CasADi SX (scalar symbolic expressions)."""
        # Create CasADi symbols for each variable
        state_syms: Dict[str, ca.SX] = {}
        for name in model.state_names:
            v = model.state_vars[name]
            state_syms[name] = ca.SX.sym(name, v.size) if v.size > 1 else ca.SX.sym(name)

        input_syms: Dict[str, ca.SX] = {}
        for name in model.input_names:
            v = model.input_vars[name]
            input_syms[name] = ca.SX.sym(name, v.size) if v.size > 1 else ca.SX.sym(name)

        param_syms: Dict[str, ca.SX] = {}
        for name in model.param_names:
            v = model.param_vars[name]
            param_syms[name] = ca.SX.sym(name, v.size) if v.size > 1 else ca.SX.sym(name)

        t_sym = ca.SX.sym("t")

        # Symbol lookup for expression compilation (NOT including outputs)
        # Outputs will be substituted with their expressions, not used as symbols
        base_syms = {**state_syms, **input_syms, **param_syms}

        # Cache for compiled output expressions (to handle interdependent outputs)
        compiled_outputs: Dict[str, ca.SX] = {}

        def expr_to_casadi(expr: Expr) -> ca.SX:
            """Recursively convert Expr tree to CasADi expression."""
            if expr.kind == ExprKind.VARIABLE:
                if expr.name in base_syms:
                    return base_syms[expr.name]
                # Check if it's an output - substitute with its expression
                if expr.name in model.output_equations:
                    if expr.name in compiled_outputs:
                        return compiled_outputs[expr.name]
                    # Compile and cache the output expression
                    out_expr = expr_to_casadi(model.output_equations[expr.name])
                    compiled_outputs[expr.name] = out_expr
                    return out_expr
                raise ValueError(f"Unknown variable: {expr.name}")

            elif expr.kind == ExprKind.DERIVATIVE:
                # For derivatives, we need the corresponding state's derivative symbol
                # This will be handled when building the xdot vector
                raise ValueError("DERIVATIVE nodes should not appear in RHS expressions")

            elif expr.kind == ExprKind.CONSTANT:
                return ca.SX(expr.value)

            elif expr.kind == ExprKind.TIME:
                return t_sym

            elif expr.kind == ExprKind.NEG:
                return -expr_to_casadi(expr.children[0])

            elif expr.kind == ExprKind.ADD:
                return expr_to_casadi(expr.children[0]) + expr_to_casadi(expr.children[1])

            elif expr.kind == ExprKind.SUB:
                return expr_to_casadi(expr.children[0]) - expr_to_casadi(expr.children[1])

            elif expr.kind == ExprKind.MUL:
                return expr_to_casadi(expr.children[0]) * expr_to_casadi(expr.children[1])

            elif expr.kind == ExprKind.DIV:
                return expr_to_casadi(expr.children[0]) / expr_to_casadi(expr.children[1])

            elif expr.kind == ExprKind.POW:
                return expr_to_casadi(expr.children[0]) ** expr_to_casadi(expr.children[1])

            elif expr.kind == ExprKind.SIN:
                return ca.sin(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.COS:
                return ca.cos(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.TAN:
                return ca.tan(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.ASIN:
                return ca.asin(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.ACOS:
                return ca.acos(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.ATAN:
                return ca.atan(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.ATAN2:
                return ca.atan2(expr_to_casadi(expr.children[0]), expr_to_casadi(expr.children[1]))

            elif expr.kind == ExprKind.SQRT:
                return ca.sqrt(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.EXP:
                return ca.exp(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.LOG:
                return ca.log(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.LOG10:
                return ca.log10(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.ABS:
                return ca.fabs(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.SIGN:
                return ca.sign(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.FLOOR:
                return ca.floor(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.CEIL:
                return ca.ceil(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.SINH:
                return ca.sinh(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.COSH:
                return ca.cosh(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.TANH:
                return ca.tanh(expr_to_casadi(expr.children[0]))

            elif expr.kind == ExprKind.MIN:
                return ca.fmin(expr_to_casadi(expr.children[0]), expr_to_casadi(expr.children[1]))

            elif expr.kind == ExprKind.MAX:
                return ca.fmax(expr_to_casadi(expr.children[0]), expr_to_casadi(expr.children[1]))

            # Relational operators
            elif expr.kind == ExprKind.LT:
                return expr_to_casadi(expr.children[0]) < expr_to_casadi(expr.children[1])

            elif expr.kind == ExprKind.LE:
                return expr_to_casadi(expr.children[0]) <= expr_to_casadi(expr.children[1])

            elif expr.kind == ExprKind.GT:
                return expr_to_casadi(expr.children[0]) > expr_to_casadi(expr.children[1])

            elif expr.kind == ExprKind.GE:
                return expr_to_casadi(expr.children[0]) >= expr_to_casadi(expr.children[1])

            elif expr.kind == ExprKind.EQ:
                return expr_to_casadi(expr.children[0]) == expr_to_casadi(expr.children[1])

            elif expr.kind == ExprKind.NE:
                return expr_to_casadi(expr.children[0]) != expr_to_casadi(expr.children[1])

            # Boolean operators
            elif expr.kind == ExprKind.AND:
                return ca.logic_and(expr_to_casadi(expr.children[0]), expr_to_casadi(expr.children[1]))

            elif expr.kind == ExprKind.OR:
                return ca.logic_or(expr_to_casadi(expr.children[0]), expr_to_casadi(expr.children[1]))

            elif expr.kind == ExprKind.NOT:
                return ca.logic_not(expr_to_casadi(expr.children[0]))

            # Conditional expression
            elif expr.kind == ExprKind.IF_THEN_ELSE:
                return ca.if_else(
                    expr_to_casadi(expr.children[0]), expr_to_casadi(expr.children[1]), expr_to_casadi(expr.children[2])
                )

            elif expr.kind in (ExprKind.PRE, ExprKind.EDGE, ExprKind.CHANGE):
                raise NotImplementedError(
                    f"Discrete operator '{expr.kind.name.lower()}()' is not yet supported "
                    "in the CasADi backend. Discrete event handling (when-equations) "
                    "will be added in a future version."
                )

            else:
                raise ValueError(f"Unsupported expression kind: {expr.kind}")

        # Build state derivatives vector
        state_derivs: List[ca.SX] = []
        for name in model.state_names:
            if name in model.derivative_equations:
                deriv_expr = model.derivative_equations[name]
                state_derivs.append(expr_to_casadi(deriv_expr))
            else:
                # No derivative equation - use zero
                v = model.state_vars[name]
                size = v.size if v.size > 1 else 1
                state_derivs.append(ca.SX.zeros(size))

        # Build output expressions vector
        y_exprs: List[ca.SX] = []
        for name in model.output_names:
            if name in model.output_equations:
                out_expr = model.output_equations[name]
                y_exprs.append(expr_to_casadi(out_expr))
            else:
                # No equation for this output - warn and use zero
                import warnings

                warnings.warn(f"Output '{name}' has no equation, will be zero")
                v = model.output_vars[name]
                size = v.size if v.size > 1 else 1
                y_exprs.append(ca.SX.zeros(size))

        # Build CasADi vectors
        state_list = [state_syms[n] for n in model.state_names]
        input_list = [input_syms[n] for n in model.input_names]
        param_list = [param_syms[n] for n in model.param_names]

        x = ca.vertcat(*state_list) if state_list else ca.SX.sym("x", 0)
        xdot = ca.vertcat(*state_derivs) if state_derivs else ca.SX.sym("xdot", 0)
        u = ca.vertcat(*input_list) if input_list else ca.SX.sym("u", 0)
        p = ca.vertcat(*param_list) if param_list else ca.SX.sym("p", 0)
        y = ca.vertcat(*y_exprs) if y_exprs else ca.SX.sym("y", 0)

        # Create dynamics function f_x(x, u, p, t) -> xdot
        f_x = ca.Function("f_x", [x, u, p, t_sym], [xdot], ["x", "u", "p", "t"], ["xdot"])

        # Create output function f_y(x, u, p, t) -> y
        f_y = ca.Function("f_y", [x, u, p, t_sym], [y], ["x", "u", "p", "t"], ["y"]) if y_exprs else None

        # Build when-clause functions for event handling
        # Each when-clause becomes:
        # - f_event: (x, u, p, t) -> condition_value (for zero-crossing detection)
        # - f_reinit: (x, u, p, t) -> x_new (for state reinitialization)
        when_clause_funcs: List[Dict[str, Any]] = []

        # Create pre-event state symbols for pre() operator
        pre_syms: Dict[str, ca.SX] = {}
        for name in model.state_names:
            v = model.state_vars[name]
            pre_syms[name] = ca.SX.sym(f"pre_{name}", v.size) if v.size > 1 else ca.SX.sym(f"pre_{name}")

        # Merge pre-symbols with base symbols for when-clause expression compilation
        when_clause_syms = {**base_syms, **{f"__pre_{k}": v for k, v in pre_syms.items()}}

        def expr_to_casadi_when(expr: Expr) -> ca.SX:
            """Convert Expr to CasADi, handling pre() for when-clauses."""
            if expr.kind == ExprKind.PRE:
                # pre(x) -> use pre-event state symbol
                pre_name = expr.name
                if pre_name in pre_syms:
                    return pre_syms[pre_name]
                # If not a state, just use current value
                if pre_name in base_syms:
                    return base_syms[pre_name]
                raise ValueError(f"Unknown variable in pre(): {pre_name}")
            elif expr.kind == ExprKind.VARIABLE:
                if expr.name in base_syms:
                    return base_syms[expr.name]
                raise ValueError(f"Unknown variable: {expr.name}")
            elif expr.kind == ExprKind.CONSTANT:
                return ca.SX(expr.value)
            elif expr.kind == ExprKind.TIME:
                return t_sym
            elif expr.kind == ExprKind.NEG:
                return -expr_to_casadi_when(expr.children[0])
            elif expr.kind == ExprKind.ADD:
                return expr_to_casadi_when(expr.children[0]) + expr_to_casadi_when(expr.children[1])
            elif expr.kind == ExprKind.SUB:
                return expr_to_casadi_when(expr.children[0]) - expr_to_casadi_when(expr.children[1])
            elif expr.kind == ExprKind.MUL:
                return expr_to_casadi_when(expr.children[0]) * expr_to_casadi_when(expr.children[1])
            elif expr.kind == ExprKind.DIV:
                return expr_to_casadi_when(expr.children[0]) / expr_to_casadi_when(expr.children[1])
            elif expr.kind == ExprKind.POW:
                return expr_to_casadi_when(expr.children[0]) ** expr_to_casadi_when(expr.children[1])
            elif expr.kind == ExprKind.LT:
                return expr_to_casadi_when(expr.children[0]) - expr_to_casadi_when(expr.children[1])
            elif expr.kind == ExprKind.LE:
                return expr_to_casadi_when(expr.children[0]) - expr_to_casadi_when(expr.children[1])
            elif expr.kind == ExprKind.GT:
                return expr_to_casadi_when(expr.children[1]) - expr_to_casadi_when(expr.children[0])
            elif expr.kind == ExprKind.GE:
                return expr_to_casadi_when(expr.children[1]) - expr_to_casadi_when(expr.children[0])
            else:
                # Fallback to regular expression conversion
                return expr_to_casadi(expr)

        # Build pre-event state vector
        pre_list = [pre_syms[n] for n in model.state_names]
        x_pre = ca.vertcat(*pre_list) if pre_list else ca.SX.sym("x_pre", 0)

        for i, wc in enumerate(model.when_clauses):
            # Build event function: returns value that crosses zero
            # For conditions like (h < 0), we want (h - 0) = h as the zero-crossing function
            event_expr = expr_to_casadi_when(wc.condition)
            f_event = ca.Function(
                f"f_event_{i}",
                [x, u, p, t_sym],
                [event_expr],
                ["x", "u", "p", "t"],
                ["event"],
            )

            # Build reinit function: computes new state after event
            # Start with identity (x_new = x_pre), then apply reinits
            x_new_list = [pre_syms[n] for n in model.state_names]

            for item in wc.body:
                if isinstance(item, Reinit):
                    # Find the state index
                    var_name = item.var_name
                    if var_name in model.state_names:
                        idx = model.state_names.index(var_name)
                        # Compile the reinit expression
                        new_val = expr_to_casadi_when(item.expr)
                        x_new_list[idx] = new_val

            x_new = ca.vertcat(*x_new_list) if x_new_list else ca.SX.sym("x_new", 0)

            f_reinit = ca.Function(
                f"f_reinit_{i}",
                [x_pre, u, p, t_sym],
                [x_new],
                ["x_pre", "u", "p", "t"],
                ["x_new"],
            )

            when_clause_funcs.append({
                "condition": wc.condition,
                "f_event": f_event,
                "f_reinit": f_reinit,
            })

        return CompiledModel(
            name=model.name,
            f_x=f_x,
            f_y=f_y,
            _state_names=model.state_names,
            _input_names=model.input_names,
            _param_names=model.param_names,
            _output_names=model.output_names,
            state_defaults=model.state_defaults,
            input_defaults=model.input_defaults,
            param_defaults=model.param_defaults,
            when_clause_funcs=when_clause_funcs,
        )

    @staticmethod
    def _compile_mx(model: FlatModel) -> "CompiledModel":
        """
        Compile using CasADi MX (matrix symbolic expressions).

        This method is designed for models with arrays and matrices.
        It keeps the array structure instead of expanding to scalars,
        which is more efficient for large-scale problems.

        Use with: model.flatten(expand_arrays=False)
        """
        import math

        # Helper to compute size from shape
        def shape_to_size(shape: Tuple[int, ...]) -> int:
            if not shape:
                return 1
            result = 1
            for dim in shape:
                result *= dim
            return result

        # Create CasADi MX symbols for each variable (preserving shape)
        state_syms: Dict[str, ca.MX] = {}
        state_shapes: Dict[str, Tuple[int, ...]] = {}
        for name in model.state_names:
            v = model.state_vars[name]
            shape = v.shape if v.shape else ()
            state_shapes[name] = shape
            size = shape_to_size(shape)
            # MX.sym creates column vectors by default
            state_syms[name] = ca.MX.sym(name, size)

        input_syms: Dict[str, ca.MX] = {}
        for name in model.input_names:
            v = model.input_vars[name]
            size = shape_to_size(v.shape) if v.shape else 1
            input_syms[name] = ca.MX.sym(name, size)

        param_syms: Dict[str, ca.MX] = {}
        for name in model.param_names:
            v = model.param_vars[name]
            size = shape_to_size(v.shape) if v.shape else 1
            param_syms[name] = ca.MX.sym(name, size)

        # Algebraic variables (computed from equations, not states)
        algebraic_syms: Dict[str, ca.MX] = {}
        algebraic_shapes: Dict[str, Tuple[int, ...]] = {}
        for name in model.algebraic_names:
            v = model.algebraic_vars[name]
            shape = v.shape if v.shape else ()
            algebraic_shapes[name] = shape
            size = shape_to_size(shape)
            algebraic_syms[name] = ca.MX.sym(name, size)

        t_sym = ca.MX.sym("t")

        # Symbol lookup for expression compilation
        # Include all variable types
        all_shapes = {**state_shapes, **algebraic_shapes}
        base_syms = {**state_syms, **input_syms, **param_syms, **algebraic_syms}

        def expr_to_casadi_mx(expr: Expr) -> ca.MX:
            """Recursively convert Expr tree to CasADi MX expression."""
            if expr.kind == ExprKind.VARIABLE:
                # Handle indexed variables like "pos[0]" or "R[0,1]"
                name = expr.name
                if name in base_syms:
                    return base_syms[name]
                # Check for indexed access (name contains "[")
                if "[" in name:
                    base_name = name.split("[")[0]
                    if base_name in base_syms:
                        # Parse indices from name like "pos[0]" or "R[0,1]"
                        idx_str = name[name.index("[") + 1 : name.rindex("]")]
                        indices = tuple(int(i) for i in idx_str.split(","))
                        sym = base_syms[base_name]
                        shape = all_shapes.get(base_name, ())
                        # Convert multi-dimensional index to flat index
                        if len(indices) == 1:
                            return sym[indices[0]]
                        else:
                            # Row-major flattening
                            flat_idx = 0
                            stride = 1
                            for i in range(len(indices) - 1, -1, -1):
                                flat_idx += indices[i] * stride
                                if i > 0:
                                    stride *= shape[i]
                            return sym[flat_idx]
                raise ValueError(f"Unknown variable: {name}")

            elif expr.kind == ExprKind.DERIVATIVE:
                raise ValueError("DERIVATIVE nodes should not appear in RHS expressions")

            elif expr.kind == ExprKind.CONSTANT:
                return ca.MX(expr.value)

            elif expr.kind == ExprKind.TIME:
                return t_sym

            elif expr.kind == ExprKind.NEG:
                return -expr_to_casadi_mx(expr.children[0])

            elif expr.kind == ExprKind.ADD:
                return expr_to_casadi_mx(expr.children[0]) + expr_to_casadi_mx(expr.children[1])

            elif expr.kind == ExprKind.SUB:
                return expr_to_casadi_mx(expr.children[0]) - expr_to_casadi_mx(expr.children[1])

            elif expr.kind == ExprKind.MUL:
                return expr_to_casadi_mx(expr.children[0]) * expr_to_casadi_mx(expr.children[1])

            elif expr.kind == ExprKind.DIV:
                return expr_to_casadi_mx(expr.children[0]) / expr_to_casadi_mx(expr.children[1])

            elif expr.kind == ExprKind.POW:
                return expr_to_casadi_mx(expr.children[0]) ** expr_to_casadi_mx(expr.children[1])

            elif expr.kind == ExprKind.SIN:
                return ca.sin(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.COS:
                return ca.cos(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.TAN:
                return ca.tan(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.ASIN:
                return ca.asin(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.ACOS:
                return ca.acos(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.ATAN:
                return ca.atan(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.ATAN2:
                return ca.atan2(expr_to_casadi_mx(expr.children[0]), expr_to_casadi_mx(expr.children[1]))

            elif expr.kind == ExprKind.SQRT:
                return ca.sqrt(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.EXP:
                return ca.exp(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.LOG:
                return ca.log(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.LOG10:
                return ca.log10(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.ABS:
                return ca.fabs(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.SIGN:
                return ca.sign(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.FLOOR:
                return ca.floor(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.CEIL:
                return ca.ceil(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.SINH:
                return ca.sinh(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.COSH:
                return ca.cosh(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.TANH:
                return ca.tanh(expr_to_casadi_mx(expr.children[0]))

            elif expr.kind == ExprKind.MIN:
                return ca.fmin(expr_to_casadi_mx(expr.children[0]), expr_to_casadi_mx(expr.children[1]))

            elif expr.kind == ExprKind.MAX:
                return ca.fmax(expr_to_casadi_mx(expr.children[0]), expr_to_casadi_mx(expr.children[1]))

            # Relational operators
            elif expr.kind == ExprKind.LT:
                return expr_to_casadi_mx(expr.children[0]) < expr_to_casadi_mx(expr.children[1])

            elif expr.kind == ExprKind.LE:
                return expr_to_casadi_mx(expr.children[0]) <= expr_to_casadi_mx(expr.children[1])

            elif expr.kind == ExprKind.GT:
                return expr_to_casadi_mx(expr.children[0]) > expr_to_casadi_mx(expr.children[1])

            elif expr.kind == ExprKind.GE:
                return expr_to_casadi_mx(expr.children[0]) >= expr_to_casadi_mx(expr.children[1])

            elif expr.kind == ExprKind.EQ:
                return expr_to_casadi_mx(expr.children[0]) == expr_to_casadi_mx(expr.children[1])

            elif expr.kind == ExprKind.NE:
                return expr_to_casadi_mx(expr.children[0]) != expr_to_casadi_mx(expr.children[1])

            # Boolean operators
            elif expr.kind == ExprKind.AND:
                return ca.logic_and(expr_to_casadi_mx(expr.children[0]), expr_to_casadi_mx(expr.children[1]))

            elif expr.kind == ExprKind.OR:
                return ca.logic_or(expr_to_casadi_mx(expr.children[0]), expr_to_casadi_mx(expr.children[1]))

            elif expr.kind == ExprKind.NOT:
                return ca.logic_not(expr_to_casadi_mx(expr.children[0]))

            # Conditional expression
            elif expr.kind == ExprKind.IF_THEN_ELSE:
                return ca.if_else(
                    expr_to_casadi_mx(expr.children[0]),
                    expr_to_casadi_mx(expr.children[1]),
                    expr_to_casadi_mx(expr.children[2]),
                )

            elif expr.kind in (ExprKind.PRE, ExprKind.EDGE, ExprKind.CHANGE):
                raise NotImplementedError(
                    f"Discrete operator '{expr.kind.name.lower()}()' is not yet supported "
                    "in the CasADi backend. Discrete event handling (when-equations) "
                    "will be added in a future version."
                )

            else:
                raise ValueError(f"Unsupported expression kind: {expr.kind}")

        def symbolic_var_to_mx(sym_var: SymbolicVar) -> ca.MX:
            """Convert SymbolicVar to CasADi MX."""
            base_name = sym_var.base_name
            if base_name in base_syms:
                return base_syms[base_name]
            raise ValueError(f"Unknown symbolic variable: {base_name}")

        # Build state derivatives vector
        # For MX, we handle both scalar and array equations
        state_derivs: List[ca.MX] = []

        for name in model.state_names:
            shape = state_shapes.get(name, ())
            size = shape_to_size(shape)

            if name in model.array_derivative_equations:
                # Array equation: der(pos) = vel (whole array)
                arr_eq = model.array_derivative_equations[name]
                rhs = arr_eq["rhs"]
                # The RHS is a SymbolicVar - get its MX symbol
                state_derivs.append(symbolic_var_to_mx(rhs))
            elif name in model.derivative_equations:
                # Scalar equation: der(theta) = omega
                deriv_expr = model.derivative_equations[name]
                state_derivs.append(expr_to_casadi_mx(deriv_expr))
            else:
                # No derivative equation - check for element-wise equations
                # Look for patterns like "pos[0]", "pos[1]", etc.
                deriv_vec = ca.MX.zeros(size)
                has_any = False
                for key, deriv_expr in model.derivative_equations.items():
                    if key.startswith(f"{name}["):
                        has_any = True
                        # Parse index
                        idx_str = key[key.index("[") + 1 : key.rindex("]")]
                        indices = tuple(int(i) for i in idx_str.split(","))
                        # Convert to flat index
                        if len(indices) == 1:
                            flat_idx = indices[0]
                        else:
                            flat_idx = 0
                            stride = 1
                            for i in range(len(indices) - 1, -1, -1):
                                flat_idx += indices[i] * stride
                                if i > 0:
                                    stride *= shape[i]
                        deriv_vec[flat_idx] = expr_to_casadi_mx(deriv_expr)

                if has_any:
                    state_derivs.append(deriv_vec)
                else:
                    # No equations at all - use zeros
                    state_derivs.append(ca.MX.zeros(size))

        # Build CasADi vectors
        state_list = [state_syms[n] for n in model.state_names]
        input_list = [input_syms[n] for n in model.input_names]
        param_list = [param_syms[n] for n in model.param_names]

        x = ca.vertcat(*state_list) if state_list else ca.MX.sym("x", 0)
        xdot = ca.vertcat(*state_derivs) if state_derivs else ca.MX.sym("xdot", 0)
        u = ca.vertcat(*input_list) if input_list else ca.MX.sym("u", 0)
        p = ca.vertcat(*param_list) if param_list else ca.MX.sym("p", 0)

        # Create dynamics function f_x(x, u, p, t) -> xdot
        f_x = ca.Function("f_x", [x, u, p, t_sym], [xdot], ["x", "u", "p", "t"], ["xdot"])

        # TODO: Handle output equations for MX
        f_y = None

        return CompiledModel(
            name=model.name,
            f_x=f_x,
            f_y=f_y,
            _state_names=model.state_names,
            _input_names=model.input_names,
            _param_names=model.param_names,
            _output_names=model.output_names,
            state_defaults=model.state_defaults,
            input_defaults=model.input_defaults,
            param_defaults=model.param_defaults,
            when_clause_funcs=[],  # MX backend doesn't support when-clauses yet
        )


@dataclass
class CompiledModel(Simulator):
    """
    A compiled model ready for simulation.

    Contains CasADi functions and metadata for numerical integration.
    Implements the Simulator interface for unified simulation/plotting.

    When-Clauses (Hybrid Systems)
    -----------------------------
    If the model has when-clauses, the simulator uses event detection to
    find zero-crossings and applies state reinitializations at events.
    """

    name: str
    f_x: ca.Function  # Dynamics: f_x(x, u, p, t) -> xdot
    f_y: Optional[ca.Function]  # Outputs: f_y(x, u, p, t) -> y (None if no outputs)
    _state_names: List[str]
    _input_names: List[str]
    _param_names: List[str]
    _output_names: List[str]
    state_defaults: Dict[str, Any]
    input_defaults: Dict[str, Any]
    param_defaults: Dict[str, Any]
    when_clause_funcs: List[Dict[str, Any]] = None  # Event functions for when-clauses

    def __post_init__(self):
        if self.when_clause_funcs is None:
            self.when_clause_funcs = []

    @property
    def state_names(self) -> List[str]:
        return self._state_names

    @property
    def input_names(self) -> List[str]:
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        return self._output_names

    @property
    def param_names(self) -> List[str]:
        return self._param_names

    @property
    def has_events(self) -> bool:
        """True if this model has when-clauses (event handling)."""
        return len(self.when_clause_funcs) > 0

    @beartype
    def simulate(
        self,
        t0: Union[float, int] = 0.0,
        tf: Union[float, int] = 10.0,
        dt: Union[float, int] = 0.01,
        x0: Optional[Dict[str, Any]] = None,
        u: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        u_func: Optional[Callable[[float], Dict[str, Any]]] = None,
        max_events: int = 1000,
    ) -> SimulationResult:
        """
        Simulate the model using RK4 integration with event detection.

        For models with when-clauses, the simulator detects zero-crossings
        and applies state reinitializations at event times.

        Parameters
        ----------
        t0 : float
            Initial time
        tf : float
            Final time
        dt : float
            Time step
        x0 : dict, optional
            Initial state values (overrides defaults)
        u : dict, optional
            Constant input values
        params : dict, optional
            Parameter values (overrides defaults)
        u_func : callable, optional
            Function u_func(t) -> dict for time-varying inputs
        max_events : int, default=1000
            Maximum number of events before stopping (prevents infinite loops)

        Returns
        -------
        SimulationResult
            Simulation results with plotting utilities
        """
        # Compute number of states
        n_states = self.f_x.size1_in(0)

        # Build initial state vector
        x_init = np.zeros(n_states)
        idx = 0
        for name in self._state_names:
            if x0 and name in x0:
                val = x0[name]
            elif name in self.state_defaults:
                val = self.state_defaults[name]
            else:
                val = 0.0
            if isinstance(val, (list, np.ndarray)):
                for v in val:
                    x_init[idx] = v
                    idx += 1
            else:
                x_init[idx] = val
                idx += 1

        # Build parameter vector
        n_params = self.f_x.size1_in(2)
        p = np.zeros(n_params)
        for i, name in enumerate(self._param_names):
            if params and name in params:
                p[i] = params[name]
            elif name in self.param_defaults:
                p[i] = self.param_defaults[name]

        # Build constant input vector
        n_inputs = self.f_x.size1_in(1)
        u_const = np.zeros(n_inputs)
        for i, name in enumerate(self._input_names):
            if u and name in u:
                u_const[i] = u[name]
            elif name in self.input_defaults:
                u_const[i] = self.input_defaults[name]

        def get_input(ti: float) -> np.ndarray:
            """Get input vector at time ti."""
            if u_func is not None:
                u_dict = u_func(ti)
                u_vec = np.zeros(n_inputs)
                for j, name in enumerate(self._input_names):
                    u_vec[j] = u_dict.get(name, u_const[j])
                return u_vec
            return u_const

        def rk4_step(x: np.ndarray, ti: float, h: float) -> np.ndarray:
            """Single RK4 integration step."""
            u_vec = get_input(ti)
            k1 = np.array(self.f_x(x, u_vec, p, ti)).flatten()
            k2 = np.array(self.f_x(x + 0.5 * h * k1, u_vec, p, ti + 0.5 * h)).flatten()
            k3 = np.array(self.f_x(x + 0.5 * h * k2, u_vec, p, ti + 0.5 * h)).flatten()
            k4 = np.array(self.f_x(x + h * k3, u_vec, p, ti + h)).flatten()
            return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        def check_events(x: np.ndarray, ti: float) -> List[Tuple[int, float]]:
            """Check event functions and return list of (event_idx, value)."""
            u_vec = get_input(ti)
            events = []
            for i, wc_func in enumerate(self.when_clause_funcs):
                val = float(np.array(wc_func["f_event"](x, u_vec, p, ti)).flatten()[0])
                events.append((i, val))
            return events

        def apply_reinit(x: np.ndarray, event_idx: int, ti: float) -> np.ndarray:
            """Apply reinit for the given event."""
            u_vec = get_input(ti)
            wc_func = self.when_clause_funcs[event_idx]
            x_new = np.array(wc_func["f_reinit"](x, u_vec, p, ti)).flatten()
            return x_new

        # Use event-detecting simulation if we have when-clauses
        if self.has_events:
            return self._simulate_with_events(
                t0, tf, dt, x_init, p, n_inputs, n_states,
                get_input, rk4_step, check_events, apply_reinit, max_events
            )

        # Simple RK4 integration without events
        t = np.arange(t0, tf + dt, dt)
        n_steps = len(t)

        # Storage
        x_hist = np.zeros((n_steps, n_states))
        x_hist[0] = x_init
        u_hist = np.zeros((n_steps, n_inputs)) if n_inputs > 0 else None

        # RK4 integration
        x = x_init.copy()
        for i in range(n_steps):
            ti = t[i]
            u_vec = get_input(ti)

            # Record input
            if u_hist is not None:
                u_hist[i] = u_vec

            # RK4 step (except for last point)
            if i < n_steps - 1:
                x = rk4_step(x, ti, dt)
                x_hist[i + 1] = x

        return self._build_result(t, x_hist, u_hist, n_inputs, p, get_input)

    def _simulate_with_events(
        self,
        t0: float,
        tf: float,
        dt: float,
        x_init: np.ndarray,
        p: np.ndarray,
        n_inputs: int,
        n_states: int,
        get_input: Callable,
        rk4_step: Callable,
        check_events: Callable,
        apply_reinit: Callable,
        max_events: int,
    ) -> SimulationResult:
        """
        Simulate with event detection using bisection for zero-crossing.

        This implements a simple event-locating integrator:
        1. Take a tentative RK4 step
        2. Check if any event function changed sign (zero-crossing)
        3. If yes, use bisection to find the event time
        4. Apply reinit and continue from there
        """
        # Use variable-length lists for event-triggered simulation
        t_list: List[float] = [t0]
        x_list: List[np.ndarray] = [x_init.copy()]
        u_list: List[np.ndarray] = [get_input(t0)] if n_inputs > 0 else None

        x = x_init.copy()
        t = t0
        n_events = 0
        event_times: List[float] = []

        # Track previous event values for edge detection
        prev_events = check_events(x, t)

        while t < tf and n_events < max_events:
            # Tentative step
            h = min(dt, tf - t)
            x_next = rk4_step(x, t, h)
            t_next = t + h

            # Check for events (sign changes in event functions)
            curr_events = check_events(x_next, t_next)

            event_occurred = False
            triggered_event = -1

            for i, (_, val) in enumerate(curr_events):
                prev_val = prev_events[i][1]
                # Detect negative-going zero crossing (condition becomes True)
                # For (h < 0), we want to trigger when h goes from positive to negative
                if prev_val > 0 and val <= 0:
                    event_occurred = True
                    triggered_event = i
                    break

            if event_occurred:
                # Bisection to find event time
                t_lo, t_hi = t, t_next
                x_lo = x.copy()

                for _ in range(20):  # Max 20 bisection iterations
                    t_mid = 0.5 * (t_lo + t_hi)
                    h_mid = t_mid - t
                    x_mid = rk4_step(x, t, h_mid)
                    mid_events = check_events(x_mid, t_mid)
                    mid_val = mid_events[triggered_event][1]

                    if mid_val <= 0:
                        t_hi = t_mid
                    else:
                        t_lo = t_mid
                        x_lo = x_mid

                    if abs(t_hi - t_lo) < 1e-10:
                        break

                # Event time is t_hi, state just before event is at t_lo
                t_event = t_hi
                x_pre = rk4_step(x, t, t_event - t)

                # Apply reinit
                x_post = apply_reinit(x_pre, triggered_event, t_event)

                # Record event
                t_list.append(t_event)
                x_list.append(x_post.copy())
                if u_list is not None:
                    u_list.append(get_input(t_event))

                event_times.append(t_event)
                n_events += 1

                # Continue from post-event state
                x = x_post.copy()
                t = t_event
                prev_events = check_events(x, t)

            else:
                # No event - accept the step
                t = t_next
                x = x_next.copy()
                t_list.append(t)
                x_list.append(x.copy())
                if u_list is not None:
                    u_list.append(get_input(t))
                prev_events = curr_events

        if n_events >= max_events:
            import warnings
            warnings.warn(f"Maximum number of events ({max_events}) reached. Simulation may be incomplete.")

        # Convert lists to arrays
        t_arr = np.array(t_list)
        x_arr = np.array(x_list)
        u_arr = np.array(u_list) if u_list is not None else None

        return self._build_result(t_arr, x_arr, u_arr, n_inputs, p, get_input)

    def _build_result(
        self,
        t: np.ndarray,
        x_hist: np.ndarray,
        u_hist: Optional[np.ndarray],
        n_inputs: int,
        p: np.ndarray,
        get_input: Callable,
    ) -> SimulationResult:
        """Build SimulationResult from trajectory data."""
        n_steps = len(t)

        # Convert states to named dict
        states: Dict[str, np.ndarray] = {}
        idx = 0
        for name in self._state_names:
            states[name] = x_hist[:, idx]
            idx += 1

        # Convert inputs to named dict
        inputs: Dict[str, np.ndarray] = {}
        if u_hist is not None:
            for j, name in enumerate(self._input_names):
                inputs[name] = u_hist[:, j]

        # Compute outputs if f_y is available
        outputs: Dict[str, np.ndarray] = {}
        if self.f_y is not None:
            n_outputs = self.f_y.size1_out(0)
            y_hist = np.zeros((n_steps, n_outputs))
            for i in range(n_steps):
                ti = t[i]
                u_vec = get_input(ti)
                y_hist[i] = np.array(self.f_y(x_hist[i], u_vec, p, ti)).flatten()

            for j, name in enumerate(self._output_names):
                outputs[name] = y_hist[:, j]

        # Build data dict with all trajectories
        data: Dict[str, np.ndarray] = {}
        data.update(states)
        data.update(outputs)
        data.update(inputs)

        return SimulationResult(
            t=t,
            _data=data,
            model_name=self.name,
            state_names=list(self._state_names),
            output_names=list(self._output_names),
            input_names=list(self._input_names),
        )

    def __repr__(self) -> str:
        parts = [f"'{self.name}'", f"states={self._state_names}"]
        if self._input_names:
            parts.append(f"inputs={self._input_names}")
        if self._param_names:
            parts.append(f"params={self._param_names}")
        if self._output_names:
            parts.append(f"outputs={self._output_names}")
        if self.has_events:
            parts.append(f"events={len(self.when_clause_funcs)}")
        return f"CompiledModel({', '.join(parts)})"
