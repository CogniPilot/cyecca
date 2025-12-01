"""
SymPy backend for symbolic manipulation and analysis.

This backend converts Cyecca IR to SymPy symbolic expressions, enabling:
- Symbolic simplification
- Analytical Jacobians
- LaTeX export for documentation
- Symbolic solutions (where possible)
- Taylor series expansions
"""

from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import sympy as sp
from sympy import symbols, lambdify, Matrix, latex, simplify

from cyecca.backends.base import Backend
from cyecca.ir.model import Model
from cyecca.ir.expr import (
    Expr,
    Literal,
    VarRef,
    ComponentRef,
    ArrayRef,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    IfExpr,
    ArrayLiteral,
    Slice,
)
from cyecca.ir.equation import Equation, EquationType
from cyecca.ir.variable import Variable
from cyecca.ir.types import VariableType


class SympyBackend(Backend):
    """
    SymPy backend for symbolic computation.

    Features:
    - Symbolic expression manipulation
    - Analytical derivatives
    - LaTeX export
    - Symbolic simplification
    - Taylor series
    - Array/vector support
    """

    def __init__(self, model: Model) -> None:
        super(SympyBackend, self).__init__(model)

        # SymPy symbols for each variable
        # For scalars: symbols["x"] = Symbol("x")
        # For arrays: symbols["x"] = IndexedBase("x") and symbols[("x", 0)] = x[0], etc.
        self.symbols: dict[str, sp.Basic] = {}

        # Variable shapes for arrays (None = scalar)
        self.var_shapes: dict[str, Optional[list[int]]] = {}

        # Symbolic expressions for derivatives
        # For scalars: derivatives["x"] = expr
        # For array elements: derivatives[("x", idx)] = expr (0-based index)
        self.derivatives: dict[str | tuple[str, int], sp.Expr] = {}

        # Symbolic expressions for algebraic equations
        self.algebraic: dict[str, sp.Expr] = {}

        # Symbolic expressions for outputs
        self.outputs: dict[str, sp.Expr] = {}

        # Jacobians (computed on demand)
        self._jacobian_state: Optional[sp.Matrix] = None
        self._jacobian_input: Optional[sp.Matrix] = None
        self._jacobian_output_state: Optional[sp.Matrix] = None
        self._jacobian_output_input: Optional[sp.Matrix] = None

    def compile(self) -> None:
        """Compile the IR model to SymPy symbolic expressions."""
        # Create SymPy symbols for all variables
        for var in self.model.variables:
            self.var_shapes[var.name] = var.shape

            if var.is_array:
                shape = var.shape
                if len(shape) == 1:
                    # Vector: Real x[n] -> create individual symbols x_0, x_1, ...
                    # Using individual symbols is more compatible with SymPy operations
                    n = shape[0]
                    for i in range(n):
                        sym = sp.Symbol(f"{var.name}_{i}", real=True)
                        self.symbols[(var.name, i)] = sym
                    # Also store a MatrixSymbol for whole-array operations
                    self.symbols[var.name] = sp.MatrixSymbol(var.name, n, 1)
                elif len(shape) == 2:
                    # Matrix: Real A[n,m]
                    n, m = shape
                    for i in range(n):
                        for j in range(m):
                            sym = sp.Symbol(f"{var.name}_{i}_{j}", real=True)
                            self.symbols[(var.name, i, j)] = sym
                    self.symbols[var.name] = sp.MatrixSymbol(var.name, n, m)
                else:
                    raise NotImplementedError(
                        f"Arrays with more than 2 dimensions not supported: {var.name} has shape {shape}"
                    )
            else:
                # Scalar variable
                self.symbols[var.name] = sp.Symbol(var.name, real=True)

        # Convert equations to SymPy expressions
        for eq in self.model.equations:
            if eq.eq_type == EquationType.SIMPLE:
                # Handle: lhs = rhs
                if eq.lhs is not None and isinstance(eq.lhs, FunctionCall) and eq.lhs.func == "der":
                    # Derivative equation: der(x) = expr or der(x[i]) = expr
                    if len(eq.lhs.args) > 0:
                        arg = eq.lhs.args[0]
                        state_name = self._get_var_name(arg)
                        expr_sympy = self._convert_expr(eq.rhs)

                        # Check if this is an array element derivative
                        elem_idx = self._get_element_index(arg)
                        if elem_idx is not None:
                            # Array element: der(x[i])
                            self.derivatives[(state_name, elem_idx)] = expr_sympy
                        else:
                            # Scalar: der(x)
                            self.derivatives[state_name] = expr_sympy

                elif eq.lhs is not None:
                    # Algebraic equation: var = expr
                    var_name = self._get_var_name(eq.lhs)
                    expr_sympy = self._convert_expr(eq.rhs)
                    self.algebraic[var_name] = expr_sympy

                elif eq.lhs is None:
                    # Implicit equation: 0 = rhs
                    # Store as a constraint
                    pass
                else:
                    raise ValueError(f"Unsupported equation lhs: {eq.lhs}")

            elif eq.eq_type == EquationType.WHEN:
                # SymPy doesn't handle discrete/hybrid well - skip for now
                pass

            elif eq.eq_type == EquationType.INITIAL:
                # Initial equations - skip for now
                pass

            else:
                raise ValueError(f"Unsupported equation type: {eq.eq_type}")

        self._compiled = True

    def _flatten_component_ref(self, ref: ComponentRef) -> str:
        """
        Flatten a hierarchical component reference to a single name.

        Examples:
            x -> "x"
            vehicle.engine.temp -> "vehicle.engine.temp"
            positions[1].x -> "positions[1].x"  (for lookup, subscripts handled separately)

        For variable lookup, we use the flattened name without the final subscripts.
        """
        parts_str = []
        for i, part in enumerate(ref.parts):
            if i < len(ref.parts) - 1:
                # Intermediate parts: include subscripts in the name
                parts_str.append(str(part))
            else:
                # Final part: just the name (subscripts handled separately)
                parts_str.append(part.name)
        return ".".join(parts_str)

    def _get_var_name(self, expr: Expr) -> str:
        """Extract variable name from an expression.

        For array element references like x[1], returns the base variable name 'x'.
        For hierarchical refs like a.b.c, returns the flattened name 'a.b.c'.
        """
        if isinstance(expr, ComponentRef):
            return self._flatten_component_ref(expr)
        elif isinstance(expr, VarRef):
            return expr.name
        else:
            raise ValueError(f"Cannot extract variable name from: {expr}")

    def _get_element_index(self, expr: Expr) -> Optional[int]:
        """Extract element index from an array element reference.

        For x[1], returns 0 (0-based index).
        For a.b.c[1], returns 0 (0-based index from last part).
        For scalar x or a.b.c, returns None.

        Note: Modelica uses 1-based indexing, this returns 0-based.
        """
        if isinstance(expr, ComponentRef):
            # Check the last part for subscripts
            last_part = expr.parts[-1]
            if last_part.subscripts and len(last_part.subscripts) == 1:
                sub = last_part.subscripts[0]
                if isinstance(sub, Literal):
                    return int(sub.value) - 1  # Convert to 0-based
        return None

    def _apply_subscripts(self, var_name: str, subscripts: tuple[Expr, ...]) -> sp.Expr:
        """
        Apply subscripts to extract elements from an array symbol.

        Args:
            var_name: Variable name
            subscripts: Tuple of subscript expressions

        Returns:
            The indexed element symbol

        Note:
            Modelica uses 1-based indexing, we store symbols with 0-based keys.
        """
        shape = self.var_shapes.get(var_name)

        if shape is None:
            raise ValueError(f"Cannot index scalar variable: {var_name}")

        if len(subscripts) == 1:
            # Vector indexing: x[i]
            sub = subscripts[0]
            if isinstance(sub, Literal):
                idx_0based = int(sub.value) - 1
                key = (var_name, idx_0based)
                if key not in self.symbols:
                    raise ValueError(f"Index out of bounds: {var_name}[{sub.value}]")
                return self.symbols[key]
            elif isinstance(sub, Slice):
                raise NotImplementedError("Slice indexing not yet supported in SymPy backend")
            else:
                raise NotImplementedError(
                    f"Symbolic array indexing not yet supported in SymPy backend. "
                    f"Variable '{var_name}' indexed with non-literal: {sub}"
                )

        elif len(subscripts) == 2:
            # Matrix indexing: A[i,j]
            sub_row, sub_col = subscripts
            if isinstance(sub_row, Literal) and isinstance(sub_col, Literal):
                row_0based = int(sub_row.value) - 1
                col_0based = int(sub_col.value) - 1
                key = (var_name, row_0based, col_0based)
                if key not in self.symbols:
                    raise ValueError(
                        f"Index out of bounds: {var_name}[{sub_row.value},{sub_col.value}]"
                    )
                return self.symbols[key]
            else:
                raise NotImplementedError(
                    f"Symbolic matrix indexing not yet supported in SymPy backend"
                )

        else:
            raise NotImplementedError(f"Indexing with {len(subscripts)} dimensions not supported")

    def _convert_expr(self, expr: Expr) -> sp.Basic:
        """Convert a Cyecca IR expression to a SymPy expression."""
        if isinstance(expr, Literal):
            if isinstance(expr.value, float):
                return sp.Float(expr.value)
            elif isinstance(expr.value, int):
                return sp.Integer(expr.value)
            elif isinstance(expr.value, bool):
                return sp.true if expr.value else sp.false
            else:
                return sp.sympify(expr.value)

        elif isinstance(expr, ComponentRef):
            # Flatten hierarchical reference to variable name
            var_name = self._flatten_component_ref(expr)
            last_part = expr.parts[-1]

            # Handle subscripts (array indexing) on the last part
            if last_part.subscripts:
                return self._apply_subscripts(var_name, last_part.subscripts)
            else:
                if var_name not in self.symbols:
                    raise ValueError(f"Unknown variable: {var_name}")
                return self.symbols[var_name]

        elif isinstance(expr, VarRef):
            # Backward compatibility
            if expr.name not in self.symbols:
                raise ValueError(f"Unknown variable: {expr.name}")
            return self.symbols[expr.name]

        elif isinstance(expr, ArrayRef):
            # Legacy ArrayRef - convert to subscript access
            return self._apply_subscripts(expr.name, expr.indices)

        elif isinstance(expr, BinaryOp):
            left = self._convert_expr(expr.left)
            right = self._convert_expr(expr.right)

            op_map = {
                "+": lambda l, r: l + r,
                "-": lambda l, r: l - r,
                "*": lambda l, r: l * r,
                "/": lambda l, r: l / r,
                "^": lambda l, r: l**r,
                "==": lambda l, r: sp.Eq(l, r),
                "!=": lambda l, r: sp.Ne(l, r),
                "<": lambda l, r: l < r,
                "<=": lambda l, r: l <= r,
                ">": lambda l, r: l > r,
                ">=": lambda l, r: l >= r,
                "and": lambda l, r: sp.And(l, r),
                "or": lambda l, r: sp.Or(l, r),
            }

            if expr.op in op_map:
                return op_map[expr.op](left, right)
            else:
                raise ValueError(f"Unsupported binary operator: {expr.op}")

        elif isinstance(expr, UnaryOp):
            operand = self._convert_expr(expr.operand)

            if expr.op == "-":
                return -operand
            elif expr.op == "+":
                return operand
            elif expr.op == "not":
                return sp.Not(operand)
            else:
                raise ValueError(f"Unsupported unary operator: {expr.op}")

        elif isinstance(expr, FunctionCall):
            # Special handling for Modelica operators that shouldn't be converted
            if expr.func in ("der", "pre", "edge"):
                raise ValueError(
                    f"Operator '{expr.func}' should not appear in RHS expressions. "
                    f"It should only appear in LHS of equations or special contexts."
                )

            args = [self._convert_expr(arg) for arg in expr.args]

            # Map Cyecca function names to SymPy functions
            func_map = {
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "asin": sp.asin,
                "acos": sp.acos,
                "atan": sp.atan,
                "atan2": sp.atan2,
                "sinh": sp.sinh,
                "cosh": sp.cosh,
                "tanh": sp.tanh,
                "asinh": sp.asinh,
                "acosh": sp.acosh,
                "atanh": sp.atanh,
                "exp": sp.exp,
                "log": sp.log,
                "ln": sp.log,
                "log10": lambda x: sp.log(x, 10),
                "sqrt": sp.sqrt,
                "abs": sp.Abs,
                "sign": sp.sign,
                "min": sp.Min,
                "max": sp.Max,
                "floor": sp.floor,
                "ceil": sp.ceiling,
                "pow": lambda x, y: x**y,
                "mod": sp.Mod,
            }

            if expr.func in func_map:
                return func_map[expr.func](*args)
            else:
                raise ValueError(f"Unsupported function: {expr.func}")

        elif isinstance(expr, IfExpr):
            cond = self._convert_expr(expr.condition)
            true_val = self._convert_expr(expr.true_expr)
            false_val = self._convert_expr(expr.false_expr)
            return sp.Piecewise((true_val, cond), (false_val, True))

        elif isinstance(expr, ArrayLiteral):
            # Convert array elements to SymPy and create a Matrix
            elements = [self._convert_expr(e) for e in expr.elements]
            return sp.Matrix(elements)

        elif isinstance(expr, Slice):
            # Slices cannot be evaluated directly - they must appear in context (array subscripts)
            raise ValueError(
                f"Slice expressions cannot be converted to SymPy directly. "
                f"They must be used as subscripts in array references."
            )

        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")

    def simplify(self, var_name: str) -> sp.Expr:
        """
        Simplify the expression for a variable.

        Args:
            var_name: Name of the variable (derivative or algebraic)

        Returns:
            Simplified SymPy expression
        """
        self._ensure_compiled()

        if var_name in self.derivatives:
            expr = self.derivatives[var_name]
        elif var_name in self.algebraic:
            expr = self.algebraic[var_name]
        else:
            raise ValueError(f"Unknown variable: {var_name}")

        return simplify(expr)

    def to_latex(self, var_name: str, simplified: bool = True) -> str:
        """
        Convert an equation to LaTeX.

        Args:
            var_name: Name of the variable
            simplified: Whether to simplify before converting

        Returns:
            LaTeX string
        """
        self._ensure_compiled()

        if simplified:
            expr = self.simplify(var_name)
        elif var_name in self.derivatives:
            expr = self.derivatives[var_name]
        elif var_name in self.algebraic:
            expr = self.algebraic[var_name]
        else:
            raise ValueError(f"Unknown variable: {var_name}")

        lhs = rf"\dot{{{var_name}}}" if var_name in self.derivatives else var_name
        return f"{lhs} = {latex(expr)}"

    def get_jacobian_state(self, simplified: bool = False) -> sp.Matrix:
        """
        Compute the Jacobian of the state derivatives with respect to states.

        This gives the A matrix: A = ∂f/∂x

        Args:
            simplified: Whether to simplify the result

        Returns:
            SymPy matrix of partial derivatives
        """
        self._ensure_compiled()

        if self._jacobian_state is None:
            # Get state variables in order
            state_vars = [self.symbols[var.name] for var in self.model.states]

            # Get derivative expressions in order
            der_exprs = [self.derivatives[var.name] for var in self.model.states]

            # Compute Jacobian
            jac = Matrix(der_exprs).jacobian(state_vars)
            self._jacobian_state = jac

        if simplified:
            return simplify(self._jacobian_state)
        return self._jacobian_state

    def get_jacobian_input(self, simplified: bool = False) -> sp.Matrix:
        """
        Compute the Jacobian of the state derivatives with respect to inputs.

        This gives the B matrix: B = ∂f/∂u

        Args:
            simplified: Whether to simplify the result

        Returns:
            SymPy matrix of partial derivatives
        """
        self._ensure_compiled()

        if self._jacobian_input is None:
            # Get input variables in order
            input_vars = [self.symbols[var.name] for var in self.model.inputs]

            # Get derivative expressions in order
            der_exprs = [self.derivatives[var.name] for var in self.model.states]

            # Compute Jacobian
            if input_vars:
                jac = Matrix(der_exprs).jacobian(input_vars)
            else:
                # No inputs - empty matrix
                jac = Matrix(len(der_exprs), 0, [])

            self._jacobian_input = jac

        if simplified:
            return simplify(self._jacobian_input)
        return self._jacobian_input

    def substitute(self, expr_name: str, substitutions: dict[str, float]) -> sp.Expr:
        """
        Substitute values into an expression.

        Args:
            expr_name: Name of the variable/expression
            substitutions: Dictionary mapping variable names to values

        Returns:
            Expression with substitutions applied
        """
        self._ensure_compiled()

        if expr_name in self.derivatives:
            expr = self.derivatives[expr_name]
        elif expr_name in self.algebraic:
            expr = self.algebraic[expr_name]
        else:
            raise ValueError(f"Unknown variable: {expr_name}")

        subs_dict = {self.symbols[name]: value for name, value in substitutions.items()}
        return expr.subs(subs_dict)

    def taylor_series(self, var_name: str, around: dict[str, float], order: int = 2) -> sp.Expr:
        """
        Compute Taylor series expansion of a derivative.

        Args:
            var_name: Name of the state variable
            around: Point to expand around (dict of var_name -> value)
            order: Order of the Taylor series

        Returns:
            Taylor series as SymPy expression
        """
        self._ensure_compiled()

        if var_name not in self.derivatives:
            raise ValueError(f"No derivative equation for: {var_name}")

        expr = self.derivatives[var_name]

        # For each variable in 'around', compute Taylor series
        for sym_name, value in around.items():
            sym = self.symbols[sym_name]
            expr = expr.series(sym, value, order + 1).removeO()

        return expr

    # Implement abstract methods from Backend

    def simulate(
        self,
        t_final: float,
        dt: float = 0.01,
        input_func: Optional[Callable[[float], dict[str, float]]] = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Simulate using numerical integration (via lambdified functions).

        Note: This converts SymPy expressions to numerical functions.
        For pure numerical simulation, use CasadiBackend instead.

        Supports both scalar and array state variables. Array states are
        flattened for integration and unflattened in the returned results.
        """
        self._ensure_compiled()

        from scipy.integrate import solve_ivp

        # Get state variables and their initial conditions
        state_vars = self.model.states
        state_names = [var.name for var in state_vars]

        # Build initial condition vector - flatten array states
        x0_parts = []
        state_shapes = {}  # Track shape for each state (None = scalar)
        for var in state_vars:
            state_shapes[var.name] = self.var_shapes.get(var.name)
            if var.is_array:
                shape = var.shape
                # Get initial value - check start first, then value
                init_val = var.start if var.start is not None else var.value
                if init_val is not None:
                    if isinstance(init_val, (list, np.ndarray)):
                        start_val = np.array(init_val).flatten()
                    else:
                        start_val = np.full(np.prod(shape), float(init_val))
                else:
                    start_val = np.zeros(np.prod(shape))
                x0_parts.append(start_val)
            else:
                start_val = float(var.start) if var.start is not None else 0.0
                x0_parts.append(np.array([start_val]))

        x0 = np.concatenate(x0_parts) if x0_parts else np.array([])

        # Get parameter values (flattened for arrays)
        param_values_flat = []
        for var in self.model.parameters:
            if var.is_array:
                shape = var.shape
                if var.value is not None:
                    if isinstance(var.value, (list, np.ndarray)):
                        param_values_flat.extend(np.array(var.value).flatten())
                    else:
                        param_values_flat.extend([float(var.value)] * np.prod(shape))
                else:
                    param_values_flat.extend([0.0] * np.prod(shape))
            else:
                val = float(var.value) if var.value is not None else 0.0
                param_values_flat.append(val)

        # Build list of symbols and derivative expressions (flattened)
        state_syms_flat = []
        der_exprs_flat = []

        for var in state_vars:
            shape = self.var_shapes.get(var.name)
            if shape is not None and len(shape) == 1:
                # Vector state - add element symbols
                n = shape[0]
                for i in range(n):
                    state_syms_flat.append(self.symbols[(var.name, i)])
                    # Look for element-wise derivatives
                    key = (var.name, i)
                    if key in self.derivatives:
                        der_exprs_flat.append(self.derivatives[key])
                    elif var.name in self.derivatives:
                        # Whole-array derivative - extract element
                        der_exprs_flat.append(self.derivatives[var.name][i])
                    else:
                        der_exprs_flat.append(sp.Float(0))
            elif shape is not None and len(shape) == 2:
                # Matrix state
                n, m = shape
                for i in range(n):
                    for j in range(m):
                        state_syms_flat.append(self.symbols[(var.name, i, j)])
                        key = (var.name, i, j)
                        if key in self.derivatives:
                            der_exprs_flat.append(self.derivatives[key])
                        else:
                            der_exprs_flat.append(sp.Float(0))
            else:
                # Scalar state
                state_syms_flat.append(self.symbols[var.name])
                der_exprs_flat.append(self.derivatives.get(var.name, sp.Float(0)))

        # Build flat input symbols
        input_syms_flat = []
        for var in self.model.inputs:
            shape = self.var_shapes.get(var.name)
            if shape is not None and len(shape) == 1:
                for i in range(shape[0]):
                    input_syms_flat.append(self.symbols[(var.name, i)])
            elif shape is not None and len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        input_syms_flat.append(self.symbols[(var.name, i, j)])
            else:
                input_syms_flat.append(self.symbols[var.name])

        # Build flat parameter symbols
        param_syms_flat = []
        for var in self.model.parameters:
            shape = self.var_shapes.get(var.name)
            if shape is not None and len(shape) == 1:
                for i in range(shape[0]):
                    param_syms_flat.append(self.symbols[(var.name, i)])
            elif shape is not None and len(shape) == 2:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        param_syms_flat.append(self.symbols[(var.name, i, j)])
            else:
                param_syms_flat.append(self.symbols[var.name])

        # Create a single lambda function for the RHS
        f_lambda = lambdify(
            [state_syms_flat, input_syms_flat, param_syms_flat],
            der_exprs_flat,
            modules=["numpy"],
        )

        def rhs(t, x):
            # Get input values (flattened)
            if input_func:
                u_dict = input_func(t)
                u = []
                for var in self.model.inputs:
                    shape = self.var_shapes.get(var.name)
                    if shape is not None and len(shape) >= 1:
                        val = u_dict.get(var.name, np.zeros(shape))
                        u.extend(np.array(val).flatten())
                    else:
                        u.append(u_dict.get(var.name, 0.0))
            else:
                u = [0.0] * len(input_syms_flat)

            # Evaluate
            result = f_lambda(list(x), u, param_values_flat)
            return np.array(result).flatten()

        # Integrate
        t_span = (0.0, t_final)
        t_eval = np.arange(0.0, t_final, dt)

        sol_obj = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method="RK45")

        # Package results - unflatten array states
        t = sol_obj.t
        sol = {}
        flat_idx = 0
        for var in state_vars:
            shape = state_shapes[var.name]
            if shape is not None and len(shape) == 1:
                n = shape[0]
                sol[var.name] = sol_obj.y[flat_idx : flat_idx + n, :]
                flat_idx += n
            elif shape is not None and len(shape) == 2:
                n, m = shape
                # Return as 3D array: (n, m, timesteps)
                sol[var.name] = sol_obj.y[flat_idx : flat_idx + n * m, :].reshape(n, m, -1)
                flat_idx += n * m
            else:
                sol[var.name] = sol_obj.y[flat_idx, :]
                flat_idx += 1

        return t, sol

    def linearize(
        self, x0: Optional[dict[str, float]] = None, u0: Optional[dict[str, float]] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize symbolically, then evaluate at operating point.

        Computes the linearization of the system around an operating point,
        returning state-space matrices (A, B, C, D) where:
        - A = ∂f/∂x (state Jacobian)
        - B = ∂f/∂u (input Jacobian)
        - C = ∂y/∂x (output-state Jacobian)
        - D = ∂y/∂u (output-input Jacobian)

        Output equations (y = g(x, u)) are taken from the algebraic equations
        that define output variables.

        Args:
            x0: State values at operating point (uses start values if None)
            u0: Input values at operating point (uses 0 if None)

        Returns:
            (A, B, C, D) state-space matrices
        """
        self._ensure_compiled()

        # Get symbolic Jacobians for state dynamics
        A_sym = self.get_jacobian_state()
        B_sym = self.get_jacobian_input()

        # Build C and D matrices from output equations
        output_vars = self.model.outputs
        state_syms = [self.symbols[var.name] for var in self.model.states]
        input_syms = [self.symbols[var.name] for var in self.model.inputs]

        if output_vars:
            # Get output expressions from algebraic equations
            output_exprs = []
            for var in output_vars:
                if var.name in self.algebraic:
                    output_exprs.append(self.algebraic[var.name])
                elif var.name in self.outputs:
                    output_exprs.append(self.outputs[var.name])
                else:
                    # Output might be a direct state passthrough
                    if var.name in self.symbols:
                        output_exprs.append(self.symbols[var.name])
                    else:
                        output_exprs.append(sp.Float(0))

            # Compute Jacobians C = ∂y/∂x and D = ∂y/∂u
            y_matrix = Matrix(output_exprs)
            C_sym = y_matrix.jacobian(state_syms) if state_syms else Matrix(len(output_vars), 0, [])
            D_sym = y_matrix.jacobian(input_syms) if input_syms else Matrix(len(output_vars), 0, [])
        else:
            C_sym = Matrix(0, len(self.model.states), [])
            D_sym = Matrix(0, len(self.model.inputs), [])

        # Prepare substitution dictionary
        subs = {}

        # State values
        if x0 is None:
            x0 = {}
        for var in self.model.states:
            value = x0.get(var.name, var.start if var.start is not None else 0.0)
            subs[self.symbols[var.name]] = value

        # Input values
        if u0 is None:
            u0 = {}
        for var in self.model.inputs:
            value = u0.get(var.name, 0.0)
            subs[self.symbols[var.name]] = value

        # Parameter values
        for var in self.model.parameters:
            value = var.value if var.value is not None else 0.0
            subs[self.symbols[var.name]] = value

        # Substitute and convert to numpy
        A = np.array(A_sym.subs(subs)).astype(float)
        B = np.array(B_sym.subs(subs)).astype(float)
        C = (
            np.array(C_sym.subs(subs)).astype(float)
            if output_vars
            else np.zeros((0, len(self.model.states)))
        )
        D = (
            np.array(D_sym.subs(subs)).astype(float)
            if output_vars
            else np.zeros((0, len(self.model.inputs)))
        )

        return A, B, C, D

    def get_rhs_function(self) -> Callable:
        """Get a numerical function for the RHS (lambdified from SymPy)."""
        self._ensure_compiled()

        state_names = [var.name for var in self.model.states]
        state_syms = [self.symbols[name] for name in state_names]
        input_syms = [self.symbols[var.name] for var in self.model.inputs]
        param_syms = [self.symbols[var.name] for var in self.model.parameters]

        der_exprs = [self.derivatives[name] for name in state_names]

        f = lambdify([state_syms, input_syms, param_syms], der_exprs, modules=["numpy"])

        return lambda t, x, u, p: np.array(f(x, u, p))

    def get_output_jacobians(self, simplified: bool = False) -> tuple[sp.Matrix, sp.Matrix]:
        """
        Get symbolic Jacobians for output equations.

        Returns:
            (C_sym, D_sym): Symbolic C and D matrices where:
                C = ∂y/∂x (output-state Jacobian)
                D = ∂y/∂u (output-input Jacobian)
        """
        self._ensure_compiled()

        output_vars = self.model.outputs
        state_syms = [self.symbols[var.name] for var in self.model.states]
        input_syms = [self.symbols[var.name] for var in self.model.inputs]

        if not output_vars:
            return Matrix(0, len(state_syms), []), Matrix(0, len(input_syms), [])

        # Get output expressions
        output_exprs = []
        for var in output_vars:
            if var.name in self.algebraic:
                output_exprs.append(self.algebraic[var.name])
            elif var.name in self.outputs:
                output_exprs.append(self.outputs[var.name])
            elif var.name in self.symbols:
                output_exprs.append(self.symbols[var.name])
            else:
                output_exprs.append(sp.Float(0))

        y_matrix = Matrix(output_exprs)
        C_sym = y_matrix.jacobian(state_syms) if state_syms else Matrix(len(output_vars), 0, [])
        D_sym = y_matrix.jacobian(input_syms) if input_syms else Matrix(len(output_vars), 0, [])

        if simplified:
            return simplify(C_sym), simplify(D_sym)
        return C_sym, D_sym

    def get_all_expressions(self) -> dict[str, sp.Expr]:
        """
        Get all symbolic expressions (derivatives + algebraic).

        Returns:
            Dictionary mapping variable names to their symbolic expressions.
        """
        self._ensure_compiled()

        result = {}
        # Add derivative expressions
        for key, expr in self.derivatives.items():
            if isinstance(key, tuple):
                # Array element: ('x', 0) -> 'der(x[1])'
                name, idx = key
                result[f"der({name}[{idx + 1}])"] = expr
            else:
                result[f"der({key})"] = expr

        # Add algebraic expressions
        for name, expr in self.algebraic.items():
            result[name] = expr

        return result

    def get_state_space_symbolic(self) -> dict[str, sp.Matrix]:
        """
        Get the full symbolic state-space representation.

        Returns:
            Dictionary with:
                'A': Symbolic state matrix (∂f/∂x)
                'B': Symbolic input matrix (∂f/∂u)
                'C': Symbolic output-state matrix (∂y/∂x)
                'D': Symbolic output-input matrix (∂y/∂u)
                'f': State derivative vector (xdot = f(x, u, p))
                'y': Output vector (y = g(x, u, p))
        """
        self._ensure_compiled()

        # Get f (state derivatives)
        state_names = [var.name for var in self.model.states]
        f_exprs = [self.derivatives.get(name, sp.Float(0)) for name in state_names]
        f_vec = Matrix(f_exprs)

        # Get y (outputs)
        output_vars = self.model.outputs
        if output_vars:
            y_exprs = []
            for var in output_vars:
                if var.name in self.algebraic:
                    y_exprs.append(self.algebraic[var.name])
                elif var.name in self.outputs:
                    y_exprs.append(self.outputs[var.name])
                elif var.name in self.symbols:
                    y_exprs.append(self.symbols[var.name])
                else:
                    y_exprs.append(sp.Float(0))
            y_vec = Matrix(y_exprs)
        else:
            y_vec = Matrix([])

        # Get Jacobians
        A = self.get_jacobian_state()
        B = self.get_jacobian_input()
        C, D = self.get_output_jacobians()

        return {
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "f": f_vec,
            "y": y_vec,
        }

    def controllability_matrix(self) -> sp.Matrix:
        """
        Compute the symbolic controllability matrix.

        The controllability matrix is [B, AB, A²B, ..., A^(n-1)B]
        where n is the number of states.

        Returns:
            Symbolic controllability matrix
        """
        self._ensure_compiled()

        A = self.get_jacobian_state()
        B = self.get_jacobian_input()
        n = A.rows

        if B.cols == 0:
            return Matrix(n, 0, [])

        # Build controllability matrix
        cols = [B]
        AB = B
        for _ in range(1, n):
            AB = A * AB
            cols.append(AB)

        return Matrix.hstack(*cols)

    def observability_matrix(self) -> sp.Matrix:
        """
        Compute the symbolic observability matrix.

        The observability matrix is [C; CA; CA²; ...; CA^(n-1)]
        where n is the number of states.

        Returns:
            Symbolic observability matrix
        """
        self._ensure_compiled()

        A = self.get_jacobian_state()
        C, _ = self.get_output_jacobians()
        n = A.rows

        if C.rows == 0:
            return Matrix(0, n, [])

        # Build observability matrix
        rows = [C]
        CA = C
        for _ in range(1, n):
            CA = CA * A
            rows.append(CA)

        return Matrix.vstack(*rows)
