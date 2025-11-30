"""
SymPy backend for symbolic manipulation and analysis.

This backend converts Cyecca IR to SymPy symbolic expressions, enabling:
- Symbolic simplification
- Analytical Jacobians
- LaTeX export for documentation
- Symbolic solutions (where possible)
- Taylor series expansions
"""

from typing import Any, Callable, Optional

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
    """

    def __init__(self, model: Model):
        super().__init__(model)

        # SymPy symbols for each variable
        self.symbols: dict[str, sp.Symbol] = {}

        # Symbolic expressions for derivatives
        self.derivatives: dict[str, sp.Expr] = {}

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
            if var.is_array:
                # For now, we'll handle arrays later
                raise NotImplementedError("Array variables not yet supported in SymPy backend")
            self.symbols[var.name] = sp.Symbol(var.name, real=True)

        # Convert equations to SymPy expressions
        for eq in self.model.equations:
            if eq.eq_type == EquationType.SIMPLE:
                # Handle: lhs = rhs
                if eq.lhs is not None and isinstance(eq.lhs, FunctionCall) and eq.lhs.func == "der":
                    # Derivative equation: der(x) = expr
                    if len(eq.lhs.args) > 0:
                        arg = eq.lhs.args[0]
                        if isinstance(arg, ComponentRef):
                            if not arg.is_simple:
                                raise NotImplementedError(
                                    f"Hierarchical der() not yet supported: der({arg})"
                                )
                            state_name = arg.simple_name
                            expr_sympy = self._convert_expr(eq.rhs)
                            self.derivatives[state_name] = expr_sympy
                        elif isinstance(arg, VarRef):
                            state_name = arg.name
                            expr_sympy = self._convert_expr(eq.rhs)
                            self.derivatives[state_name] = expr_sympy

                elif eq.lhs is not None:
                    # Algebraic equation: var = expr
                    if isinstance(eq.lhs, ComponentRef):
                        if not eq.lhs.is_simple:
                            raise NotImplementedError(
                                f"Hierarchical algebraic equations not yet supported: {eq.lhs}"
                            )
                        var_name = eq.lhs.simple_name
                        expr_sympy = self._convert_expr(eq.rhs)
                        self.algebraic[var_name] = expr_sympy
                    elif isinstance(eq.lhs, VarRef):
                        # Backward compatibility
                        var_name = eq.lhs.name
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

    def _convert_expr(self, expr: Expr) -> sp.Expr:
        """Convert a Cyecca IR expression to a SymPy expression."""
        if isinstance(expr, Literal):
            return sp.Float(expr.value) if isinstance(expr.value, float) else expr.value

        elif isinstance(expr, ComponentRef):
            # For now, only support simple component references
            if not expr.is_simple:
                raise NotImplementedError(
                    f"Hierarchical component references not yet supported: {expr}"
                )
            var_name = expr.simple_name
            if var_name not in self.symbols:
                raise ValueError(f"Unknown variable: {var_name}")
            return self.symbols[var_name]

        elif isinstance(expr, VarRef):
            # Backward compatibility
            if expr.name not in self.symbols:
                raise ValueError(f"Unknown variable: {expr.name}")
            return self.symbols[expr.name]

        elif isinstance(expr, ArrayRef):
            raise NotImplementedError("Array indexing not yet supported")

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
        """
        self._ensure_compiled()

        from scipy.integrate import solve_ivp

        # Get state variables and their initial conditions
        state_vars = self.model.states
        state_names = [var.name for var in state_vars]
        x0 = np.array([var.start if var.start is not None else 0.0 for var in state_vars])

        # Get parameter values
        param_values = {}
        for var in self.model.parameters:
            param_values[var.name] = var.value if var.value is not None else 0.0

        # Create lambdified function for derivatives
        state_syms = [self.symbols[name] for name in state_names]
        input_syms = [self.symbols[var.name] for var in self.model.inputs]
        param_syms = [self.symbols[var.name] for var in self.model.parameters]

        der_exprs = [self.derivatives[name] for name in state_names]

        # Create a single lambda function for the RHS
        f_lambda = lambdify(
            [state_syms, input_syms, param_syms],
            der_exprs,
            modules=["numpy"],
        )

        def rhs(t, x):
            # Get input values
            if input_func:
                u_dict = input_func(t)
                u = [u_dict.get(var.name, 0.0) for var in self.model.inputs]
            else:
                u = [0.0] * len(self.model.inputs)

            # Get parameter values
            p = [param_values[var.name] for var in self.model.parameters]

            # Evaluate
            return f_lambda(x, u, p)

        # Integrate
        t_span = (0.0, t_final)
        t_eval = np.arange(0.0, t_final, dt)

        sol_obj = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method="RK45")

        # Package results
        t = sol_obj.t
        sol = {}
        for i, name in enumerate(state_names):
            sol[name] = sol_obj.y[i, :]

        return t, sol

    def linearize(
        self, x0: Optional[dict[str, float]] = None, u0: Optional[dict[str, float]] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize symbolically, then evaluate at operating point.

        Returns:
            (A, B, C, D) state-space matrices
        """
        self._ensure_compiled()

        # Get symbolic Jacobians
        A_sym = self.get_jacobian_state()
        B_sym = self.get_jacobian_input()

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

        # For now, assume no outputs (C, D are empty)
        n_states = len(self.model.states)
        n_inputs = len(self.model.inputs)
        C = np.zeros((0, n_states))
        D = np.zeros((0, n_inputs))

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
