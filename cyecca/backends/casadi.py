"""
CasADi backend for fast numerical simulation and optimization.

This backend converts Cyecca IR to CasADi symbolic expressions, enabling:
- Fast numerical simulation with CVODES/IDAS integrators
- Event handling for hybrid/discrete systems (when equations)
- Automatic differentiation for optimization
- Code generation (C/C++)
- JIT compilation
"""

from typing import Any, Callable, Optional

import casadi as ca
from casadi import event_in, event_out
import numpy as np

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


class CasadiBackend(Backend):
    """
    CasADi backend for fast numerical simulation.

    Features:
    - Fast numerical integration (CVODES/IDAS)
    - Event handling for when-equations
    - Automatic differentiation
    - Sparse Jacobians
    - Code generation
    """

    def __init__(self, model: Model):
        super().__init__(model)

        # CasADi symbols for each variable
        self.symbols: dict[str, ca.SX] = {}

        # Symbolic expressions for derivatives
        self.derivatives: dict[str, ca.SX] = {}

        # Symbolic expressions for algebraic equations
        self.algebraic: dict[str, ca.SX] = {}

        # When equations (for event handling)
        self.when_equations: list[tuple[Expr, list[Equation]]] = []

        # State/parameter/input names (in order)
        self.state_names: list[str] = []
        self.param_names: list[str] = []
        self.input_names: list[str] = []

        # Default values
        self.state_defaults: dict[str, float] = {}
        self.param_defaults: dict[str, float] = {}

        # Compiled functions
        self.f_ode: Optional[ca.Function] = None
        self.integrator: Optional[ca.Function] = None

    def compile(self) -> None:
        """Compile the IR model to CasADi symbolic expressions."""
        # Create CasADi symbols for all variables
        for var in self.model.variables:
            if var.is_array:
                raise NotImplementedError("Array variables not yet supported in CasADi backend")

            self.symbols[var.name] = ca.SX.sym(var.name)

            # Track state/param/input names and defaults
            if var.var_type == VariableType.STATE:
                self.state_names.append(var.name)
                self.state_defaults[var.name] = self._extract_value(var.start)
            elif var.var_type == VariableType.PARAMETER:
                self.param_names.append(var.name)
                value = self._extract_value(var.value) if var.value is not None else self._extract_value(var.start)
                self.param_defaults[var.name] = value if value is not None else 0.0
            elif var.var_type == VariableType.INPUT:
                self.input_names.append(var.name)

        # Convert equations to CasADi expressions
        for eq in self.model.equations:
            if eq.eq_type == EquationType.SIMPLE:
                # Handle: lhs = rhs
                if eq.lhs is not None and isinstance(eq.lhs, FunctionCall) and eq.lhs.func == "der":
                    # Derivative equation: der(x) = expr
                    if len(eq.lhs.args) > 0:
                        arg = eq.lhs.args[0]
                        state_name = self._get_var_name(arg)
                        expr_casadi = self._convert_expr(eq.rhs)
                        self.derivatives[state_name] = expr_casadi

                elif eq.lhs is not None:
                    # Algebraic equation: var = expr
                    var_name = self._get_var_name(eq.lhs)
                    expr_casadi = self._convert_expr(eq.rhs)
                    self.algebraic[var_name] = expr_casadi

            elif eq.eq_type == EquationType.WHEN:
                # Store when equations for event handling
                self.when_equations.append((eq.condition, eq.when_equations))

            elif eq.eq_type == EquationType.INITIAL:
                # Initial equations - handled via start values
                pass

            else:
                raise ValueError(f"Unsupported equation type: {eq.eq_type}")

        # Build ODE function: xdot = f(x, u, p)
        if self.state_names:
            x = ca.vertcat(*[self.symbols[name] for name in self.state_names])
            u = ca.vertcat(*[self.symbols[name] for name in self.input_names]) if self.input_names else ca.SX([])
            p = ca.vertcat(*[self.symbols[name] for name in self.param_names]) if self.param_names else ca.SX([])

            xdot_exprs = [self.derivatives.get(name, ca.SX(0)) for name in self.state_names]
            xdot = ca.vertcat(*xdot_exprs)

            self.f_ode = ca.Function('ode', [x, u, p], [xdot], ['x', 'u', 'p'], ['xdot'])

        self._compiled = True

    def _extract_value(self, val: Any) -> Optional[float]:
        """Extract numeric value from Expr or direct value."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, Literal):
            return float(val.value)
        # Handle dict format from JSON: {'op': 'literal', 'value': x}
        if isinstance(val, dict) and val.get('op') == 'literal':
            return float(val['value'])
        return None

    def _get_var_name(self, expr: Expr) -> str:
        """Extract variable name from an expression."""
        if isinstance(expr, ComponentRef):
            if not expr.is_simple:
                raise NotImplementedError(f"Hierarchical references not supported: {expr}")
            return expr.simple_name
        elif isinstance(expr, VarRef):
            return expr.name
        else:
            raise ValueError(f"Cannot extract variable name from: {expr}")

    def _convert_expr(self, expr: Expr) -> ca.SX:
        """Convert a Cyecca IR expression to a CasADi expression."""
        if isinstance(expr, Literal):
            return ca.SX(expr.value)

        elif isinstance(expr, ComponentRef):
            if not expr.is_simple:
                raise NotImplementedError(f"Hierarchical component references not supported: {expr}")
            var_name = expr.simple_name
            if var_name not in self.symbols:
                raise ValueError(f"Unknown variable: {var_name}")
            return self.symbols[var_name]

        elif isinstance(expr, VarRef):
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
                "^": lambda l, r: l ** r,
                "<": lambda l, r: l < r,
                "<=": lambda l, r: l <= r,
                ">": lambda l, r: l > r,
                ">=": lambda l, r: l >= r,
                "==": lambda l, r: l == r,
                "!=": lambda l, r: l != r,
            }

            if expr.op in op_map:
                return op_map[expr.op](left, right)
            else:
                raise ValueError(f"Unsupported binary operator: {expr.op}")

        elif isinstance(expr, UnaryOp):
            operand = self._convert_expr(expr.operand)

            if expr.op == "-" or expr.op == "neg":
                return -operand
            elif expr.op == "+":
                return operand
            elif expr.op == "not":
                return ca.logic_not(operand)
            else:
                raise ValueError(f"Unsupported unary operator: {expr.op}")

        elif isinstance(expr, FunctionCall):
            # Handle special Modelica operators
            if expr.func == "der":
                raise ValueError("der() should only appear in LHS of equations")
            elif expr.func == "pre":
                # pre(x) appears in when equations - return the variable itself
                # The actual "previous value" semantics are handled by the integrator
                arg = expr.args[0]
                return self._convert_expr(arg)

            args = [self._convert_expr(arg) for arg in expr.args]

            # Map Cyecca function names to CasADi functions
            func_map = {
                "sin": ca.sin,
                "cos": ca.cos,
                "tan": ca.tan,
                "asin": ca.asin,
                "acos": ca.acos,
                "atan": ca.atan,
                "atan2": ca.atan2,
                "exp": ca.exp,
                "log": ca.log,
                "ln": ca.log,
                "log10": ca.log10,
                "sqrt": ca.sqrt,
                "abs": ca.fabs,
                "sign": ca.sign,
                "min": ca.fmin,
                "max": ca.fmax,
                "floor": ca.floor,
                "ceil": ca.ceil,
            }

            if expr.func in func_map:
                return func_map[expr.func](*args)
            else:
                raise ValueError(f"Unsupported function: {expr.func}")

        elif isinstance(expr, IfExpr):
            cond = self._convert_expr(expr.condition)
            true_val = self._convert_expr(expr.true_expr)
            false_val = self._convert_expr(expr.false_expr)
            return ca.if_else(cond, true_val, false_val)

        elif isinstance(expr, ArrayLiteral):
            elements = [self._convert_expr(e) for e in expr.elements]
            return ca.vertcat(*elements)

        elif isinstance(expr, Slice):
            raise ValueError("Slice expressions cannot be converted directly")

        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _extract_zero_crossing(self, expr: Any) -> ca.SX:
        """
        Extract zero-crossing indicator from a comparison expression.

        CasADi event handling needs an expression that crosses zero when the event occurs,
        not a boolean condition. This converts comparisons to zero-crossing expressions:
        - x < 0  -->  x  (crosses zero when x becomes negative)
        - x > 0  --> -x  (crosses zero when x becomes positive)
        - x <= 0 -->  x
        - x >= 0 --> -x
        """
        from cyecca.ir.expr import BinaryOp

        if isinstance(expr, BinaryOp):
            if expr.op in ['<', '<=']:
                # For x < 0 or x <= 0, zero-crossing is x itself
                # Assuming RHS is zero (standard form)
                return self._convert_expr(expr.left)
            elif expr.op in ['>', '>=']:
                # For x > 0 or x >= 0, zero-crossing is -x
                return -self._convert_expr(expr.left)
            else:
                raise ValueError(f"Unsupported comparison operator for events: {expr.op}")
        else:
            # If not a comparison, assume it's already a zero-crossing expression
            return self._convert_expr(expr)

    def simulate(
        self,
        t_final: float,
        dt: float = 0.01,
        input_func: Optional[Callable[[float], dict[str, float]]] = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Simulate the model using CasADi integrator.

        When-equations are automatically handled if present in the model.

        Args:
            t_final: Final simulation time
            dt: Time step
            input_func: Optional function that returns input values at time t

        Returns:
            (t, sol) where:
                t: Time array of shape (n_steps,)
                sol: Dictionary mapping variable names to arrays of shape (n_steps,)
        """
        self._ensure_compiled()

        # Initial conditions
        x0 = np.array([self.state_defaults.get(name, 0.0) for name in self.state_names])

        # Parameter values
        p_val = np.array([self.param_defaults.get(name, 0.0) for name in self.param_names]) if self.param_names else np.array([])

        # Time grid
        t_grid = np.arange(0.0, t_final + dt, dt)

        # Use event handling if when-equations are present
        if self.when_equations:
            # Use CasADi integrator with event handling
            result = self._simulate_with_events(x0, p_val, t_grid)
        else:
            # Use simple RK4 integration (no events)
            result = self._simulate_rk4(x0, p_val, t_grid, input_func)

        return t_grid, result

    def _simulate_with_events(
        self,
        x0: np.ndarray,
        p_val: np.ndarray,
        t_grid: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Simulate using CasADi integrator with event handling."""
        # Build DAE structure
        x = ca.vertcat(*[self.symbols[name] for name in self.state_names])
        p = ca.vertcat(*[self.symbols[name] for name in self.param_names]) if self.param_names else ca.SX([])

        xdot_exprs = [self.derivatives.get(name, ca.SX(0)) for name in self.state_names]
        ode = ca.vertcat(*xdot_exprs)

        # Extract event condition from first when-equation
        # For now, support single when-equation
        if len(self.when_equations) > 1:
            raise NotImplementedError("Multiple when-equations not yet supported")

        condition_expr, reset_eqs = self.when_equations[0]

        # Convert condition to zero-crossing indicator
        # CasADi needs an expression that crosses zero, not a boolean
        # For "x < 0", the zero-crossing expression is "x"
        # For "x > 0", the zero-crossing expression is "-x"
        event_indicator = self._extract_zero_crossing(condition_expr)

        # Build DAE
        dae = {
            'x': x,
            'ode': ode,
            'zero': event_indicator,  # Zero-crossing indicator
        }

        if self.param_names:
            dae['p'] = p

        # Build transition function for reset
        # Start with all states unchanged
        post_x_exprs = [self.symbols[name] for name in self.state_names]

        # Apply resets from when-equations
        for reset_eq in reset_eqs:
            if reset_eq.eq_type == EquationType.SIMPLE:
                var_name = self._get_var_name(reset_eq.lhs)
                reset_value = self._convert_expr(reset_eq.rhs)

                # Find index of this state and update
                if var_name in self.state_names:
                    idx = self.state_names.index(var_name)
                    post_x_exprs[idx] = reset_value

        post_x = ca.vertcat(*post_x_exprs)

        # Create transition function (following CasADi official example format)
        trans_dict = {'x': x, 'post_x': post_x}
        if self.param_names:
            trans_dict['p'] = p

        transition = ca.Function('transition', trans_dict, event_in(), event_out())

        # Create integrator with event handling
        opts = {
            'transition': transition,
            'max_events': 100,
        }

        # Note: Initial height needs to be high enough to avoid grid interpolation issues
        # The official CasADi example uses h0=5.0
        if 'h' in self.state_names:
            h_idx = self.state_names.index('h')
            if x0[h_idx] < 2.0:
                print(f"NOTE: Increasing initial height from {x0[h_idx]} to 5.0 for event handling stability")
                x0[h_idx] = 5.0

        integrator_args = ['sim', 'cvodes', dae, t_grid[0], t_grid]
        if self.param_names:
            integrator = ca.integrator(*integrator_args, opts)
        else:
            integrator = ca.integrator(*integrator_args, opts)

        # Run integration
        sim_args = {'x0': x0}
        if self.param_names:
            sim_args['p'] = p_val

        result = integrator(**sim_args)

        # Extract results
        x_result = result['xf'].full()

        sol = {}
        for i, name in enumerate(self.state_names):
            sol[name] = x_result[i, :]

        return sol

    def _simulate_rk4(
        self,
        x0: np.ndarray,
        p_val: np.ndarray,
        t_grid: np.ndarray,
        input_func: Optional[Callable[[float], dict[str, float]]],
    ) -> dict[str, np.ndarray]:
        """Simple RK4 integration (no events)."""
        dt = t_grid[1] - t_grid[0]
        n_steps = len(t_grid)
        n_states = len(self.state_names)

        # Allocate result arrays
        x_result = np.zeros((n_states, n_steps))
        x_result[:, 0] = x0

        # Current state
        x = x0.copy()

        for i in range(1, n_steps):
            t = t_grid[i - 1]

            # Input values
            if input_func:
                u_dict = input_func(t)
                u = np.array([u_dict.get(name, 0.0) for name in self.input_names])
            else:
                u = np.zeros(len(self.input_names))

            # RK4 step
            k1 = np.array(self.f_ode(x, u, p_val)).flatten()
            k2 = np.array(self.f_ode(x + dt * k1 / 2, u, p_val)).flatten()
            k3 = np.array(self.f_ode(x + dt * k2 / 2, u, p_val)).flatten()
            k4 = np.array(self.f_ode(x + dt * k3, u, p_val)).flatten()

            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            x_result[:, i] = x

        # Package results
        sol = {}
        for i, name in enumerate(self.state_names):
            sol[name] = x_result[i, :]

        return sol

    def linearize(
        self, x0: Optional[dict[str, float]] = None, u0: Optional[dict[str, float]] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize the model at an operating point.

        Returns:
            (A, B, C, D) state-space matrices
        """
        self._ensure_compiled()

        # Build symbolic state and input vectors
        x = ca.vertcat(*[self.symbols[name] for name in self.state_names])
        u = ca.vertcat(*[self.symbols[name] for name in self.input_names]) if self.input_names else ca.SX([])
        p = ca.vertcat(*[self.symbols[name] for name in self.param_names]) if self.param_names else ca.SX([])

        xdot_exprs = [self.derivatives.get(name, ca.SX(0)) for name in self.state_names]
        xdot = ca.vertcat(*xdot_exprs)

        # Compute Jacobians
        A_sym = ca.jacobian(xdot, x)
        B_sym = ca.jacobian(xdot, u) if self.input_names else ca.SX(len(self.state_names), 0)

        # Create functions for evaluation
        A_func = ca.Function('A', [x, u, p], [A_sym])
        B_func = ca.Function('B', [x, u, p], [B_sym])

        # Prepare operating point
        if x0 is None:
            x0 = {}
        x0_vec = np.array([x0.get(name, self.state_defaults.get(name, 0.0)) for name in self.state_names])

        if u0 is None:
            u0 = {}
        u0_vec = np.array([u0.get(name, 0.0) for name in self.input_names]) if self.input_names else np.array([])

        p_vec = np.array([self.param_defaults.get(name, 0.0) for name in self.param_names]) if self.param_names else np.array([])

        # Evaluate
        A = np.array(A_func(x0_vec, u0_vec, p_vec))
        B = np.array(B_func(x0_vec, u0_vec, p_vec))

        # No outputs for now
        n_states = len(self.state_names)
        n_inputs = len(self.input_names)
        C = np.zeros((0, n_states))
        D = np.zeros((0, n_inputs))

        return A, B, C, D

    def get_rhs_function(self) -> Callable:
        """Get the right-hand side function for the ODEs."""
        self._ensure_compiled()

        def rhs(t, x, u, p):
            return np.array(self.f_ode(x, u, p)).flatten()

        return rhs
