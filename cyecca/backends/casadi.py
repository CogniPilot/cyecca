"""
CasADi backend for fast numerical simulation and optimization.

This backend converts Cyecca IR to CasADi symbolic expressions, enabling:
- Fast numerical simulation with CVODES/IDAS integrators
- Event handling for hybrid/discrete systems (when equations)
- Automatic differentiation for optimization
- Code generation (C/C++)
- JIT compilation
"""

from collections.abc import Callable, Iterator, MutableMapping
from typing import Any, Optional, Union

import casadi as ca
import numpy as np

# event_in/event_out are only available in CasADi 3.6.4+
# They're needed for proper event handling with integrators
try:
    from casadi import event_in, event_out

    HAS_EVENT_FUNCTIONS = True
except ImportError:
    HAS_EVENT_FUNCTIONS = False
    event_in = None
    event_out = None

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


class LazySimulationResult(MutableMapping):
    """
    A dictionary-like object that lazily computes algebraic variables.

    State variables are stored directly, while algebraic variables are
    computed on-demand when accessed. This avoids unnecessary computation
    for variables that are never queried.
    """

    def __init__(
        self,
        state_data: dict[str, np.ndarray],
        algebraic_funcs: dict[str, ca.Function],
        state_names: list[str],
        input_names: list[str],
        param_names: list[str],
        param_values: np.ndarray,
        t_grid: np.ndarray,
        input_func: Optional[Callable[[float], dict[str, float]]] = None,
    ) -> None:
        self._state_data = state_data
        self._algebraic_funcs = algebraic_funcs
        self._state_names = state_names
        self._input_names = input_names
        self._param_names = param_names
        self._param_values = param_values
        self._t_grid = t_grid
        self._input_func = input_func
        self._cache: dict[str, np.ndarray] = {}

    def _compute_algebraic(self, name: str) -> np.ndarray:
        """Compute a single algebraic variable from state trajectory."""
        if name not in self._algebraic_funcs:
            raise KeyError(name)

        func = self._algebraic_funcs[name]
        n_steps = next(iter(self._state_data.values())).shape[0]
        result = np.zeros(n_steps)

        # Build state array for each timestep and evaluate
        # Algebraic funcs have signature (t, x..., u..., p...)
        for i in range(n_steps):
            t = self._t_grid[i]
            state_vals = [self._state_data[s][i] for s in self._state_names]
            # Get input values at this time
            if self._input_func and self._input_names:
                u_dict = self._input_func(t)
                input_vals = [u_dict.get(name, 0.0) for name in self._input_names]
            else:
                input_vals = [0.0] * len(self._input_names)
            args = [t] + state_vals + input_vals + list(self._param_values)
            result[i] = float(func(*args))

        return result

    def compute_all_algebraic(self) -> None:
        """
        Compute all algebraic variables in a single pass through the timesteps.

        This is more efficient than accessing algebraic variables individually,
        as it only loops through the timesteps once and reuses the state arrays.
        Results are cached for subsequent access.
        """
        if not self._algebraic_funcs:
            return

        # Skip if all already computed
        uncached = [name for name in self._algebraic_funcs if name not in self._cache]
        if not uncached:
            return

        n_steps = next(iter(self._state_data.values())).shape[0]

        # Pre-allocate result arrays for uncached variables
        results: dict[str, np.ndarray] = {name: np.zeros(n_steps) for name in uncached}

        # Single pass through all timesteps
        # Algebraic funcs have signature (t, x..., u..., p...)
        for i in range(n_steps):
            t = self._t_grid[i]
            state_vals = [self._state_data[s][i] for s in self._state_names]
            # Get input values at this time
            if self._input_func and self._input_names:
                u_dict = self._input_func(t)
                input_vals = [u_dict.get(name, 0.0) for name in self._input_names]
            else:
                input_vals = [0.0] * len(self._input_names)
            args = [t] + state_vals + input_vals + list(self._param_values)

            # Evaluate all uncached algebraic functions at this timestep
            for name in uncached:
                results[name][i] = float(self._algebraic_funcs[name](*args))

        # Store in cache
        self._cache.update(results)

    def __getitem__(self, key: str) -> np.ndarray:
        # Check state data first
        if key in self._state_data:
            return self._state_data[key]

        # Check cache
        if key in self._cache:
            return self._cache[key]

        # Compute algebraic variable lazily
        if key in self._algebraic_funcs:
            result = self._compute_algebraic(key)
            self._cache[key] = result
            return result

        raise KeyError(key)

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        self._state_data[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._state_data:
            del self._state_data[key]
        elif key in self._cache:
            del self._cache[key]
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        # Iterate over all available keys (states + algebraic)
        yield from self._state_data.keys()
        yield from self._algebraic_funcs.keys()

    def __len__(self) -> int:
        return len(self._state_data) + len(self._algebraic_funcs)

    def __contains__(self, key: object) -> bool:
        return key in self._state_data or key in self._algebraic_funcs

    def keys(self):
        """Return all available variable names."""
        return list(self._state_data.keys()) + list(self._algebraic_funcs.keys())


class CasadiBackend(Backend):
    """
    CasADi backend for fast numerical simulation.

    Features:
    - Fast numerical integration (CVODES/IDAS)
    - Event handling for when-equations
    - Automatic differentiation
    - Sparse Jacobians
    - Code generation

    Args:
        model: The Cyecca IR model to compile (optional for empty() pattern)
        sym_type: Type of CasADi symbols to use - 'SX' (scalar, default) or 'MX' (matrix)
    """

    def __init__(self, model: Optional[Model] = None, sym_type: str = "SX") -> None:
        if model is not None:
            super(CasadiBackend, self).__init__(model)
        else:
            # Empty/placeholder mode - don't call super().__init__
            self.model = None  # type: ignore

        if sym_type not in ["SX", "MX"]:
            raise ValueError(f"sym_type must be 'SX' or 'MX', got '{sym_type}'")

        self.sym_type = sym_type
        self.sym_class: type[Union[ca.SX, ca.MX]] = ca.SX if sym_type == "SX" else ca.MX

        # CasADi symbols for each variable
        self.symbols: dict[str, Union[ca.SX, ca.MX]] = {}

        # Variable shapes for array variables (None = scalar)
        self.var_shapes: dict[str, Optional[list[int]]] = {}

        # Symbolic expressions for derivatives
        # For scalars: derivatives["x"] = expr
        # For array elements: derivatives[("x", idx)] = expr (0-based index)
        self.derivatives: dict[Union[str, tuple[str, int]], Union[ca.SX, ca.MX]] = {}

        # Symbolic expressions for algebraic equations
        self.algebraic: dict[str, Union[ca.SX, ca.MX]] = {}

        # When equations (for event handling)
        self.when_equations: list[tuple[Expr, list[Equation]]] = []

        # State/parameter/input names (in order)
        self.state_names: list[str] = []
        self.param_names: list[str] = []
        self.input_names: list[str] = []

        # Default values (for scalars: float, for arrays: list or np.ndarray)
        self.state_defaults: dict[str, Any] = {}
        self.param_defaults: dict[str, float] = {}
        self.const_values: dict[str, float] = {}

        # Compiled functions
        self.f_ode: Optional[ca.Function] = None
        self.integrator: Optional[ca.Function] = None
        self.algebraic_funcs: dict[str, ca.Function] = {}

    @classmethod
    def empty(cls) -> "CasadiBackend":
        """
        Create an empty/placeholder CasadiBackend.

        Use this for pre-declaration to satisfy type checkers like beartype/Pylance.
        Call from_modelica() to populate the backend.

        Example:
            >>> model = CasadiBackend.empty()
            >>> # Later, in a magic cell or function:
            >>> model.from_modelica('''
            ...     model MyModel
            ...         Real x;
            ...     equation
            ...         der(x) = -x;
            ...     end MyModel;
            ... ''')
        """
        return cls(model=None)

    def from_modelica(self, source: str, model_name: Optional[str] = None) -> "CasadiBackend":
        """
        Populate this backend from Modelica source code.

        Args:
            source: Modelica source code as a string
            model_name: Name of the model to compile. If None, auto-detects from source.

        Returns:
            self (for chaining)

        Example:
            >>> model = CasadiBackend.empty()
            >>> model.from_modelica('''
            ...     model MyModel
            ...         Real x(start=1);
            ...     equation
            ...         der(x) = -x;
            ...     end MyModel;
            ... ''')
            >>> t, sol = model.simulate(10.0)
        """
        import re

        # Auto-detect model name if not provided
        if model_name is None:
            match = re.search(r"\b(?:model|class)\s+(\w+)", source)
            if match:
                model_name = match.group(1)
            else:
                raise ValueError("Could not detect model name. Please provide model_name argument.")

        # Import here to avoid circular imports
        from cyecca.io import compile_modelica as _compile_modelica

        ir_model = _compile_modelica(source, model_name)

        # Re-initialize with the model
        super(CasadiBackend, self).__init__(ir_model)
        self.__init__(ir_model, self.sym_type)
        self.compile()
        return self

    def compile(self) -> "CasadiBackend":
        """Compile the IR model to CasADi symbolic expressions."""
        if self.model is None:
            raise RuntimeError("Cannot compile empty backend. Use from_modelica() first.")
        # Create built-in 'time' symbol (Modelica's independent variable)
        self.symbols["time"] = self.sym_class.sym("time")

        # Infer array shapes from derivative equations (workaround for rumoca not exporting shapes)
        inferred_shapes = {}
        for eq in self.model.equations:
            if eq.eq_type == EquationType.SIMPLE and eq.lhs is not None:
                if (
                    isinstance(eq.lhs, FunctionCall)
                    and eq.lhs.func == "der"
                    and len(eq.lhs.args) > 0
                ):
                    from cyecca.ir import ArrayLiteral

                    if isinstance(eq.rhs, ArrayLiteral):
                        # der(x) = array(...) means x must be an array
                        arg = eq.lhs.args[0]
                        if isinstance(arg, ComponentRef):
                            state_name = self._flatten_component_ref(arg)
                            inferred_shapes[state_name] = (len(eq.rhs.elements),)

        # Create CasADi symbols for all variables (including algebraic)
        for var in self.model.variables:
            # Use inferred shape if available, otherwise use variable's shape
            var_shape = inferred_shapes.get(var.name, var.shape)
            self.var_shapes[var.name] = var_shape

            if var.is_array or var_shape is not None:
                # Create array/matrix symbol
                shape = var_shape
                if len(shape) == 1:
                    # Vector: Real x[n]
                    self.symbols[var.name] = self.sym_class.sym(var.name, shape[0])
                elif len(shape) == 2:
                    # Matrix: Real A[n,m]
                    self.symbols[var.name] = self.sym_class.sym(var.name, shape[0], shape[1])
                else:
                    raise NotImplementedError(
                        f"Arrays with more than 2 dimensions not supported: {var.name} has shape {shape}"
                    )
            else:
                # Scalar variable
                self.symbols[var.name] = self.sym_class.sym(var.name)

            # Track state/param/input/algebraic/constant names and defaults
            if var.var_type == VariableType.STATE:
                self.state_names.append(var.name)
                self.state_defaults[var.name] = self._extract_default(var, var_shape)
            elif var.var_type == VariableType.PARAMETER:
                self.param_names.append(var.name)
                self.param_defaults[var.name] = self._extract_default(var, var_shape)
            elif var.var_type == VariableType.INPUT:
                self.input_names.append(var.name)
            elif var.var_type == VariableType.CONSTANT:
                # Constants have fixed values - store them for substitution
                self.const_values[var.name] = self._extract_default(var, var_shape)
            elif var.var_type == VariableType.ALGEBRAIC:
                # Algebraic variables also need symbols for proper substitution
                pass  # Symbol already created above

        # Convert equations to CasADi expressions
        for eq in self.model.equations:
            if eq.eq_type == EquationType.SIMPLE:
                # Handle: lhs = rhs
                if eq.lhs is not None and isinstance(eq.lhs, FunctionCall) and eq.lhs.func == "der":
                    # Derivative equation: der(x) = expr or der(x[i]) = expr
                    if len(eq.lhs.args) > 0:
                        arg = eq.lhs.args[0]
                        state_name = self._get_var_name(arg)
                        expr_casadi = self._convert_expr(eq.rhs)

                        # Check if this is an array element derivative
                        elem_idx = self._get_element_index(arg)
                        if elem_idx is not None:
                            # Array element: der(x[i]) - store with (name, idx) key
                            self.derivatives[(state_name, elem_idx)] = expr_casadi
                        else:
                            # Scalar: der(x)
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
                # Initial equations are handled separately below
                pass

            else:
                raise ValueError(f"Unsupported equation type: {eq.eq_type}")

        # Evaluate initial equations to compute parameter values
        # Initial equations assign values to parameters based on other parameters/constants
        # They are already in topological order from rumoca
        self._evaluate_initial_equations()

        # Substitute algebraic expressions in topological order
        # The equations from rumoca are already sorted in dependency order,
        # so we process algebraic equations in the order they were added,
        # substituting each one into all subsequent equations.
        #
        # This is more efficient and avoids exponential growth from
        # repeatedly substituting the same expressions.
        alg_names_in_order = list(self.algebraic.keys())

        for i, alg_name in enumerate(alg_names_in_order):
            # Substitute all previously defined algebraic variables into this one
            expr = self.algebraic[alg_name]
            for prev_name in alg_names_in_order[:i]:
                if prev_name in self.symbols:
                    expr = ca.substitute(expr, self.symbols[prev_name], self.algebraic[prev_name])
            self.algebraic[alg_name] = expr

        # Then substitute algebraic expressions into derivative expressions
        for state_name in list(self.derivatives.keys()):
            expr = self.derivatives[state_name]
            for alg_name in alg_names_in_order:
                if alg_name in self.symbols:
                    expr = ca.substitute(expr, self.symbols[alg_name], self.algebraic[alg_name])
            self.derivatives[state_name] = expr

        # Build ODE function: xdot = f(t, x, u, p)
        # Include time as an explicit input for time-varying systems
        if self.state_names:
            t = self.symbols["time"]  # Time is always available
            x = ca.vertcat(*[self.symbols[name] for name in self.state_names])
            u = (
                ca.vertcat(*[self.symbols[name] for name in self.input_names])
                if self.input_names
                else self.sym_class([])
            )
            p = (
                ca.vertcat(*[self.symbols[name] for name in self.param_names])
                if self.param_names
                else self.sym_class([])
            )

            # Build xdot expressions, handling both scalar and array states
            xdot_parts = []
            for name in self.state_names:
                shape = self.var_shapes.get(name)
                if shape is not None and len(shape) == 1:
                    # Vector state: look for element-wise derivatives
                    n = shape[0]
                    elem_exprs = []
                    for i in range(n):
                        key = (name, i)  # 0-based index
                        if key in self.derivatives:
                            elem_exprs.append(self.derivatives[key])
                        elif name in self.derivatives:
                            # Whole-array derivative given (e.g., der(x) = v)
                            # Extract the i-th element
                            elem_exprs.append(self.derivatives[name][i])
                        else:
                            elem_exprs.append(self.sym_class(0))
                    xdot_parts.extend(elem_exprs)
                else:
                    # Scalar state
                    xdot_parts.append(self.derivatives.get(name, self.sym_class(0)))
            xdot = ca.vertcat(*xdot_parts)

            self.f_ode = ca.Function("ode", [t, x, u, p], [xdot], ["t", "x", "u", "p"], ["xdot"])

        # Build algebraic functions: z = g(t, x, u, p)
        # These are used for lazy evaluation of algebraic variables in simulation results
        if self.algebraic:
            t = self.symbols["time"]
            x_syms = [self.symbols[name] for name in self.state_names]
            u_syms = [self.symbols[name] for name in self.input_names]
            p_syms = [self.symbols[name] for name in self.param_names]
            for name, expr in self.algebraic.items():
                # CasADi function names cannot contain dots, so replace with underscores
                func_name = f"alg_{name}".replace(".", "_")
                self.algebraic_funcs[name] = ca.Function(
                    func_name, [t] + x_syms + u_syms + p_syms, [expr]
                )

        self._compiled = True
        return self

    def set_parameter(self, name: str, value: float) -> None:
        """
        Set a parameter value.

        Args:
            name: Parameter name
            value: New parameter value
        """
        if name not in self.param_names:
            raise ValueError(f"Unknown parameter: {name}. Available: {self.param_names}")
        self.param_defaults[name] = value

    def _extract_default(self, var: Variable, inferred_shape: Optional[tuple] = None) -> Any:
        """
        Extract default value for a variable (scalar or array).

        Args:
            var: The variable
            inferred_shape: Optional inferred shape (overrides var.shape if provided)

        Returns:
            For scalars: float or 0.0
            For arrays: np.ndarray of appropriate shape
        """
        shape = inferred_shape if inferred_shape is not None else var.shape
        if var.is_array or shape is not None:
            # Try to get value from var.value first, then var.start
            val = var.value if var.value is not None else var.start
            if val is not None:
                if isinstance(val, (list, np.ndarray)):
                    return np.array(val).reshape(shape)
                else:
                    # Scalar value - broadcast to array shape
                    scalar = self._extract_value(val)
                    if scalar is not None:
                        return np.full(shape, scalar)
            # Default to zeros
            return np.zeros(shape)
        else:
            # Scalar variable
            val = var.value if var.value is not None else var.start
            result = self._extract_value(val)
            return result if result is not None else 0.0

    def _extract_value(self, val: Any) -> Optional[float]:
        """Extract numeric value from Expr or direct value."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, Literal):
            return float(val.value)
        # Handle unary operations (e.g., -5.0)
        if isinstance(val, UnaryOp):
            operand_val = self._extract_value(val.operand)
            if operand_val is not None:
                if val.op == "-":
                    return -operand_val
                elif val.op == "+":
                    return operand_val
        # Handle dict format from JSON: {'op': 'literal', 'value': x}
        if isinstance(val, dict):
            if val.get("op") == "literal":
                return float(val["value"])
            # Handle dict format for unary operations: {'op': 'neg', 'args': [...]}
            elif val.get("op") == "neg" and "args" in val and len(val["args"]) > 0:
                operand_val = self._extract_value(val["args"][0])
                if operand_val is not None:
                    return -operand_val
            elif val.get("op") == "pos" and "args" in val and len(val["args"]) > 0:
                operand_val = self._extract_value(val["args"][0])
                if operand_val is not None:
                    return operand_val
        return None

    def _evaluate_initial_equations(self) -> None:
        """
        Evaluate initial equations to compute parameter values.

        Initial equations are of the form `param = expr` where expr can depend
        on constants and other parameters. They are assumed to be in topological
        order from rumoca, so we can evaluate them sequentially.
        """
        if not hasattr(self.model, "initial_equations") or not self.model.initial_equations:
            return

        # Build a dictionary of all known values
        # Start with constants and parameters that have start values
        known_values: dict[str, float] = {}

        # Add constant values
        known_values.update(self.const_values)

        # Add all parameter default values
        # Parameters with explicit start values will have those values,
        # parameters without will have 0.0 as default
        for name, val in self.param_defaults.items():
            if isinstance(val, (int, float)):
                known_values[name] = float(val)

        # Process initial equations in order (they're topologically sorted)
        for eq in self.model.initial_equations:
            if eq.eq_type != EquationType.SIMPLE:
                continue
            if eq.lhs is None:
                continue

            # Get the variable name being assigned
            var_name = self._get_var_name(eq.lhs)

            # Convert RHS to CasADi expression
            try:
                rhs_expr = self._convert_expr(eq.rhs)
            except ValueError:
                # If conversion fails (unknown variable), skip for now
                continue

            # Substitute all known values into the expression
            for name, value in known_values.items():
                if name in self.symbols:
                    rhs_expr = ca.substitute(rhs_expr, self.symbols[name], self.sym_class(value))

            # Try to evaluate the expression numerically
            try:
                # Create a function with no inputs to evaluate the expression
                eval_func = ca.Function("eval", [], [rhs_expr])
                eval_result = eval_func()
                # CasADi returns a dict with 'o0' as the output key
                result = float(eval_result["o0"])

                # Store the computed value
                if var_name in self.param_names:
                    self.param_defaults[var_name] = result
                elif var_name in self.const_values:
                    self.const_values[var_name] = result

                # Add to known values for subsequent equations
                known_values[var_name] = result
            except Exception:
                # If evaluation fails, the expression may still contain free variables
                # This can happen if dependencies aren't fully resolved
                pass

    def _reduce_minmax(self, args: list, func: Callable) -> Union[ca.SX, ca.MX]:
        """
        Reduce a list of arguments using min/max function.

        Handles both:
        - max(a, b) with two scalar arguments
        - max([a, b, c]) with a single array argument

        Args:
            args: List of CasADi expressions (may be scalars or vectors)
            func: ca.fmin or ca.fmax

        Returns:
            Reduced result
        """
        if len(args) == 0:
            raise ValueError("min/max requires at least one argument")

        if len(args) == 1:
            # Single argument - could be an array/vector
            arg = args[0]
            if arg.is_vector() and arg.numel() > 1:
                # It's a vector - reduce over elements
                result = arg[0]
                for i in range(1, arg.numel()):
                    result = func(result, arg[i])
                return result
            else:
                # Single scalar - just return it
                return arg

        if len(args) == 2:
            # Two arguments - standard binary min/max
            return func(args[0], args[1])

        # More than 2 arguments - reduce pairwise
        result = args[0]
        for arg in args[1:]:
            result = func(result, arg)
        return result

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

    def _apply_subscripts(
        self, sym: Union[ca.SX, ca.MX], var_name: str, subscripts: tuple[Expr, ...]
    ) -> Union[ca.SX, ca.MX]:
        """
        Apply subscripts to extract elements/slices from an array symbol.

        Args:
            sym: The CasADi symbol (vector or matrix)
            var_name: Variable name for error messages
            subscripts: Tuple of subscript expressions (indices or slices)

        Returns:
            The indexed element or sliced subarray

        Note:
            Modelica uses 1-based indexing, CasADi uses 0-based.
            We convert from Modelica to CasADi indexing here.
        """
        shape = self.var_shapes.get(var_name)

        if shape is None:
            raise ValueError(f"Cannot index scalar variable: {var_name}")

        if len(subscripts) == 1:
            # Vector indexing: x[i] or x[1:3]
            sub = subscripts[0]
            if isinstance(sub, Slice):
                # Slice: x[1:3] -> extract range
                return self._apply_slice_1d(sym, sub)
            elif isinstance(sub, Literal):
                # Literal index: x[1] -> use integer directly
                # Convert from Modelica 1-based to CasADi 0-based
                idx_0based = int(sub.value) - 1
                return sym[idx_0based]
            else:
                # Symbolic index - CasADi can't index with SX directly
                # This requires special handling (e.g., ca.if_else cascade)
                raise NotImplementedError(
                    f"Symbolic array indexing not yet supported. "
                    f"Variable '{var_name}' indexed with non-literal: {sub}"
                )

        elif len(subscripts) == 2:
            # Matrix indexing: A[i,j] or A[1:2, 3] or A[:, j]
            sub_row, sub_col = subscripts

            # Convert row subscript
            if isinstance(sub_row, Slice):
                row_idx = self._slice_to_range(sub_row, shape[0])
            elif isinstance(sub_row, Literal):
                row_idx = int(sub_row.value) - 1  # 1-based to 0-based
            else:
                raise NotImplementedError(
                    f"Symbolic matrix row indexing not yet supported for '{var_name}'"
                )

            # Convert column subscript
            if isinstance(sub_col, Slice):
                col_idx = self._slice_to_range(sub_col, shape[1])
            elif isinstance(sub_col, Literal):
                col_idx = int(sub_col.value) - 1  # 1-based to 0-based
            else:
                raise NotImplementedError(
                    f"Symbolic matrix column indexing not yet supported for '{var_name}'"
                )

            return sym[row_idx, col_idx]

        else:
            raise NotImplementedError(f"Indexing with {len(subscripts)} dimensions not supported")

    def _apply_slice_1d(self, sym: Union[ca.SX, ca.MX], slc: Slice) -> Union[ca.SX, ca.MX]:
        """Apply a slice to a 1D array (vector)."""
        n = sym.shape[0]

        # Handle different slice cases
        if slc.start is None and slc.stop is None:
            # Full slice: x[:]
            return sym

        # Convert start/stop to indices (1-based Modelica to 0-based CasADi)
        if slc.start is not None:
            start_val = self._convert_expr(slc.start)
            start_idx = int(float(start_val)) - 1  # 1-based to 0-based
        else:
            start_idx = 0

        if slc.stop is not None:
            stop_val = self._convert_expr(slc.stop)
            stop_idx = int(float(stop_val))  # Modelica is inclusive, keep as-is
        else:
            stop_idx = n

        # Extract the slice
        return sym[start_idx:stop_idx]

    def _slice_to_range(self, slc: Slice, dim_size: int) -> slice:
        """Convert a Slice expression to a Python slice for matrix indexing."""
        if slc.start is None and slc.stop is None:
            # Full slice: :
            return slice(None)

        # Convert start/stop (1-based Modelica to 0-based)
        if slc.start is not None:
            start_val = self._convert_expr(slc.start)
            start_idx = int(float(start_val)) - 1
        else:
            start_idx = None

        if slc.stop is not None:
            stop_val = self._convert_expr(slc.stop)
            stop_idx = int(float(stop_val))  # Modelica inclusive, but Python exclusive
        else:
            stop_idx = None

        return slice(start_idx, stop_idx)

    def _convert_expr(self, expr: Expr) -> Union[ca.SX, ca.MX]:
        """Convert a Cyecca IR expression to a CasADi expression."""
        if isinstance(expr, Literal):
            return self.sym_class(expr.value)

        elif isinstance(expr, ComponentRef):
            # Flatten hierarchical reference to variable name
            var_name = self._flatten_component_ref(expr)
            last_part = expr.parts[-1]

            if var_name not in self.symbols:
                available = sorted(self.symbols.keys())
                raise ValueError(
                    f"Unknown variable: '{var_name}'\n"
                    f"Available variables: {', '.join(available)}\n"
                    f"Hint: Check for typos in your Modelica code."
                )

            sym = self.symbols[var_name]

            # Handle subscripts (array indexing) on the last part
            if last_part.subscripts:
                return self._apply_subscripts(sym, var_name, last_part.subscripts)
            else:
                return sym

        elif isinstance(expr, VarRef):
            if expr.name not in self.symbols:
                available = sorted(self.symbols.keys())
                raise ValueError(
                    f"Unknown variable: '{expr.name}'\n"
                    f"Available variables: {', '.join(available)}\n"
                    f"Hint: Check for typos in your Modelica code."
                )
            return self.symbols[expr.name]

        elif isinstance(expr, ArrayRef):
            # Legacy ArrayRef - convert to subscript access
            if expr.name not in self.symbols:
                available = sorted(self.symbols.keys())
                raise ValueError(
                    f"Unknown variable: '{expr.name}'\n"
                    f"Available variables: {', '.join(available)}\n"
                    f"Hint: Check for typos in your Modelica code."
                )
            sym = self.symbols[expr.name]
            return self._apply_subscripts(sym, expr.name, expr.indices)

        elif isinstance(expr, BinaryOp):
            left = self._convert_expr(expr.left)
            right = self._convert_expr(expr.right)

            op_map = {
                "+": lambda l, r: l + r,
                "-": lambda l, r: l - r,
                "*": lambda l, r: l * r,
                "/": lambda l, r: l / r,
                "^": lambda l, r: l**r,
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

            # Handle min/max specially - they can take arrays or multiple args
            if expr.func == "min":
                return self._reduce_minmax(args, ca.fmin)
            elif expr.func == "max":
                return self._reduce_minmax(args, ca.fmax)

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

    def _extract_zero_crossing(self, expr: Expr) -> Union[ca.SX, ca.MX]:
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
            if expr.op in ["<", "<="]:
                # For x < 0 or x <= 0, zero-crossing is x itself
                # Assuming RHS is zero (standard form)
                return self._convert_expr(expr.left)
            elif expr.op in [">", ">="]:
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
    ) -> tuple[np.ndarray, LazySimulationResult]:
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
                sol: LazySimulationResult that provides state variables directly
                     and computes algebraic variables on-demand when accessed
        """
        self._ensure_compiled()

        # Re-evaluate initial equations to update dependent parameters
        # (e.g., pid.Gain.k = pid.k needs to be recomputed if pid.k was changed via set_parameter)
        self._evaluate_initial_equations()

        # Initial conditions - flatten array states into a single vector
        x0_parts = []
        for name in self.state_names:
            val = self.state_defaults.get(name, 0.0)
            if isinstance(val, np.ndarray):
                x0_parts.append(val.flatten())
            else:
                x0_parts.append(np.array([val]))
        x0 = np.concatenate(x0_parts) if x0_parts else np.array([])

        # Parameter values - flatten array parameters
        p_parts = []
        for name in self.param_names:
            val = self.param_defaults.get(name, 0.0)
            if isinstance(val, np.ndarray):
                p_parts.append(val.flatten())
            else:
                p_parts.append(np.array([val]))
        p_val = np.concatenate(p_parts) if p_parts else np.array([])

        # Time grid
        t_grid = np.arange(0.0, t_final + dt, dt)

        # Use event handling if when-equations are present
        if self.when_equations:
            # Use CasADi integrator with event handling
            state_data = self._simulate_with_events(x0, p_val, t_grid)
        else:
            # Use simple RK4 integration (no events)
            state_data = self._simulate_rk4(x0, p_val, t_grid, input_func)

        # Wrap in LazySimulationResult for lazy algebraic variable computation
        result = LazySimulationResult(
            state_data=state_data,
            algebraic_funcs=self.algebraic_funcs,
            state_names=self.state_names,
            input_names=self.input_names,
            param_names=self.param_names,
            param_values=p_val,
            t_grid=t_grid,
            input_func=input_func,
        )

        return t_grid, result

    def _simulate_with_events(
        self,
        x0: np.ndarray,
        p_val: np.ndarray,
        t_grid: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Simulate using CasADi integrator with event handling.

        Supports multiple when-equations. Each when-equation defines:
        - A zero-crossing condition that triggers the event
        - Reset equations that are applied when the event occurs

        CasADi requires event indicators to be expressions that cross zero.
        Multiple events are handled by providing a vector of zero-crossing
        indicators and a transition function that handles all events.
        """
        # Check if event handling functions are available
        if not HAS_EVENT_FUNCTIONS:
            raise RuntimeError(
                "CasADi event handling requires CasADi 3.6.4+ with event_in/event_out functions. "
                f"Current version: {ca.__version__}. "
                "When-equations cannot be simulated with this CasADi version."
            )

        # Build DAE structure
        x = ca.vertcat(*[self.symbols[name] for name in self.state_names])
        p = (
            ca.vertcat(*[self.symbols[name] for name in self.param_names])
            if self.param_names
            else ca.SX([])
        )

        xdot_exprs = [self.derivatives.get(name, ca.SX(0)) for name in self.state_names]
        ode = ca.vertcat(*xdot_exprs)

        # Build zero-crossing indicators for all when-equations
        # Each when-equation contributes one zero-crossing indicator
        event_indicators = []
        for condition_expr, _ in self.when_equations:
            indicator = self._extract_zero_crossing(condition_expr)
            event_indicators.append(indicator)

        # Stack all event indicators into a vector
        zero = ca.vertcat(*event_indicators)

        # Build DAE
        dae = {
            "x": x,
            "ode": ode,
            "zero": zero,  # Vector of zero-crossing indicators
        }

        if self.param_names:
            dae["p"] = p

        # Build combined transition function for all events
        # For each event i, we need to check which event triggered and apply
        # the corresponding reset equations
        #
        # CasADi provides event_in() with 'i' indicating which event triggered
        # We use if_else to select the appropriate reset based on event index
        post_x_exprs = [self.symbols[name] for name in self.state_names]

        if len(self.when_equations) == 1:
            # Single event - simple case
            _, reset_eqs = self.when_equations[0]
            for reset_eq in reset_eqs:
                if reset_eq.eq_type == EquationType.SIMPLE:
                    var_name = self._get_var_name(reset_eq.lhs)
                    reset_value = self._convert_expr(reset_eq.rhs)
                    if var_name in self.state_names:
                        idx = self.state_names.index(var_name)
                        post_x_exprs[idx] = reset_value
        else:
            # Multiple events - use if_else to select reset based on event index
            # event_in() provides 'i' which indicates which zero crossing triggered
            i_event = ca.SX.sym("i")  # Event index from CasADi

            # For each state, build a cascaded if_else based on event index
            for state_idx, name in enumerate(self.state_names):
                state_sym = self.symbols[name]
                result_expr = state_sym  # Default: unchanged

                # Go through events in reverse order to build nested if_else
                for event_idx in range(len(self.when_equations) - 1, -1, -1):
                    _, reset_eqs = self.when_equations[event_idx]

                    # Find reset for this state in this event's equations
                    reset_value = state_sym  # Default: unchanged
                    for reset_eq in reset_eqs:
                        if reset_eq.eq_type == EquationType.SIMPLE:
                            var_name = self._get_var_name(reset_eq.lhs)
                            if var_name == name:
                                reset_value = self._convert_expr(reset_eq.rhs)
                                break

                    # Build condition: if i == event_idx then reset_value else result_expr
                    result_expr = ca.if_else(i_event == event_idx, reset_value, result_expr)

                post_x_exprs[state_idx] = result_expr

        post_x = ca.vertcat(*post_x_exprs)

        # Create transition function (following CasADi official example format)
        trans_dict = {"x": x, "post_x": post_x}
        if self.param_names:
            trans_dict["p"] = p

        # For multiple events, include the event index in transition function
        if len(self.when_equations) > 1:
            trans_dict["i"] = i_event

        transition = ca.Function("transition", trans_dict, event_in(), event_out())

        # Create integrator with event handling
        opts = {
            "transition": transition,
            "max_events": 100 * len(self.when_equations),  # Scale with number of events
        }

        integrator_args = ["sim", "cvodes", dae, t_grid[0], t_grid]
        if self.param_names:
            integrator = ca.integrator(*integrator_args, opts)
        else:
            integrator = ca.integrator(*integrator_args, opts)

        # Run integration
        sim_args = {"x0": x0}
        if self.param_names:
            sim_args["p"] = p_val

        result = integrator(**sim_args)

        # Extract results
        x_result = result["xf"].full()

        sol = {}
        flat_idx = 0
        for name in self.state_names:
            shape = self.var_shapes.get(name)
            if shape is not None and len(shape) == 1:
                # Vector state: extract n elements and transpose to [n_steps, dim]
                n = shape[0]
                sol[name] = x_result[flat_idx : flat_idx + n, :].T
                flat_idx += n
            else:
                # Scalar state: shape is [n_steps]
                sol[name] = x_result[flat_idx, :]
                flat_idx += 1

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
        n_flat_states = len(x0)  # Total flattened state size

        # Allocate result arrays
        x_result = np.zeros((n_flat_states, n_steps))
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

            # RK4 step - f_ode signature is (t, x, u, p)
            k1 = np.array(self.f_ode(t, x, u, p_val)).flatten()
            k2 = np.array(self.f_ode(t + dt / 2, x + dt * k1 / 2, u, p_val)).flatten()
            k3 = np.array(self.f_ode(t + dt / 2, x + dt * k2 / 2, u, p_val)).flatten()
            k4 = np.array(self.f_ode(t + dt, x + dt * k3, u, p_val)).flatten()

            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            x_result[:, i] = x

        # Package results - unflatten array states back to their original form
        sol = {}
        flat_idx = 0
        for name in self.state_names:
            shape = self.var_shapes.get(name)
            if shape is not None and len(shape) == 1:
                # Vector state: extract n elements and transpose to [n_steps, dim]
                n = shape[0]
                sol[name] = x_result[flat_idx : flat_idx + n, :].T
                flat_idx += n
            else:
                # Scalar state: shape is [n_steps]
                sol[name] = x_result[flat_idx, :]
                flat_idx += 1

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
        u = (
            ca.vertcat(*[self.symbols[name] for name in self.input_names])
            if self.input_names
            else ca.SX([])
        )
        p = (
            ca.vertcat(*[self.symbols[name] for name in self.param_names])
            if self.param_names
            else ca.SX([])
        )

        xdot_exprs = [self.derivatives.get(name, ca.SX(0)) for name in self.state_names]
        xdot = ca.vertcat(*xdot_exprs)

        # Compute Jacobians
        A_sym = ca.jacobian(xdot, x)
        B_sym = ca.jacobian(xdot, u) if self.input_names else ca.SX(len(self.state_names), 0)

        # Create functions for evaluation
        A_func = ca.Function("A", [x, u, p], [A_sym])
        B_func = ca.Function("B", [x, u, p], [B_sym])

        # Prepare operating point
        if x0 is None:
            x0 = {}
        x0_vec = np.array(
            [x0.get(name, self.state_defaults.get(name, 0.0)) for name in self.state_names]
        )

        if u0 is None:
            u0 = {}
        u0_vec = (
            np.array([u0.get(name, 0.0) for name in self.input_names])
            if self.input_names
            else np.array([])
        )

        p_vec = (
            np.array([self.param_defaults.get(name, 0.0) for name in self.param_names])
            if self.param_names
            else np.array([])
        )

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


def compile_modelica(source: str, model_name: Optional[str] = None) -> CasadiBackend:
    """
    Compile Modelica source code to a ready-to-use CasadiBackend.

    This is the simplest way to go from Modelica code to simulation.

    Args:
        source: Modelica source code as a string
        model_name: Name of the model to compile. If None, auto-detects from source.

    Returns:
        Compiled CasadiBackend ready for simulation

    Example:
        >>> from cyecca.backends.casadi import compile_modelica
        >>>
        >>> model = compile_modelica('''
        ...     model MyModel
        ...         Real x(start=1);
        ...     equation
        ...         der(x) = -x;
        ...     end MyModel;
        ... ''')
        >>> t, sol = model.simulate(10.0)
    """
    import re

    # Auto-detect model name if not provided
    if model_name is None:
        match = re.search(r"\b(?:model|class)\s+(\w+)", source)
        if match:
            model_name = match.group(1)
        else:
            raise ValueError("Could not detect model name. Please provide model_name argument.")

    # Import here to avoid circular imports
    from cyecca.io import compile_modelica as _compile_modelica

    ir_model = _compile_modelica(source, model_name)
    backend = CasadiBackend(ir_model)
    backend.compile()
    return backend
