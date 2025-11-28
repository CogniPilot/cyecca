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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import casadi as ca
import numpy as np
from beartype import beartype

from cyecca.dsl.equations import Equation, Reinit, WhenClause
from cyecca.dsl.expr import Expr, ExprKind
from cyecca.dsl.flat_model import FlatModel
from cyecca.dsl.simulation import SimulationResult, Simulator
from cyecca.dsl.variables import SymbolicVar


class SymbolicType(Enum):
    """CasADi symbolic type selection."""

    SX = auto()  # Scalar symbolic - expands arrays, good for small models
    MX = auto()  # Matrix symbolic - keeps array structure, good for large models


class Integrator(Enum):
    """Integration method selection."""

    RK4 = auto()  # Fixed-step 4th-order Runge-Kutta (default, simple)
    CVODES = auto()  # SUNDIALS CVODES - variable-step BDF/Adams (stiff/non-stiff ODEs)
    IDAS = auto()  # SUNDIALS IDAS - variable-step BDF for DAEs


# Type variable for SX or MX
SymT = TypeVar("SymT", ca.SX, ca.MX)


# =============================================================================
# Expression Conversion - Dispatch Table (shared by SX and MX)
# =============================================================================


def _make_expr_handlers():
    """Create dispatch table for expression conversion.

    This is the single source of truth for converting Expr to CasADi.
    Works identically for both SX and MX types.
    """
    # Unary math operations
    unary_math = {
        ExprKind.NEG: lambda c, e: -c(e.children[0]),
        ExprKind.SIN: lambda c, e: ca.sin(c(e.children[0])),
        ExprKind.COS: lambda c, e: ca.cos(c(e.children[0])),
        ExprKind.TAN: lambda c, e: ca.tan(c(e.children[0])),
        ExprKind.ASIN: lambda c, e: ca.asin(c(e.children[0])),
        ExprKind.ACOS: lambda c, e: ca.acos(c(e.children[0])),
        ExprKind.ATAN: lambda c, e: ca.atan(c(e.children[0])),
        ExprKind.SQRT: lambda c, e: ca.sqrt(c(e.children[0])),
        ExprKind.EXP: lambda c, e: ca.exp(c(e.children[0])),
        ExprKind.LOG: lambda c, e: ca.log(c(e.children[0])),
        ExprKind.LOG10: lambda c, e: ca.log10(c(e.children[0])),
        ExprKind.ABS: lambda c, e: ca.fabs(c(e.children[0])),
        ExprKind.SIGN: lambda c, e: ca.sign(c(e.children[0])),
        ExprKind.FLOOR: lambda c, e: ca.floor(c(e.children[0])),
        ExprKind.CEIL: lambda c, e: ca.ceil(c(e.children[0])),
        ExprKind.SINH: lambda c, e: ca.sinh(c(e.children[0])),
        ExprKind.COSH: lambda c, e: ca.cosh(c(e.children[0])),
        ExprKind.TANH: lambda c, e: ca.tanh(c(e.children[0])),
        ExprKind.NOT: lambda c, e: ca.logic_not(c(e.children[0])),
    }

    # Binary math operations
    binary_math = {
        ExprKind.ADD: lambda c, e: c(e.children[0]) + c(e.children[1]),
        ExprKind.SUB: lambda c, e: c(e.children[0]) - c(e.children[1]),
        ExprKind.MUL: lambda c, e: c(e.children[0]) * c(e.children[1]),
        ExprKind.DIV: lambda c, e: c(e.children[0]) / c(e.children[1]),
        ExprKind.POW: lambda c, e: c(e.children[0]) ** c(e.children[1]),
        ExprKind.ATAN2: lambda c, e: ca.atan2(c(e.children[0]), c(e.children[1])),
        ExprKind.MIN: lambda c, e: ca.fmin(c(e.children[0]), c(e.children[1])),
        ExprKind.MAX: lambda c, e: ca.fmax(c(e.children[0]), c(e.children[1])),
        ExprKind.AND: lambda c, e: ca.logic_and(c(e.children[0]), c(e.children[1])),
        ExprKind.OR: lambda c, e: ca.logic_or(c(e.children[0]), c(e.children[1])),
    }

    # Relational operations
    relational = {
        ExprKind.LT: lambda c, e: c(e.children[0]) < c(e.children[1]),
        ExprKind.LE: lambda c, e: c(e.children[0]) <= c(e.children[1]),
        ExprKind.GT: lambda c, e: c(e.children[0]) > c(e.children[1]),
        ExprKind.GE: lambda c, e: c(e.children[0]) >= c(e.children[1]),
        ExprKind.EQ: lambda c, e: c(e.children[0]) == c(e.children[1]),
        ExprKind.NE: lambda c, e: c(e.children[0]) != c(e.children[1]),
    }

    # Ternary
    ternary = {
        ExprKind.IF_THEN_ELSE: lambda c, e: ca.if_else(c(e.children[0]), c(e.children[1]), c(e.children[2])),
    }

    return {**unary_math, **binary_math, **relational, **ternary}


# Global dispatch table
_EXPR_HANDLERS = _make_expr_handlers()


# =============================================================================
# Unified Compiler Class
# =============================================================================


class CasadiCompiler:
    """
    Unified compiler for both SX and MX symbolic types.

    This class contains the shared compilation logic. The only differences
    between SX and MX are:
    1. Symbol creation (ca.SX.sym vs ca.MX.sym)
    2. Zero vector creation (ca.SX.zeros vs ca.MX.zeros)
    3. MX supports indexed array access from unexpanded names
    """

    def __init__(self, sym_type: Type[SymT], model: FlatModel):
        """
        Initialize compiler with symbolic type and model.

        Parameters
        ----------
        sym_type : Type[ca.SX] or Type[ca.MX]
            The CasADi symbolic type to use
        model : FlatModel
            The flattened model to compile
        """
        self.sym_type = sym_type
        self.model = model
        self.is_mx = sym_type == ca.MX

        # Symbol dictionaries (populated during compilation)
        self.state_syms: Dict[str, SymT] = {}
        self.input_syms: Dict[str, SymT] = {}
        self.param_syms: Dict[str, SymT] = {}
        self.algebraic_syms: Dict[str, SymT] = {}
        self.pre_syms: Dict[str, SymT] = {}  # For when-clauses
        self.t_sym: SymT = None

        # Shape tracking (for MX indexed access)
        self.state_shapes: Dict[str, Tuple[int, ...]] = {}
        self.algebraic_shapes: Dict[str, Tuple[int, ...]] = {}

        # Combined symbol lookup
        self.base_syms: Dict[str, SymT] = {}

        # Output expression cache
        self.compiled_outputs: Dict[str, SymT] = {}

    def _sym(self, name: str, size: int = 1) -> SymT:
        """Create a symbol of the appropriate type."""
        return self.sym_type.sym(name, size) if size > 1 else self.sym_type.sym(name)

    def _zeros(self, size: int) -> SymT:
        """Create a zero vector of the appropriate type."""
        return self.sym_type.zeros(size)

    def _const(self, value: float) -> SymT:
        """Create a constant of the appropriate type."""
        return self.sym_type(value)

    @staticmethod
    def _shape_to_size(shape: Tuple[int, ...]) -> int:
        """Compute total size from shape."""
        if not shape:
            return 1
        result = 1
        for dim in shape:
            result *= dim
        return result

    def _create_symbols(self) -> None:
        """Create all CasADi symbols for the model."""
        model = self.model

        # State symbols
        for name in model.state_names:
            v = model.state_vars[name]
            shape = v.shape if v.shape else ()
            self.state_shapes[name] = shape
            size = self._shape_to_size(shape)
            self.state_syms[name] = self._sym(name, size)

        # Input symbols
        for name in model.input_names:
            v = model.input_vars[name]
            size = v.size if v.size > 1 else 1
            self.input_syms[name] = self._sym(name, size)

        # Parameter symbols
        for name in model.param_names:
            v = model.param_vars[name]
            size = v.size if v.size > 1 else 1
            self.param_syms[name] = self._sym(name, size)

        # Algebraic variable symbols
        for name in model.algebraic_names:
            v = model.algebraic_vars[name]
            shape = v.shape if v.shape else ()
            self.algebraic_shapes[name] = shape
            size = self._shape_to_size(shape)
            self.algebraic_syms[name] = self._sym(name, size)

        # Time symbol
        self.t_sym = self._sym("t")

        # Combined lookup (NOT including outputs - they get substituted)
        self.base_syms = {
            **self.state_syms,
            **self.input_syms,
            **self.param_syms,
            **self.algebraic_syms,
        }

    def _resolve_indexed_variable(self, name: str) -> SymT:
        """
        Resolve an indexed variable name like 'pos[0]' or 'R[0,1]'.

        Only used for MX backend when arrays aren't expanded.
        """
        if "[" not in name:
            raise ValueError(f"Not an indexed name: {name}")

        base_name = name.split("[")[0]
        if base_name not in self.base_syms:
            raise ValueError(f"Unknown variable base: {base_name}")

        # Parse indices
        idx_str = name[name.index("[") + 1 : name.rindex("]")]
        indices = tuple(int(i) for i in idx_str.split(","))

        sym = self.base_syms[base_name]
        all_shapes = {**self.state_shapes, **self.algebraic_shapes}
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

    def expr_to_casadi(self, expr: Expr) -> SymT:
        """
        Convert an Expr tree to a CasADi expression.

        This is the main expression compiler. It handles:
        - Variable lookup (with output substitution)
        - Constants and time
        - All math/logic operations via dispatch table
        - MX-specific indexed variable access
        """
        # Variable reference
        if expr.kind == ExprKind.VARIABLE:
            name = expr.name

            # Direct lookup
            if name in self.base_syms:
                return self.base_syms[name]

            # Output substitution (compile the output's expression)
            if name in self.model.output_equations:
                if name in self.compiled_outputs:
                    return self.compiled_outputs[name]
                out_expr = self.expr_to_casadi(self.model.output_equations[name])
                self.compiled_outputs[name] = out_expr
                return out_expr

            # MX indexed access (e.g., 'pos[0]')
            if self.is_mx and "[" in name:
                return self._resolve_indexed_variable(name)

            raise ValueError(f"Unknown variable: {name}")

        # Derivative (should not appear in RHS)
        if expr.kind == ExprKind.DERIVATIVE:
            raise ValueError("DERIVATIVE nodes should not appear in RHS expressions")

        # Constant
        if expr.kind == ExprKind.CONSTANT:
            return self._const(expr.value)

        # Time
        if expr.kind == ExprKind.TIME:
            return self.t_sym

        # Discrete operators (not supported in continuous equations)
        if expr.kind in (ExprKind.PRE, ExprKind.EDGE, ExprKind.CHANGE):
            raise NotImplementedError(
                f"Discrete operator '{expr.kind.name.lower()}()' is not yet supported "
                "in continuous equations. Use inside when-clauses only."
            )

        # Dispatch table lookup
        handler = _EXPR_HANDLERS.get(expr.kind)
        if handler:
            return handler(self.expr_to_casadi, expr)

        raise ValueError(f"Unsupported expression kind: {expr.kind}")

    def expr_to_casadi_when(self, expr: Expr) -> SymT:
        """
        Convert Expr for when-clause context (handles pre() operator).
        """
        if expr.kind == ExprKind.PRE:
            pre_name = expr.name
            if pre_name in self.pre_syms:
                return self.pre_syms[pre_name]
            if pre_name in self.base_syms:
                return self.base_syms[pre_name]
            raise ValueError(f"Unknown variable in pre(): {pre_name}")

        # For relational operators in event conditions, convert to zero-crossing form
        if expr.kind == ExprKind.LT:
            return self.expr_to_casadi_when(expr.children[0]) - self.expr_to_casadi_when(expr.children[1])
        if expr.kind == ExprKind.LE:
            return self.expr_to_casadi_when(expr.children[0]) - self.expr_to_casadi_when(expr.children[1])
        if expr.kind == ExprKind.GT:
            return self.expr_to_casadi_when(expr.children[1]) - self.expr_to_casadi_when(expr.children[0])
        if expr.kind == ExprKind.GE:
            return self.expr_to_casadi_when(expr.children[1]) - self.expr_to_casadi_when(expr.children[0])

        # Variable reference
        if expr.kind == ExprKind.VARIABLE:
            if expr.name in self.base_syms:
                return self.base_syms[expr.name]
            raise ValueError(f"Unknown variable: {expr.name}")

        # Constant
        if expr.kind == ExprKind.CONSTANT:
            return self._const(expr.value)

        # Time
        if expr.kind == ExprKind.TIME:
            return self.t_sym

        # Dispatch for other operations
        handler = _EXPR_HANDLERS.get(expr.kind)
        if handler:
            return handler(self.expr_to_casadi_when, expr)

        # Fallback to regular conversion
        return self.expr_to_casadi(expr)

    def _build_state_derivatives(self) -> List[SymT]:
        """Build the state derivative vector."""
        model = self.model
        state_derivs: List[SymT] = []

        for name in model.state_names:
            shape = self.state_shapes.get(name, ())
            size = self._shape_to_size(shape)

            # Check for array derivative equation (MX backend)
            if self.is_mx and name in model.array_derivative_equations:
                arr_eq = model.array_derivative_equations[name]
                rhs = arr_eq["rhs"]
                # RHS is a SymbolicVar - get its symbol
                if rhs.base_name in self.base_syms:
                    state_derivs.append(self.base_syms[rhs.base_name])
                else:
                    state_derivs.append(self._zeros(size))
            elif name in model.derivative_equations:
                # Scalar/element equation
                deriv_expr = model.derivative_equations[name]
                state_derivs.append(self.expr_to_casadi(deriv_expr))
            else:
                # Check for element-wise equations (MX)
                if self.is_mx:
                    deriv_vec = self._zeros(size)
                    has_any = False
                    for key, deriv_expr in model.derivative_equations.items():
                        if key.startswith(f"{name}["):
                            has_any = True
                            idx_str = key[key.index("[") + 1 : key.rindex("]")]
                            indices = tuple(int(i) for i in idx_str.split(","))
                            flat_idx = indices[0] if len(indices) == 1 else self._compute_flat_index(indices, shape)
                            deriv_vec[flat_idx] = self.expr_to_casadi(deriv_expr)
                    if has_any:
                        state_derivs.append(deriv_vec)
                    else:
                        state_derivs.append(self._zeros(size))
                else:
                    state_derivs.append(self._zeros(size))

        return state_derivs

    def _compute_flat_index(self, indices: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
        """Convert multi-dimensional indices to flat index (row-major)."""
        flat_idx = 0
        stride = 1
        for i in range(len(indices) - 1, -1, -1):
            flat_idx += indices[i] * stride
            if i > 0:
                stride *= shape[i]
        return flat_idx

    def _build_outputs(self) -> List[SymT]:
        """Build the output expressions vector."""
        model = self.model
        y_exprs: List[SymT] = []

        for name in model.output_names:
            if name in model.output_equations:
                out_expr = model.output_equations[name]
                y_exprs.append(self.expr_to_casadi(out_expr))
            else:
                import warnings

                warnings.warn(f"Output '{name}' has no equation, will be zero")
                v = model.output_vars[name]
                size = v.size if v.size > 1 else 1
                y_exprs.append(self._zeros(size))

        return y_exprs

    def _build_algebraic_residuals(self) -> List[SymT]:
        """Build algebraic equation residuals (0 = lhs - rhs)."""
        residuals: List[SymT] = []
        for eq in self.model.algebraic_equations:
            lhs = self.expr_to_casadi(eq.lhs)
            rhs = self.expr_to_casadi(eq.rhs)
            residuals.append(lhs - rhs)
        return residuals

    def _build_when_clause_funcs(self, x: SymT, u: SymT, p: SymT) -> List[Dict[str, Any]]:
        """Build event and reinit functions for when-clauses."""
        model = self.model
        when_clause_funcs: List[Dict[str, Any]] = []

        if not model.when_clauses:
            return when_clause_funcs

        # Create pre-event state symbols
        for name in model.state_names:
            v = model.state_vars[name]
            size = v.size if v.size > 1 else 1
            self.pre_syms[name] = self._sym(f"pre_{name}", size)

        pre_list = [self.pre_syms[n] for n in model.state_names]
        x_pre = ca.vertcat(*pre_list) if pre_list else self._sym("x_pre", 0)

        for i, wc in enumerate(model.when_clauses):
            # Event function: returns value that crosses zero
            event_expr = self.expr_to_casadi_when(wc.condition)
            f_event = ca.Function(
                f"f_event_{i}",
                [x, u, p, self.t_sym],
                [event_expr],
                ["x", "u", "p", "t"],
                ["event"],
            )

            # Reinit function: computes new state after event
            x_new_list = [self.pre_syms[n] for n in model.state_names]

            for item in wc.body:
                if isinstance(item, Reinit):
                    var_name = item.var_name
                    if var_name in model.state_names:
                        idx = model.state_names.index(var_name)
                        new_val = self.expr_to_casadi_when(item.expr)
                        x_new_list[idx] = new_val

            x_new = ca.vertcat(*x_new_list) if x_new_list else self._sym("x_new", 0)

            f_reinit = ca.Function(
                f"f_reinit_{i}",
                [x_pre, u, p, self.t_sym],
                [x_new],
                ["x_pre", "u", "p", "t"],
                ["x_new"],
            )

            when_clause_funcs.append(
                {
                    "condition": wc.condition,
                    "f_event": f_event,
                    "f_reinit": f_reinit,
                }
            )

        return when_clause_funcs

    def compile(self) -> "CompiledModel":
        """
        Compile the model to CasADi functions.

        Returns
        -------
        CompiledModel
            Compiled model ready for simulation
        """
        model = self.model

        # Create all symbols
        self._create_symbols()

        # Build derivative vector
        state_derivs = self._build_state_derivatives()

        # Build output vector
        y_exprs = self._build_outputs()

        # Build algebraic residuals
        alg_residuals = self._build_algebraic_residuals()

        # Build CasADi vectors
        state_list = [self.state_syms[n] for n in model.state_names]
        input_list = [self.input_syms[n] for n in model.input_names]
        param_list = [self.param_syms[n] for n in model.param_names]
        algebraic_list = [self.algebraic_syms[n] for n in model.algebraic_names]

        x = ca.vertcat(*state_list) if state_list else self._sym("x", 0)
        xdot = ca.vertcat(*state_derivs) if state_derivs else self._sym("xdot", 0)
        u = ca.vertcat(*input_list) if input_list else self._sym("u", 0)
        p = ca.vertcat(*param_list) if param_list else self._sym("p", 0)
        y = ca.vertcat(*y_exprs) if y_exprs else self._sym("y", 0)
        z = ca.vertcat(*algebraic_list) if algebraic_list else self._sym("z", 0)
        alg = ca.vertcat(*alg_residuals) if alg_residuals else self._sym("alg", 0)

        # Create dynamics function f_x(x, z, u, p, t) -> xdot
        f_x = ca.Function("f_x", [x, z, u, p, self.t_sym], [xdot], ["x", "z", "u", "p", "t"], ["xdot"])

        # Create algebraic residual function (for DAE)
        f_alg = (
            ca.Function("f_alg", [x, z, u, p, self.t_sym], [alg], ["x", "z", "u", "p", "t"], ["alg"])
            if alg_residuals
            else None
        )

        # Create output function
        f_y = ca.Function("f_y", [x, z, u, p, self.t_sym], [y], ["x", "z", "u", "p", "t"], ["y"]) if y_exprs else None

        # Build when-clause functions
        when_clause_funcs = self._build_when_clause_funcs(x, u, p)

        # Build algebraic defaults
        algebraic_defaults: Dict[str, Any] = {}
        for name in model.algebraic_names:
            v = model.algebraic_vars[name]
            algebraic_defaults[name] = v.start if v.start is not None else 0.0

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
            _algebraic_names=model.algebraic_names,
            algebraic_defaults=algebraic_defaults,
            f_alg=f_alg,
        )


# =============================================================================
# Backend Interface
# =============================================================================


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
        sym_class = ca.MX if symbolic_type == SymbolicType.MX else ca.SX
        compiler = CasadiCompiler(sym_class, model)
        return compiler.compile()


# =============================================================================
# CompiledModel (unchanged from original)
# =============================================================================


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

    DAE Support (IDAS)
    ------------------
    For models with algebraic variables, use integrator=Integrator.IDAS.
    The model is formulated as:
        ẋ = f(x, z, u, p, t)    (differential equations)
        0 = g(x, z, u, p, t)    (algebraic equations)
    where x are states, z are algebraic variables.
    """

    name: str
    f_x: ca.Function  # Dynamics: f_x(x, z, u, p, t) -> xdot
    f_y: Optional[ca.Function]  # Outputs: f_y(x, z, u, p, t) -> y
    _state_names: List[str]
    _input_names: List[str]
    _param_names: List[str]
    _output_names: List[str]
    state_defaults: Dict[str, Any]
    input_defaults: Dict[str, Any]
    param_defaults: Dict[str, Any]
    when_clause_funcs: List[Dict[str, Any]] = None
    _algebraic_names: List[str] = None
    algebraic_defaults: Dict[str, Any] = None
    f_alg: Optional[ca.Function] = None

    def __post_init__(self):
        if self.when_clause_funcs is None:
            self.when_clause_funcs = []
        if self._algebraic_names is None:
            self._algebraic_names = []
        if self.algebraic_defaults is None:
            self.algebraic_defaults = {}

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
    def algebraic_names(self) -> List[str]:
        return self._algebraic_names

    @property
    def has_algebraic(self) -> bool:
        """Check if model has algebraic variables (DAE system)."""
        return len(self._algebraic_names) > 0

    @property
    def has_events(self) -> bool:
        """Check if model has when-clauses (hybrid system)."""
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
        integrator: Integrator = Integrator.RK4,
    ) -> SimulationResult:
        """
        Simulate the model.

        For models with when-clauses, the simulator detects zero-crossings
        and applies state reinitializations at event times.

        Parameters
        ----------
        t0 : float
            Initial time
        tf : float
            Final time
        dt : float
            Time step (output interval for CVODES, fixed step for RK4)
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
        integrator : Integrator, default=Integrator.RK4
            Integration method:
            - RK4: Fixed-step 4th-order Runge-Kutta
            - CVODES: SUNDIALS variable-step BDF/Adams method
            - IDAS: SUNDIALS variable-step DAE solver

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
        # f_x signature: f_x(x, z, u, p, t) -> xdot
        # Input indices: 0=x, 1=z, 2=u, 3=p, 4=t
        n_params = self.f_x.size1_in(3)  # p is input 3
        p = np.zeros(n_params)
        for i, name in enumerate(self._param_names):
            if params and name in params:
                p[i] = params[name]
            elif name in self.param_defaults:
                p[i] = self.param_defaults[name]

        # Build algebraic variable vector (for DAE systems)
        n_alg = len(self._algebraic_names)
        z_init = np.zeros(n_alg)
        for i, name in enumerate(self._algebraic_names):
            if name in self.algebraic_defaults:
                z_init[i] = self.algebraic_defaults[name]

        # Build constant input vector
        n_inputs = self.f_x.size1_in(2)  # u is input 2
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

        # For pure ODE systems, z is empty
        z_empty = np.zeros(0) if n_alg == 0 else z_init

        def rk4_step(x: np.ndarray, ti: float, h: float) -> np.ndarray:
            """Single RK4 integration step (ODE only, no algebraic)."""
            u_vec = get_input(ti)
            k1 = np.array(self.f_x(x, z_empty, u_vec, p, ti)).flatten()
            k2 = np.array(self.f_x(x + 0.5 * h * k1, z_empty, u_vec, p, ti + 0.5 * h)).flatten()
            k3 = np.array(self.f_x(x + 0.5 * h * k2, z_empty, u_vec, p, ti + 0.5 * h)).flatten()
            k4 = np.array(self.f_x(x + h * k3, z_empty, u_vec, p, ti + h)).flatten()
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

        # For CVODES (ODE), we'll create the integrator inside the step function
        cvodes_dae = None
        cvodes_opts = None
        if integrator == Integrator.CVODES:
            if self.has_algebraic:
                raise ValueError(
                    "CVODES does not support DAE systems with algebraic variables. "
                    "Use integrator=Integrator.IDAS for DAE systems."
                )
            x_sym = ca.SX.sym("x", n_states)
            z_sym = ca.SX.sym("z", max(n_alg, 1))  # Dummy if no algebraic
            u_sym = ca.SX.sym("u", max(n_inputs, 1))
            p_sym = ca.SX.sym("p", n_params)
            t_sym = ca.SX.sym("t")

            # Build the ODE right-hand side (z is empty for pure ODE)
            z_in = z_sym[:n_alg] if n_alg > 0 else ca.SX()
            u_in = u_sym[:n_inputs] if n_inputs > 0 else ca.SX()
            xdot = self.f_x(x_sym, z_in, u_in, p_sym, t_sym)

            # Store DAE definition for use in step function
            cvodes_dae = {"x": x_sym, "t": t_sym, "p": ca.vertcat(u_sym, p_sym), "ode": xdot}
            cvodes_opts = {"abstol": 1e-8, "reltol": 1e-6}

        # For IDAS (DAE), create integrator that handles both differential and algebraic
        idas_dae = None
        idas_opts = None
        if integrator == Integrator.IDAS:
            x_sym = ca.SX.sym("x", n_states)
            z_sym = ca.SX.sym("z", max(n_alg, 1))
            u_sym = ca.SX.sym("u", max(n_inputs, 1))
            p_sym = ca.SX.sym("p", n_params)
            t_sym = ca.SX.sym("t")

            z_in = z_sym[:n_alg] if n_alg > 0 else ca.SX()
            u_in = u_sym[:n_inputs] if n_inputs > 0 else ca.SX()

            # Differential equations: ẋ = f(x, z, u, p, t)
            xdot = self.f_x(x_sym, z_in, u_in, p_sym, t_sym)

            # Algebraic equations: 0 = g(x, z, u, p, t)
            if self.f_alg is not None:
                alg = self.f_alg(x_sym, z_in, u_in, p_sym, t_sym)
            else:
                alg = ca.SX()

            # IDAS DAE formulation
            idas_dae = {
                "x": x_sym,
                "z": z_in,
                "t": t_sym,
                "p": ca.vertcat(u_sym, p_sym),
                "ode": xdot,
                "alg": alg,
            }
            idas_opts = {"abstol": 1e-8, "reltol": 1e-6}

        def cvodes_step(x: np.ndarray, ti: float, h: float) -> np.ndarray:
            """Single CVODES integration step using CasADi integrator."""
            u_vec = get_input(ti)
            p_combined = np.concatenate([u_vec if n_inputs > 0 else np.array([0.0]), p])
            # Create integrator with t0 and tf as positional args (new API)
            integ = ca.integrator("integ", "cvodes", cvodes_dae, ti, ti + h, cvodes_opts)
            result = integ(x0=x, p=p_combined)
            return np.array(result["xf"]).flatten()

        def idas_step(x: np.ndarray, ti: float, h: float) -> np.ndarray:
            """Single IDAS integration step for DAE systems."""
            u_vec = get_input(ti)
            p_combined = np.concatenate([u_vec if n_inputs > 0 else np.array([0.0]), p])
            # Create IDAS integrator
            integ = ca.integrator("integ", "idas", idas_dae, ti, ti + h, idas_opts)
            result = integ(x0=x, z0=z_init, p=p_combined)
            return np.array(result["xf"]).flatten()

        # Select step function based on integrator
        if integrator == Integrator.IDAS:
            step_func = idas_step
        elif integrator == Integrator.CVODES:
            step_func = cvodes_step
        else:
            step_func = rk4_step

        # Use event-detecting simulation if we have when-clauses
        if self.has_events:
            return self._simulate_with_events(
                t0,
                tf,
                dt,
                x_init,
                p,
                n_inputs,
                n_states,
                get_input,
                step_func,
                check_events,
                apply_reinit,
                max_events,
                integrator,
                z_empty,
            )

        # Integration without events
        t = np.arange(t0, tf + dt, dt)
        n_steps = len(t)

        # Storage
        x_hist = np.zeros((n_steps, n_states))
        x_hist[0] = x_init
        u_hist = np.zeros((n_steps, n_inputs)) if n_inputs > 0 else None

        # Integration loop
        x = x_init.copy()
        for i in range(n_steps):
            ti = t[i]
            u_vec = get_input(ti)

            # Record input
            if u_hist is not None:
                u_hist[i] = u_vec

            # Step (except for last point)
            if i < n_steps - 1:
                x = step_func(x, ti, dt)
                x_hist[i + 1] = x

        return self._build_result(t, x_hist, u_hist, n_inputs, p, get_input, z_empty)

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
        step_func: Callable,
        check_events: Callable,
        apply_reinit: Callable,
        max_events: int,
        integrator: Integrator = Integrator.RK4,
        z_empty: Optional[np.ndarray] = None,
    ) -> SimulationResult:
        """
        Simulate with event detection using bisection for zero-crossing.
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
            x_next = step_func(x, t, h)
            t_next = t + h

            # Check for events (sign changes in event functions)
            curr_events = check_events(x_next, t_next)

            event_occurred = False
            triggered_event = -1

            for i, ((_, prev_val), (_, curr_val)) in enumerate(zip(prev_events, curr_events)):
                # Check for zero-crossing (sign change from positive to negative)
                if prev_val > 0 and curr_val <= 0:
                    event_occurred = True
                    triggered_event = i
                    break

            if event_occurred:
                # Bisection to find event time
                t_lo, t_hi = t, t_next
                x_lo = x.copy()

                for _ in range(20):  # Max bisection iterations
                    t_mid = (t_lo + t_hi) / 2
                    x_mid = step_func(x_lo, t_lo, t_mid - t_lo)
                    mid_events = check_events(x_mid, t_mid)
                    mid_val = mid_events[triggered_event][1]

                    if mid_val > 0:
                        t_lo = t_mid
                        x_lo = x_mid
                    else:
                        t_hi = t_mid

                    if abs(t_hi - t_lo) < 1e-10:
                        break

                # Use midpoint as event time
                t_event = (t_lo + t_hi) / 2
                x_event = step_func(x_lo, t_lo, t_event - t_lo)

                # Apply reinit
                x_new = apply_reinit(x_event, triggered_event, t_event)

                # Record pre-event state
                t_list.append(t_event)
                x_list.append(x_event.copy())
                if u_list is not None:
                    u_list.append(get_input(t_event))

                # Record post-event state (same time, different state)
                t_list.append(t_event)
                x_list.append(x_new.copy())
                if u_list is not None:
                    u_list.append(get_input(t_event))

                # Continue from post-event state
                x = x_new
                t = t_event
                n_events += 1
                event_times.append(t_event)
                prev_events = check_events(x, t)
            else:
                # No event - accept the step
                t = t_next
                x = x_next
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

        return self._build_result(t_arr, x_arr, u_arr, n_inputs, p, get_input, z_empty)

    def _build_result(
        self,
        t: np.ndarray,
        x_hist: np.ndarray,
        u_hist: Optional[np.ndarray],
        n_inputs: int,
        p: np.ndarray,
        get_input: Callable,
        z: Optional[np.ndarray] = None,
    ) -> SimulationResult:
        """Build SimulationResult from trajectory data."""
        n_steps = len(t)

        # Algebraic variables (empty if not DAE)
        z_vec = z if z is not None else np.zeros(len(self._algebraic_names))

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
                y_hist[i] = np.array(self.f_y(x_hist[i], z_vec, u_vec, p, ti)).flatten()

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
