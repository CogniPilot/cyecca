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

# DSL imports (equation types still in DSL for now)
from cyecca.dsl.equations import Equation, Reinit, WhenClause
from cyecca.dsl.variables import SymbolicVar
from cyecca.ir.causality import SortedSystem

# IR imports
from cyecca.ir.expr import Expr, ExprKind, find_derivatives
from cyecca.ir.flat_model import FlatModel
from cyecca.ir.simulation import SimulationResult, Simulator

# Type for model input - either raw FlatModel or analyzed SortedSystem
ModelInput = Union[FlatModel, SortedSystem]


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
        ExprKind.ASINH: lambda c, e: ca.asinh(c(e.children[0])),
        ExprKind.ACOSH: lambda c, e: ca.acosh(c(e.children[0])),
        ExprKind.ATANH: lambda c, e: ca.atanh(c(e.children[0])),
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
        ExprKind.MOD: lambda c, e: ca.fmod(c(e.children[0]), c(e.children[1])),
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
        self.xdot_syms: Dict[str, SymT] = {}  # State derivative symbols for implicit DAE
        self.input_syms: Dict[str, SymT] = {}
        self.param_syms: Dict[str, SymT] = {}
        self.algebraic_syms: Dict[str, SymT] = {}
        self.discrete_syms: Dict[str, SymT] = {}  # Discrete variables
        self.pre_syms: Dict[str, SymT] = {}  # For when-clauses
        self.t_sym: SymT = None

        # Shape tracking (for indexed access)
        self.state_shapes: Dict[str, Tuple[int, ...]] = {}
        self.param_shapes: Dict[str, Tuple[int, ...]] = {}
        self.input_shapes: Dict[str, Tuple[int, ...]] = {}
        self.algebraic_shapes: Dict[str, Tuple[int, ...]] = {}
        self.discrete_shapes: Dict[str, Tuple[int, ...]] = {}

        # Implicit DAE flag (set during _create_symbols if der() appears in RHS)
        self.is_implicit_dae: bool = False

        # Combined symbol lookup
        self.base_syms: Dict[str, SymT] = {}

        # Output expression cache
        self.compiled_outputs: Dict[str, SymT] = {}

    def _sym(self, name: str, size: int = 1) -> SymT:
        """Create a symbol of the appropriate type."""
        return self.sym_type.sym(name, size) if size != 1 else self.sym_type.sym(name)

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
            # Also create xdot symbols for implicit DAE support
            self.xdot_syms[name] = self._sym(f"der_{name}", size)

        # Input symbols
        for name in model.input_names:
            v = model.input_vars[name]
            shape = v.shape if v.shape else ()
            self.input_shapes[name] = shape
            size = self._shape_to_size(shape)
            self.input_syms[name] = self._sym(name, size)

        # Parameter symbols
        for name in model.param_names:
            v = model.param_vars[name]
            shape = v.shape if v.shape else ()
            self.param_shapes[name] = shape
            size = self._shape_to_size(shape)
            self.param_syms[name] = self._sym(name, size)

        # Algebraic variable symbols
        for name in model.algebraic_names:
            v = model.algebraic_vars[name]
            shape = v.shape if v.shape else ()
            self.algebraic_shapes[name] = shape
            size = self._shape_to_size(shape)
            self.algebraic_syms[name] = self._sym(name, size)

        # Discrete variable symbols
        for name in model.discrete_names:
            v = model.discrete_vars[name]
            shape = v.shape if v.shape else ()
            self.discrete_shapes[name] = shape
            size = self._shape_to_size(shape)
            self.discrete_syms[name] = self._sym(name, size)

        # Time symbol
        self.t_sym = self._sym("t")

        # Run BLT causality analysis to solve equations
        from cyecca.ir.causality import analyze_causality

        self.sorted_system = analyze_causality(self.model)

        # Use BLT result to determine if system is explicit
        self.is_implicit_dae = not self.sorted_system.is_ode_explicit

        # Combined lookup (NOT including outputs - they get substituted)
        self.base_syms = {
            **self.state_syms,
            **self.input_syms,
            **self.param_syms,
            **self.algebraic_syms,
            **self.discrete_syms,
        }

    def _detect_implicit_dae(self) -> bool:
        """Detect if the model requires implicit DAE form.

        Returns True if any equation:
        1. Has der() on RHS (e.g., output == der(x) + der(y))
        2. Has der() on LHS but not in pure form (e.g., m * der(v) == g)
        """
        # find_derivatives is imported at module level from cyecca.ir.expr

        for eq in self.model.equations:
            # Skip output equations
            if eq.lhs.kind == ExprKind.VARIABLE and eq.lhs.name in self.model.output_equations:
                continue

            rhs_derivs = find_derivatives(eq.rhs)
            if rhs_derivs:
                # der() on RHS - implicit
                return True

            lhs_derivs = find_derivatives(eq.lhs)
            if lhs_derivs and not eq.is_derivative:
                # der() on LHS but not pure der(x) == rhs form
                return True

        return False

    def _resolve_indexed_variable(self, name: str) -> SymT:
        """
        Resolve an indexed variable name like 'pos[0]' or 'R[0,1]'.

        Works for both SX and MX backends.
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
        all_shapes = {
            **self.state_shapes,
            **self.param_shapes,
            **self.input_shapes,
            **self.algebraic_shapes,
            **self.discrete_shapes,
        }
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
        - Indexed variable access
        - Derivative nodes (for implicit DAE form)
        """
        # Variable reference
        if expr.kind == ExprKind.VARIABLE:
            name = expr.name
            indices = getattr(expr, "indices", None)

            # Handle indexed access (e.g., x[0,0]) - indices must be non-empty
            if indices:
                if name not in self.base_syms:
                    raise ValueError(f"Unknown variable base: {name}")
                sym = self.base_syms[name]
                all_shapes = {
                    **self.state_shapes,
                    **self.param_shapes,
                    **self.input_shapes,
                    **self.algebraic_shapes,
                    **self.discrete_shapes,
                }
                shape = all_shapes.get(name, ())
                if len(indices) == 1:
                    return sym[indices[0]]
                else:
                    flat_idx = self._compute_flat_index(indices, shape)
                    return sym[flat_idx]

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

            # Indexed access via string name (e.g., 'pos[0]' or 'x[0,0]')
            if "[" in name:
                return self._resolve_indexed_variable(name)

            raise ValueError(f"Unknown variable: {name}")

        # Derivative node - for implicit DAE form
        if expr.kind == ExprKind.DERIVATIVE:
            var_name = expr.name
            if var_name in self.xdot_syms:
                return self.xdot_syms[var_name]
            # Handle indexed derivatives like der(x[0])
            if "[" in var_name:
                base_name = var_name.split("[")[0]
                if base_name in self.xdot_syms:
                    idx_str = var_name[var_name.index("[") + 1 : var_name.rindex("]")]
                    indices = tuple(int(i) for i in idx_str.split(","))
                    shape = self.state_shapes.get(base_name, ())
                    if len(indices) == 1:
                        return self.xdot_syms[base_name][indices[0]]
                    else:
                        flat_idx = self._compute_flat_index(indices, shape)
                        return self.xdot_syms[base_name][flat_idx]
            raise ValueError(f"Unknown derivative variable: {var_name}")

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

        # Array literal - convert to CasADi array
        # Vectors: [1, 2, 3] -> ca.vertcat(1, 2, 3) -> 3x1 column vector
        # Matrices: [[1, 2], [3, 4], [5, 6]] -> 3x2 matrix (each inner list is a row)
        if expr.kind == ExprKind.ARRAY_LITERAL:
            if not expr.children:
                return ca.DM([])  # Empty array

            # Check if this is a matrix (all children are ARRAY_LITERALs)
            first_child = expr.children[0]
            if isinstance(first_child, Expr) and first_child.kind == ExprKind.ARRAY_LITERAL:
                # Matrix: each child is a row, convert to row vectors and stack
                rows = []
                expected_cols = len(first_child.children)
                for i, child in enumerate(expr.children):
                    if not (isinstance(child, Expr) and child.kind == ExprKind.ARRAY_LITERAL):
                        raise ValueError(f"Inconsistent array literal: row {i} is not an array")
                    if len(child.children) != expected_cols:
                        raise ValueError(
                            f"Inconsistent array literal: row {i} has {len(child.children)} "
                            f"elements, expected {expected_cols}"
                        )
                    # Convert row elements and create a row vector using horzcat
                    row_elements = [self.expr_to_casadi(elem) for elem in child.children]
                    rows.append(ca.horzcat(*row_elements))
                # Stack rows vertically to form the matrix
                return ca.vertcat(*rows)
            else:
                # Vector: simple list of scalars -> column vector
                elements = [self.expr_to_casadi(child) for child in expr.children]
                return ca.vertcat(*elements)

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

        # initial() - returns True only at t=0
        # For event detection, we use a zero-crossing: -t goes from negative to zero
        # This triggers the event at t=0 when simulation starts
        if expr.kind == ExprKind.INITIAL:
            # Return -t so at t=0, the value is 0 (event triggers)
            # For t>0, value is negative (no event)
            # Note: This creates an instantaneous event at the start
            return -self.t_sym

        # sample(start, interval) - periodic events
        # For zero-crossing detection, we need an event at t = start + n*interval
        # Use -sin(2*pi*(t-start)/interval) which crosses zero going DOWN at sample times
        if expr.kind == ExprKind.SAMPLE:
            start = self.expr_to_casadi_when(expr.children[0])
            interval = self.expr_to_casadi_when(expr.children[1])

            # -sin crosses zero from positive to negative at t = start + n*interval
            # At t = start: -sin(0) = 0, and just before (from t<start) we return 1
            # At t = start + interval/2: -sin(pi) = 0, but coming from negative
            # At t = start + interval: -sin(2*pi) = 0, coming from positive (event!)
            rel_time = self.t_sym - start
            neg_sin = -ca.sin(2 * ca.pi * rel_time / interval)

            # Before start time, return positive value to prevent events
            # This also sets up the first event at t=start (transition from 1 to 0)
            return ca.if_else(self.t_sym < start, 1.0, neg_sin)

        # terminal() - not fully implemented yet
        if expr.kind == ExprKind.TERMINAL:
            raise NotImplementedError(
                "terminal() is not yet implemented in the CasADi backend. " "It requires end-of-simulation detection."
            )

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

    def expr_to_casadi_reinit(self, expr: Expr) -> SymT:
        """
        Convert Expr for reinit() context - maps variables to pre-event values.

        In a reinit expression like: reinit(x, pre(x) + y)
        - pre(x) explicitly requests the pre-event value
        - y (without pre) should also use the pre-event value per Modelica semantics

        This method maps all state/discrete variables to their pre_ symbols.
        """
        if expr.kind == ExprKind.PRE:
            pre_name = expr.name
            if pre_name in self.pre_syms:
                return self.pre_syms[pre_name]
            if pre_name in self.base_syms:
                return self.base_syms[pre_name]
            raise ValueError(f"Unknown variable in pre(): {pre_name}")

        # Variable reference - use pre_syms for state/discrete, base_syms for params
        if expr.kind == ExprKind.VARIABLE:
            # First check pre_syms (for state/discrete variables)
            if expr.name in self.pre_syms:
                return self.pre_syms[expr.name]
            # Fall back to base_syms for parameters/constants
            if expr.name in self.base_syms:
                return self.base_syms[expr.name]
            raise ValueError(f"Unknown variable: {expr.name}")

        # Constant
        if expr.kind == ExprKind.CONSTANT:
            return self._const(expr.value)

        # Time
        if expr.kind == ExprKind.TIME:
            return self.t_sym

        # Dispatch for other operations (recursively use reinit converter)
        handler = _EXPR_HANDLERS.get(expr.kind)
        if handler:
            return handler(self.expr_to_casadi_reinit, expr)

        # Fallback to regular conversion
        return self.expr_to_casadi(expr)

    def _build_state_derivatives(self) -> List[SymT]:
        """Build the state derivative vector for explicit ODE form.

        Uses BLT-solved equations from causality analysis. Each state must have
        exactly one solved equation for der(state).
        """
        model = self.model
        state_derivs: List[SymT] = []

        # Build lookup from var_name to solved expression for derivatives
        # Use BLT-solved equations which handle cases like C*der(v) == i
        deriv_rhs_map: Dict[str, "Expr"] = {}
        for solved in self.sorted_system.solved:
            if solved.is_derivative:
                deriv_rhs_map[solved.var_name] = solved.expr

        # Also check original equations for pure der(x) == rhs form
        for eq in model.equations:
            if eq.is_derivative and eq.var_name:
                if eq.var_name not in deriv_rhs_map:
                    deriv_rhs_map[eq.var_name] = eq.rhs

        for name in model.state_names:
            shape = self.state_shapes.get(name, ())
            size = self._shape_to_size(shape)

            # Check for array equation (MX backend)
            if self.is_mx and name in model.array_equations:
                arr_eq = model.array_equations[name]
                rhs = arr_eq["rhs"]
                # RHS is a SymbolicVar - get its symbol
                if rhs.base_name in self.base_syms:
                    state_derivs.append(self.base_syms[rhs.base_name])
                else:
                    state_derivs.append(self._zeros(size))
            elif name in deriv_rhs_map:
                # Scalar/element equation: der(x) == rhs
                state_derivs.append(self.expr_to_casadi(deriv_rhs_map[name]))
            else:
                # Check for element-wise equations
                deriv_vec_parts: List[SymT] = []
                for key, deriv_expr in deriv_rhs_map.items():
                    if key.startswith(f"{name}["):
                        idx_str = key[key.index("[") + 1 : key.rindex("]")]
                        indices = tuple(int(i) for i in idx_str.split(","))
                        flat_idx = indices[0] if len(indices) == 1 else self._compute_flat_index(indices, shape)
                        # Extend list to accommodate index
                        while len(deriv_vec_parts) <= flat_idx:
                            deriv_vec_parts.append(None)
                        deriv_vec_parts[flat_idx] = self.expr_to_casadi(deriv_expr)

                if deriv_vec_parts:
                    # Fill any None entries with zeros
                    for i in range(len(deriv_vec_parts)):
                        if deriv_vec_parts[i] is None:
                            deriv_vec_parts[i] = self._const(0.0)
                    # Concatenate for both SX and MX
                    state_derivs.append(ca.vertcat(*deriv_vec_parts))
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
        """Build algebraic equation residuals (0 = lhs - rhs).

        Uses BLT-solved equations for algebraic variables. Only includes
        equations that were successfully matched and solved.
        """
        residuals: List[SymT] = []

        # Use BLT-solved algebraic equations (those that aren't derivatives)
        for solved in self.sorted_system.solved:
            if not solved.is_derivative:
                # Build residual: var - expr = 0
                if solved.var_name in self.algebraic_syms:
                    var_sym = self.algebraic_syms[solved.var_name]
                    expr_sym = self.expr_to_casadi(solved.expr)
                    residuals.append(var_sym - expr_sym)

        return residuals

    def _build_differential_residuals(self) -> List[SymT]:
        """
        Build differential equation residuals for implicit DAE form.

        For implicit form, equations are: 0 = f(xdot, x, z, u, p, t)

        All differential equations are converted to residual form:
        - Explicit eq "der(x) == rhs" → residual: xdot - rhs
        - Implicit eq "lhs == rhs" → residual: lhs - rhs (with xdot symbols for der())

        The expr_to_casadi() method converts der(x) nodes to xdot_syms[x].
        """
        # find_derivatives is imported at module level from cyecca.ir.expr
        model = self.model
        residuals: List[SymT] = []

        # Filter equations that involve derivatives (differential equations)
        for eq in model.equations:
            # Check if equation involves any derivatives
            lhs_derivs = find_derivatives(eq.lhs)
            rhs_derivs = find_derivatives(eq.rhs)
            if lhs_derivs or rhs_derivs:
                # This is a differential equation
                # Convert both sides (der() nodes become xdot symbols)
                lhs = self.expr_to_casadi(eq.lhs)
                rhs = self.expr_to_casadi(eq.rhs)
                residuals.append(lhs - rhs)

        return residuals

    def _build_when_clause_funcs(self, x: SymT, d: SymT, u: SymT, p: SymT) -> List[Dict[str, Any]]:
        """Build event and reinit functions for when-clauses.

        Parameters
        ----------
        x : SymT
            Continuous state vector
        d : SymT
            Discrete state vector
        u : SymT
            Input vector
        p : SymT
            Parameter vector

        Returns
        -------
        List[Dict[str, Any]]
            List of when-clause function dictionaries with:
            - f_event: Event detection function (zero-crossing)
            - f_reinit: State/discrete update function at event
        """
        model = self.model
        when_clause_funcs: List[Dict[str, Any]] = []

        if not model.when_clauses:
            return when_clause_funcs

        # Create pre-event symbols for states
        for name in model.state_names:
            v = model.state_vars[name]
            size = v.size if v.size > 1 else 1
            self.pre_syms[name] = self._sym(f"pre_{name}", size)

        # Create pre-event symbols for discrete variables
        for name in model.discrete_names:
            v = model.discrete_vars[name]
            size = v.size if v.size > 1 else 1
            self.pre_syms[name] = self._sym(f"pre_{name}", size)

        # Build combined pre-vectors
        state_pre_list = [self.pre_syms[n] for n in model.state_names]
        discrete_pre_list = [self.pre_syms[n] for n in model.discrete_names]

        x_pre = ca.vertcat(*state_pre_list) if state_pre_list else self._sym("x_pre", 0)
        d_pre = ca.vertcat(*discrete_pre_list) if discrete_pre_list else self._sym("d_pre", 0)

        for i, wc in enumerate(model.when_clauses):
            # Event function: returns value that crosses zero
            # Events can depend on both x and d
            event_expr = self.expr_to_casadi_when(wc.condition)
            f_event = ca.Function(
                f"f_event_{i}",
                [x, d, u, p, self.t_sym],
                [event_expr],
                ["x", "d", "u", "p", "t"],
                ["event"],
            )

            # Reinit function: computes new state AND discrete values after event
            # Start with pre-values (identity mapping)
            x_new_list = [self.pre_syms[n] for n in model.state_names]
            d_new_list = [self.pre_syms[n] for n in model.discrete_names]

            for reinit in wc.reinits:
                var_name = reinit.var_name
                # Use expr_to_casadi_reinit which maps vars to pre-event values
                new_val = self.expr_to_casadi_reinit(reinit.expr)

                if var_name in model.state_names:
                    idx = model.state_names.index(var_name)
                    x_new_list[idx] = new_val
                elif var_name in model.discrete_names:
                    idx = model.discrete_names.index(var_name)
                    d_new_list[idx] = new_val

            x_new = ca.vertcat(*x_new_list) if x_new_list else self._sym("x_new", 0)
            # For d_new, if there are no discrete vars, use d_pre (same empty symbol)
            d_new = ca.vertcat(*d_new_list) if d_new_list else d_pre

            f_reinit = ca.Function(
                f"f_reinit_{i}",
                [x_pre, d_pre, u, p, self.t_sym],
                [x_new, d_new],
                ["x_pre", "d_pre", "u", "p", "t"],
                ["x_new", "d_new"],
            )

            when_clause_funcs.append(
                {
                    "condition": wc.condition,
                    "f_event": f_event,
                    "f_reinit": f_reinit,
                }
            )

        return when_clause_funcs

    def _build_initial_solver(self, x: SymT, z: SymT, u: SymT, p: SymT) -> Optional[ca.Function]:
        """
        Build a rootfinder to solve initial equations.

        Initial equations of the form `var == expr` are converted to
        residuals `var - expr = 0` and solved using CasADi's rootfinder.

        Parameters
        ----------
        x : SymT
            State vector symbol
        z : SymT
            Algebraic variable vector symbol
        u : SymT
            Input vector symbol
        p : SymT
            Parameter vector symbol

        Returns
        -------
        Optional[ca.Function]
            A rootfinder function f_init(x0_guess, z, u, p) -> x0_solved,
            or None if there are no initial equations.
        """
        model = self.model

        if not model.initial_equations:
            return None

        # Build residuals for initial equations
        # Each equation lhs == rhs becomes residual = lhs - rhs
        residuals = []
        for eq in model.initial_equations:
            lhs_expr = self.expr_to_casadi(eq.lhs)
            rhs_expr = self.expr_to_casadi(eq.rhs)
            residual = lhs_expr - rhs_expr
            # Handle vector residuals (from array literals)
            if hasattr(residual, "shape") and residual.shape[0] > 1:
                for i in range(residual.shape[0]):
                    residuals.append(residual[i])
            else:
                residuals.append(residual)

        if not residuals:
            return None

        # Create residual vector
        res = ca.vertcat(*residuals)

        # Create residual function: g(x, z, u, p) = 0
        # x is what we're solving for
        g = ca.Function(
            "g_init",
            [x, z, u, p],
            [res],
            ["x", "z", "u", "p"],
            ["residual"],
        )

        # Create rootfinder
        # Uses Newton method to find x such that g(x, z, u, p) = 0
        try:
            solver = ca.rootfinder("init_solver", "newton", g)
            return solver
        except Exception:
            # If Newton fails, try nlpsol-based approach
            return None

    def compile(self) -> "CompiledModel":
        """
        Compile the model to CasADi functions.

        Supports two forms:
        - Explicit ODE: der(x) = f(x, z, u, p, t)
        - Implicit DAE: 0 = f(xdot, x, z, u, p, t)

        The implicit form is automatically detected when der(x) appears on
        the RHS of equations. IDAS can use either form, but the implicit
        form is more general and allows for index-1 DAEs where the derivatives
        cannot be explicitly isolated.

        Returns
        -------
        CompiledModel
            Compiled model ready for simulation
        """
        model = self.model

        # Create all symbols (also detects implicit DAE)
        self._create_symbols()

        # Build output vector
        y_exprs = self._build_outputs()

        # Build algebraic residuals
        alg_residuals = self._build_algebraic_residuals()

        # Build CasADi vectors
        state_list = [self.state_syms[n] for n in model.state_names]
        xdot_list = [self.xdot_syms[n] for n in model.state_names]
        input_list = [self.input_syms[n] for n in model.input_names]
        param_list = [self.param_syms[n] for n in model.param_names]
        algebraic_list = [self.algebraic_syms[n] for n in model.algebraic_names]
        discrete_list = [self.discrete_syms[n] for n in model.discrete_names]

        x = ca.vertcat(*state_list) if state_list else self._sym("x", 0)
        xdot_sym = ca.vertcat(*xdot_list) if xdot_list else self._sym("xdot", 0)
        u = ca.vertcat(*input_list) if input_list else self._sym("u", 0)
        p = ca.vertcat(*param_list) if param_list else self._sym("p", 0)
        y = ca.vertcat(*y_exprs) if y_exprs else self._sym("y", 0)
        z = ca.vertcat(*algebraic_list) if algebraic_list else self._sym("z", 0)
        alg = ca.vertcat(*alg_residuals) if alg_residuals else self._sym("alg", 0)
        d = ca.vertcat(*discrete_list) if discrete_list else self._sym("d", 0)

        # Build differential equation representation
        if self.is_implicit_dae:
            # Implicit DAE form: 0 = f(xdot, x, z, u, p, t)
            diff_residuals = self._build_differential_residuals()
            diff_res = ca.vertcat(*diff_residuals) if diff_residuals else self._sym("diff_res", 0)

            # Create residual function f_res(xdot, x, z, u, p, t) -> residual
            f_res = ca.Function(
                "f_res",
                [xdot_sym, x, z, u, p, self.t_sym],
                [diff_res],
                ["xdot", "x", "z", "u", "p", "t"],
                ["residual"],
            )
            f_x = None  # No explicit form for implicit DAE
        else:
            # Explicit ODE form: xdot = f(x, z, d, u, p, t)
            # Includes d because discrete variables may affect ODEs
            state_derivs = self._build_state_derivatives()
            xdot_expr = ca.vertcat(*state_derivs) if state_derivs else self._sym("xdot", 0)

            # Create dynamics function f_x(x, z, d, u, p, t) -> xdot
            f_x = ca.Function("f_x", [x, z, d, u, p, self.t_sym], [xdot_expr], ["x", "z", "d", "u", "p", "t"], ["xdot"])
            f_res = None  # No residual form for explicit ODE

        # Create algebraic residual function (for DAE)
        f_alg = (
            ca.Function("f_alg", [x, z, u, p, self.t_sym], [alg], ["x", "z", "u", "p", "t"], ["alg"])
            if alg_residuals
            else None
        )

        # Create output function (now includes d for discrete-dependent outputs)
        f_y = (
            ca.Function("f_y", [x, z, d, u, p, self.t_sym], [y], ["x", "z", "d", "u", "p", "t"], ["y"])
            if y_exprs
            else None
        )

        # Build when-clause functions (handle both state and discrete reinit)
        when_clause_funcs = self._build_when_clause_funcs(x, d, u, p)

        # Build initial equation solver
        f_init = self._build_initial_solver(x, z, u, p)

        # Build algebraic defaults
        algebraic_defaults: Dict[str, Any] = {}
        for name in model.algebraic_names:
            v = model.algebraic_vars[name]
            algebraic_defaults[name] = v.start if v.start is not None else 0.0

        # Build state shapes from model
        state_shapes: Dict[str, Tuple[int, ...]] = {}
        for name in model.state_names:
            if name in model.state_vars:
                v = model.state_vars[name]
                if v.shape:
                    state_shapes[name] = v.shape

        # Build input shapes from model
        input_shapes: Dict[str, Tuple[int, ...]] = {}
        for name in model.input_names:
            if name in model.input_vars:
                v = model.input_vars[name]
                if v.shape:
                    input_shapes[name] = v.shape

        # Build discrete shapes from model
        discrete_shapes: Dict[str, Tuple[int, ...]] = {}
        for name in model.discrete_names:
            if name in model.discrete_vars:
                v = model.discrete_vars[name]
                if v.shape:
                    discrete_shapes[name] = v.shape

        return CompiledModel(
            name=model.name,
            f_x=f_x,
            f_res=f_res,
            f_y=f_y,
            is_implicit=self.is_implicit_dae,
            _state_names=model.state_names,
            _input_names=model.input_names,
            _param_names=model.param_names,
            _output_names=model.output_names,
            _discrete_names=model.discrete_names,
            state_defaults=model.state_defaults,
            input_defaults=model.input_defaults,
            param_defaults=model.param_defaults,
            discrete_defaults=model.discrete_defaults,
            when_clause_funcs=when_clause_funcs,
            _algebraic_names=model.algebraic_names,
            algebraic_defaults=algebraic_defaults,
            f_alg=f_alg,
            f_init=f_init,
            state_shapes=state_shapes,
            input_shapes=input_shapes,
            discrete_shapes=discrete_shapes,
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

    Discrete Variables
    ------------------
    Discrete variables are piecewise-constant: they only change at events
    via reinit() statements in when-clauses. They are tracked separately
    from continuous states and included in simulation results.

    DAE Support (IDAS)
    ------------------
    For models with algebraic variables, use integrator=Integrator.IDAS.

    Explicit form (default):
        ẋ = f(x, z, u, p, t)    (differential equations)
        0 = g(x, z, u, p, t)    (algebraic equations)

    Implicit form (when der() appears on RHS):
        0 = F(ẋ, x, z, u, p, t)  (implicit differential-algebraic equations)

    The implicit form is required for index-1 DAEs where the derivatives
    cannot be explicitly isolated (e.g., mass matrix systems: M(x)*ẋ = f(x)).
    """

    name: str
    f_x: Optional[ca.Function] = None  # Explicit dynamics: f_x(x, z, d, u, p, t) -> xdot
    f_res: Optional[ca.Function] = None  # Implicit residual: f_res(xdot, x, z, u, p, t) -> 0
    f_y: Optional[ca.Function] = None  # Outputs: f_y(x, z, d, u, p, t) -> y
    f_init: Optional[ca.Function] = None  # Initial equation solver
    is_implicit: bool = False  # True if model uses implicit DAE form
    _state_names: List[str] = None
    _input_names: List[str] = None
    _param_names: List[str] = None
    _output_names: List[str] = None
    _discrete_names: List[str] = None
    state_defaults: Dict[str, Any] = None
    input_defaults: Dict[str, Any] = None
    param_defaults: Dict[str, Any] = None
    discrete_defaults: Dict[str, Any] = None
    when_clause_funcs: List[Dict[str, Any]] = None
    _algebraic_names: List[str] = None
    algebraic_defaults: Dict[str, Any] = None
    f_alg: Optional[ca.Function] = None
    # Shape tracking for array variables
    state_shapes: Dict[str, Tuple[int, ...]] = None
    input_shapes: Dict[str, Tuple[int, ...]] = None
    discrete_shapes: Dict[str, Tuple[int, ...]] = None

    def __post_init__(self):
        if self.when_clause_funcs is None:
            self.when_clause_funcs = []
        if self._algebraic_names is None:
            self._algebraic_names = []
        if self.algebraic_defaults is None:
            self.algebraic_defaults = {}
        if self._discrete_names is None:
            self._discrete_names = []
        if self.discrete_defaults is None:
            self.discrete_defaults = {}
        if self.state_defaults is None:
            self.state_defaults = {}
        if self.input_defaults is None:
            self.input_defaults = {}
        if self.param_defaults is None:
            self.param_defaults = {}
        if self.state_shapes is None:
            self.state_shapes = {}
        if self.input_shapes is None:
            self.input_shapes = {}
        if self.discrete_shapes is None:
            self.discrete_shapes = {}

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
    def discrete_names(self) -> List[str]:
        return self._discrete_names

    @property
    def has_algebraic(self) -> bool:
        """Check if model has algebraic variables (DAE system)."""
        return len(self._algebraic_names) > 0

    @property
    def is_dae(self) -> bool:
        """Check if model is a DAE (has algebraic constraints)."""
        return self.has_algebraic

    @property
    def has_discrete(self) -> bool:
        """Check if model has discrete variables."""
        return len(self._discrete_names) > 0

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
        # Get actual vector sizes from the CasADi functions (which account for arrays)
        # f_x has inputs: x, z, d, u, p, t (explicit)
        # f_res has inputs: xdot, x, z, u, p, t (implicit)
        if self.f_x is not None:
            n_states = self.f_x.size_in(0)[0]  # x
            n_alg = self.f_x.size_in(1)[0]  # z
            n_discrete = self.f_x.size_in(2)[0]  # d
            n_inputs = self.f_x.size_in(3)[0]  # u
            n_params = self.f_x.size_in(4)[0]  # p
        elif self.f_res is not None:
            n_states = self.f_res.size_in(1)[0]  # x (xdot is at 0)
            n_alg = self.f_res.size_in(2)[0]  # z
            n_discrete = 0  # f_res doesn't have d in signature
            n_inputs = self.f_res.size_in(3)[0]  # u
            n_params = self.f_res.size_in(4)[0]  # p
        else:
            # Fallback: compute sizes by summing element counts from defaults
            def compute_size(names, defaults):
                total = 0
                for name in names:
                    if name in defaults:
                        val = defaults[name]
                        if isinstance(val, (list, np.ndarray)):
                            total += len(np.array(val).flatten())
                        else:
                            total += 1
                    else:
                        total += 1  # Scalar default
                return total

            n_states = compute_size(self._state_names, self.state_defaults)
            n_params = compute_size(self._param_names, self.param_defaults)
            n_inputs = compute_size(self._input_names, self.input_defaults)
            n_alg = len(self._algebraic_names)
            n_discrete = len(self._discrete_names)

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
        p = np.zeros(n_params)
        idx = 0
        for name in self._param_names:
            if params and name in params:
                val = params[name]
            elif name in self.param_defaults:
                val = self.param_defaults[name]
            else:
                val = 0.0
            if isinstance(val, (list, np.ndarray)):
                for v in np.array(val).flatten():
                    p[idx] = v
                    idx += 1
            else:
                p[idx] = val
                idx += 1

        # Build algebraic variable vector (for DAE systems)
        z_init = np.zeros(n_alg)
        idx = 0
        for name in self._algebraic_names:
            if name in self.algebraic_defaults:
                val = self.algebraic_defaults[name]
            else:
                val = 0.0
            if isinstance(val, (list, np.ndarray)):
                for v in np.array(val).flatten():
                    z_init[idx] = v
                    idx += 1
            else:
                z_init[idx] = val
                idx += 1

        # Build initial discrete variable vector
        d_init = np.zeros(n_discrete)
        idx = 0
        for name in self._discrete_names:
            if name in self.discrete_defaults:
                val = self.discrete_defaults[name]
            else:
                val = 0.0
            if isinstance(val, (list, np.ndarray)):
                for v in np.array(val).flatten():
                    d_init[idx] = v
                    idx += 1
            else:
                d_init[idx] = val
                idx += 1

        # Build constant input vector
        u_const = np.zeros(n_inputs)
        idx = 0
        for name in self._input_names:
            if u and name in u:
                val = u[name]
            elif name in self.input_defaults:
                val = self.input_defaults[name]
            else:
                val = 0.0
            if isinstance(val, (list, np.ndarray)):
                for v in np.array(val).flatten():
                    u_const[idx] = v
                    idx += 1
            else:
                u_const[idx] = val
                idx += 1

        # Solve initial equations if present
        # This uses CasADi rootfinder to solve equations like y == [3.0, 4.0, 5.0]
        if self.f_init is not None:
            try:
                # f_init(x0_guess, z, u, p) -> x0_solved
                x_solved = self.f_init(x_init, z_init, u_const, p)
                x_init = np.array(x_solved).flatten()
            except Exception as e:
                import warnings

                warnings.warn(f"Initial equation solver failed: {e}. Using default initial values.")

        def get_input(ti: float) -> np.ndarray:
            """Get input vector at time ti."""
            if u_func is not None:
                u_dict = u_func(ti)
                u_vec = np.zeros(n_inputs)
                idx = 0
                for name in self._input_names:
                    if name in u_dict:
                        val = u_dict[name]
                    else:
                        # Determine size from defaults or assume scalar
                        if name in self.input_defaults:
                            default_val = self.input_defaults[name]
                            if isinstance(default_val, (list, np.ndarray)):
                                val = np.zeros(len(np.array(default_val).flatten()))
                            else:
                                val = u_const[idx]
                        else:
                            val = u_const[idx]
                    if isinstance(val, (list, np.ndarray)):
                        for v in np.array(val).flatten():
                            u_vec[idx] = v
                            idx += 1
                    else:
                        u_vec[idx] = val
                        idx += 1
                return u_vec
            return u_const

        # For pure ODE systems, z is empty
        z_empty = np.zeros(0) if n_alg == 0 else z_init

        def rk4_step(x: np.ndarray, d: np.ndarray, ti: float, h: float) -> np.ndarray:
            """Single RK4 integration step (ODE only, no algebraic)."""
            u_vec = get_input(ti)
            k1 = np.array(self.f_x(x, z_empty, d, u_vec, p, ti)).flatten()
            k2 = np.array(self.f_x(x + 0.5 * h * k1, z_empty, d, u_vec, p, ti + 0.5 * h)).flatten()
            k3 = np.array(self.f_x(x + 0.5 * h * k2, z_empty, d, u_vec, p, ti + 0.5 * h)).flatten()
            k4 = np.array(self.f_x(x + h * k3, z_empty, d, u_vec, p, ti + h)).flatten()
            return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        def check_events(x: np.ndarray, d: np.ndarray, ti: float) -> List[Tuple[int, float]]:
            """Check event functions and return list of (event_idx, value)."""
            u_vec = get_input(ti)
            events = []
            for i, wc_func in enumerate(self.when_clause_funcs):
                val = float(np.array(wc_func["f_event"](x, d, u_vec, p, ti)).flatten()[0])
                events.append((i, val))
            return events

        def apply_reinit(x: np.ndarray, d: np.ndarray, event_idx: int, ti: float) -> Tuple[np.ndarray, np.ndarray]:
            """Apply reinit for the given event. Returns (x_new, d_new)."""
            u_vec = get_input(ti)
            wc_func = self.when_clause_funcs[event_idx]
            result = wc_func["f_reinit"](x, d, u_vec, p, ti)
            x_new = np.array(result[0]).flatten()
            d_new = np.array(result[1]).flatten() if n_discrete > 0 else d.copy()
            return x_new, d_new

        # RK4 doesn't support algebraic variables - it would silently use zeros
        if integrator == Integrator.RK4:
            if self.has_algebraic:
                raise ValueError(
                    "RK4 does not support DAE systems with algebraic variables. "
                    "Use integrator=Integrator.IDAS for DAE systems."
                )

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
            d_sym = ca.SX.sym("d", max(n_discrete, 1))  # Discrete vars
            u_sym = ca.SX.sym("u", max(n_inputs, 1))
            p_sym = ca.SX.sym("p", n_params)
            t_sym = ca.SX.sym("t")

            # Build the ODE right-hand side (includes discrete vars)
            z_in = z_sym[:n_alg] if n_alg > 0 else ca.SX()
            d_in = d_sym[:n_discrete] if n_discrete > 0 else ca.SX()
            u_in = u_sym[:n_inputs] if n_inputs > 0 else ca.SX()
            xdot = self.f_x(x_sym, z_in, d_in, u_in, p_sym, t_sym)

            # Store DAE definition for use in step function (d is passed as part of p)
            cvodes_dae = {"x": x_sym, "t": t_sym, "p": ca.vertcat(d_sym, u_sym, p_sym), "ode": xdot}
            cvodes_opts = {"abstol": 1e-8, "reltol": 1e-6}

        # For IDAS (DAE), create integrator that handles both differential and algebraic
        idas_dae = None
        idas_opts = None
        if integrator == Integrator.IDAS:
            x_sym = ca.SX.sym("x", n_states)
            z_sym = ca.SX.sym("z", max(n_alg, 1))
            d_sym = ca.SX.sym("d", max(n_discrete, 1))  # Discrete vars
            u_sym = ca.SX.sym("u", max(n_inputs, 1))
            p_sym = ca.SX.sym("p", n_params)
            t_sym = ca.SX.sym("t")

            z_in = z_sym[:n_alg] if n_alg > 0 else ca.SX()
            d_in = d_sym[:n_discrete] if n_discrete > 0 else ca.SX()
            u_in = u_sym[:n_inputs] if n_inputs > 0 else ca.SX()

            if self.is_implicit:
                # Implicit DAE form: use f_res to build the residual
                # f_res signature: f_res(xdot, x, z, u, p, t) -> residual
                xdot_sym = ca.SX.sym("xdot", n_states)
                diff_res = self.f_res(xdot_sym, x_sym, z_in, u_in, p_sym, t_sym)

                # IDAS can also handle implicit form, but it needs the ODE
                # For now, we need to have explicit ODE form for IDAS
                # Let's raise an error for implicit forms for now
                raise ValueError(
                    "Implicit DAE form is not yet fully supported by the IDAS integrator. "
                    "Try reformulating your model to have explicit differential equations "
                    "(der(x) == expr form, not implicit in der(x))."
                )
            else:
                # Explicit ODE form: ẋ = f(x, z, d, u, p, t)
                xdot = self.f_x(x_sym, z_in, d_in, u_in, p_sym, t_sym)

            # Algebraic equations: 0 = g(x, z, u, p, t)
            if self.f_alg is not None:
                alg = self.f_alg(x_sym, z_in, u_in, p_sym, t_sym)
            else:
                alg = ca.SX()

            # IDAS DAE formulation (d is passed as part of p)
            idas_dae = {
                "x": x_sym,
                "z": z_in,
                "t": t_sym,
                "p": ca.vertcat(d_sym, u_sym, p_sym),
                "ode": xdot,
                "alg": alg,
            }
            idas_opts = {"abstol": 1e-8, "reltol": 1e-6}

        def cvodes_step(x: np.ndarray, d: np.ndarray, ti: float, h: float) -> np.ndarray:
            """Single CVODES integration step using CasADi integrator."""
            u_vec = get_input(ti)
            # Combined parameters: [d, u, p]
            d_vec = d if n_discrete > 0 else np.array([0.0])
            u_part = u_vec if n_inputs > 0 else np.array([0.0])
            p_combined = np.concatenate([d_vec, u_part, p])
            # Create integrator with t0 and tf as positional args (new API)
            integ = ca.integrator("integ", "cvodes", cvodes_dae, ti, ti + h, cvodes_opts)
            result = integ(x0=x, p=p_combined)
            return np.array(result["xf"]).flatten()

        def idas_step(x: np.ndarray, d: np.ndarray, ti: float, h: float) -> np.ndarray:
            """Single IDAS integration step for DAE systems."""
            u_vec = get_input(ti)
            # Combined parameters: [d, u, p]
            d_vec = d if n_discrete > 0 else np.array([0.0])
            u_part = u_vec if n_inputs > 0 else np.array([0.0])
            p_combined = np.concatenate([d_vec, u_part, p])
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
                d_init,
                p,
                n_inputs,
                n_states,
                n_discrete,
                get_input,
                step_func,
                check_events,
                apply_reinit,
                max_events,
                integrator,
                z_empty,
            )

        # Integration without events (discrete remains constant)
        t = np.arange(t0, tf + dt, dt)
        n_steps = len(t)

        # Storage
        x_hist = np.zeros((n_steps, n_states))
        x_hist[0] = x_init
        d_hist = np.zeros((n_steps, n_discrete)) if n_discrete > 0 else None
        if d_hist is not None:
            d_hist[:] = d_init  # Discrete stays constant without events
        u_hist = np.zeros((n_steps, n_inputs)) if n_inputs > 0 else None

        # Integration loop
        x = x_init.copy()
        d = d_init.copy()
        for i in range(n_steps):
            ti = t[i]
            u_vec = get_input(ti)

            # Record input
            if u_hist is not None:
                u_hist[i] = u_vec

            # Step (except for last point)
            if i < n_steps - 1:
                x = step_func(x, d, ti, dt)
                x_hist[i + 1] = x

        return self._build_result(t, x_hist, d_hist, u_hist, n_inputs, n_discrete, p, get_input, z_empty, d_init)

    def _simulate_with_events(
        self,
        t0: float,
        tf: float,
        dt: float,
        x_init: np.ndarray,
        d_init: np.ndarray,
        p: np.ndarray,
        n_inputs: int,
        n_states: int,
        n_discrete: int,
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

        Discrete variables are updated only at events via reinit().
        """
        # Use variable-length lists for event-triggered simulation
        t_list: List[float] = [t0]
        x_list: List[np.ndarray] = [x_init.copy()]
        d_list: List[np.ndarray] = [d_init.copy()] if n_discrete > 0 else None
        u_list: List[np.ndarray] = [get_input(t0)] if n_inputs > 0 else None

        x = x_init.copy()
        d = d_init.copy()
        t = t0
        n_events = 0
        event_times: List[float] = []

        # Check for initial() events at t=t0
        # initial() events should fire at the very start of simulation
        for i, wc_func in enumerate(self.when_clause_funcs):
            condition = wc_func.get("condition")
            # Check if this when-clause uses initial()
            if condition is not None and condition.kind == ExprKind.INITIAL:
                # Apply the reinit at t0
                x, d = apply_reinit(x, d, i, t0)
                n_events += 1
                event_times.append(t0)
                # Update stored values
                x_list[0] = x.copy()
                if d_list is not None:
                    d_list[0] = d.copy()

        # Check for sample() events at t=start (if start == t0)
        # sample(start, interval) fires at t=start, start+interval, etc.
        for i, wc_func in enumerate(self.when_clause_funcs):
            condition = wc_func.get("condition")
            # Check if this when-clause uses sample()
            if condition is not None and condition.kind == ExprKind.SAMPLE:
                # Extract start time from sample(start, interval)
                start_expr = condition.children[0]
                if start_expr.kind == ExprKind.CONSTANT:
                    start_time = start_expr.value
                    # If start time equals t0, fire the event
                    if abs(start_time - t0) < 1e-10:
                        x, d = apply_reinit(x, d, i, t0)
                        n_events += 1
                        event_times.append(t0)
                        # Update stored values
                        x_list[0] = x.copy()
                        if d_list is not None:
                            d_list[0] = d.copy()

        # Track previous event values for edge detection
        prev_events = check_events(x, d, t)

        while t < tf and n_events < max_events:
            # Tentative step (discrete d stays constant between events)
            h = min(dt, tf - t)
            x_next = step_func(x, d, t, h)
            t_next = t + h

            # Check for events (sign changes in event functions)
            curr_events = check_events(x_next, d, t_next)

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
                    x_mid = step_func(x_lo, d, t_lo, t_mid - t_lo)
                    mid_events = check_events(x_mid, d, t_mid)
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
                x_event = step_func(x_lo, d, t_lo, t_event - t_lo)

                # Apply reinit (updates both x and d)
                x_new, d_new = apply_reinit(x_event, d, triggered_event, t_event)

                # Check for other events that fire at the same time
                # This handles multi-rate sampling where multiple events coincide
                simultaneous_events = [triggered_event]
                for j, ((_, prev_j), (_, curr_j)) in enumerate(zip(prev_events, curr_events)):
                    if j != triggered_event and prev_j > 0 and curr_j <= 0:
                        # This event also triggers - check if it's at the same time
                        # (within tolerance)
                        # Apply its reinit too
                        x_new, d_new = apply_reinit(x_new, d_new, j, t_event)
                        simultaneous_events.append(j)
                        n_events += 1

                # Record pre-event state
                t_list.append(t_event)
                x_list.append(x_event.copy())
                if d_list is not None:
                    d_list.append(d.copy())
                if u_list is not None:
                    u_list.append(get_input(t_event))

                # Record post-event state (same time, different state)
                t_list.append(t_event)
                x_list.append(x_new.copy())
                if d_list is not None:
                    d_list.append(d_new.copy())
                if u_list is not None:
                    u_list.append(get_input(t_event))

                # Continue from post-event state
                x = x_new
                d = d_new
                t = t_event
                n_events += 1
                event_times.append(t_event)
                prev_events = check_events(x, d, t)
            else:
                # No event - accept the step
                t = t_next
                x = x_next
                t_list.append(t)
                x_list.append(x.copy())
                if d_list is not None:
                    d_list.append(d.copy())
                if u_list is not None:
                    u_list.append(get_input(t))
                prev_events = curr_events

        if n_events >= max_events:
            import warnings

            warnings.warn(f"Maximum number of events ({max_events}) reached. Simulation may be incomplete.")

        # Convert lists to arrays
        t_arr = np.array(t_list)
        x_arr = np.array(x_list)
        d_arr = np.array(d_list) if d_list is not None else None
        u_arr = np.array(u_list) if u_list is not None else None

        return self._build_result(t_arr, x_arr, d_arr, u_arr, n_inputs, n_discrete, p, get_input, z_empty, d_init)

    def _build_result(
        self,
        t: np.ndarray,
        x_hist: np.ndarray,
        d_hist: Optional[np.ndarray],
        u_hist: Optional[np.ndarray],
        n_inputs: int,
        n_discrete: int,
        p: np.ndarray,
        get_input: Callable,
        z: Optional[np.ndarray] = None,
        d_current: Optional[np.ndarray] = None,
    ) -> SimulationResult:
        """Build SimulationResult from trajectory data."""
        n_steps = len(t)

        # Algebraic variables (empty if not DAE)
        z_vec = z if z is not None else np.zeros(len(self._algebraic_names))

        # Helper to compute size from shape or defaults
        def get_var_size(name: str, shapes: Dict[str, Tuple[int, ...]], defaults: Dict[str, Any]) -> int:
            """Get the flattened size of a variable from shapes or defaults."""
            if name in shapes:
                shape = shapes[name]
                size = 1
                for dim in shape:
                    size *= dim
                return size
            if name in defaults:
                default_val = defaults[name]
                if isinstance(default_val, (list, np.ndarray)):
                    return len(np.array(default_val).flatten())
            return 1

        # Convert states to named dict (handle array states)
        states: Dict[str, np.ndarray] = {}
        idx = 0
        for name in self._state_names:
            size = get_var_size(name, self.state_shapes, self.state_defaults)

            if size == 1:
                states[name] = x_hist[:, idx]
            else:
                states[name] = x_hist[:, idx : idx + size]
            idx += size

        # Convert discrete variables to named dict (handle arrays)
        discrete: Dict[str, np.ndarray] = {}
        if d_hist is not None and n_discrete > 0:
            idx = 0
            for name in self._discrete_names:
                size = get_var_size(name, self.discrete_shapes, self.discrete_defaults)
                if size == 1:
                    discrete[name] = d_hist[:, idx]
                else:
                    discrete[name] = d_hist[:, idx : idx + size]
                idx += size

        # Convert inputs to named dict (handle arrays)
        inputs: Dict[str, np.ndarray] = {}
        if u_hist is not None:
            idx = 0
            for name in self._input_names:
                size = get_var_size(name, self.input_shapes, self.input_defaults)
                if size == 1:
                    inputs[name] = u_hist[:, idx]
                else:
                    inputs[name] = u_hist[:, idx : idx + size]
                idx += size

        # Compute outputs if f_y is available
        outputs: Dict[str, np.ndarray] = {}
        if self.f_y is not None:
            n_outputs = self.f_y.size1_out(0)
            y_hist = np.zeros((n_steps, n_outputs))

            # f_y signature: f_y(x, z, d, u, p, t)
            for i in range(n_steps):
                ti = t[i]
                u_vec = get_input(ti)
                d_vec = d_hist[i] if d_hist is not None else (d_current if d_current is not None else np.zeros(0))
                y_hist[i] = np.array(self.f_y(x_hist[i], z_vec, d_vec, u_vec, p, ti)).flatten()

            # Extract outputs by name (handle arrays)
            idx = 0
            for name in self._output_names:
                # For outputs, we need to determine size from the function output
                # For now, assume scalar outputs - array outputs would need tracking
                outputs[name] = y_hist[:, idx]
                idx += 1

        # Build data dict with all trajectories
        data: Dict[str, np.ndarray] = {}
        data.update(states)
        data.update(discrete)
        data.update(outputs)
        data.update(inputs)

        return SimulationResult(
            t=t,
            _data=data,
            model_name=self.name,
            state_names=list(self._state_names),
            output_names=list(self._output_names),
            input_names=list(self._input_names),
            discrete_names=list(self._discrete_names),
        )

    def __repr__(self) -> str:
        parts = [f"'{self.name}'", f"states={self._state_names}"]
        if self._discrete_names:
            parts.append(f"discrete={self._discrete_names}")
        if self._input_names:
            parts.append(f"inputs={self._input_names}")
        if self._param_names:
            parts.append(f"params={self._param_names}")
        if self._output_names:
            parts.append(f"outputs={self._output_names}")
        if self.has_events:
            parts.append(f"events={len(self.when_clause_funcs)}")
        return f"CompiledModel({', '.join(parts)})"
