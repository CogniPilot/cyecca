"""
SymPy backend for the Cyecca DSL.

Compiles FlatModel representations into SymPy expressions for
symbolic analysis, simplification, and code generation.

================================================================================
PROTOTYPE MODE - API IS IN FLUX
================================================================================

This DSL is in active prototype development. The API may change significantly
between versions. Do NOT maintain backward compatibility - iterate rapidly.

================================================================================
SymPy Backend Features
================================================================================

- **Symbolic Analysis**: Full symbolic manipulation and simplification
- **Jacobian Computation**: Symbolic derivatives for linearization
- **Code Generation**: Export to C, Fortran, NumPy, etc.
- **LaTeX Output**: Pretty mathematical expressions
- **Equation Solving**: Symbolic equation solving and manipulation

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp
from sympy import (
    Abs,
    And,
    Derivative,
    Eq,
    Function,
    Matrix,
    Max,
    Min,
    Not,
    Or,
    Piecewise,
    Symbol,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    ceiling,
    cos,
    cosh,
    exp,
    floor,
)
from sympy import log
from sympy import log as ln
from sympy import sign, sin, sinh, sqrt, tan, tanh

# DSL imports (equation types still in DSL for now)
from cyecca.dsl.equations import Equation
from cyecca.dsl.variables import SymbolicVar
from cyecca.ir.causality import SortedSystem, analyze_causality

# IR imports
from cyecca.ir.expr import Expr, ExprKind
from cyecca.ir.flat_model import FlatModel

# Type for model input
ModelInput = Union[FlatModel, SortedSystem]


# =============================================================================
# Expression Conversion - Dispatch Table
# =============================================================================


def _make_expr_handlers():
    """Create dispatch table for expression conversion to SymPy."""
    # Unary math operations
    unary_math = {
        ExprKind.NEG: lambda c, e: -c(e.children[0]),
        ExprKind.SIN: lambda c, e: sin(c(e.children[0])),
        ExprKind.COS: lambda c, e: cos(c(e.children[0])),
        ExprKind.TAN: lambda c, e: tan(c(e.children[0])),
        ExprKind.ASIN: lambda c, e: asin(c(e.children[0])),
        ExprKind.ACOS: lambda c, e: acos(c(e.children[0])),
        ExprKind.ATAN: lambda c, e: atan(c(e.children[0])),
        ExprKind.SQRT: lambda c, e: sqrt(c(e.children[0])),
        ExprKind.EXP: lambda c, e: exp(c(e.children[0])),
        ExprKind.LOG: lambda c, e: ln(c(e.children[0])),
        ExprKind.LOG10: lambda c, e: log(c(e.children[0]), 10),
        ExprKind.ABS: lambda c, e: Abs(c(e.children[0])),
        ExprKind.SIGN: lambda c, e: sign(c(e.children[0])),
        ExprKind.FLOOR: lambda c, e: floor(c(e.children[0])),
        ExprKind.CEIL: lambda c, e: ceiling(c(e.children[0])),
        ExprKind.SINH: lambda c, e: sinh(c(e.children[0])),
        ExprKind.COSH: lambda c, e: cosh(c(e.children[0])),
        ExprKind.TANH: lambda c, e: tanh(c(e.children[0])),
        ExprKind.ASINH: lambda c, e: asinh(c(e.children[0])),
        ExprKind.ACOSH: lambda c, e: acosh(c(e.children[0])),
        ExprKind.ATANH: lambda c, e: atanh(c(e.children[0])),
    }

    # Binary math operations
    binary_math = {
        ExprKind.ADD: lambda c, e: c(e.children[0]) + c(e.children[1]),
        ExprKind.SUB: lambda c, e: c(e.children[0]) - c(e.children[1]),
        ExprKind.MUL: lambda c, e: c(e.children[0]) * c(e.children[1]),
        ExprKind.DIV: lambda c, e: c(e.children[0]) / c(e.children[1]),
        ExprKind.POW: lambda c, e: c(e.children[0]) ** c(e.children[1]),
        ExprKind.ATAN2: lambda c, e: atan2(c(e.children[0]), c(e.children[1])),
        ExprKind.MIN: lambda c, e: Min(c(e.children[0]), c(e.children[1])),
        ExprKind.MAX: lambda c, e: Max(c(e.children[0]), c(e.children[1])),
        ExprKind.MOD: lambda c, e: sp.Mod(c(e.children[0]), c(e.children[1])),
    }

    # Comparison operations
    comparisons = {
        ExprKind.LT: lambda c, e: c(e.children[0]) < c(e.children[1]),
        ExprKind.LE: lambda c, e: c(e.children[0]) <= c(e.children[1]),
        ExprKind.GT: lambda c, e: c(e.children[0]) > c(e.children[1]),
        ExprKind.GE: lambda c, e: c(e.children[0]) >= c(e.children[1]),
        ExprKind.EQ: lambda c, e: sp.Eq(c(e.children[0]), c(e.children[1])),
        ExprKind.NE: lambda c, e: sp.Ne(c(e.children[0]), c(e.children[1])),
    }

    # Logical operations
    logical = {
        ExprKind.AND: lambda c, e: And(c(e.children[0]), c(e.children[1])),
        ExprKind.OR: lambda c, e: Or(c(e.children[0]), c(e.children[1])),
        ExprKind.NOT: lambda c, e: Not(c(e.children[0])),
    }

    return {**unary_math, **binary_math, **comparisons, **logical}


EXPR_HANDLERS = _make_expr_handlers()


# =============================================================================
# SymPy Backend Compiler
# =============================================================================


@dataclass
class SymPyCompiler:
    """Compiles a FlatModel into SymPy expressions.

    Parameters
    ----------
    model : FlatModel
        The flattened model to compile
    """

    model: FlatModel

    # Symbol mappings
    state_syms: Dict[str, Symbol] = field(default_factory=dict)
    state_dot_syms: Dict[str, Symbol] = field(default_factory=dict)
    input_syms: Dict[str, Symbol] = field(default_factory=dict)
    param_syms: Dict[str, Symbol] = field(default_factory=dict)
    algebraic_syms: Dict[str, Symbol] = field(default_factory=dict)
    output_syms: Dict[str, Symbol] = field(default_factory=dict)

    # Time symbol
    t_sym: Symbol = field(default_factory=lambda: Symbol("t"))

    # BLT analysis result
    sorted_system: Optional[SortedSystem] = None

    def __post_init__(self):
        """Initialize symbols after dataclass creation."""
        self._create_symbols()

    def _create_symbols(self) -> None:
        """Create SymPy symbols for all model variables."""
        model = self.model

        # State variables and their derivatives
        for name in model.state_names:
            # Clean name for SymPy (replace dots with underscores for valid Python)
            clean_name = name.replace(".", "_")
            self.state_syms[name] = Symbol(clean_name, real=True)
            self.state_dot_syms[name] = Symbol(f"d{clean_name}_dt", real=True)

        # Input variables
        for name in model.input_names:
            clean_name = name.replace(".", "_")
            self.input_syms[name] = Symbol(clean_name, real=True)

        # Parameter variables
        for name in model.param_names:
            clean_name = name.replace(".", "_")
            self.param_syms[name] = Symbol(clean_name, real=True, positive=True)

        # Algebraic variables
        for name in model.algebraic_names:
            clean_name = name.replace(".", "_")
            self.algebraic_syms[name] = Symbol(clean_name, real=True)

        # Output variables
        for name in model.output_names:
            clean_name = name.replace(".", "_")
            self.output_syms[name] = Symbol(clean_name, real=True)

        # Run BLT causality analysis
        self.sorted_system = analyze_causality(self.model)

    def expr_to_sympy(self, expr: Expr) -> sp.Basic:
        """Convert an Expr tree to a SymPy expression.

        Parameters
        ----------
        expr : Expr
            The expression tree to convert

        Returns
        -------
        sp.Basic
            The SymPy expression
        """
        kind = expr.kind

        # Constant
        if kind == ExprKind.CONSTANT:
            return sp.Float(expr.value) if isinstance(expr.value, float) else sp.Integer(expr.value)

        # Variable lookup
        if kind == ExprKind.VARIABLE:
            name = expr.name
            if name in self.state_syms:
                return self.state_syms[name]
            if name in self.input_syms:
                return self.input_syms[name]
            if name in self.param_syms:
                return self.param_syms[name]
            if name in self.algebraic_syms:
                return self.algebraic_syms[name]
            if name in self.output_syms:
                return self.output_syms[name]
            if name == "time":
                return self.t_sym
            raise ValueError(f"Unknown variable: {name}")

        # Derivative: der(x) -> dx_dt symbol
        if kind == ExprKind.DERIVATIVE:
            name = expr.name
            if name in self.state_dot_syms:
                return self.state_dot_syms[name]
            raise ValueError(f"Unknown state for derivative: {name}")

        # Time variable
        if kind == ExprKind.TIME:
            return self.t_sym

        # If-then-else -> Piecewise
        if kind == ExprKind.IF_THEN_ELSE:
            cond = self.expr_to_sympy(expr.children[0])
            then_val = self.expr_to_sympy(expr.children[1])
            else_val = self.expr_to_sympy(expr.children[2])
            return Piecewise((then_val, cond), (else_val, True))

        # Dispatch table
        if kind in EXPR_HANDLERS:
            return EXPR_HANDLERS[kind](self.expr_to_sympy, expr)

        raise ValueError(f"Unsupported expression kind: {kind}")

    def compile(self) -> "CompiledSymPyModel":
        """Compile the model into SymPy expressions.

        Returns
        -------
        CompiledSymPyModel
            The compiled model with SymPy expressions
        """
        model = self.model

        # Build state derivative expressions using BLT-solved equations
        state_dots: Dict[str, sp.Basic] = {}
        for solved in self.sorted_system.solved:
            if solved.is_derivative:
                state_dots[solved.var_name] = self.expr_to_sympy(solved.expr)

        # Also check original equations for pure der(x) == rhs form
        for eq in model.equations:
            if eq.is_derivative and eq.var_name:
                if eq.var_name not in state_dots:
                    state_dots[eq.var_name] = self.expr_to_sympy(eq.rhs)

        # Build algebraic equations using BLT-solved equations
        algebraic_eqs: Dict[str, sp.Basic] = {}
        for solved in self.sorted_system.solved:
            if not solved.is_derivative and solved.var_name in self.algebraic_syms:
                algebraic_eqs[solved.var_name] = self.expr_to_sympy(solved.expr)

        # Build output expressions
        output_exprs: Dict[str, sp.Basic] = {}
        for name in model.output_names:
            if name in model.output_equations:
                output_exprs[name] = self.expr_to_sympy(model.output_equations[name])

        # Build all equations in raw form
        raw_equations: List[sp.Eq] = []
        for eq in model.equations:
            lhs = self.expr_to_sympy(eq.lhs)
            rhs = self.expr_to_sympy(eq.rhs)
            raw_equations.append(sp.Eq(lhs, rhs))

        return CompiledSymPyModel(
            name=model.name,
            t=self.t_sym,
            states=self.state_syms,
            state_dots=self.state_dot_syms,
            inputs=self.input_syms,
            params=self.param_syms,
            algebraics=self.algebraic_syms,
            outputs=self.output_syms,
            f_x=state_dots,
            f_z=algebraic_eqs,
            f_y=output_exprs,
            equations=raw_equations,
            state_defaults=model.state_defaults,
            param_defaults=model.param_defaults,
        )

    @classmethod
    def from_model(cls, model: ModelInput) -> "SymPyCompiler":
        """Create a compiler from a model.

        Parameters
        ----------
        model : FlatModel or SortedSystem
            The model to compile

        Returns
        -------
        SymPyCompiler
            The compiler instance
        """
        if isinstance(model, SortedSystem):
            return cls(model=model.model)
        return cls(model=model)


# =============================================================================
# Compiled Model
# =============================================================================


@dataclass
class CompiledSymPyModel:
    """A compiled SymPy model ready for symbolic analysis.

    This class holds the SymPy expressions derived from a FlatModel
    and provides methods for symbolic manipulation, Jacobian computation,
    code generation, and LaTeX output.
    """

    name: str
    t: Symbol
    states: Dict[str, Symbol]
    state_dots: Dict[str, Symbol]
    inputs: Dict[str, Symbol]
    params: Dict[str, Symbol]
    algebraics: Dict[str, Symbol]
    outputs: Dict[str, Symbol]
    f_x: Dict[str, sp.Basic]  # State derivatives: der(x) = f_x[x]
    f_z: Dict[str, sp.Basic]  # Algebraic equations: z = f_z[z]
    f_y: Dict[str, sp.Basic]  # Outputs: y = f_y[y]
    equations: List[sp.Eq]  # All raw equations
    state_defaults: Dict[str, Any] = field(default_factory=dict)
    param_defaults: Dict[str, Any] = field(default_factory=dict)

    @property
    def state_names(self) -> List[str]:
        """Get state variable names."""
        return list(self.states.keys())

    @property
    def param_names(self) -> List[str]:
        """Get parameter names."""
        return list(self.params.keys())

    @property
    def state_vector(self) -> Matrix:
        """Get state vector as SymPy Matrix."""
        return Matrix([self.states[n] for n in self.state_names])

    @property
    def state_dot_vector(self) -> Matrix:
        """Get state derivative vector as SymPy Matrix."""
        return Matrix([self.state_dots[n] for n in self.state_names])

    @property
    def f_vector(self) -> Matrix:
        """Get dynamics vector f(x) where dx/dt = f(x)."""
        return Matrix([self.f_x.get(n, sp.Integer(0)) for n in self.state_names])

    def jacobian(self, simplify: bool = True) -> Matrix:
        """Compute the Jacobian matrix df/dx.

        Parameters
        ----------
        simplify : bool
            Whether to simplify the result

        Returns
        -------
        Matrix
            The Jacobian matrix
        """
        f = self.f_vector
        x = self.state_vector
        J = f.jacobian(x)
        if simplify:
            J = sp.simplify(J)
        return J

    def linearize(self, equilibrium: Optional[Dict[str, float]] = None) -> Tuple[Matrix, Matrix]:
        """Linearize the system around an equilibrium point.

        Parameters
        ----------
        equilibrium : dict, optional
            Equilibrium point {var_name: value}. If None, uses defaults.

        Returns
        -------
        A : Matrix
            System matrix (df/dx evaluated at equilibrium)
        B : Matrix
            Input matrix (df/du evaluated at equilibrium) if inputs exist
        """
        J = self.jacobian(simplify=True)

        # Build substitution dict
        subs = {}
        if equilibrium:
            for name, val in equilibrium.items():
                if name in self.states:
                    subs[self.states[name]] = val
                elif name in self.params:
                    subs[self.params[name]] = val
                elif name in self.inputs:
                    subs[self.inputs[name]] = val

        # Add parameter defaults
        for name, val in self.param_defaults.items():
            if name in self.params and self.params[name] not in subs:
                subs[self.params[name]] = val

        A = J.subs(subs)

        # Compute B if we have inputs
        if self.inputs:
            f = self.f_vector
            u = Matrix([self.inputs[n] for n in self.inputs.keys()])
            B = f.jacobian(u).subs(subs)
        else:
            B = Matrix([[]])

        return A, B

    def substitute(self, subs: Dict[str, float]) -> "CompiledSymPyModel":
        """Substitute parameter values into the model.

        Parameters
        ----------
        subs : dict
            Substitutions {var_name: value}

        Returns
        -------
        CompiledSymPyModel
            New model with substitutions applied
        """
        # Build symbol substitution dict
        sym_subs = {}
        for name, val in subs.items():
            if name in self.params:
                sym_subs[self.params[name]] = val
            elif name in self.states:
                sym_subs[self.states[name]] = val
            elif name in self.inputs:
                sym_subs[self.inputs[name]] = val

        # Apply to f_x
        new_f_x = {k: v.subs(sym_subs) for k, v in self.f_x.items()}
        new_f_z = {k: v.subs(sym_subs) for k, v in self.f_z.items()}
        new_f_y = {k: v.subs(sym_subs) for k, v in self.f_y.items()}
        new_eqs = [eq.subs(sym_subs) for eq in self.equations]

        return CompiledSymPyModel(
            name=self.name,
            t=self.t,
            states=self.states,
            state_dots=self.state_dots,
            inputs=self.inputs,
            params=self.params,
            algebraics=self.algebraics,
            outputs=self.outputs,
            f_x=new_f_x,
            f_z=new_f_z,
            f_y=new_f_y,
            equations=new_eqs,
            state_defaults=self.state_defaults,
            param_defaults=self.param_defaults,
        )

    def to_latex(self, wrap: bool = True) -> str:
        """Generate LaTeX representation of the system equations.

        Parameters
        ----------
        wrap : bool
            Whether to wrap in align environment

        Returns
        -------
        str
            LaTeX string
        """
        lines = []

        # State equations
        for name in self.state_names:
            if name in self.f_x:
                lhs = sp.latex(self.state_dots[name])
                rhs = sp.latex(self.f_x[name])
                lines.append(f"{lhs} &= {rhs}")

        if wrap:
            body = " \\\\\n".join(lines)
            return f"\\begin{{align}}\n{body}\n\\end{{align}}"
        return "\n".join(lines)

    def to_numpy_func(self) -> Callable:
        """Generate a NumPy-compatible function for the dynamics.

        Returns
        -------
        Callable
            Function f(t, x, u, p) -> dx/dt as numpy arrays
        """
        from sympy.utilities.lambdify import lambdify

        # Build ordered symbol lists
        state_list = [self.states[n] for n in self.state_names]
        param_list = [self.params[n] for n in self.param_names]
        input_list = [self.inputs[n] for n in self.inputs.keys()]

        # Build f vector
        f = [self.f_x.get(n, sp.Integer(0)) for n in self.state_names]

        # Create lambdified function
        all_symbols = [self.t] + state_list + input_list + param_list
        f_numpy = lambdify(all_symbols, f, modules=["numpy"])

        def dynamics(t: float, x: np.ndarray, u: np.ndarray = None, p: np.ndarray = None) -> np.ndarray:
            """Evaluate dynamics at given state.

            Parameters
            ----------
            t : float
                Time
            x : ndarray
                State vector
            u : ndarray, optional
                Input vector
            p : ndarray, optional
                Parameter vector

            Returns
            -------
            ndarray
                State derivative dx/dt
            """
            if u is None:
                u = np.zeros(len(input_list))
            if p is None:
                p = np.array([self.param_defaults.get(n, 0.0) for n in self.param_names])

            args = [t] + list(x) + list(u) + list(p)
            return np.array(f_numpy(*args)).flatten()

        return dynamics

    def simplify(self) -> "CompiledSymPyModel":
        """Return a new model with simplified expressions.

        Returns
        -------
        CompiledSymPyModel
            Model with simplified expressions
        """
        new_f_x = {k: sp.simplify(v) for k, v in self.f_x.items()}
        new_f_z = {k: sp.simplify(v) for k, v in self.f_z.items()}
        new_f_y = {k: sp.simplify(v) for k, v in self.f_y.items()}

        return CompiledSymPyModel(
            name=self.name,
            t=self.t,
            states=self.states,
            state_dots=self.state_dots,
            inputs=self.inputs,
            params=self.params,
            algebraics=self.algebraics,
            outputs=self.outputs,
            f_x=new_f_x,
            f_z=new_f_z,
            f_y=new_f_y,
            equations=self.equations,
            state_defaults=self.state_defaults,
            param_defaults=self.param_defaults,
        )

    def __repr__(self) -> str:
        return f"CompiledSymPyModel('{self.name}', " f"states={self.state_names}, " f"params={self.param_names})"


# =============================================================================
# Public API
# =============================================================================


class SymPyBackend:
    """SymPy backend for symbolic analysis of Cyecca models.

    This backend compiles FlatModel representations into SymPy expressions,
    enabling symbolic manipulation, Jacobian computation, linearization,
    code generation, and LaTeX output.

    Examples
    --------
    .. code-block:: python

        from cyecca.dsl import der, equations, model, var
        from cyecca.backends import SymPyBackend

        @model
        class Oscillator:
            x = var(start=1.0)
            v = var(start=0.0)
            k = var(1.0, parameter=True)

            @equations
            def _(m):
                der(m.x) == m.v
                der(m.v) == -m.k * m.x

        osc = Oscillator()
        flat = osc.flatten()
        compiled = SymPyBackend.compile(flat)
        jacobian = compiled.jacobian()
        print(jacobian)
        print(compiled.to_latex())
    """

    @staticmethod
    def compile(model: ModelInput) -> CompiledSymPyModel:
        """Compile a model to SymPy expressions.

        Parameters
        ----------
        model : FlatModel or SortedSystem
            The model to compile

        Returns
        -------
        CompiledSymPyModel
            The compiled model
        """
        compiler = SymPyCompiler.from_model(model)
        return compiler.compile()
