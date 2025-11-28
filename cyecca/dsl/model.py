"""
Base Model class and @model decorator for the Cyecca DSL.

Provides the decorator and base class that enable Modelica-like syntax
for defining dynamic systems in Python using yield-based equations.

================================================================================
PROTOTYPE MODE - API IS IN FLUX
================================================================================

This DSL is in active prototype development. The API may change significantly
between versions. Do NOT maintain backward compatibility - iterate rapidly.

================================================================================
DESIGN PRINCIPLES - DO NOT REMOVE OR IGNORE
================================================================================

1. MODELICA CONFORMANCE: This DSL conforms as closely as possible to the
   Modelica Language Specification v3.7-dev.
   Reference: https://specification.modelica.org/master/
   
   - Variable flags (parameter, input, output, constant) follow Modelica semantics
   - Attributes (start, fixed, min, max, nominal, unit) match Modelica definitions
   - Equation-based modeling with der() operator
   - Model flattening follows Modelica Chapter 5
   - Automatic state/algebraic classification based on der() usage
   - Connector/connection semantics (future) will follow Modelica Chapter 9

2. TYPE SAFETY: All functions MUST use beartype for runtime type checking.
   - Use @beartype decorator on ALL public functions and methods
   - Use proper type hints for all parameters and return values
   - Do NOT remove beartype decorators - they ensure runtime safety
   - WHEN ADDING NEW FUNCTIONS: Always add @beartype decorator
   - Use type aliases (Shape, Indices) from types.py for consistency

3. SELF-CONTAINED: This DSL module uses NO external libraries except:
   - Python standard library
   - beartype (for runtime type checking)
   - numpy (for NumericValue type hints only)
   
   The DSL builds an abstract model representation (expression trees, equation
   graphs) that can be compiled by separate backends (CasADi, JAX, etc.).
   
   DO NOT import CasADi, JAX, or other compute backends in this module.

4. IMMUTABILITY: Prefer immutable data structures where possible.

5. EXPLICIT > IMPLICIT: All behavior should be explicit and documented.

================================================================================

Example
-------
>>> from cyecca.dsl import model, var, der, sin
>>>
>>> @model
... class Pendulum:
...     g = var(9.81, parameter=True)
...     theta = var(start=0.5)
...     omega = var()
...     x = var(output=True)
...
...     def equations(m):
...         yield der(m.theta) == m.omega
...         yield der(m.omega) == -m.g * sin(m.theta)
...         yield m.x == sin(m.theta)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
from beartype import beartype

from cyecca.dsl.types import (
    DType,
    Indices,
    Shape,
    Var,
    VarKind,
    NumericValue,
    SubmodelField,
)


# =============================================================================
# Expression Tree - Abstract Symbolic Representation
# =============================================================================


class ExprKind(Enum):
    """Kinds of expression nodes."""
    # Leaf nodes
    VARIABLE = auto()      # Named variable (state, param, input, etc.)
    DERIVATIVE = auto()    # der(x) - derivative of a variable
    CONSTANT = auto()      # Numeric constant
    TIME = auto()          # Time variable t
    
    # Unary operations
    NEG = auto()           # -x
    NOT = auto()           # not x (Boolean negation)
    
    # Binary arithmetic operations  
    ADD = auto()           # x + y
    SUB = auto()           # x - y
    MUL = auto()           # x * y
    DIV = auto()           # x / y
    POW = auto()           # x ** y
    
    # Relational operations (Modelica MLS 3.5)
    LT = auto()            # x < y
    LE = auto()            # x <= y
    GT = auto()            # x > y
    GE = auto()            # x >= y
    EQ = auto()            # x == y (equality test, not equation)
    NE = auto()            # x != y (or x <> y in Modelica)
    
    # Boolean operations (Modelica MLS 3.5)
    AND = auto()           # x and y
    OR = auto()            # x or y
    
    # Conditional expression (Modelica MLS 3.6.5)
    IF_THEN_ELSE = auto()  # if cond then expr1 else expr2
    
    # Array operations
    INDEX = auto()         # x[i] - array indexing (stores index in 'value' field)
    
    # Discrete/event operators (Modelica MLS 3.8)
    PRE = auto()           # pre(x) - previous value of discrete variable
    EDGE = auto()          # edge(x) - True when x changes from False to True
    CHANGE = auto()        # change(x) - True when x changes value
    
    # Math functions
    SIN = auto()
    COS = auto()
    TAN = auto()
    ASIN = auto()
    ACOS = auto()
    ATAN = auto()
    ATAN2 = auto()
    SQRT = auto()
    EXP = auto()
    LOG = auto()
    ABS = auto()


@dataclass(frozen=True)
class Expr:
    """
    Immutable expression tree node.
    
    Represents symbolic mathematical expressions that can be compiled
    to different backends (CasADi, JAX, NumPy, etc.).
    
    This is the core abstraction that makes the DSL backend-agnostic.
    The DSL builds expression trees, and backends compile them to 
    executable functions.
    
    For indexed variables, use VARIABLE kind with indices set.
    """
    kind: ExprKind
    children: Tuple["Expr", ...] = ()
    name: Optional[str] = None       # For VARIABLE, DERIVATIVE
    value: Optional[float] = None    # For CONSTANT
    indices: Indices = ()            # For indexed VARIABLE: (i,), (i,j), etc.
    
    def __repr__(self) -> str:
        if self.kind == ExprKind.VARIABLE:
            if self.indices:
                idx_str = ",".join(str(i) for i in self.indices)
                return f"{self.name}[{idx_str}]"
            return f"{self.name}"
        elif self.kind == ExprKind.DERIVATIVE:
            return f"der({self.name})"
        elif self.kind == ExprKind.CONSTANT:
            return f"{self.value}"
        elif self.kind == ExprKind.TIME:
            return "t"
        elif self.kind == ExprKind.NEG:
            return f"(-{self.children[0]})"
        elif self.kind == ExprKind.ADD:
            return f"({self.children[0]} + {self.children[1]})"
        elif self.kind == ExprKind.SUB:
            return f"({self.children[0]} - {self.children[1]})"
        elif self.kind == ExprKind.MUL:
            return f"({self.children[0]} * {self.children[1]})"
        elif self.kind == ExprKind.DIV:
            return f"({self.children[0]} / {self.children[1]})"
        elif self.kind == ExprKind.POW:
            return f"({self.children[0]} ** {self.children[1]})"
        elif self.kind == ExprKind.INDEX:
            # Legacy INDEX kind - prefer using VARIABLE with indices
            return f"{self.name}[{int(self.value)}]"
        elif self.kind in (ExprKind.SIN, ExprKind.COS, ExprKind.TAN,
                          ExprKind.ASIN, ExprKind.ACOS, ExprKind.ATAN,
                          ExprKind.SQRT, ExprKind.EXP, ExprKind.LOG, ExprKind.ABS):
            return f"{self.kind.name.lower()}({self.children[0]})"
        elif self.kind == ExprKind.ATAN2:
            return f"atan2({self.children[0]}, {self.children[1]})"
        elif self.kind == ExprKind.PRE:
            return f"pre({self.name})"
        elif self.kind == ExprKind.EDGE:
            return f"edge({self.name})"
        elif self.kind == ExprKind.CHANGE:
            return f"change({self.name})"
        elif self.kind == ExprKind.LT:
            return f"({self.children[0]} < {self.children[1]})"
        elif self.kind == ExprKind.LE:
            return f"({self.children[0]} <= {self.children[1]})"
        elif self.kind == ExprKind.GT:
            return f"({self.children[0]} > {self.children[1]})"
        elif self.kind == ExprKind.GE:
            return f"({self.children[0]} >= {self.children[1]})"
        elif self.kind == ExprKind.EQ:
            return f"({self.children[0]} == {self.children[1]})"
        elif self.kind == ExprKind.NE:
            return f"({self.children[0]} != {self.children[1]})"
        elif self.kind == ExprKind.AND:
            return f"({self.children[0]} and {self.children[1]})"
        elif self.kind == ExprKind.OR:
            return f"({self.children[0]} or {self.children[1]})"
        elif self.kind == ExprKind.NOT:
            return f"(not {self.children[0]})"
        elif self.kind == ExprKind.IF_THEN_ELSE:
            return f"(if {self.children[0]} then {self.children[1]} else {self.children[2]})"
        return f"Expr({self.kind})"
    
    @property
    def indexed_name(self) -> str:
        """Get the full name including indices: 'x' or 'x[0,1]'."""
        if self.indices:
            idx_str = ",".join(str(i) for i in self.indices)
            return f"{self.name}[{idx_str}]"
        return self.name or ""
    
    # Arithmetic operators - return new Expr nodes
    def __add__(self, other: Any) -> "Expr":
        return Expr(ExprKind.ADD, (self, _to_expr(other)))
    
    def __radd__(self, other: Any) -> "Expr":
        return Expr(ExprKind.ADD, (_to_expr(other), self))
    
    def __sub__(self, other: Any) -> "Expr":
        return Expr(ExprKind.SUB, (self, _to_expr(other)))
    
    def __rsub__(self, other: Any) -> "Expr":
        return Expr(ExprKind.SUB, (_to_expr(other), self))
    
    def __mul__(self, other: Any) -> "Expr":
        return Expr(ExprKind.MUL, (self, _to_expr(other)))
    
    def __rmul__(self, other: Any) -> "Expr":
        return Expr(ExprKind.MUL, (_to_expr(other), self))
    
    def __truediv__(self, other: Any) -> "Expr":
        return Expr(ExprKind.DIV, (self, _to_expr(other)))
    
    def __rtruediv__(self, other: Any) -> "Expr":
        return Expr(ExprKind.DIV, (_to_expr(other), self))
    
    def __pow__(self, other: Any) -> "Expr":
        return Expr(ExprKind.POW, (self, _to_expr(other)))
    
    def __rpow__(self, other: Any) -> "Expr":
        return Expr(ExprKind.POW, (_to_expr(other), self))
    
    def __neg__(self) -> "Expr":
        return Expr(ExprKind.NEG, (self,))
    
    def __pos__(self) -> "Expr":
        return self
    
    # Relational operators - return Boolean Expr
    def __lt__(self, other: Any) -> "Expr":
        return Expr(ExprKind.LT, (self, _to_expr(other)))
    
    def __le__(self, other: Any) -> "Expr":
        return Expr(ExprKind.LE, (self, _to_expr(other)))
    
    def __gt__(self, other: Any) -> "Expr":
        return Expr(ExprKind.GT, (self, _to_expr(other)))
    
    def __ge__(self, other: Any) -> "Expr":
        return Expr(ExprKind.GE, (self, _to_expr(other)))


@beartype
def _to_expr(x: Any) -> Expr:
    """Convert various types to Expr."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, SymbolicVar):
        return x._expr
    if isinstance(x, DerivativeExpr):
        return x._expr
    if isinstance(x, TimeVar):
        return x._expr
    # Check for AlgorithmVar by attribute (defined later in file)
    if hasattr(x, '_expr') and hasattr(x, '_name') and type(x).__name__ == 'AlgorithmVar':
        return x._expr
    if isinstance(x, (int, float)):
        return Expr(ExprKind.CONSTANT, value=float(x))
    if isinstance(x, np.ndarray) and x.size == 1:
        return Expr(ExprKind.CONSTANT, value=float(x.flat[0]))
    raise TypeError(f"Cannot convert {type(x)} to Expr")


@beartype
def _find_derivatives(expr: Expr) -> set[str]:
    """
    Find all variable names whose derivative (der) appears in an expression.
    
    This is used for automatic state detection: if der(x) appears anywhere
    in the equations, then x is a state variable.
    
    For indexed variables like der(pos[0]), returns "pos[0]".
    """
    result: set[str] = set()
    
    if expr.kind == ExprKind.DERIVATIVE and expr.name:
        result.add(expr.name)
    
    for child in expr.children:
        result.update(_find_derivatives(child))
    
    return result


def _prefix_expr(expr: Expr, prefix: str) -> Expr:
    """
    Create a new Expr with all variable names prefixed.
    
    This is used when flattening submodels to give all variables
    their fully qualified names (e.g., 'x' -> 'spring.x').
    """
    if expr.kind == ExprKind.VARIABLE:
        new_name = f"{prefix}.{expr.name}"
        return Expr(
            kind=ExprKind.VARIABLE,
            name=new_name,
            value=expr.value,
            children=tuple(_prefix_expr(c, prefix) for c in expr.children),
        )
    elif expr.kind == ExprKind.DERIVATIVE:
        new_name = f"{prefix}.{expr.name}" if expr.name else None
        return Expr(
            kind=ExprKind.DERIVATIVE,
            name=new_name,
            value=expr.value,
            children=tuple(_prefix_expr(c, prefix) for c in expr.children),
        )
    elif expr.kind == ExprKind.CONSTANT:
        return expr  # Constants don't need prefixing
    else:
        # Recursively prefix children for operators, functions, etc.
        return Expr(
            kind=expr.kind,
            name=expr.name,
            value=expr.value,
            children=tuple(_prefix_expr(c, prefix) for c in expr.children),
        )


@beartype
def _get_base_name(name: str) -> str:
    """Extract base name from indexed name: 'pos[0,1]' -> 'pos'."""
    if '[' in name:
        return name.split('[')[0]
    return name


@beartype
def _parse_indices(name: str) -> Tuple[str, Indices]:
    """Parse indexed name: 'pos[0,1]' -> ('pos', (0, 1))."""
    if '[' not in name:
        return name, ()
    base = name.split('[')[0]
    idx_str = name.split('[')[1].rstrip(']')
    indices = tuple(int(i) for i in idx_str.split(','))
    return base, indices


@beartype
def _format_indices(indices: Indices) -> str:
    """Format indices as string: (0, 1) -> '[0,1]'."""
    if not indices:
        return ""
    return "[" + ",".join(str(i) for i in indices) + "]"


@beartype
def _iter_indices(shape: Shape) -> Generator[Indices, None, None]:
    """Iterate over all valid index tuples for a given shape."""
    if not shape:
        yield ()
        return
    import itertools
    for idx in itertools.product(*(range(dim) for dim in shape)):
        yield idx


@beartype
def _is_array_state(name: str, shape: Shape, derivatives_used: set[str]) -> bool:
    """
    Check if an array variable is a state by checking if any element's
    derivative is used.
    
    For a variable 'pos' with shape=(3,), checks if 'pos[0]', 'pos[1]', or 'pos[2]'
    appear in derivatives_used.
    """
    if not shape:  # Scalar
        return False
    for indices in _iter_indices(shape):
        indexed_name = f"{name}{_format_indices(indices)}"
        if indexed_name in derivatives_used:
            return True
    return False


# =============================================================================
# Factory function for var() - the unified variable declaration
# =============================================================================
# NOTE: All public functions below MUST have @beartype decorator.
# When adding new functions, always include @beartype.


@beartype
def var(
    default: Optional[Union[float, int, List[float], np.ndarray]] = None,
    dtype: DType = DType.REAL,
    shape: Shape = (),
    unit: Optional[str] = None,
    desc: str = "",
    start: Optional[Union[float, int, List[float], np.ndarray]] = None,
    fixed: bool = False,
    min: Optional[Union[float, int]] = None,
    max: Optional[Union[float, int]] = None,
    nominal: Optional[Union[float, int]] = None,
    # Variability flags (Modelica-style)
    parameter: bool = False,
    discrete: bool = False,
    input: bool = False,
    output: bool = False,
    constant: bool = False,
    # Visibility (Modelica-style)
    protected: bool = False,
) -> Var:
    """
    Declare a variable in a Cyecca model.
    
    This is the unified way to declare all types of variables. The classification
    (state, algebraic, parameter, input, output) is determined automatically based
    on the flags and equation analysis.
    
    Parameters
    ----------
    default : float or array, optional
        Default value (used if start not specified)
    dtype : DType, optional
        Variable type (REAL, INTEGER, BOOLEAN). Default: REAL
    shape : tuple of int, optional
        Array shape: () for scalar, (n,) for vector, (m,n) for matrix, etc.
        Default: () (scalar)
    unit : str, optional
        Physical unit (e.g., "m", "rad/s")
    desc : str, optional
        Description
    start : float or array, optional
        Initial value (Modelica-style, takes precedence over default)
    fixed : bool, optional
        If True, start value is used as fixed initial condition
    min : float, optional
        Minimum bound for the variable (only for REAL/INTEGER)
    max : float, optional
        Maximum bound for the variable (only for REAL/INTEGER)
    nominal : float, optional
        Nominal value for scaling (only for REAL)
        
    Variability/Causality Prefixes (Modelica-style)
    ------------------------------------------------
    These are Modelica prefixes that indicate how the variable interfaces
    with the outside world. In Modelica, these are just prefixes on variables,
    NOT separate equation categories.
    
    parameter : bool, optional
        If True, variable is constant during simulation (can change between runs)
    discrete : bool, optional
        If True, variable is piecewise constant (changes only at events).
        Can use pre() to access previous value. Typical for counters, state machines.
    input : bool, optional
        If True, variable value is provided externally (causality: input)
    output : bool, optional
        If True, variable value is computed internally and exposed (causality: output)
    constant : bool, optional
        If True, variable is a compile-time constant
        
    Visibility (Modelica-style)
    ---------------------------
    protected : bool, optional
        If True, variable is in the protected section (internal implementation).
        Protected variables are not part of the public interface.
        For blocks, only public (non-protected) variables must have input/output.
        
    Automatic Classification
    ------------------------
    If none of the flags are True:
    - If der(var) appears in equations → state variable
    - Otherwise → algebraic variable
    
    Examples
    --------
    >>> @model
    ... class Pendulum:
    ...     # Parameters (constant during simulation)
    ...     g = var(9.81, parameter=True, unit="m/s^2")
    ...     l = var(1.0, parameter=True, min=0.1)
    ...     
    ...     # State variables (der() used in equations)  
    ...     theta = var(start=0.5, fixed=True)
    ...     omega = var(start=0, min=-100, max=100)
    ...     
    ...     # Vector state (3D position and velocity)
    ...     pos = var(shape=(3,))
    ...     vel = var(shape=(3,))
    ...     
    ...     # Matrix parameter (rotation)
    ...     R = var(shape=(3, 3), parameter=True)
    ...     
    ...     # Outputs (computed, exposed)
    ...     x = var(output=True)
    ...     y = var(output=True)
    ...     
    ...     # Inputs (externally controlled)
    ...     torque = var(input=True, min=-10, max=10)
    ...     
    ...     # Integer variable
    ...     mode = var(0, dtype=DType.INTEGER, parameter=True)
    ...     
    ...     # Boolean variable
    ...     enabled = var(True, dtype=DType.BOOLEAN, parameter=True)
    """
    return Var(
        default=default,
        dtype=dtype,
        shape=shape,
        unit=unit,
        desc=desc,
        start=start,
        fixed=fixed,
        min=min,
        max=max,
        nominal=nominal,
        parameter=parameter,
        discrete=discrete,
        input=input,
        output=output,
        constant=constant,
        protected=protected,
    )


@beartype
def submodel(model_class: Type) -> SubmodelField:
    """
    Declare a submodel (nested model).

    Submodels allow hierarchical composition of models.

    Example
    -------
    >>> from cyecca.dsl import model, var, submodel
    >>> @model
    ... class Controller:
    ...     gain = var(1.0, parameter=True)
    >>> @model
    ... class System:
    ...     ctrl = submodel(Controller)
    """
    return SubmodelField(model_class=model_class)


# =============================================================================
# Equation representation
# =============================================================================


@dataclass(frozen=True)
class Equation:
    """
    Represents an equation: lhs == rhs.
    
    Immutable representation of a model equation that can be processed
    by backends for compilation.
    """
    lhs: Expr
    rhs: Expr
    is_derivative: bool = False  # True if LHS is der(x)
    var_name: Optional[str] = None  # Name of variable if LHS is der(x)

    def __repr__(self) -> str:
        return f"Eq({self.lhs} == {self.rhs})"
    
    def _prefix_names(self, prefix: str) -> "Equation":
        """Create a new equation with all variable names prefixed."""
        new_lhs = _prefix_expr(self.lhs, prefix)
        new_rhs = _prefix_expr(self.rhs, prefix)
        new_var_name = f"{prefix}.{self.var_name}" if self.var_name else None
        return Equation(
            lhs=new_lhs,
            rhs=new_rhs,
            is_derivative=self.is_derivative,
            var_name=new_var_name,
        )


@dataclass(frozen=True)
class Assignment:
    """
    Represents an algorithm assignment: var := expr.
    
    Unlike equations (which are declarative and can be solved in any order),
    assignments are imperative and executed in sequence.
    
    In Modelica, assignments use := operator (vs == for equations).
    In this DSL, use the assign() function or <<= operator in algorithm sections.
    
    Assignments can target:
    - Model variables (m.x := expr)
    - Local algorithm variables (temp := expr)
    """
    target: str           # Variable name being assigned
    expr: Expr            # Right-hand side expression
    is_local: bool = False  # True if target is a local algorithm variable
    
    def __repr__(self) -> str:
        return f"Assign({self.target} := {self.expr})"


# =============================================================================
# Symbolic variable wrappers (user-facing)
# =============================================================================
# NOTE: Public methods that accept external input should use @beartype.
# Class __dunder__ methods are exempt but type hints are still required.


class SymbolicVar:
    """
    Symbolic variable proxy for building equations.

    Wraps an Expr and supports arithmetic operations.
    This is the user-facing object accessed via self.x in equations.
    
    Unified representation for both:
    - Base variables: shape=(), indices=() for scalar; shape=(3,), indices=() for vector
    - Indexed elements: shape=(3,), indices=(0,) for first element of vector
    
    For array variables, supports N-dimensional indexing:
        x[0]      - 1D indexing
        x[0, 1]   - 2D indexing  
        x[0, 1, 2] - 3D indexing
    """

    def __init__(
        self,
        name: str,
        var: Var,
        model: "ModelInstance",
        indices: Indices = (),
    ):
        self._base_name = name
        self._var = var
        self._model = model
        self._shape = var.shape
        self._indices = indices
        
        # Compute effective shape after indexing
        # e.g., shape=(3,3), indices=(0,) -> remaining_shape=(3,)
        self._remaining_shape = self._shape[len(indices):]
        
        # Build the expression
        self._expr = Expr(ExprKind.VARIABLE, name=name, indices=indices)
        
        # Full name including indices
        self._name = name + _format_indices(indices)
        
        # Cache for indexed elements
        self._indexed_cache: Dict[Indices, "SymbolicVar"] = {}

    @property
    def name(self) -> str:
        """The variable name including indices (for use with SimulationResult)."""
        return self._name
    
    @property
    def base_name(self) -> str:
        """The base variable name without indices."""
        return self._base_name
    
    @property
    def shape(self) -> Shape:
        """The original shape of the variable."""
        return self._shape
    
    @property
    def remaining_shape(self) -> Shape:
        """The remaining shape after current indexing."""
        return self._remaining_shape
    
    @property
    def indices(self) -> Indices:
        """Current indices applied to this variable."""
        return self._indices
    
    @property
    def size(self) -> int:
        """Total number of scalar elements."""
        return self._var.size
    
    @property 
    def ndim(self) -> int:
        """Number of remaining dimensions."""
        return len(self._remaining_shape)
    
    def is_scalar(self) -> bool:
        """Return True if fully indexed to a scalar."""
        return self._remaining_shape == ()

    def __repr__(self) -> str:
        return self._name
    
    def __getitem__(self, index: Union[int, Indices]) -> "SymbolicVar":
        """
        Index into an array variable (N-dimensional).
        
        Parameters
        ----------
        index : int or tuple of int
            Index or indices into the array.
            x[0] for 1D, x[0, 1] for 2D, etc.
            
        Returns
        -------
        SymbolicVar
            Symbolic representation of x[index] or x[i,j,...]
            
        Raises
        ------
        IndexError
            If index is out of bounds
        TypeError
            If variable has no remaining dimensions to index
        """
        # Normalize index to tuple
        if isinstance(index, int):
            new_indices = (index,)
        elif isinstance(index, tuple):
            new_indices = index
        else:
            raise TypeError(f"Index must be int or tuple, got {type(index).__name__}")
        
        # Check we have dimensions to index
        if len(new_indices) > len(self._remaining_shape):
            raise TypeError(
                f"Too many indices for '{self._name}': got {len(new_indices)}, "
                f"but remaining dimensions is {len(self._remaining_shape)}"
            )
        
        # Validate each index
        for i, (idx, dim) in enumerate(zip(new_indices, self._remaining_shape)):
            if not isinstance(idx, int):
                raise TypeError(f"Index {i} must be an integer, got {type(idx).__name__}")
            if idx < 0 or idx >= dim:
                raise IndexError(
                    f"Index {idx} out of bounds for dimension {i} of '{self._name}' with size {dim}"
                )
        
        # Combine with existing indices
        full_indices = self._indices + new_indices
        
        # Use cache
        if full_indices not in self._indexed_cache:
            self._indexed_cache[full_indices] = SymbolicVar(
                name=self._base_name,
                var=self._var,
                model=self._model,
                indices=full_indices,
            )
        
        return self._indexed_cache[full_indices]
    
    def __len__(self) -> int:
        """Return the size of the first remaining dimension."""
        if not self._remaining_shape:
            raise TypeError(f"Scalar variable '{self._name}' has no length")
        return self._remaining_shape[0]
    
    def __iter__(self):
        """Iterate over the first remaining dimension."""
        if not self._remaining_shape:
            raise TypeError(f"Cannot iterate over scalar variable '{self._name}'")
        for i in range(self._remaining_shape[0]):
            yield self[i]

    # Comparison operators for equations
    def __eq__(self, other: Any) -> "Equation":  # type: ignore[override]
        """Capture equation: x == expr."""
        rhs = _to_expr(other)
        return Equation(lhs=self._expr, rhs=rhs)

    # Arithmetic operations - return Expr
    def __add__(self, other: Any) -> Expr:
        return self._expr + other

    def __radd__(self, other: Any) -> Expr:
        return _to_expr(other) + self._expr

    def __sub__(self, other: Any) -> Expr:
        return self._expr - other

    def __rsub__(self, other: Any) -> Expr:
        return _to_expr(other) - self._expr

    def __mul__(self, other: Any) -> Expr:
        return self._expr * other

    def __rmul__(self, other: Any) -> Expr:
        return _to_expr(other) * self._expr

    def __truediv__(self, other: Any) -> Expr:
        return self._expr / other

    def __rtruediv__(self, other: Any) -> Expr:
        return _to_expr(other) / self._expr

    def __neg__(self) -> Expr:
        return -self._expr

    def __pow__(self, other: Any) -> Expr:
        return self._expr ** other
    
    # Relational operators - return Boolean Expr
    def __lt__(self, other: Any) -> Expr:
        return self._expr < other
    
    def __le__(self, other: Any) -> Expr:
        return self._expr <= other
    
    def __gt__(self, other: Any) -> Expr:
        return self._expr > other
    
    def __ge__(self, other: Any) -> Expr:
        return self._expr >= other
    
    # Assignment operator for algorithm sections
    def __matmul__(self, other: Any) -> "Assignment":
        """
        Assignment operator for algorithm sections: m.x @ expr
        
        This creates an Assignment object that can be yielded in algorithm().
        The @ operator is used because := is not valid Python syntax for this,
        and @ is free since we use * for matrix multiplication (like Modelica).
        
        Note: We use @ (not @=) because augmented assignment is a statement,
        not an expression, so it can't be used with yield.
        
        Example
        -------
        >>> def algorithm(m):
        ...     yield m.temp @ m.x * 2
        ...     yield m.y @ m.temp + 1
        """
        return Assignment(target=self._name, expr=_to_expr(other), is_local=False)


class DerivativeExpr:
    """Represents der(x) - the derivative of a state variable."""

    def __init__(self, var_name: str):
        self._var_name = var_name
        self._expr = Expr(ExprKind.DERIVATIVE, name=var_name)

    def __repr__(self) -> str:
        return f"der({self._var_name})"

    def __eq__(self, other: Any) -> Equation:  # type: ignore[override]
        """Capture equation: der(x) == expr."""
        rhs = _to_expr(other)
        return Equation(
            lhs=self._expr,
            rhs=rhs,
            is_derivative=True,
            var_name=self._var_name,
        )

    # Arithmetic (for expressions like der(x) + y)
    def __add__(self, other: Any) -> Expr:
        return self._expr + other

    def __radd__(self, other: Any) -> Expr:
        return _to_expr(other) + self._expr

    def __sub__(self, other: Any) -> Expr:
        return self._expr - other

    def __rsub__(self, other: Any) -> Expr:
        return _to_expr(other) - self._expr

    def __neg__(self) -> Expr:
        return -self._expr


class TimeVar:
    """Represents the time variable t."""
    
    def __init__(self) -> None:
        self._expr = Expr(ExprKind.TIME)
    
    def __repr__(self) -> str:
        return "t"
    
    # Arithmetic operations
    def __add__(self, other: Any) -> Expr:
        return self._expr + other

    def __radd__(self, other: Any) -> Expr:
        return _to_expr(other) + self._expr

    def __sub__(self, other: Any) -> Expr:
        return self._expr - other

    def __rsub__(self, other: Any) -> Expr:
        return _to_expr(other) - self._expr

    def __mul__(self, other: Any) -> Expr:
        return self._expr * other

    def __rmul__(self, other: Any) -> Expr:
        return _to_expr(other) * self._expr

    def __truediv__(self, other: Any) -> Expr:
        return self._expr / other

    def __rtruediv__(self, other: Any) -> Expr:
        return _to_expr(other) / self._expr


# =============================================================================
# Free function: der()
# =============================================================================


class ArrayDerivativeExpr:
    """
    Represents der(x) for an array variable with remaining dimensions.
    
    When used in an equation like `der(pos) == vel`, this expands to
    multiple scalar equations during flattening.
    """
    
    def __init__(self, var: SymbolicVar):
        self._var = var
        self._base_name = var._base_name
        self._indices = var._indices
        self._remaining_shape = var._remaining_shape
    
    def __repr__(self) -> str:
        return f"der({self._var._name})"
    
    def __eq__(self, other: Any) -> "ArrayEquation":  # type: ignore[override]
        """Capture array equation: der(pos) == vel."""
        return ArrayEquation(
            lhs_var=self._var,
            rhs=other,
            is_derivative=True,
        )
    
    def __getitem__(self, index: Union[int, Indices]) -> "DerivativeExpr":
        """Allow der(pos)[i] syntax as alternative to der(pos[i])."""
        indexed_var = self._var[index]
        if not indexed_var.is_scalar():
            # Return another ArrayDerivativeExpr for partial indexing
            return ArrayDerivativeExpr(indexed_var)
        return DerivativeExpr(indexed_var._name)


@dataclass
class ArrayEquation:
    """
    Represents an array equation that expands to multiple scalar equations.
    
    For example: der(pos) == vel with shape=(3,) expands to:
        der(pos[0]) == vel[0]
        der(pos[1]) == vel[1]
        der(pos[2]) == vel[2]
        
    For 2D arrays with shape=(2,3):
        der(pos[0,0]) == vel[0,0]
        der(pos[0,1]) == vel[0,1]
        ...
    """
    lhs_var: SymbolicVar  # The LHS array variable
    rhs: Any              # The RHS (could be SymbolicVar, Expr, etc.)
    is_derivative: bool = False
    
    def expand(self) -> List[Equation]:
        """Expand array equation to scalar equations."""
        remaining_shape = self.lhs_var._remaining_shape
        base_indices = self.lhs_var._indices
        equations = []
        
        for rel_indices in _iter_indices(remaining_shape):
            full_indices = base_indices + rel_indices
            indexed_name = self.lhs_var._base_name + _format_indices(full_indices)
            
            # Create LHS for this element
            if self.is_derivative:
                lhs = Expr(ExprKind.DERIVATIVE, name=indexed_name)
            else:
                lhs = Expr(ExprKind.VARIABLE, name=self.lhs_var._base_name, indices=full_indices)
            
            # Create RHS for this element
            if isinstance(self.rhs, SymbolicVar):
                if self.rhs._remaining_shape != remaining_shape:
                    raise ValueError(
                        f"Shape mismatch in array equation: "
                        f"{self.lhs_var._name} has remaining shape {remaining_shape}, "
                        f"but {self.rhs._name} has remaining shape {self.rhs._remaining_shape}"
                    )
                rhs_indices = self.rhs._indices + rel_indices
                rhs = Expr(ExprKind.VARIABLE, name=self.rhs._base_name, indices=rhs_indices)
            elif isinstance(self.rhs, (int, float)):
                # Scalar broadcast
                rhs = Expr(ExprKind.CONSTANT, value=float(self.rhs))
            elif isinstance(self.rhs, Expr):
                # Expression - use as-is (assumed to be scalar broadcast)
                rhs = self.rhs
            else:
                raise TypeError(f"Cannot expand array equation with RHS type {type(self.rhs)}")
            
            eq = Equation(
                lhs=lhs,
                rhs=rhs,
                is_derivative=self.is_derivative,
                var_name=indexed_name if self.is_derivative else None,
            )
            equations.append(eq)
        
        return equations
    
    def __repr__(self) -> str:
        prefix = "der(" if self.is_derivative else ""
        suffix = ")" if self.is_derivative else ""
        return f"ArrayEq({prefix}{self.lhs_var._name}{suffix} == {self.rhs})"


@beartype
def der(var: SymbolicVar) -> Union[DerivativeExpr, ArrayDerivativeExpr]:
    """
    Return the derivative of a state variable.
    
    This is a free function for use in equations. Works with both scalar
    and array variables:
    
        def equations(m):
            yield der(m.theta) == m.omega      # Scalar
            yield der(m.pos) == m.vel          # Array (element-wise)
            yield der(m.pos[0]) == m.vel[0]    # Single element
            yield der(m.R[0,0]) == m.R_dot[0,0]  # Matrix element
    
    Parameters
    ----------
    var : SymbolicVar
        The state variable (scalar or array, possibly indexed)
    
    Returns
    -------
    DerivativeExpr or ArrayDerivativeExpr
        An expression representing the derivative
    
    Example
    -------
    >>> @model
    ... class Pendulum:
    ...     theta = var()
    ...     omega = var()
    ...
    ...     def equations(m):
    ...         yield der(m.theta) == m.omega
    ...
    >>> @model
    ... class Particle:
    ...     pos = var(shape=(3,))
    ...     vel = var(shape=(3,))
    ...
    ...     def equations(m):
    ...         yield der(m.pos) == m.vel  # Expands to 3 equations
    """
    if not isinstance(var, SymbolicVar):
        raise TypeError(f"der() expects a SymbolicVar, got {type(var)}")
    
    if var.is_scalar():
        # Fully indexed or scalar variable
        return DerivativeExpr(var._name)
    else:
        # Array with remaining dimensions
        return ArrayDerivativeExpr(var)


# =============================================================================
# Discrete operators: pre(), edge(), change()
# =============================================================================


@beartype
def pre(var: SymbolicVar) -> Expr:
    """
    Return the previous value of a discrete variable.
    
    In Modelica, pre(x) returns the value of x at the previous event instant.
    This is only valid for discrete variables (variables with discrete=True
    or variables assigned in when-equations).
    
    Parameters
    ----------
    var : SymbolicVar
        A discrete variable
    
    Returns
    -------
    Expr
        An expression representing pre(var)
    
    Example
    -------
    >>> @model
    ... class Counter:
    ...     count = var(0, discrete=True)
    ...     tick = var(False, dtype=DType.BOOLEAN, input=True)
    ...
    ...     def when_equations(m):
    ...         with when(m.tick):
    ...             yield m.count == pre(m.count) + 1
    """
    if not isinstance(var, SymbolicVar):
        raise TypeError(f"pre() expects a SymbolicVar, got {type(var)}")
    if not var.is_scalar():
        raise TypeError(f"pre() currently only supports scalar variables, got shape {var.shape}")
    return Expr(ExprKind.PRE, name=var._name)


@beartype
def edge(var: SymbolicVar) -> Expr:
    """
    Return True when a Boolean variable changes from False to True.
    
    Equivalent to: `var and not pre(var)`
    
    Parameters
    ----------
    var : SymbolicVar
        A Boolean discrete variable
    
    Returns
    -------
    Expr
        An expression representing edge(var)
    
    Example
    -------
    >>> @model
    ... class EdgeDetector:
    ...     trigger = var(False, dtype=DType.BOOLEAN, input=True)
    ...     count = var(0, discrete=True)
    ...
    ...     def when_equations(m):
    ...         with when(edge(m.trigger)):
    ...             yield m.count == pre(m.count) + 1
    """
    if not isinstance(var, SymbolicVar):
        raise TypeError(f"edge() expects a SymbolicVar, got {type(var)}")
    if not var.is_scalar():
        raise TypeError(f"edge() currently only supports scalar variables")
    return Expr(ExprKind.EDGE, name=var._name)


@beartype
def change(var: SymbolicVar) -> Expr:
    """
    Return True when a variable changes its value.
    
    Equivalent to: `var != pre(var)`
    
    Parameters
    ----------
    var : SymbolicVar
        A discrete variable
    
    Returns
    -------
    Expr
        An expression representing change(var)
    
    Example
    -------
    >>> @model
    ... class ChangeDetector:
    ...     mode = var(0, discrete=True, input=True)
    ...     mode_changes = var(0, discrete=True)
    ...
    ...     def when_equations(m):
    ...         with when(change(m.mode)):
    ...             yield m.mode_changes == pre(m.mode_changes) + 1
    """
    if not isinstance(var, SymbolicVar):
        raise TypeError(f"change() expects a SymbolicVar, got {type(var)}")
    if not var.is_scalar():
        raise TypeError(f"change() currently only supports scalar variables")
    return Expr(ExprKind.CHANGE, name=var._name)


# =============================================================================
# Boolean operators: and_, or_, not_
# =============================================================================
# NOTE: Python's `and`, `or`, `not` are keywords and cannot be overloaded.
# We use trailing underscores per PEP 8 convention for avoiding conflicts.


@beartype
def and_(a: Any, b: Any) -> Expr:
    """
    Logical AND of two Boolean expressions.
    
    Since Python's `and` keyword cannot be overloaded, use this function
    for Boolean conjunction in model equations.
    
    Parameters
    ----------
    a : Expr or SymbolicVar or bool
        First Boolean operand
    b : Expr or SymbolicVar or bool
        Second Boolean operand
    
    Returns
    -------
    Expr
        Boolean expression representing `a and b`
    
    Example
    -------
    >>> @model
    ... class SafetyCheck:
    ...     enabled = var(dtype=DType.BOOLEAN, input=True)
    ...     in_range = var(dtype=DType.BOOLEAN)
    ...     safe = var(dtype=DType.BOOLEAN, output=True)
    ...
    ...     def equations(m):
    ...         yield m.in_range == and_(m.x > 0, m.x < 100)
    ...         yield m.safe == and_(m.enabled, m.in_range)
    """
    return Expr(ExprKind.AND, (_to_expr(a), _to_expr(b)))


@beartype
def or_(a: Any, b: Any) -> Expr:
    """
    Logical OR of two Boolean expressions.
    
    Since Python's `or` keyword cannot be overloaded, use this function
    for Boolean disjunction in model equations.
    
    Parameters
    ----------
    a : Expr or SymbolicVar or bool
        First Boolean operand
    b : Expr or SymbolicVar or bool
        Second Boolean operand
    
    Returns
    -------
    Expr
        Boolean expression representing `a or b`
    
    Example
    -------
    >>> @model
    ... class AlarmSystem:
    ...     temp_high = var(dtype=DType.BOOLEAN)
    ...     pressure_high = var(dtype=DType.BOOLEAN)
    ...     alarm = var(dtype=DType.BOOLEAN, output=True)
    ...
    ...     def equations(m):
    ...         yield m.alarm == or_(m.temp_high, m.pressure_high)
    """
    return Expr(ExprKind.OR, (_to_expr(a), _to_expr(b)))


@beartype
def not_(a: Any) -> Expr:
    """
    Logical NOT of a Boolean expression.
    
    Since Python's `not` keyword cannot be overloaded, use this function
    for Boolean negation in model equations.
    
    Parameters
    ----------
    a : Expr or SymbolicVar or bool
        Boolean operand
    
    Returns
    -------
    Expr
        Boolean expression representing `not a`
    
    Example
    -------
    >>> @model
    ... class Inverter:
    ...     input_signal = var(dtype=DType.BOOLEAN, input=True)
    ...     output_signal = var(dtype=DType.BOOLEAN, output=True)
    ...
    ...     def equations(m):
    ...         yield m.output_signal == not_(m.input_signal)
    """
    return Expr(ExprKind.NOT, (_to_expr(a),))


# =============================================================================
# Conditional expression: if_then_else
# =============================================================================


@beartype
def if_then_else(condition: Any, then_expr: Any, else_expr: Any) -> Expr:
    """
    Conditional expression: if condition then then_expr else else_expr.
    
    This is the Modelica if-expression (MLS 3.6.5). Unlike if-statements,
    if-expressions always return a value and both branches must be provided.
    
    Parameters
    ----------
    condition : Expr or SymbolicVar or bool
        Boolean condition
    then_expr : Expr or SymbolicVar or numeric
        Value if condition is True
    else_expr : Expr or SymbolicVar or numeric
        Value if condition is False
    
    Returns
    -------
    Expr
        Conditional expression
    
    Example
    -------
    >>> @model
    ... class SaturatedGain:
    ...     u = var(input=True)
    ...     y = var(output=True)
    ...     K = var(2.0, parameter=True)
    ...     limit = var(10.0, parameter=True)
    ...
    ...     def equations(m):
    ...         raw = m.K * m.u
    ...         yield m.y == if_then_else(
    ...             raw > m.limit,
    ...             m.limit,
    ...             if_then_else(raw < -m.limit, -m.limit, raw)
    ...         )
    
    Notes
    -----
    For smooth simulation, consider using smooth conditional functions
    like `smooth_if` (not yet implemented) to avoid discontinuities.
    """
    return Expr(
        ExprKind.IF_THEN_ELSE,
        (_to_expr(condition), _to_expr(then_expr), _to_expr(else_expr))
    )


# =============================================================================
# Algorithm section support
# =============================================================================


class AlgorithmVar:
    """
    Local variable for use in algorithm sections.
    
    Algorithm sections can define local variables that exist only within
    the algorithm block. These are created using the `local()` function.
    
    Example
    -------
    >>> def algorithm(m):  # doctest: +SKIP
    ...     temp = local("temp")
    ...     yield temp @ (m.x * 2)
    ...     yield m.y @ (temp + 1)
    """
    
    def __init__(self, name: str):
        self._name = name
        self._expr = Expr(ExprKind.VARIABLE, name=name)
    
    @property
    def name(self) -> str:
        return self._name
    
    def __repr__(self) -> str:
        return f"local({self._name})"
    
    # Assignment operator
    def __matmul__(self, other: Any) -> Assignment:
        """Create an assignment: local_var @ expr"""
        return Assignment(target=self._name, expr=_to_expr(other), is_local=True)
    
    # Arithmetic operators - return Expr
    def __add__(self, other: Any) -> Expr:
        return self._expr + other

    def __radd__(self, other: Any) -> Expr:
        return _to_expr(other) + self._expr

    def __sub__(self, other: Any) -> Expr:
        return self._expr - other

    def __rsub__(self, other: Any) -> Expr:
        return _to_expr(other) - self._expr

    def __mul__(self, other: Any) -> Expr:
        return self._expr * other

    def __rmul__(self, other: Any) -> Expr:
        return _to_expr(other) * self._expr

    def __truediv__(self, other: Any) -> Expr:
        return self._expr / other

    def __rtruediv__(self, other: Any) -> Expr:
        return _to_expr(other) / self._expr

    def __neg__(self) -> Expr:
        return -self._expr

    def __pow__(self, other: Any) -> Expr:
        return self._expr ** other
    
    # Relational operators
    def __lt__(self, other: Any) -> Expr:
        return self._expr < other
    
    def __le__(self, other: Any) -> Expr:
        return self._expr <= other
    
    def __gt__(self, other: Any) -> Expr:
        return self._expr > other
    
    def __ge__(self, other: Any) -> Expr:
        return self._expr >= other


@beartype
def local(name: str) -> AlgorithmVar:
    """
    Create a local variable for use in algorithm sections.
    
    Local variables are temporary variables that exist only within
    an algorithm block. They are useful for storing intermediate
    calculations.
    
    Parameters
    ----------
    name : str
        Name of the local variable (for debugging/display)
    
    Returns
    -------
    AlgorithmVar
        A local variable that can be assigned and used in expressions
    
    Example
    -------
    >>> @model
    ... class Controller:
    ...     u = var(input=True)
    ...     y = var(output=True)
    ...     limit = var(10.0, parameter=True)
    ...
    ...     def algorithm(m):
    ...         temp = local("temp")
    ...         yield temp @ (m.u * 2)
    ...         yield m.y @ if_then_else(temp > m.limit, m.limit, temp)
    """
    return AlgorithmVar(name)


@beartype  
def assign(target: Union[SymbolicVar, AlgorithmVar, str], value: Any) -> Assignment:
    """
    Create an assignment for algorithm sections.
    
    This is an alternative to the @ operator for creating assignments.
    
    Parameters
    ----------
    target : SymbolicVar, AlgorithmVar, or str
        The variable to assign to
    value : Any
        The value to assign (will be converted to Expr)
    
    Returns
    -------
    Assignment
        An assignment that can be yielded in algorithm()
    
    Example
    -------
    >>> def algorithm(m):
    ...     yield assign(m.y, m.x * 2)
    ...     # Equivalent to: yield m.y @ (m.x * 2)
    """
    if isinstance(target, SymbolicVar):
        return Assignment(target=target._name, expr=_to_expr(value), is_local=False)
    elif isinstance(target, AlgorithmVar):
        return Assignment(target=target._name, expr=_to_expr(value), is_local=True)
    elif isinstance(target, str):
        return Assignment(target=target, expr=_to_expr(value), is_local=True)
    else:
        raise TypeError(f"Cannot assign to {type(target)}")


# =============================================================================
# Submodel proxy for dot-access
# =============================================================================


class SubmodelProxy:
    """Proxy for accessing submodel variables with dot notation."""

    def __init__(self, name: str, instance: ModelInstance, parent: ModelInstance):
        self._name = name
        self._instance = instance
        self._parent = parent

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        # Access submodel's symbolic variable with prefixed name
        full_name = f"{self._name}.{attr}"
        if full_name in self._parent._sym_vars:
            return self._parent._sym_vars[full_name]
        raise AttributeError(f"Submodel '{self._name}' has no attribute '{attr}'")


# =============================================================================
# Model metadata
# =============================================================================


@dataclass
class ModelMetadata:
    """Metadata extracted from a model class by the @model decorator."""

    variables: Dict[str, Var] = field(default_factory=dict)
    submodels: Dict[str, SubmodelField] = field(default_factory=dict)


# =============================================================================
# Model instance (runtime)
# =============================================================================


class ModelInstance:
    """
    Runtime instance of a model for building equations.

    Created when a @model decorated class is instantiated.
    """

    _dsl_metadata: ModelMetadata  # Set by @model decorator

    def __init__(self, model_class: Type[Any], name: str = ""):
        self._model_class = model_class
        self._name = name or model_class.__name__
        self._metadata: ModelMetadata = model_class._dsl_metadata

        # Symbolic storage - unified
        self._sym_vars: Dict[str, SymbolicVar] = {}
        self._submodels: Dict[str, ModelInstance] = {}

        self._t = TimeVar()  # Time variable

        self._create_symbols()

    def _create_symbols(self) -> None:
        """Create symbolic variables for all fields."""
        md = self._metadata

        # Create symbols for all variables
        for name, v in md.variables.items():
            self._sym_vars[name] = SymbolicVar(name, v, self)

        # Submodels - create nested instances and flatten their symbols
        for name, subfld in md.submodels.items():
            # Instantiate the submodel class (which is already decorated with @model)
            # This creates a ModelClass instance with proper equations() method
            sub_instance = subfld.model_class(name=name)
            self._submodels[name] = sub_instance

            # Flatten submodel symbols into parent with prefixed names
            for var_name, sym_var in sub_instance._sym_vars.items():
                full_name = f"{name}.{var_name}"
                self._sym_vars[full_name] = SymbolicVar(full_name, sym_var._var, self)

    def __getattr__(self, name: str) -> Any:
        """Provide access to symbolic variables and submodels."""
        if name.startswith("_"):
            raise AttributeError(name)

        # Check submodels first
        if name in self._submodels:
            return SubmodelProxy(name, self._submodels[name], self)

        # Check variables (unified storage)
        if name in self._sym_vars:
            return self._sym_vars[name]

        raise AttributeError(f"'{self._model_class.__name__}' has no attribute '{name}'")

    @property
    def t(self) -> TimeVar:
        """Time variable."""
        return self._t

    def equations(m) -> Generator[Equation, None, None]:
        """
        Override this method to define model equations.

        Use 'm' as the first parameter (model namespace) instead of 'self'.
        Yield equations using the == operator.

        Example
        -------
        >>> def equations(m):
        ...     yield der(m.theta) == m.omega
        ...     yield der(m.omega) == -m.g / m.l * sin(m.theta)
        ...     yield m.x == m.l * sin(m.theta)  # Output equation
        """
        return
        yield  # Make this a generator

    def algorithm(m) -> Generator[Assignment, None, None]:
        """
        Override this method to define algorithm sections.
        
        Algorithm sections contain imperative assignments that are executed
        in order, unlike equations which are declarative. Use @ for assignments.
        
        Algorithm sections are useful for:
        - Computing intermediate values
        - Implementing control logic with if/else
        - Breaking complex expressions into readable steps
        
        Example
        -------
        >>> def algorithm(m):
        ...     temp = local("temp")
        ...     yield temp @ (m.u * 2)
        ...     yield m.y @ if_then_else(temp > m.limit, m.limit, temp)
        
        Notes
        -----
        In Modelica, algorithm sections use := for assignment (vs == for equations).
        In this DSL, we use @ since := is Python's walrus operator and @= can't
        be used with yield (augmented assignment is a statement, not expression).
        """
        return
        yield  # Make this a generator

    def flatten(self, expand_arrays: bool = True) -> "FlatModel":
        """
        Flatten the model into a backend-agnostic representation.
        
        This method performs automatic variable classification:
        1. Variables with parameter=True → parameters
        2. Variables with constant=True → constants (treated as parameters)
        3. Variables with discrete=True → discrete (piecewise constant)
        4. Variables with input=True → inputs
        5. Variables with output=True → outputs
        6. Variables whose der() appears in equations → states
        7. Remaining variables → algebraic

        Parameters
        ----------
        expand_arrays : bool, default=True
            If True, array equations like `der(pos) == vel` are expanded to
            scalar equations: `der(pos[0]) == vel[0]`, `der(pos[1]) == vel[1]`, etc.
            This is suitable for CasADi SX backend.
            
            If False, array equations are kept as-is with the base variable name.
            The derivative_equations dict will have entries like `{'pos': vel_expr}`
            where `vel_expr` represents the whole array expression.
            This is suitable for CasADi MX backend which operates on matrices.

        Returns
        -------
        FlatModel
            A flattened model with all variables and equations.
        """
        md = self._metadata

        # Collect equations by iterating the generator
        # Include both this model's equations AND submodel equations
        raw_equations = self.equations()
        equations: List[Equation] = []
        array_equations: List[ArrayEquation] = []
        
        for eq in raw_equations:
            if isinstance(eq, ArrayEquation):
                if expand_arrays:
                    # Expand array equation to scalar equations
                    equations.extend(eq.expand())
                else:
                    # Keep array equations separate for MX backend
                    array_equations.append(eq)
            elif isinstance(eq, Equation):
                equations.append(eq)
            else:
                raise TypeError(f"Expected Equation or ArrayEquation, got {type(eq)}")
        
        # Collect equations from submodels (with prefixed variable names)
        for sub_name, sub_instance in self._submodels.items():
            for eq in sub_instance.equations():
                if isinstance(eq, Equation):
                    # Prefix variable names in the equation
                    prefixed_eq = eq._prefix_names(sub_name)
                    equations.append(prefixed_eq)
                elif isinstance(eq, ArrayEquation):
                    if expand_arrays:
                        for scalar_eq in eq.expand():
                            prefixed_eq = scalar_eq._prefix_names(sub_name)
                            equations.append(prefixed_eq)
                    else:
                        # TODO: prefix array equations
                        array_equations.append(eq)

        # Collect algorithm assignments
        raw_algorithm = self.algorithm()
        algorithm_assignments: List[Assignment] = []
        algorithm_locals: List[str] = []
        
        for assign in raw_algorithm:
            if isinstance(assign, Assignment):
                algorithm_assignments.append(assign)
                if assign.is_local and assign.target not in algorithm_locals:
                    algorithm_locals.append(assign.target)
            else:
                raise TypeError(f"Expected Assignment in algorithm(), got {type(assign)}")

        # Find all derivatives (der(x)) used in equations to identify states
        derivatives_used: set[str] = set()
        for eq in equations:
            derivatives_used.update(_find_derivatives(eq.lhs))
            derivatives_used.update(_find_derivatives(eq.rhs))
        
        # Also check array equations for derivatives (when not expanding)
        # Array equations like der(pos) == vel mark 'pos' as a state
        array_state_names: set[str] = set()
        for arr_eq in array_equations:
            if arr_eq.is_derivative:
                # The base variable name is a state
                base_name = arr_eq.lhs_var.base_name
                array_state_names.add(base_name)

        # Separate derivative equations from algebraic and output
        derivative_equations: Dict[str, Expr] = {}
        # For non-expanded arrays, store the whole array expression
        # Key is base variable name, value is Expr for RHS array
        array_derivative_equations: Dict[str, Any] = {}  # name -> (shape, rhs_var)
        algebraic_equations: List[Equation] = []
        output_equations_map: Dict[str, Expr] = {}

        # Classify variables from new unified Var storage
        state_names: List[str] = []
        state_vars: Dict[str, Var] = {}
        state_defaults: Dict[str, Any] = {}
        
        param_names: List[str] = []
        param_vars: Dict[str, Var] = {}
        param_defaults: Dict[str, Any] = {}
        
        input_names: List[str] = []
        input_vars: Dict[str, Var] = {}
        input_defaults: Dict[str, Any] = {}
        
        discrete_names: List[str] = []
        discrete_vars: Dict[str, Var] = {}
        discrete_defaults: Dict[str, Any] = {}
        
        output_names: List[str] = []
        output_vars: Dict[str, Var] = {}
        
        algebraic_names: List[str] = []
        algebraic_vars: Dict[str, Var] = {}
        
        # First pass: classify based on flags
        output_name_set: set[str] = set()
        for name, v in md.variables.items():
            if v.constant or v.parameter:
                v.kind = VarKind.CONSTANT if v.constant else VarKind.PARAMETER
                param_names.append(name)
                param_vars[name] = v
                if v.default is not None:
                    param_defaults[name] = v.default
                elif v.start is not None:
                    param_defaults[name] = v.start
            elif v.input:
                v.kind = VarKind.INPUT
                input_names.append(name)
                input_vars[name] = v
                val = v.get_initial_value()
                if val is not None:
                    input_defaults[name] = val
            elif v.discrete:
                v.kind = VarKind.DISCRETE
                discrete_names.append(name)
                discrete_vars[name] = v
                val = v.get_initial_value()
                if val is not None:
                    discrete_defaults[name] = val
            elif v.output:
                v.kind = VarKind.OUTPUT
                output_names.append(name)
                output_vars[name] = v
                output_name_set.add(name)
            elif (name in derivatives_used 
                  or _is_array_state(name, v.shape, derivatives_used)
                  or name in array_state_names):
                # der(name) is used → state variable
                # For arrays: check if any der(name[i,j,...]) is used, or
                # if the whole array der(name) appears in array_equations
                v.kind = VarKind.STATE
                state_names.append(name)
                state_vars[name] = v
                val = v.get_initial_value()
                if val is not None:
                    state_defaults[name] = val
            else:
                # No der(), not flagged → algebraic
                v.kind = VarKind.ALGEBRAIC
                algebraic_names.append(name)
                algebraic_vars[name] = v

        # Classify scalar equations
        for eq in equations:
            if eq.is_derivative and eq.var_name:
                derivative_equations[eq.var_name] = eq.rhs
            elif eq.lhs.kind == ExprKind.VARIABLE and eq.lhs.name in output_name_set:
                # This is an output equation (output_var = expr)
                output_equations_map[eq.lhs.name] = eq.rhs
            else:
                algebraic_equations.append(eq)
        
        # Classify array equations (when expand_arrays=False)
        for arr_eq in array_equations:
            if arr_eq.is_derivative:
                # Store the array derivative equation
                # Key is base variable name, value is the RHS SymbolicVar
                base_name = arr_eq.lhs_var.base_name
                array_derivative_equations[base_name] = {
                    'shape': arr_eq.lhs_var.shape,
                    'rhs': arr_eq.rhs,  # The RHS SymbolicVar (e.g., vel for der(pos) == vel)
                }
            # TODO: Handle non-derivative array equations (output, algebraic)

        # Add submodel variables (flattened with prefixed names)
        # We need to classify them based on der() usage in submodel equations
        for sub_name, sub in self._submodels.items():
            # Find derivatives in submodel equations (with prefixed names)
            sub_derivatives: set[str] = set()
            for eq in sub.equations():
                if isinstance(eq, Equation):
                    # Find derivatives and prefix them
                    for der_name in _find_derivatives(eq.lhs):
                        sub_derivatives.add(f"{sub_name}.{der_name}")
                    for der_name in _find_derivatives(eq.rhs):
                        sub_derivatives.add(f"{sub_name}.{der_name}")
            
            for name, v in sub._metadata.variables.items():
                full_name = f"{sub_name}.{name}"
                
                # Skip if already processed
                if (full_name in state_vars or full_name in input_vars or 
                    full_name in output_vars or full_name in param_vars or
                    full_name in algebraic_vars):
                    continue
                
                # Create a copy of the Var with the full name
                sub_v = Var(
                    dtype=v.dtype,
                    default=v.default, shape=v.shape, unit=v.unit, desc=v.desc,
                    start=v.start, fixed=v.fixed, min=v.min, max=v.max,
                    nominal=v.nominal,
                    parameter=v.parameter, discrete=v.discrete,
                    input=v.input, output=v.output, constant=v.constant,
                    protected=v.protected,
                    name=full_name
                )
                
                # Classify based on flags and der() usage
                if v.constant or v.parameter:
                    sub_v.kind = VarKind.CONSTANT if v.constant else VarKind.PARAMETER
                    param_names.append(full_name)
                    param_vars[full_name] = sub_v
                    if v.default is not None:
                        param_defaults[full_name] = v.default
                    elif v.start is not None:
                        param_defaults[full_name] = v.start
                elif v.input:
                    sub_v.kind = VarKind.INPUT
                    input_names.append(full_name)
                    input_vars[full_name] = sub_v
                    val = sub_v.get_initial_value()
                    if val is not None:
                        input_defaults[full_name] = val
                elif v.discrete:
                    sub_v.kind = VarKind.DISCRETE
                    discrete_names.append(full_name)
                    discrete_vars[full_name] = sub_v
                    val = sub_v.get_initial_value()
                    if val is not None:
                        discrete_defaults[full_name] = val
                elif v.output:
                    sub_v.kind = VarKind.OUTPUT
                    output_names.append(full_name)
                    output_vars[full_name] = sub_v
                elif full_name in sub_derivatives or full_name in derivatives_used:
                    # der() is used -> state variable
                    sub_v.kind = VarKind.STATE
                    state_names.append(full_name)
                    state_vars[full_name] = sub_v
                    val = sub_v.get_initial_value()
                    if val is not None:
                        state_defaults[full_name] = val
                else:
                    # Algebraic
                    sub_v.kind = VarKind.ALGEBRAIC
                    algebraic_names.append(full_name)
                    algebraic_vars[full_name] = sub_v

        return FlatModel(
            name=self._name,
            state_names=state_names,
            input_names=input_names,
            output_names=output_names,
            param_names=param_names,
            discrete_names=discrete_names,
            algebraic_names=algebraic_names,
            state_vars=state_vars,
            input_vars=input_vars,
            output_vars=output_vars,
            param_vars=param_vars,
            discrete_vars=discrete_vars,
            algebraic_vars=algebraic_vars,
            derivative_equations=derivative_equations,
            array_derivative_equations=array_derivative_equations,
            output_equations=output_equations_map,
            algebraic_equations=algebraic_equations,
            state_defaults=state_defaults,
            input_defaults=input_defaults,
            discrete_defaults=discrete_defaults,
            param_defaults=param_defaults,
            algorithm_assignments=algorithm_assignments,
            algorithm_locals=algorithm_locals,
            expand_arrays=expand_arrays,
        )


# =============================================================================
# @model decorator
# =============================================================================
# NOTE: The @model decorator and helper functions MUST have @beartype.


@beartype
def model(cls: Type[Any]) -> Type[Any]:
    """
    Decorator to convert a class into a Cyecca model.

    Processes field descriptors (var, submodel) and sets up the class
    for equation-based modeling with automatic variable classification.

    Example
    -------
    >>> @model
    ... class Pendulum:
    ...     g = var(9.81, parameter=True)
    ...     l = var(1.0, parameter=True)
    ...     theta = var(start=0.5)
    ...     omega = var()
    ...     x = var(output=True)
    ...
    ...     def equations(m):
    ...         yield der(m.theta) == m.omega
    ...         yield der(m.omega) == -m.g / m.l * sin(m.theta)
    ...         yield m.x == m.l * sin(m.theta)
    """
    metadata = ModelMetadata()

    # Process class attributes
    for name, value in list(vars(cls).items()):
        if isinstance(value, Var):
            value.name = name
            metadata.variables[name] = value
        elif isinstance(value, SubmodelField):
            value.name = name
            metadata.submodels[name] = value

    # Store metadata on the class
    cls._dsl_metadata = metadata

    # Get original equations method
    # NOTE: In Modelica, all equations (including output definitions) are in one
    # equation section. There is no separate output_equations() - that would be
    # non-conformant. Output variables are just vars with output=True flag.
    original_equations = getattr(cls, "equations", None)
    original_algorithm = getattr(cls, "algorithm", None)
    
    # Deprecation check: warn if user defines output_equations (non-Modelica)
    if hasattr(cls, "output_equations"):
        import warnings
        warnings.warn(
            f"Model '{cls.__name__}' defines output_equations(). This is deprecated "
            "and non-Modelica-conformant. In Modelica, all equations (including output "
            "definitions) go in the equations() method. The output prefix just marks "
            "which variables are exposed externally.",
            DeprecationWarning,
            stacklevel=2,
        )

    class ModelClass(ModelInstance):
        __doc__ = cls.__doc__
        __name__ = cls.__name__
        __qualname__ = cls.__qualname__
        __module__ = cls.__module__

        def __init__(self, name: str = ""):
            super().__init__(cls, name=name or cls.__name__)

        def equations(self) -> Generator[Equation, None, None]:
            if original_equations is not None:
                yield from original_equations(self)
        
        def algorithm(self) -> Generator[Assignment, None, None]:
            if original_algorithm is not None:
                yield from original_algorithm(self)

    # Copy the metadata reference
    ModelClass._dsl_metadata = metadata

    return ModelClass


# Alias for type hints
Model = ModelInstance


# =============================================================================
# @function decorator - Modelica functions (Ch. 12)
# =============================================================================
# In Modelica, a function is a restricted class (like block). It is essentially
# a model with:
# - Only algorithm sections (no equations)
# - All public non-parameter variables must be input or output
# - No states (no der())
# - Evaluated once when called, not continuously simulated


@dataclass
class FunctionMetadata:
    """Metadata for a function, extracted from FlatModel."""
    name: str
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    protected_names: List[str] = field(default_factory=list)
    algorithm_assignments: List[Assignment] = field(default_factory=list)
    algorithm_locals: List[str] = field(default_factory=list)


@beartype
def function(cls: Type[Any]) -> Type[Any]:
    """
    Decorator to convert a class into a Cyecca function.
    
    A function is a restricted model (Modelica Ch. 12) that:
    - Uses only algorithm sections (no equations)
    - All public non-parameter variables must be input or output
    - Cannot have states (no der())
    - Is evaluated once when called, not continuously simulated
    
    This is similar to @block but even more restricted - blocks can have
    equations, functions can only have algorithms.
    
    Example
    -------
    >>> @function
    ... class Saturate:
    ...     '''Saturate a value between lo and hi.'''
    ...     x = var(input=True)
    ...     lo = var(input=True)
    ...     hi = var(input=True)
    ...     y = var(output=True)
    ...
    ...     def algorithm(f):
    ...         yield f.y @ if_then_else(
    ...             f.x < f.lo,
    ...             f.lo,
    ...             if_then_else(f.x > f.hi, f.hi, f.x)
    ...         )
    
    >>> @function
    ... class Quadratic:
    ...     '''Solve quadratic equation ax^2 + bx + c = 0.'''
    ...     a = var(input=True)
    ...     b = var(input=True)
    ...     c = var(input=True)
    ...     x1 = var(output=True)
    ...     x2 = var(output=True)
    ...     
    ...     # Protected variable for intermediate calculation
    ...     d = var(protected=True)
    ...     
    ...     def algorithm(f):
    ...         yield f.d @ sqrt(f.b**2 - 4*f.a*f.c)
    ...         yield f.x1 @ (-f.b + f.d) / (2*f.a)
    ...         yield f.x2 @ (-f.b - f.d) / (2*f.a)
    
    Notes
    -----
    Unlike models, functions:
    - Use algorithm sections (imperative), not equations (declarative)
    - Are called/evaluated once, not continuously simulated
    - Cannot have states (no der())
    - Can be called from model equations (future feature)
    
    Raises
    ------
    TypeError
        If a public non-parameter variable lacks input or output prefix.
        If equations() method is defined (functions use algorithm only).
    """
    # Validate function constraints
    errors = []
    
    # Check for equations method (not allowed in functions)
    if hasattr(cls, 'equations'):
        # Check if it's overridden (not just inherited empty generator)
        equations_method = getattr(cls, 'equations')
        # Try to detect if it's a real implementation
        import inspect
        source_lines = []
        try:
            source_lines = inspect.getsourcelines(equations_method)[0]
        except (OSError, TypeError):
            pass
        # If there's more than just "def equations(m): pass" or similar
        if len(source_lines) > 2:
            errors.append(
                f"Functions cannot have equations() method - use algorithm() only"
            )
    
    # Validate all public non-parameter variables have input/output
    for name, value in vars(cls).items():
        if isinstance(value, Var):
            # Skip protected variables
            if value.protected:
                continue
            # Skip parameters and constants
            if value.parameter or value.constant:
                continue
            # Public non-parameter variable must have input or output
            if not value.input and not value.output:
                errors.append(
                    f"  - '{name}': must have input=True or output=True "
                    f"(or use protected=True for intermediate variables)"
                )
    
    if errors:
        error_msg = (
            f"Function '{cls.__name__}' violates Modelica function constraints.\n"
            + "\n".join(errors)
        )
        raise TypeError(error_msg)
    
    # Check algorithm method exists
    if not hasattr(cls, 'algorithm'):
        raise TypeError(
            f"Function '{cls.__name__}' must have an algorithm() method"
        )
    
    # Use the model decorator to do the actual work
    model_cls = model(cls)
    
    # Mark it as a function
    model_cls._is_function = True
    
    # Add helper method to get function metadata
    def get_function_metadata(self) -> FunctionMetadata:
        """Get function-specific metadata."""
        flat = self.flatten()
        
        # Collect protected variable names
        protected_names = [
            name for name, v in self._metadata.variables.items()
            if v.protected
        ]
        
        return FunctionMetadata(
            name=flat.name,
            input_names=flat.input_names,
            output_names=flat.output_names,
            param_names=flat.param_names,
            protected_names=protected_names,
            algorithm_assignments=flat.algorithm_assignments,
            algorithm_locals=flat.algorithm_locals,
        )
    
    model_cls.get_function_metadata = get_function_metadata
    
    return model_cls


# =============================================================================
# @block decorator - Modelica block (signal-flow)
# =============================================================================


@beartype
def block(cls: Type[Any]) -> Type[Any]:
    """
    Decorator to convert a class into a Cyecca block.
    
    A block is a specialized model for signal-flow (causal) modeling.
    In Modelica, blocks have a key restriction: all public (non-protected)
    variables that are not parameters must have input or output prefix.
    
    This is enforced at decoration time to catch errors early.
    
    Use blocks for:
    - Control systems (PID controllers, filters, state machines)
    - Signal processing (gains, limiters, delays)
    - Any causal input-output system
    
    Use models (not blocks) for:
    - Physical systems with energy exchange (electrical, mechanical)
    - Acausal connections (connectors with flow variables)
    
    Example
    -------
    >>> @block
    ... class PIDController:
    ...     # Public interface (must have input/output)
    ...     setpoint = var(input=True)
    ...     measurement = var(input=True)
    ...     command = var(output=True)
    ...     
    ...     # Parameters (allowed without input/output)
    ...     Kp = var(1.0, parameter=True)
    ...     Ki = var(0.1, parameter=True)
    ...     
    ...     # Protected variables (internal, no input/output required)
    ...     integral = var(start=0.0, protected=True)
    ...     
    ...     def equations(m):
    ...         error = m.setpoint - m.measurement
    ...         yield der(m.integral) == error
    ...         yield m.command == m.Kp * error + m.Ki * m.integral
    
    Raises
    ------
    TypeError
        If a public non-parameter variable lacks input or output prefix.
    """
    # Validate block constraints before creating the model
    errors = []
    for name, value in vars(cls).items():
        if isinstance(value, Var):
            # Skip protected variables - they don't need input/output
            if value.protected:
                continue
            # Skip parameters and constants - they don't need input/output
            if value.parameter or value.constant:
                continue
            # Public non-parameter variable must have input or output
            if not value.input and not value.output:
                errors.append(
                    f"  - '{name}': public variable must have input=True or output=True "
                    f"(or use protected=True for internal variables)"
                )
    
    if errors:
        error_msg = (
            f"Block '{cls.__name__}' violates Modelica block constraints.\n"
            f"All public non-parameter variables must have input or output prefix:\n"
            + "\n".join(errors)
        )
        raise TypeError(error_msg)
    
    # Use the model decorator to do the actual work
    model_cls = model(cls)
    
    # Mark it as a block for potential future use
    model_cls._is_block = True
    
    return model_cls


# =============================================================================
# Flat Model - Backend-agnostic representation
# =============================================================================


@dataclass
class FlatModel:
    """
    Flattened model representation - the output of the DSL.
    
    This is a backend-agnostic representation of the model that contains:
    - All variables (states, inputs, outputs, params) with metadata
    - All equations as expression trees
    - Default values for initialization
    
    Variable Classification (Modelica-conformant)
    ---------------------------------------------
    In Modelica, `input` and `output` are just **prefixes** on variables that
    indicate causality (how the variable interfaces with the outside world).
    They are NOT separate equation categories.
    
    - parameter=True → parameter (constant during simulation)
    - discrete=True → discrete (piecewise constant, changes at events)
    - input=True → input (value provided externally)
    - output=True → output (value computed internally, exposed externally)
    - der(var) in equations → state
    - otherwise → algebraic
    
    All equations come from a single equations() method. The `output_equations`
    field is populated by extracting equations of the form `output_var == expr`
    for backend convenience.
    
    Backends (CasADi, JAX, etc.) compile this into executable functions.
    """
    name: str
    
    # Variable lists (ordered)
    state_names: List[str]
    input_names: List[str]
    output_names: List[str]
    param_names: List[str]
    discrete_names: List[str]
    algebraic_names: List[str]
    
    # Variable metadata (using unified Var type)
    state_vars: Dict[str, Var]
    input_vars: Dict[str, Var]
    output_vars: Dict[str, Var]
    param_vars: Dict[str, Var]
    discrete_vars: Dict[str, Var]
    algebraic_vars: Dict[str, Var]
    
    # Equations
    # NOTE: In Modelica, all equations are in one section. The separation here
    # is for backend convenience only. Output equations are extracted from
    # equations that define output variables (var with output=True).
    derivative_equations: Dict[str, Expr]  # state_name -> rhs expression for der(state)
    output_equations: Dict[str, Expr]      # output_name -> rhs (extracted from equations())
    algebraic_equations: List[Equation]    # 0 = f(x, z) equations
    
    # Default values
    state_defaults: Dict[str, Any]
    input_defaults: Dict[str, Any]
    discrete_defaults: Dict[str, Any]
    param_defaults: Dict[str, Any]
    
    # Array derivative equations (when expand_arrays=False)
    # For CasADi MX backend: keeps array structure for efficient matrix operations
    # Key is base variable name (e.g., 'pos'), value is {'shape': (3,), 'rhs': SymbolicVar}
    array_derivative_equations: Dict[str, Any] = field(default_factory=dict)
    
    # Algorithm section
    # Ordered list of assignments from algorithm() method
    algorithm_assignments: List[Assignment] = field(default_factory=list)
    # Local variables declared in algorithm section
    algorithm_locals: List[str] = field(default_factory=list)
    
    # Flattening mode
    expand_arrays: bool = True  # If False, array equations are kept as-is for MX backend
    
    def __repr__(self) -> str:
        parts = [f"'{self.name}'"]
        if self.state_names:
            parts.append(f"states={self.state_names}")
        if self.discrete_names:
            parts.append(f"discrete={self.discrete_names}")
        if self.input_names:
            parts.append(f"inputs={self.input_names}")
        if self.output_names:
            parts.append(f"outputs={self.output_names}")
        if self.param_names:
            parts.append(f"params={self.param_names}")
        return f"FlatModel({', '.join(parts)})"
