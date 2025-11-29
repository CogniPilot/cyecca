"""
Equation context and when-clause support for the Cyecca DSL.

This module provides the infrastructure for collecting equations:
- EquationContext: Thread-local context for collecting equations via side-effects
- WhenContext: Context manager for building when-clauses
- @equations decorator: Mark methods as equation blocks
- when(): Create when-clauses for event handling
- reinit(): Reinitialize state variables at events

================================================================================
PROTOTYPE MODE - API IS IN FLUX
================================================================================

This DSL is in active prototype development. The API may change significantly
between versions. Do NOT maintain backward compatibility - iterate rapidly.

================================================================================
DESIGN PRINCIPLES - DO NOT REMOVE OR IGNORE
================================================================================

1. MODELICA CONFORMANCE: This DSL conforms to Modelica Language Spec v3.7-dev.
2. TYPE SAFETY: All functions MUST use beartype for runtime type checking.
3. SELF-CONTAINED: NO external compute libraries (CasADi, JAX) in core DSL.
4. IMMUTABILITY: Prefer immutable data structures where possible.
5. EXPLICIT > IMPLICIT: All behavior should be explicit and documented.

================================================================================
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from beartype import beartype

from cyecca.dsl.equations import IfEquation, IfEquationBranch, Reinit, WhenClause
from cyecca.dsl.expr import Expr, ExprLike, to_expr
from cyecca.dsl.instance import SubmodelProxy
from cyecca.dsl.variables import SymbolicVar

if TYPE_CHECKING:
    from cyecca.dsl.equations import ArrayEquation, Equation
    from cyecca.dsl.instance import ModelInstance


# Thread-local storage for equation context stack
_equation_context = threading.local()


class EquationContext:
    """
    Context for collecting equations via side effects.

    When active, expressions using == (like `der(m.x) == m.y`) automatically
    register themselves as equations instead of returning Equation objects.

    This enables the clean @equations decorator syntax without yield:

        @equations
        def _(m):
            der(m.theta) == m.omega
            der(m.omega) == -9.81 * sin(m.theta)
    """

    def __init__(self) -> None:
        self.equations: List[Union["Equation", "ArrayEquation", "WhenClause"]] = []
        self.assignments: List["Assignment"] = []  # For @algorithm blocks
        self._current_when: Optional["WhenContext"] = None
        self._building_expr_depth: int = 0  # Track if we're building a subexpression

    def add_equation(self, eq: Union["Equation", "ArrayEquation"]) -> None:
        """Add an equation to the current context."""
        if self._current_when is not None:
            # If we're inside a when-clause, add to that instead
            self._current_when.body.append(eq)
        else:
            self.equations.append(eq)

    def add_when_clause(self, wc: "WhenClause") -> None:
        """Add a completed when-clause to the context."""
        self.equations.append(wc)

    @property
    def is_building_expr(self) -> bool:
        """Check if we're currently building a subexpression."""
        return self._building_expr_depth > 0

    def enter_expr(self) -> None:
        """Enter subexpression building mode."""
        self._building_expr_depth += 1

    def exit_expr(self) -> None:
        """Exit subexpression building mode."""
        self._building_expr_depth -= 1

    def add_assignment(self, assign: "Assignment") -> None:
        """Add an assignment to the current context (for @algorithm blocks)."""
        self.assignments.append(assign)

    def add_reinit(self, r: "Reinit") -> None:
        """Add a reinit statement (must be inside a when-clause)."""
        if self._current_when is None:
            raise RuntimeError("reinit() can only be used inside a when() block")
        self._current_when.body.append(r)

    def enter_when(self, when_ctx: "WhenContext") -> None:
        """Enter a when-clause context."""
        self._current_when = when_ctx

    def exit_when(self) -> None:
        """Exit a when-clause context and register it."""
        if self._current_when is not None:
            wc = self._current_when.to_when_clause()
            self.equations.append(wc)
            self._current_when = None


def get_current_equation_context() -> Optional[EquationContext]:
    """Get the current equation context, or None if not in an @equations block."""
    if not hasattr(_equation_context, "stack") or not _equation_context.stack:
        return None
    return _equation_context.stack[-1]


def push_equation_context() -> EquationContext:
    """Push a new equation context onto the stack."""
    if not hasattr(_equation_context, "stack"):
        _equation_context.stack = []
    ctx = EquationContext()
    _equation_context.stack.append(ctx)
    return ctx


def pop_equation_context() -> EquationContext:
    """Pop the current equation context from the stack."""
    return _equation_context.stack.pop()


# Marker attribute for @equations decorated methods
_EQUATIONS_MARKER = "_cyecca_equations_method"

# Track decorated method names to detect overwrites (name -> decorator type)
# This is keyed by the code object's filename and first line number of the class
_decorated_methods_registry: dict = {}


def _get_class_key() -> tuple:
    """Get a key identifying the current class being defined."""
    import inspect

    # Walk up the stack to find the class definition frame
    for frame_info in inspect.stack():
        # Look for a frame that's defining a class (has __qualname__ being built)
        if "__qualname__" in frame_info.frame.f_locals:
            return (frame_info.filename, frame_info.frame.f_locals.get("__qualname__", ""))
    return None


def _check_and_register_decorated_method(func: Callable, decorator_name: str) -> None:
    """Check if this method name was already decorated and warn if so."""
    import warnings

    func_name = func.__name__
    class_key = _get_class_key()

    if class_key is None:
        return  # Not in a class definition, skip tracking

    registry_key = (class_key, func_name)

    if registry_key in _decorated_methods_registry:
        prev_decorator = _decorated_methods_registry[registry_key]
        warnings.warn(
            f"Method '{func_name}' was already decorated with @{prev_decorator}, "
            f"but is now being decorated with @{decorator_name}. "
            f"The previous method will be overwritten! "
            f"Use unique method names for each decorated block.",
            UserWarning,
            stacklevel=4,  # Point to the actual decorator usage
        )

    _decorated_methods_registry[registry_key] = decorator_name


def equations(func: Callable) -> Callable:
    """
    Decorator to mark a method as an equations block.

    Methods decorated with @equations use side-effect based equation capture.
    Instead of yielding equations, simply write them as statements:

        @equations
        def _(m):
            der(m.theta) == m.omega
            der(m.omega) == -9.81 * sin(m.theta)
            m.x == sin(m.theta)

            with when(m.h < 0):
                reinit(m.v, -m.e * pre(m.v))

    The == operator auto-registers equations when inside an @equations block.
    The reinit() function auto-registers when inside a when() block.

    Parameters
    ----------
    func : Callable
        The equations method (conventionally named `_` since self isn't used)

    Returns
    -------
    Callable
        The decorated method with equation capture marker
    """
    _check_and_register_decorated_method(func, "equations")
    setattr(func, _EQUATIONS_MARKER, True)
    return func


def is_equations_method(func: Any) -> bool:
    """Check if a function is marked as an @equations method."""
    return getattr(func, _EQUATIONS_MARKER, False)


# Marker attribute for @initial_equations decorated methods
_INITIAL_EQUATIONS_MARKER = "_cyecca_initial_equations_method"


def initial_equations(func: Callable) -> Callable:
    """
    Decorator to mark a method as an initial equations block.

    Methods decorated with @initial_equations use side-effect based capture,
    just like @equations. Initial equations specify values at t=0.

        @initial_equations
        def _(m):
            m.x == 1.0
            m.v == 0.0

    Parameters
    ----------
    func : Callable
        The initial equations method

    Returns
    -------
    Callable
        The decorated method with initial equation capture marker
    """
    _check_and_register_decorated_method(func, "initial_equations")
    setattr(func, _INITIAL_EQUATIONS_MARKER, True)
    return func


def is_initial_equations_method(func: Any) -> bool:
    """Check if a function is marked as an @initial_equations method."""
    return getattr(func, _INITIAL_EQUATIONS_MARKER, False)


def execute_equations_method(
    func: Callable, model_instance: "ModelInstance"
) -> List[Union["Equation", "ArrayEquation", "WhenClause"]]:
    """
    Execute an @equations method and collect equations via context.

    Parameters
    ----------
    func : Callable
        The @equations decorated method
    model_instance : ModelInstance
        The model instance to pass to the method

    Returns
    -------
    List[Union[Equation, ArrayEquation, WhenClause]]
        The collected equations
    """
    ctx = push_equation_context()
    try:
        func(model_instance)
    finally:
        pop_equation_context()
    return ctx.equations


def execute_initial_equations_method(func: Callable, model_instance: "ModelInstance") -> List["Equation"]:
    """
    Execute an @initial_equations method and collect equations via context.

    Parameters
    ----------
    func : Callable
        The @initial_equations decorated method
    model_instance : ModelInstance
        The model instance to pass to the method

    Returns
    -------
    List[Equation]
        The collected initial equations (including ArrayEquation)
    """
    ctx = push_equation_context()
    try:
        func(model_instance)
    finally:
        pop_equation_context()
    # Filter to Equation or ArrayEquation (no WhenClause in initial equations)
    from cyecca.dsl.equations import ArrayEquation, Equation

    return [eq for eq in ctx.equations if isinstance(eq, (Equation, ArrayEquation))]


# =============================================================================
# Algorithm decorator (Modelica MLS Ch. 11)
# =============================================================================

# Marker attribute for @algorithm decorated methods
_ALGORITHM_MARKER = "_cyecca_algorithm_method"


def algorithm(func: Callable) -> Callable:
    """
    Decorator to mark a method as an algorithm block.

    Methods decorated with @algorithm use side-effect based capture
    for assignments. Use the @ operator for assignments:

        @algorithm
        def _(m):
            m.y @ m.x * 2.0
            m.z @ m.y + 1.0

    Parameters
    ----------
    func : Callable
        The algorithm method

    Returns
    -------
    Callable
        The decorated method with algorithm capture marker
    """
    _check_and_register_decorated_method(func, "algorithm")
    setattr(func, _ALGORITHM_MARKER, True)
    return func


def is_algorithm_method(func: Any) -> bool:
    """Check if a function is marked as an @algorithm method."""
    return getattr(func, _ALGORITHM_MARKER, False)


def execute_algorithm_method(func: Callable, model_instance: "ModelInstance") -> List["Assignment"]:
    """
    Execute an @algorithm method and collect assignments via context.

    Parameters
    ----------
    func : Callable
        The @algorithm decorated method
    model_instance : ModelInstance
        The model instance to pass to the method

    Returns
    -------
    List[Assignment]
        The collected assignments
    """
    ctx = push_equation_context()
    try:
        func(model_instance)
    finally:
        pop_equation_context()
    return ctx.assignments


# =============================================================================
# When-clause support (Modelica MLS 8.5)
# =============================================================================


class WhenContext:
    """
    Context manager for building when-clauses.

    This is used internally by the `when()` function to collect
    reinit statements within a when-clause.

    Use inside an @equations block:

        @equations
        def _(m):
            der(m.h) == m.v
            with when(m.h < 0):
                reinit(m.v, -m.e * pre(m.v))
    """

    def __init__(self, condition: Expr):
        self.condition = condition
        self.body: List[Union["Equation", "Reinit"]] = []
        self._in_equations_context = False

    def __enter__(self) -> "WhenContext":
        # Check if we're inside an @equations block
        eq_ctx = get_current_equation_context()
        if eq_ctx is not None:
            self._in_equations_context = True
            eq_ctx.enter_when(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # If in @equations context, auto-register the when-clause
        if self._in_equations_context:
            eq_ctx = get_current_equation_context()
            if eq_ctx is not None:
                eq_ctx.exit_when()

        return None

    def to_when_clause(self) -> "WhenClause":
        """Convert to a WhenClause object."""
        from cyecca.dsl.equations import WhenClause

        return WhenClause(condition=self.condition, body=self.body)


@beartype
def when(condition: ExprLike) -> WhenContext:
    """
    Create a when-clause for event handling (Modelica MLS 8.5).

    When-clauses define event-driven behavior in hybrid systems. The body
    of a when-clause is executed when the condition changes from False to
    True (rising edge).

    Use this as a context manager in an @equations method:

        @equations
        def _(m):
            der(m.h) == m.v
            der(m.v) == -9.81

            with when(m.h < 0):
                reinit(m.v, -m.e * pre(m.v))

    The condition can be:
    - A relational expression (m.h < 0)
    - A Boolean variable (m.trigger)
    - edge(m.trigger) - rising edge of Boolean
    - change(m.mode) - any value change

    Parameters
    ----------
    condition : ExprLike
        Boolean condition that triggers the when-clause

    Returns
    -------
    WhenContext
        A context manager for the when-clause body

    Notes
    -----
    Modelica Spec: Section 8.5 - When-Equations

    In Modelica, when-equations are triggered on the rising edge of the
    condition. The condition is typically a relational expression that
    crosses zero (like h < 0), and the simulator uses root-finding to
    detect the exact crossing time.
    """
    return WhenContext(to_expr(condition))


@beartype
def reinit(var: SymbolicVar, expr: ExprLike) -> Optional[Reinit]:
    """
    Reinitialize a continuous-time state variable at an event.

    This is used within when-clauses to set a new value for a state
    variable when an event occurs. The reinit takes effect instantaneously
    at the event time.

    When used inside an @equations block with a when() context, the reinit
    is automatically registered. No yield or explicit add() is needed.

    Parameters
    ----------
    var : SymbolicVar
        The state variable to reinitialize
    expr : ExprLike
        The new value expression (can use pre() for previous values)

    Returns
    -------
    Reinit or None
        Returns None when auto-registered in @equations context,
        otherwise returns a Reinit object.

    Notes
    -----
    Modelica Spec: Section 8.5 - reinit()

    The reinit() function:
    - Can only be used within when-clauses
    - Can only reinitialize continuous-time state variables
    - The new value is computed using values from just before the event
    - Use pre(x) to access the value of x just before the event
    """
    if not isinstance(var, SymbolicVar):
        raise TypeError(f"reinit() expects a SymbolicVar as first argument, got {type(var)}")
    if not var.is_scalar():
        raise TypeError(f"reinit() currently only supports scalar variables, got shape {var.shape}")

    r = Reinit(var_name=var._name, expr=to_expr(expr))

    # Auto-register if in @equations context
    ctx = get_current_equation_context()
    if ctx is not None:
        ctx.add_reinit(r)
        return None
    return r


# =============================================================================
# Connect equations (Modelica MLS Ch. 9)
# =============================================================================


@beartype
def connect(a: SubmodelProxy, b: SubmodelProxy) -> None:
    """
    Connect two connectors, generating appropriate connection equations.

    This implements Modelica connection semantics (MLS Chapter 9):
    - Potential (non-flow) variables: equality equations (a.v == b.v)
    - Flow variables: sum-to-zero equations (a.i + b.i == 0)

    Must be used inside an @equations block:

        @equations
        def _(m):
            connect(m.resistor.p, m.ground.p)
            connect(m.resistor.n, m.capacitor.p)

    Parameters
    ----------
    a : SubmodelProxy
        First connector (e.g., m.resistor.p)
    b : SubmodelProxy
        Second connector (e.g., m.ground.p)

    Raises
    ------
    RuntimeError
        If called outside an @equations block.
    TypeError
        If a or b are not connector proxies.

    Example
    -------
    >>> from cyecca.dsl import model, connector, var, equations, connect, submodel
    >>> @connector
    ... class Pin:
    ...     v = var()           # Potential (voltage)
    ...     i = var(flow=True)  # Flow (current)

    >>> @model
    ... class Resistor:
    ...     p = submodel(Pin)   # Positive pin
    ...     n = submodel(Pin)   # Negative pin
    ...     R = var(1000.0, parameter=True)
    ...
    ...     @equations
    ...     def _(m):
    ...         m.p.v - m.n.v == m.R * m.p.i
    ...         m.p.i + m.n.i == 0  # Current conservation

    >>> @model
    ... class Circuit:
    ...     r1 = submodel(Resistor)
    ...     r2 = submodel(Resistor)
    ...
    ...     @equations
    ...     def _(m):
    ...         connect(m.r1.n, m.r2.p)  # Series connection

    Notes
    -----
    Modelica Spec: Section 9.2 - Generation of Connection Equations

    For each pair of corresponding primitive components:
    - a.v == b.v (for non-flow variables, equality)
    - a.i + b.i == 0 (for flow variables, sum to zero)

    The sign convention for flow variables assumes "inside" connectors
    where positive flow is into the component.
    """
    from cyecca.dsl.equations import Equation

    ctx = get_current_equation_context()
    if ctx is None:
        raise RuntimeError("connect() can only be used inside an @equations block")

    # Get the connector metadata (beartype already validates SubmodelProxy type)
    a_instance = a._instance
    b_instance = b._instance
    a_metadata = a_instance._metadata
    b_metadata = b_instance._metadata

    # Check that both are connectors (check metadata, not class attribute)
    if not a_metadata.is_connector:
        raise TypeError(f"connect() first argument '{a._name}' is not a connector")
    if not b_metadata.is_connector:
        raise TypeError(f"connect() second argument '{b._name}' is not a connector")

    # Verify connectors have matching structure
    a_vars = set(a_metadata.variables.keys())
    b_vars = set(b_metadata.variables.keys())

    if a_vars != b_vars:
        raise TypeError(f"Connectors '{a._name}' and '{b._name}' have different variables: " f"{a_vars} vs {b_vars}")

    # Generate connection equations for each variable
    for var_name, a_var in a_metadata.variables.items():
        b_var = b_metadata.variables[var_name]

        # Skip parameters and constants
        if a_var.parameter or a_var.constant:
            continue

        # Get the full qualified names
        a_full_name = f"{a._name}.{var_name}"
        b_full_name = f"{b._name}.{var_name}"

        # Get symbolic variables from parent
        a_sym = a._parent._sym_vars[a_full_name]
        b_sym = b._parent._sym_vars[b_full_name]

        if a_var.flow:
            # Flow variable: sum-to-zero (a.i + b.i == 0)
            # Using the inside connector convention
            eq = Equation(
                lhs=to_expr(a_sym) + to_expr(b_sym),
                rhs=to_expr(0.0),
            )
        else:
            # Potential variable: equality (a.v == b.v)
            eq = Equation(
                lhs=to_expr(a_sym),
                rhs=to_expr(b_sym),
            )

        ctx.add_equation(eq)


# =============================================================================
# If-equation support (Modelica MLS 8.3.4)
# =============================================================================


class IfContext:
    """
    Context manager for building if-equations.

    This is used internally by the `if_eq()` function to collect
    equations within if/elseif/else branches.

    Use inside an @equations block:

        @equations
        def _(m):
            with if_eq(m.use_linear):
                m.y == m.a * m.x + m.b
            with else_eq():
                m.y == m.a * m.x**2 + m.b * m.x + m.c

    Note: elseif/else branches are linked to the preceding if_eq via
    thread-local state, so they must immediately follow the if_eq block.

    Modelica Spec: Section 8.3.4 - If-Equations
    """

    # Thread-local storage for the current if-equation being built
    _current: threading.local = threading.local()

    def __init__(self, condition: Expr, is_elseif: bool = False, is_else: bool = False):
        self.condition: Optional[Expr] = condition if not is_else else None
        self.is_elseif = is_elseif
        self.is_else = is_else
        self.body: List[Union["Equation", "WhenClause", "Reinit"]] = []
        self._in_equations_context = False
        self._saved_when_ctx: Optional["WhenContext"] = None
        self._parent_if_eq: Optional["IfEquation"] = None

    def __enter__(self) -> "IfContext":
        eq_ctx = get_current_equation_context()
        if eq_ctx is not None:
            self._in_equations_context = True
            # Save and clear any current when context (nested when not allowed)
            self._saved_when_ctx = eq_ctx._current_when
            eq_ctx._current_when = None
            # Enter if context
            eq_ctx._current_if = self

        # For elseif/else, link to parent if-equation
        if self.is_elseif or self.is_else:
            if not hasattr(IfContext._current, "pending_if") or IfContext._current.pending_if is None:
                raise RuntimeError("elseif_eq()/else_eq() must immediately follow an if_eq() or elseif_eq() block")
            self._parent_if_eq = IfContext._current.pending_if
        else:
            # Start new if-equation chain
            IfContext._current.pending_if = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        eq_ctx = get_current_equation_context()

        if self._in_equations_context and eq_ctx is not None:
            # Restore when context
            eq_ctx._current_when = self._saved_when_ctx
            eq_ctx._current_if = None

        # Create branch for this block
        branch = IfEquationBranch(condition=self.condition, body=self.body)

        if self.is_elseif or self.is_else:
            # Add to existing if-equation
            if self._parent_if_eq is not None:
                self._parent_if_eq.branches.append(branch)
                if self.is_else:
                    # Complete the if-equation
                    if eq_ctx is not None:
                        eq_ctx.add_equation(self._parent_if_eq)
                    IfContext._current.pending_if = None
                else:
                    # Keep chain open for more elseif/else
                    IfContext._current.pending_if = self._parent_if_eq
        else:
            # Start new if-equation
            if_eq = IfEquation(branches=[branch])
            # Store for potential elseif/else
            IfContext._current.pending_if = if_eq
            # If no elseif/else follows (determined in next call or finalize),
            # register immediately in add_if_standalone()

        return None


# Store reference to current if context on EquationContext for equation collection
def _add_if_support_to_context():
    """Monkey-patch EquationContext to support if-equations."""
    original_init = EquationContext.__init__

    def new_init(self) -> None:
        original_init(self)
        self._current_if: Optional[IfContext] = None

    EquationContext.__init__ = new_init

    # Also patch add_equation to support if-context
    original_add_equation = EquationContext.add_equation

    def new_add_equation(self, eq) -> None:
        if self._current_if is not None:
            self._current_if.body.append(eq)
        elif self._current_when is not None:
            self._current_when.body.append(eq)
        else:
            self.equations.append(eq)

    EquationContext.add_equation = new_add_equation


_add_if_support_to_context()


def _finalize_pending_if() -> None:
    """Finalize any pending if-equation without else/elseif.

    Called at end of @equations block to register standalone if-equations.
    """
    if hasattr(IfContext._current, "pending_if") and IfContext._current.pending_if is not None:
        eq_ctx = get_current_equation_context()
        if eq_ctx is not None:
            eq_ctx.equations.append(IfContext._current.pending_if)
        IfContext._current.pending_if = None


# Patch execute_equations_method to finalize pending if-equations
_original_execute_equations_method = execute_equations_method


def execute_equations_method(
    func: Callable, model_instance: "ModelInstance"
) -> List[Union["Equation", "ArrayEquation", "WhenClause"]]:
    """
    Execute an @equations method and collect equations via context.

    Parameters
    ----------
    func : Callable
        The @equations decorated method
    model_instance : ModelInstance
        The model instance to pass to the method

    Returns
    -------
    List[Union[Equation, ArrayEquation, WhenClause]]
        The collected equations
    """
    ctx = push_equation_context()
    try:
        func(model_instance)
        _finalize_pending_if()
    finally:
        pop_equation_context()
    return ctx.equations


@beartype
def if_eq(condition: ExprLike) -> IfContext:
    """
    Create an if-equation for conditional equations (Modelica MLS 8.3.4).

    If-equations select which equations are active based on a condition.
    Use this as a context manager in an @equations method:

        @equations
        def _(m):
            with if_eq(m.use_linear):
                m.y == m.a * m.x + m.b
            with else_eq():
                m.y == m.a * m.x**2 + m.b * m.x + m.c

    For multiple conditions:

        @equations
        def _(m):
            with if_eq(m.mode == 1):
                m.y == m.x
            with elseif_eq(m.mode == 2):
                m.y == 2 * m.x
            with else_eq():
                m.y == 3 * m.x

    Parameters
    ----------
    condition : ExprLike
        Boolean condition that selects which equations are active

    Returns
    -------
    IfContext
        A context manager for the if-equation body

    Notes
    -----
    Modelica Spec: Section 8.3.4 - If-Equations

    Key constraints:
    - For non-parameter conditions, all branches MUST have the same
      number of equations (the DAE structure cannot change dynamically).
    - For parameter conditions, branches can differ (structure is
      determined at compile time and only one branch is kept).
    """
    return IfContext(to_expr(condition), is_elseif=False, is_else=False)


@beartype
def elseif_eq(condition: ExprLike) -> IfContext:
    """
    Create an elseif branch for an if-equation.

    Must immediately follow an if_eq() or another elseif_eq() block:

        @equations
        def _(m):
            with if_eq(m.mode == 1):
                m.y == m.x
            with elseif_eq(m.mode == 2):
                m.y == 2 * m.x
            with else_eq():
                m.y == 0

    Parameters
    ----------
    condition : ExprLike
        Boolean condition for this elseif branch

    Returns
    -------
    IfContext
        A context manager for the elseif body

    Raises
    ------
    RuntimeError
        If not immediately following an if_eq() or elseif_eq() block.
    """
    return IfContext(to_expr(condition), is_elseif=True, is_else=False)


def else_eq() -> IfContext:
    """
    Create an else branch for an if-equation.

    Must immediately follow an if_eq() or elseif_eq() block:

        @equations
        def _(m):
            with if_eq(m.use_linear):
                m.y == m.a * m.x + m.b
            with else_eq():
                m.y == m.a * m.x**2 + m.b * m.x + m.c

    Returns
    -------
    IfContext
        A context manager for the else body

    Raises
    ------
    RuntimeError
        If not immediately following an if_eq() or elseif_eq() block.
    """
    # Create a dummy True condition that will be ignored (marked as is_else=True)
    from cyecca.dsl.expr import Expr, ExprKind

    dummy_condition = Expr(ExprKind.CONSTANT, value=True)
    return IfContext(dummy_condition, is_elseif=False, is_else=True)
