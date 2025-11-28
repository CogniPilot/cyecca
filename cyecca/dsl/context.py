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
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from beartype import beartype

from cyecca.dsl.equations import Reinit, WhenClause
from cyecca.dsl.expr import Expr, to_expr
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
        The collected initial equations
    """
    ctx = push_equation_context()
    try:
        func(model_instance)
    finally:
        pop_equation_context()
    # Filter to only Equation (no WhenClause in initial equations)
    from cyecca.dsl.equations import Equation

    return [eq for eq in ctx.equations if isinstance(eq, Equation)]


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
def when(condition: Any) -> WhenContext:
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
    condition : Expr or SymbolicVar
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
def reinit(var: SymbolicVar, expr: Any) -> Optional[Reinit]:
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
    expr : Expr or numeric
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
