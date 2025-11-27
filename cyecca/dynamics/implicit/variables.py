"""Symbolic variable wrapper classes for implicit DAE modeling.

These classes wrap CasADi symbolic variables and provide convenient
interfaces for equation writing, including .dot() for time derivatives.
"""

from typing import Union

import casadi as ca


class ImplicitVariable:
    """Base class for symbolic variables in implicit DAE models.

    Wraps a CasADi symbolic variable and provides operator overloading
    for natural equation writing.
    """

    def __init__(self, name: str, var_type: str, shape: int = 1, sym_type=ca.SX):
        """Initialize symbolic variable.

        Args:
            name: Variable name
            var_type: Type ('state', 'alg', 'param', 'var')
            shape: Number of elements (1 for scalar, N for vector)
            sym_type: CasADi symbol type (SX or MX)
        """
        self.name = name
        self.var_type = var_type
        self.shape = shape
        self.sym = sym_type.sym(name, shape)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"

    # Operator overloading for CasADi operations
    def __add__(self, other):
        return self.sym + (other.sym if isinstance(other, ImplicitVariable) else other)

    def __radd__(self, other):
        return (other.sym if isinstance(other, ImplicitVariable) else other) + self.sym

    def __sub__(self, other):
        return self.sym - (other.sym if isinstance(other, ImplicitVariable) else other)

    def __rsub__(self, other):
        return (other.sym if isinstance(other, ImplicitVariable) else other) - self.sym

    def __mul__(self, other):
        return self.sym * (other.sym if isinstance(other, ImplicitVariable) else other)

    def __rmul__(self, other):
        return (other.sym if isinstance(other, ImplicitVariable) else other) * self.sym

    def __truediv__(self, other):
        return self.sym / (other.sym if isinstance(other, ImplicitVariable) else other)

    def __rtruediv__(self, other):
        return (other.sym if isinstance(other, ImplicitVariable) else other) / self.sym

    def __pow__(self, other):
        return self.sym ** (other.sym if isinstance(other, ImplicitVariable) else other)

    def __neg__(self):
        return -self.sym

    def __getitem__(self, key):
        """Support indexing for vector variables."""
        return self.sym[key]

    # Delegate unknown attributes to the underlying symbol
    # This allows CasADi functions like sin(), cos(), etc. to work
    def __getattr__(self, name):
        # Avoid infinite recursion for defined attributes
        if name in ["sym", "name", "var_type", "shape"]:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.sym, name)


class ImplicitVar(ImplicitVariable):
    """General variable with automatic state inference.

    This is the Modelica-style variable class. Variables are not explicitly
    declared as state or algebraic - instead, calling .dot() marks the
    variable as a state, and the model infers the variable types during build().

    Provides .dot() method to access the time derivative symbol. When .dot()
    is called, the variable is marked as having a derivative (i.e., it's a state).
    """

    def __init__(self, name: str, shape: int = 1, sym_type=ca.SX):
        """Initialize variable.

        Args:
            name: Variable name
            shape: Number of elements (1 for scalar, N for vector)
            sym_type: CasADi symbol type (SX or MX)
        """
        super().__init__(name, "var", shape, sym_type)
        # The derivative symbol is created lazily when .dot() is called
        self._dot_sym = None
        self._has_derivative = False
        self._sym_type = sym_type

    def dot(self):
        """Return time derivative symbol and mark this variable as a state.

        When .dot() is called, this variable becomes a state variable
        (it has a time derivative). Variables without .dot() called are
        treated as algebraic variables.

        Returns:
            CasADi symbol representing dvar/dt

        Example:
            >>> from cyecca.dynamics._doctest_examples import get_unbuilt_implicit_model
            >>> model = get_unbuilt_implicit_model()
            >>> model.v.theta.dot()  # doctest: +ELLIPSIS
            SX(theta_dot)
        """
        if self._dot_sym is None:
            self._dot_sym = self._sym_type.sym(f"{self.name}_dot", self.shape)
        self._has_derivative = True
        return self._dot_sym

    @property
    def has_derivative(self) -> bool:
        """Check if .dot() has been called on this variable."""
        return self._has_derivative


class ImplicitState(ImplicitVariable):
    """State variable with time derivative support.

    Provides .dot() method to access the time derivative symbol.

    Note: For Modelica-style models, prefer ImplicitVar which automatically
    infers state vs algebraic based on .dot() usage.
    """

    def __init__(self, name: str, shape: int = 1, sym_type=ca.SX):
        """Initialize state variable.

        Args:
            name: Variable name
            shape: Number of elements (1 for scalar, N for vector)
            sym_type: CasADi symbol type (SX or MX)
        """
        super().__init__(name, "state", shape, sym_type)
        self.dot_sym = sym_type.sym(f"{name}_dot", shape)

    def dot(self):
        """Return time derivative symbol.

        Returns:
            CasADi symbol representing dvar/dt

        Example:
            >>> from cyecca.dynamics.implicit.variables import ImplicitState
            >>> state = ImplicitState('x')
            >>> state.dot()  # doctest: +ELLIPSIS
            SX(x_dot)
        """
        return self.dot_sym


class ImplicitAlg(ImplicitVariable):
    """Algebraic variable (no time derivative)."""

    def __init__(self, name: str, shape: int = 1, sym_type=ca.SX):
        """Initialize algebraic variable.

        Args:
            name: Variable name
            shape: Number of elements (1 for scalar, N for vector)
            sym_type: CasADi symbol type (SX or MX)
        """
        super().__init__(name, "alg", shape, sym_type)


class ImplicitParam(ImplicitVariable):
    """Parameter variable (constant)."""

    def __init__(self, name: str, default: float = 0.0, shape: int = 1, sym_type=ca.SX):
        """Initialize parameter variable.

        Args:
            name: Variable name
            default: Default parameter value
            shape: Number of elements (1 for scalar, N for vector)
            sym_type: CasADi symbol type (SX or MX)
        """
        super().__init__(name, "param", shape, sym_type)
        self.default = default
