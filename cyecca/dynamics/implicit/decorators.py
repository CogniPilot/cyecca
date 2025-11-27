"""Decorator for creating symbolic implicit DAE dataclasses.

The @implicit decorator transforms dataclasses with field descriptors
into symbolic variable containers suitable for equation-based modeling.

DESIGN PRINCIPLE: All type classes must be explicitly declared - no dynamic generation.
This ensures autocomplete works and type safety is maintained.
"""

from dataclasses import MISSING, dataclass, fields

import casadi as ca
import numpy as np

from .fields import VarDescriptor as ImplicitVarDescriptor
from .variables import ImplicitParam, ImplicitVar


def implicit(cls):
    """Decorator to create symbolic implicit DAE dataclass.

    Transforms a dataclass with var(), param() fields into
    a class that instantiates symbolic variables. Mirrors the @explicit
    decorator pattern for consistency.

    For Modelica-style models, use var() for variables and param() for
    parameters. The Model class will automatically infer which variables
    are states (have .dot() called) vs algebraic (no .dot() call).

    Args:
        cls: Dataclass with ImplicitVarDescriptor fields

    Returns:
        Enhanced class with symbolic(), numeric(), as_vec(), from_vec() methods

    Example:
        >>> from cyecca.dynamics.implicit import implicit, var, param
        >>> @implicit
        ... class Pendulum:
        ...     theta: float = var()
        ...     omega: float = var()
        ...     g: float = param(default=9.81)
        >>> x = Pendulum.symbolic()  # Creates symbolic instance
        >>> x.theta  # doctest: +ELLIPSIS
        ImplicitVar('theta')
        >>> x0 = Pendulum.numeric()  # Creates numeric defaults
        >>> x0.theta
        0.0
    """
    # First apply dataclass decorator if not already applied
    if not hasattr(cls, "__dataclass_fields__"):
        cls = dataclass(cls)

    # Extract field metadata from VarDescriptor instances - STATIC, stored on class
    # Create a temporary instance to extract VarDescriptor metadata
    temp_instance = cls()
    field_info = {}
    for f in fields(cls):
        descriptor = getattr(temp_instance, f.name, None)

        if isinstance(descriptor, ImplicitVarDescriptor):
            field_info[f.name] = {
                "dim": descriptor.shape,
                "default": descriptor.default,
                "type": descriptor.var_type,
            }
        elif hasattr(descriptor, "var_type"):
            # This is a VarDescriptor from explicit fields - store its type for error checking
            field_info[f.name] = {
                "dim": getattr(descriptor, "shape", 1),
                "default": getattr(descriptor, "default", 0.0),
                "type": descriptor.var_type,  # Will trigger error in create_symbolic
            }
        else:
            # Fallback for non-VarDescriptor fields
            field_info[f.name] = {
                "dim": 1,
                "default": 0.0,
                "type": "unknown",
            }

    # Store field_info STATICALLY on the class - critical for IDE autocomplete
    cls._field_info = field_info

    # Create symbolic instance factory (matches @explicit pattern)
    @classmethod
    def create_symbolic(cls_obj, sym_type=ca.SX):
        """Create instance with symbolic CasADi variables.

        Returns:
            Instance with ImplicitVar/ImplicitParam fields
        """
        instance = object.__new__(cls_obj)
        instance._field_names = []
        instance._sym_type = sym_type

        for field_name, info in cls_obj._field_info.items():
            var_type = info["type"]
            dim = info["dim"]
            default = info["default"]

            if var_type == "var":
                # Modelica-style variable - state/alg inferred from .dot() usage
                v = ImplicitVar(field_name, dim, sym_type)
                setattr(instance, field_name, v)
                instance._field_names.append(field_name)
            elif var_type in ("param", "parameter"):
                v = ImplicitParam(field_name, default, dim, sym_type)
                setattr(instance, field_name, v)
                instance._field_names.append(field_name)
            elif var_type in ("state", "alg", "algebraic", "input", "output"):
                raise ValueError(
                    f"Field '{field_name}' uses explicit-style field type '{var_type}' which is not allowed in implicit models. "
                    f"Use var() for variables and param() for parameters - states are automatically inferred from .dot() usage."
                )
            # Skip unknown types

        return instance

    # Create numeric instance factory (matches @explicit pattern)
    @classmethod
    def create_numeric(cls_obj):
        """Create instance with numeric default values.

        Returns:
            Instance with float/numpy values for each field
        """
        instance = object.__new__(cls_obj)
        instance._field_names = []
        instance._sym_type = None

        for field_name, info in cls_obj._field_info.items():
            default = info["default"]
            dim = info["dim"]

            if dim == 1:
                val = float(default) if default is not None else 0.0
            else:
                if isinstance(default, (list, tuple, np.ndarray)):
                    val = np.array(default, dtype=float)
                else:
                    val = np.zeros(dim, dtype=float)

            setattr(instance, field_name, val)
            instance._field_names.append(field_name)

        return instance

    def as_vec(self):
        """Convert all fields to a single CasADi vector.

        Returns:
            CasADi vector containing all variable symbols
        """
        parts = []
        for name in self._field_names:
            val = getattr(self, name)
            if hasattr(val, "sym"):
                parts.append(ca.vec(val.sym))
            else:
                parts.append(ca.vec(val))
        return ca.vertcat(*parts) if parts else ca.DM.zeros(0, 1)

    def dot_vec(self):
        """Convert all state derivatives to a vector.

        Only valid for state variables (those with .dot_sym).

        Returns:
            CasADi vector containing all derivative symbols

        Raises:
            AttributeError: If called on non-state variables
        """
        dots = []
        for name in self._field_names:
            var = getattr(self, name)
            if hasattr(var, "dot_sym"):
                dots.append(var.dot_sym)
            else:
                raise AttributeError(f"Variable {name} does not have derivative (not a state)")
        return ca.vertcat(*dots)

    @classmethod
    def from_vec(cls_obj, vec):
        """Reconstruct from CasADi vector (works with both numeric and symbolic).

        For numeric vectors (DM), converts to float/numpy arrays.
        For symbolic vectors (SX/MX), returns symbolic expressions.

        Args:
            vec: CasADi vector (DM, SX, or MX)

        Returns:
            Instance with values extracted from vector
        """
        # Handle CasADi Function dict outputs
        if isinstance(vec, dict):
            if len(vec) == 1:
                vec = list(vec.values())[0]
            else:
                raise ValueError(f"from_vec() received dict with multiple outputs: {list(vec.keys())}")

        # Check if this is a symbolic type (SX or MX)
        is_symbolic = hasattr(vec, "__class__") and vec.__class__.__name__ in ("SX", "MX")

        # Convert to proper vector (only for numeric)
        if not is_symbolic and not hasattr(vec, "shape"):
            vec = ca.DM(vec)

        # Ensure column vector
        if hasattr(vec, "shape") and len(vec.shape) == 2 and vec.shape[1] != 1:
            vec = vec.T

        instance = object.__new__(cls_obj)
        instance._field_names = list(cls_obj._field_info.keys())
        instance._sym_type = type(vec) if is_symbolic else None

        offset = 0
        for name, info in cls_obj._field_info.items():
            dim = info["dim"]

            if dim == 1:
                val = vec[offset]
                # Convert DM scalar to float (numeric only)
                if not is_symbolic and hasattr(val, "numel") and val.numel() == 1:
                    val = float(val)
            else:
                val = vec[offset : offset + dim]
                # Convert DM vector to numpy array (numeric only)
                if not is_symbolic and hasattr(val, "__class__") and val.__class__.__name__ == "DM":
                    val = np.array(val, dtype=float).flatten()

            setattr(instance, name, val)
            offset += dim

        return instance

    def size(self):
        """Return total number of scalar variables.

        Returns:
            Sum of dimensions of all fields
        """
        return sum(self._field_info[name]["dim"] for name in self._field_names)

    def size1(self):
        """Get number of rows when converted to vector (CasADi compat)."""
        return self.size()

    def size2(self):
        """Get number of columns (always 1, CasADi compat)."""
        return 1

    @classmethod
    def from_trajectory(cls_obj, matrix):
        """Create single instance with ndarray fields from trajectory matrix.

        For efficient trajectory storage and plotting. Each field becomes
        an ndarray with shape (n_steps,) for scalars or (n_steps, dim) for vectors.

        Args:
            matrix: NumPy array or CasADi matrix with shape (n_states, n_steps)

        Returns:
            Instance where each field is an ndarray (n_steps,) or (n_steps, dim)

        Example:
            >>> from cyecca.dynamics._doctest_examples import Pendulum
            >>> import numpy as np
            >>> x_history = np.array([[0.5, 0.4, 0.3], [0.0, 0.1, 0.2]])  # 2 vars, 3 timesteps
            >>> x_traj = Pendulum.from_trajectory(x_history)
            >>> x_traj.theta.shape
            (3,)
        """
        # Convert to NumPy if needed
        if hasattr(matrix, "__class__") and matrix.__class__.__name__ in ("SX", "MX", "DM"):
            matrix = np.array(ca.DM(matrix))
        else:
            matrix = np.asarray(matrix)

        instance = object.__new__(cls_obj)
        instance._field_names = list(cls_obj._field_info.keys())
        instance._sym_type = None
        instance._is_trajectory = True  # Flag to indicate trajectory mode

        offset = 0
        for name, info in cls_obj._field_info.items():
            dim = info["dim"]

            if dim == 1:
                # Scalar field: extract row, shape becomes (n_steps,)
                val = matrix[offset, :]
                offset += 1
            else:
                # Vector field: extract rows, transpose to (n_steps, dim)
                val = matrix[offset : offset + dim, :].T
                offset += dim

            setattr(instance, name, val)

        return instance

    def getitem(self, idx):
        """Index into trajectory to get a numeric instance at timestep.

        Only valid for trajectory instances (created via simulate() or from_trajectory()).

        Args:
            idx: Integer index (supports negative indexing like -1 for last)

        Returns:
            Numeric instance with scalar values at the given timestep

        Example:
            >>> from cyecca.dynamics._doctest_examples import get_built_implicit_model
            >>> model = get_built_implicit_model()
            >>> model.v0.theta = 0.5
            >>> t, data = model.simulate(0.0, 0.5, 0.1)
            >>> final = data[-1]  # Get final state
            >>> hasattr(final, 'theta')
            True
        """
        if not getattr(self, "_is_trajectory", False):
            raise TypeError("Indexing only supported on trajectory instances")

        # Create numeric instance
        instance = object.__new__(self.__class__)
        instance._field_names = list(self._field_names)
        instance._sym_type = None
        instance._is_trajectory = False

        for name in self._field_names:
            info = self.__class__._field_info[name]

            # Skip if field not in trajectory
            if not hasattr(self, name):
                continue

            val = getattr(self, name)
            dim = info["dim"]

            if dim == 1:
                # Scalar: val is (n_steps,), extract single value
                setattr(instance, name, float(val[idx]))
            else:
                # Vector: val is (n_steps, dim), extract row
                setattr(instance, name, val[idx].copy())

        return instance

    # Attach methods to class
    cls.symbolic = create_symbolic
    cls.numeric = create_numeric
    cls.as_vec = as_vec
    cls.dot_vec = dot_vec
    cls.from_vec = from_vec
    cls.from_trajectory = from_trajectory
    cls.__getitem__ = getitem
    cls.size = size
    cls.size1 = size1
    cls.size2 = size2

    return cls
