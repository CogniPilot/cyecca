"""Decorators for the modeling framework.

Provides the @explicit decorator for creating type-safe dataclasses with
CasADi symbolic variable support.

DESIGN PRINCIPLES:
- Single unified namespace - all variable types in one class
- Autocomplete must ALWAYS work - IDEs need static type information
- Type safety via beartype - all functions must have proper type annotations
"""

from dataclasses import dataclass, field, fields

import casadi as ca
import numpy as np

__all__ = ["explicit", "symbolic"]


def compose_states(*state_types):
    """Compose multiple state types into a single combined state type.

    This programmatically creates a new state class that includes all fields
    from the input state types, preserving type safety and field metadata.

    Args:
        *state_types: State type classes to combine

    Returns:
        New combined state class with all fields from input types

    Example:
        >>> ClosedLoopStates = compose_states(PlantStates, ControllerStates)  # doctest: +SKIP
    """
    from dataclasses import MISSING
    from dataclasses import fields as get_fields

    # Collect all fields from input types, preserving metadata
    combined_annotations = {}
    combined_fields = {}

    for state_type in state_types:
        # Ensure the type is a dataclass
        if not hasattr(state_type, "__dataclass_fields__"):
            continue

        # Get fields from the dataclass
        for fld in get_fields(state_type):
            if fld.name.startswith("_"):
                continue  # Skip private fields
            combined_annotations[fld.name] = fld.type
            # Create a new field with the same metadata
            if fld.default is not MISSING:
                combined_fields[fld.name] = field(default=fld.default, metadata=fld.metadata)
            elif fld.default_factory is not MISSING:
                combined_fields[fld.name] = field(default_factory=fld.default_factory, metadata=fld.metadata)
            else:
                combined_fields[fld.name] = field(default=None, metadata=fld.metadata)

    # Create new class dynamically
    combined_class = type(
        "ComposedStates",
        (),
        {"__annotations__": combined_annotations, **combined_fields},
    )

    # Apply @dataclass first, then @explicit
    combined_class = dataclass(combined_class)
    return explicit(combined_class)


def explicit(cls):
    """Combined decorator: applies @dataclass and adds CasADi symbolic methods.

    Adds methods:
        - cls.symbolic(sym_type=ca.SX) -> instance with symbolic variables
        - cls.numeric() -> instance with numeric defaults
        - instance.as_vec() -> ca.SX/MX column vector
        - cls.from_vec(vec) -> instance from vector
        - instance.size1() -> number of rows
        - instance.size2() -> number of columns (always 1)

    Usage:
        @explicit
        class States:
            x: ca.SX = state(1, 0.0, "position")
            v: ca.SX = state(1, 0.0, "velocity")

        # Create symbolic instance
        x_sym = States.symbolic()  # x_sym.x is ca.SX.sym('x')

        # Create numeric instance
        x0 = States.numeric()  # x0.x == 0.0

        # Convert to vector
        vec = x_sym.as_vec()  # ca.vertcat(x_sym.x, x_sym.v)

        # Access instances directly
        x = model.x  # Access as attribute
    """
    # Apply @dataclass - always reapply to ensure child class fields are processed
    # Even if parent was a dataclass, child needs its own dataclass processing
    cls = dataclass(cls)

    # Extract field metadata from VarDescriptor instances - STATIC, stored on class
    from ..fields import VarDescriptor
    
    # Start with inherited _field_info from parent classes
    field_info = {}
    for base in cls.__mro__[1:]:  # Skip the class itself
        if hasattr(base, '_field_info'):
            for name, info in base._field_info.items():
                if name not in field_info:
                    field_info[name] = info.copy()
    
    # Create a temporary instance to extract VarDescriptor metadata
    temp_instance = cls()
    for f in fields(cls):
        descriptor = getattr(temp_instance, f.name, None)
        
        if isinstance(descriptor, VarDescriptor):
            dim = descriptor.shape
            default = descriptor.default
            var_type = descriptor.var_type
        else:
            # Fallback for non-VarDescriptor fields (shouldn't happen in normal use)
            dim = 1
            default = 0.0
            var_type = "unknown"

        field_info[f.name] = {
            "dim": dim,
            "default": default,
            "type": var_type,
            "casadi_type": f.type,
        }
    
    # Store field_info STATICALLY on the class - critical for performance and _build_index_maps()
    cls._field_info = field_info

    # Create symbolic instance factory
    @classmethod
    def create_symbolic(cls_obj, sym_type=ca.SX):
        """Create instance with symbolic CasADi variables."""
        kwargs = {}
        for name, info in cls_obj._field_info.items():
            dim = info["dim"]
            if dim == 1:
                kwargs[name] = sym_type.sym(name)
            else:
                kwargs[name] = sym_type.sym(name, dim)
        return cls_obj(**kwargs)

    # Create numeric instance factory
    @classmethod
    def create_numeric(cls_obj):
        """Create instance with numeric default values."""
        kwargs = {}
        for name, info in cls_obj._field_info.items():
            default = info["default"]
            if isinstance(default, (list, tuple)):
                kwargs[name] = np.array(default, dtype=float)
            else:
                kwargs[name] = float(default)
        return cls_obj(**kwargs)

    # Vector conversion
    def as_vec(self):
        """Convert to CasADi column vector."""
        parts = []
        for f in fields(self.__class__):
            val = getattr(self, f.name)
            parts.append(ca.vec(val))
        return ca.vertcat(*parts) if parts else ca.DM.zeros(0, 1)

    # Reconstruct from vector
    @classmethod
    def from_vec(cls_obj, vec):
        """Reconstruct from CasADi vector (works with both numeric and symbolic).

        For numeric vectors (DM), converts to float/numpy arrays.
        For symbolic vectors (SX/MX), preserves symbolic expressions.
        """
        
        # Handle CasADi Function dict outputs
        if isinstance(vec, dict):
            if len(vec) == 1:
                vec = list(vec.values())[0]
            else:
                raise ValueError(f"from_vec() received dict with multiple outputs: {list(vec.keys())}")

        # Check if this is a symbolic type (SX or MX)
        is_symbolic = hasattr(vec, "__class__") and vec.__class__.__name__ in (
            "SX",
            "MX",
        )

        # Convert to proper vector (only for numeric)
        if not is_symbolic and not hasattr(vec, "shape"):
            vec = ca.DM(vec)

        # Ensure column vector
        if hasattr(vec, "shape") and len(vec.shape) == 2 and vec.shape[1] != 1:
            vec = vec.T

        kwargs = {}
        offset = 0
        for f in fields(cls_obj):
            name = f.name
            dim = cls_obj._field_info[name]["dim"]

            if dim == 1:
                val = vec[offset]
                # Convert DM scalar to float (numeric only)
                if not is_symbolic and hasattr(val, "numel") and val.numel() == 1:
                    val = float(val)
                kwargs[name] = val
                offset += 1
            else:
                val = vec[offset : offset + dim]
                # Convert DM vector to numpy array (numeric only)
                if not is_symbolic and hasattr(val, "__class__") and val.__class__.__name__ == "DM":
                    val = np.array(val, dtype=float).flatten()
                kwargs[name] = val
                offset += dim

        return cls_obj(**kwargs)

    # Matrix reconstruction (for trajectories - returns list of instances)
    @classmethod
    def from_matrix(cls_obj, matrix):
        """Create instances from matrix where columns are timesteps.

        Args:
            matrix: CasADi matrix or NumPy array with shape (n_vars, n_steps)

        Returns:
            List of instances, one per timestep
        """
        # Convert to NumPy
        if hasattr(matrix, "__class__") and matrix.__class__.__name__ in (
            "SX",
            "MX",
            "DM",
        ):
            matrix = np.array(ca.DM(matrix))
        else:
            matrix = np.asarray(matrix)

        n_steps = matrix.shape[1]
        instances = []

        for i in range(n_steps):
            col = matrix[:, i : i + 1]
            instances.append(cls_obj.from_vec(col))

        return instances

    # Efficient trajectory storage (single instance with ndarray fields)
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
            >>> x_traj = States.from_trajectory(x_history)
            >>> plt.plot(t, x_traj.theta)  # theta is ndarray(n_steps,)
            >>> plt.plot(t, x_traj.pos)    # pos is ndarray(n_steps, 3)
        """
        # Convert to NumPy if needed
        if hasattr(matrix, "__class__") and matrix.__class__.__name__ in (
            "SX",
            "MX",
            "DM",
        ):
            matrix = np.array(ca.DM(matrix))
        else:
            matrix = np.asarray(matrix)
        
        kwargs = {}
        offset = 0
        for f in fields(cls_obj):
            name = f.name
            dim = cls_obj._field_info[name]["dim"]
            
            if dim == 1:
                # Scalar field: extract row, shape becomes (n_steps,)
                kwargs[name] = matrix[offset, :]
                offset += 1
            else:
                # Vector field: extract rows, transpose to (n_steps, dim)
                kwargs[name] = matrix[offset:offset + dim, :].T
                offset += dim
        
        instance = cls_obj(**kwargs)
        instance._is_trajectory = True  # Flag to indicate trajectory mode
        return instance

    # CasADi compatibility methods
    def size1(self):
        """Get number of rows when converted to vector."""
        return self.as_vec().size1()

    def size2(self):
        """Get number of columns (always 1)."""
        return 1

    # Custom repr with descriptions
    def custom_repr(self):
        """String representation with field descriptions."""
        parts = []
        for f in fields(self.__class__):
            val = getattr(self, f.name)
            desc = self.__class__._field_info[f.name].get("desc", "")
            if desc:
                parts.append(f"{f.name}={val!r}  # {desc}")
            else:
                parts.append(f"{f.name}={val!r}")
        return f"{cls.__name__}(" + ", ".join(parts) + ")"

    # Attach methods to class
    cls.symbolic = create_symbolic
    cls.numeric = create_numeric
    cls.as_vec = as_vec
    cls.from_vec = from_vec
    cls.from_matrix = from_matrix
    cls.from_trajectory = from_trajectory
    cls.size1 = size1
    cls.size2 = size2
    cls.__repr__ = custom_repr

    return cls


# Alias for internal compatibility
symbolic = explicit
