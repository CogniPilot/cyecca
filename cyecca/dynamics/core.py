"""Type-safe CasADi modeling framework with full hybrid DAE support.

Provides declarative API for building hybrid dynamical systems without dynamic
class generation. Full autocomplete and type safety throughout.

Features:
- Continuous states (x): dx/dt = f_x(...)
- Algebraic variables (z_alg): 0 = g(x, z_alg, u, p) for DAE constraints
- Dependent variables (dep): dep = f_dep(x, u, p) for computed quantities
- Quadrature states (q): dq/dt = f_q(x, u, p) for path integrals
- Discrete states (z): z⁺ = f_z(...) updated at events
- Discrete variables (m): m⁺ = f_m(...) for integers/booleans
- Event indicators (c): event when c crosses zero
- Inputs (u): control signals
- Parameters (p): time-independent constants
- Outputs (y): y = f_y(x, u, p) observables/diagnostics

Example:
    @symbolic
    class States:
        h: ca.SX = state(1, 10.0, "height (m)")
        v: ca.SX = state(1, 0.0, "velocity (m/s)")

    @symbolic
    class Inputs:
        thrust: ca.SX = input_var(desc="thrust (N)")

    @symbolic
    class Params:
        m: ca.SX = param(1.0, "mass (kg)")
        g: ca.SX = param(9.81, "gravity (m/s^2)")

    model = ModelSX.create(States, Inputs, Params)

    x, u, p = model.x, model.u, model.p

    f_x = ca.vertcat(x.v, u.thrust / p.m - p.g)
    model.build(f_x=f_x, integrator='rk4')
"""

import copy
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Generic, TypeVar, Union

import casadi as ca
import numpy as np
from beartype import beartype

from .composition import CompositionMixin, SignalRef, SubmodelProxy
from .decorators import compose_states, symbolic
from .fields import (
    algebraic_var,
    dependent_var,
    discrete_state,
    discrete_var,
    event_indicator,
    input_var,
    output_var,
    param,
    quadrature_var,
    state,
)

__all__ = [
    "ModelSX",
    "ModelMX",
    "Trajectory",
    "EmptyOutputs",
]


@symbolic
class EmptyOutputs:
    """Empty outputs class for models without outputs.
    
    Use this instead of defining an empty @symbolic class Outputs: pass
    to avoid boilerplate while being explicit about having no outputs.
    
    Example:
        from cyecca.dynamics import ModelSX, EmptyOutputs
        
        model = ModelSX.create(States, Inputs, Params, EmptyOutputs)
    """
    pass


# Type variables for generic model classes
TState = TypeVar("TState")
TInput = TypeVar("TInput")
TParam = TypeVar("TParam")
TOutput = TypeVar("TOutput")
TAlgebraic = TypeVar("TAlgebraic")
TDependent = TypeVar("TDependent")
TQuadrature = TypeVar("TQuadrature")
TDiscreteState = TypeVar("TDiscreteState")
TDiscreteVar = TypeVar("TDiscreteVar")
TEventIndicator = TypeVar("TEventIndicator")

# Type variables for trajectory classes
TStateTrajectory = TypeVar("TStateTrajectory")
TOutputTrajectory = TypeVar("TOutputTrajectory")
TDiscreteStateTrajectory = TypeVar("TDiscreteStateTrajectory")
TDiscreteVarTrajectory = TypeVar("TDiscreteVarTrajectory")
TQuadratureTrajectory = TypeVar("TQuadratureTrajectory")


def _create_trajectory_class(original_class, class_suffix="Trajectory"):
    """Create a trajectory version of a dataclass where each field is an array.
    
    For a field that was (dim, 1), it becomes (n_steps, dim).
    This enables convenient matplotlib plotting: plt.plot(traj.t, traj.x.p) plots all components.
    
    Parameters
    ----------
    original_class : type
        The original dataclass with _field_info
    class_suffix : str
        Suffix to append to class name
        
    Returns
    -------
    type
        New dataclass with array fields
    """
    if not hasattr(original_class, '_field_info'):
        raise ValueError(f"Class {original_class} must have _field_info attribute")
    
    # Build new class with array fields
    new_class_name = original_class.__name__ + class_suffix
    
    # Create class dynamically with proper field storage
    class_dict = {
        '__annotations__': {},
        '_original_class': original_class,
        '_field_info': {},
    }
    
    # Add fields as numpy arrays
    for fname, finfo in original_class._field_info.items():
        class_dict['__annotations__'][fname] = np.ndarray
        class_dict['_field_info'][fname] = finfo.copy()
    
    # Add from_matrix classmethod to construct from simulation results
    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        """Create trajectory from matrix (n_states x n_steps).
        
        Parameters
        ----------
        matrix : np.ndarray
            State matrix with shape (n_states, n_steps)
            
        Returns
        -------
        instance
            Trajectory instance with fields populated as (n_steps, dim) arrays
        """
        instance = cls()
        offset = 0
        for fname, finfo in cls._field_info.items():
            dim = finfo['dim']
            if dim == 1:
                # Single dimension: extract row and transpose to (n_steps,)
                setattr(instance, fname, matrix[offset:offset+1, :].T.squeeze())
                offset += 1
            else:
                # Multi-dimension: extract rows and transpose to (n_steps, dim)
                setattr(instance, fname, matrix[offset:offset+dim, :].T)
                offset += dim
        return instance
    
    class_dict['from_matrix'] = from_matrix
    
    # Create the new class
    TrajClass = type(new_class_name, (), class_dict)
    
    return TrajClass


@dataclass
class Trajectory(Generic[TStateTrajectory, TDiscreteStateTrajectory, TDiscreteVarTrajectory, 
                         TQuadratureTrajectory, TOutputTrajectory]):
    """Container for simulation trajectory data with type-safe field access.
    
    Holds time history and state/output trajectories from simulation.
    Each state field becomes an array with shape (n_steps, dim) for convenient plotting.
    
    The generic type parameters ensure full type safety and autocomplete for all fields.
    
    Example
    -------
    >>> model = model.simulate(0, 10, 0.01)  # doctest: +SKIP
    >>> traj = model.trajectory  # doctest: +SKIP
    >>> plt.plot(traj.t, traj.x.p)  # Plot all position components (n_steps, dim)  # doctest: +SKIP
    """
    
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    x: TStateTrajectory | None = None  # State trajectory with full type info
    z: TDiscreteStateTrajectory | None = None  # Discrete state trajectory  
    m: TDiscreteVarTrajectory | None = None  # Discrete variable trajectory
    q: TQuadratureTrajectory | None = None  # Quadrature trajectory
    y: TOutputTrajectory | None = None  # Output trajectory
    
    @classmethod
    def from_history(cls, sim_history: dict, model: 'ModelSX'):
        """Create Trajectory from simulation history dict.
        
        Parameters
        ----------
        sim_history : dict
            Dictionary with 't', 'x', and optionally 'z', 'm', 'q', 'y' arrays
        model : ModelSX
            Model instance with type information
            
        Returns
        -------
        Trajectory
            Trajectory instance with typed field access
        """
        traj = cls()
        traj.t = sim_history['t']
        
        # Create state trajectory class if not already cached
        if not hasattr(model, '_StateTrajectory'):
            model._StateTrajectory = _create_trajectory_class(model.state_type, "Trajectory")
        
        traj.x = model._StateTrajectory.from_matrix(sim_history['x'])
        
        if 'z' in sim_history and model.discrete_state_type is not None:
            if not hasattr(model, '_DiscreteStateTrajectory'):
                model._DiscreteStateTrajectory = _create_trajectory_class(
                    model.discrete_state_type, "Trajectory"
                )
            traj.z = model._DiscreteStateTrajectory.from_matrix(sim_history['z'])
        
        if 'm' in sim_history and model.discrete_var_type is not None:
            if not hasattr(model, '_DiscreteVarTrajectory'):
                model._DiscreteVarTrajectory = _create_trajectory_class(
                    model.discrete_var_type, "Trajectory"
                )
            traj.m = model._DiscreteVarTrajectory.from_matrix(sim_history['m'])
        
        if 'q' in sim_history and model.quadrature_type is not None:
            if not hasattr(model, '_QuadratureTrajectory'):
                model._QuadratureTrajectory = _create_trajectory_class(
                    model.quadrature_type, "Trajectory"
                )
            traj.q = model._QuadratureTrajectory.from_matrix(sim_history['q'])
        
        if 'y' in sim_history and model.output_type is not None:
            if not hasattr(model, '_OutputTrajectory'):
                model._OutputTrajectory = _create_trajectory_class(
                    model.output_type, "Trajectory"
                )
            traj.y = model._OutputTrajectory.from_matrix(sim_history['y'])
        
        return traj


@beartype
class ModelSX(CompositionMixin, Generic[TState, TInput, TParam]):
    """Type-safe SX model with full hybrid DAE support.

    Supports:
    - Continuous states (x): dx/dt = f_x(...)
    - Algebraic variables (z_alg): 0 = g(x, z_alg, u, p)
    - Outputs (y): y = f_y(x, u, p)
    - Dependent variables (dep): dep = f_dep(x, u, p)
    - Quadratures (q): dq/dt = f_q(x, u, p)
    - Discrete states (z): z⁺ = f_z(...) at events
    - Discrete variables (m): m⁺ = f_m(...) at events
    - Event indicators (c): event when c = 0

    Example:
        model = ModelSX.create(States, Inputs, Params)

        x = model.x()
        u = model.u()
        p = model.p()

        f_x = ca.vertcat(x.v, u.thrust / p.m - p.g)
        model.build(f_x=f_x, integrator='rk4')
    """

    # Required types
    x: TState
    u: TInput
    p: TParam
    x0: TState
    u0: TInput
    p0: TParam

    # Optional types (set if provided)
    z_alg: Any = None  # Algebraic variables
    y: Any = None  # Dependent variables
    q: Any = None  # Quadrature states
    z: Any = None  # Discrete states
    m: Any = None  # Discrete variables
    c: Any = None  # Event indicators
    out: Any = None  # Outputs

    def __init__(
        self,
        state_type: type[TState],
        input_type: type[TInput],
        param_type: type[TParam],
        output_type: type[TOutput],
        algebraic_type: type[TAlgebraic] | None = None,
        dependent_type: type[TDependent] | None = None,
        quadrature_type: type[TQuadrature] | None = None,
        discrete_state_type: type[TDiscreteState] | None = None,
        discrete_var_type: type[TDiscreteVar] | None = None,
        event_indicator_type: type[TEventIndicator] | None = None,
    ):
        """Initialize typed hybrid DAE model.

        Prefer using ModelSX.create() for automatic type inference.
        """
        self._sym = ca.SX
        self.state_type = state_type
        self.input_type = input_type
        self.param_type = param_type
        self.output_type = output_type
        self.algebraic_type = algebraic_type
        self.dependent_type = dependent_type
        self.quadrature_type = quadrature_type
        self.discrete_state_type = discrete_state_type
        self.discrete_var_type = discrete_var_type
        self.event_indicator_type = event_indicator_type

        # Create required symbolic instances
        self.x = state_type.symbolic(ca.SX)
        self.u = input_type.symbolic(ca.SX)
        self.p = param_type.symbolic(ca.SX)

        # Create required numeric defaults
        self.x0 = state_type.numeric()
        self.u0 = input_type.numeric()
        self.p0 = param_type.numeric()

        # Create required output
        self.y = output_type.symbolic(ca.SX)
        self.y0 = output_type.numeric()

        # Create optional types if provided

        if algebraic_type:
            self.z_alg = algebraic_type.symbolic(ca.SX)
            self.z_alg0 = algebraic_type.numeric()

        if dependent_type:
            self.dep = dependent_type.symbolic(ca.SX)
            self.dep0 = dependent_type.numeric()

        if quadrature_type:
            self.q = quadrature_type.symbolic(ca.SX)
            self.q0 = quadrature_type.numeric()

        if discrete_state_type:
            self.z = discrete_state_type.symbolic(ca.SX)
            self.z0 = discrete_state_type.numeric()

        if discrete_var_type:
            self.m = discrete_var_type.symbolic(ca.SX)
            self.m0 = discrete_var_type.numeric()

        if event_indicator_type:
            self.c = event_indicator_type.symbolic(ca.SX)

        # Create connection helpers for autocomplete
        # These are separate from the actual u/y objects used for building
        class ConnectionHelper:
            """Helper for accessing signals in connect() calls."""

            def __init__(self, prefix: str, obj):
                object.__setattr__(self, '_prefix', prefix)
                object.__setattr__(self, '_obj', obj)

            def __getattr__(self, attr: str):
                # Handle special attributes that deepcopy/pickle looks for
                if attr in ('__setstate__', '__getstate__', '__dict__',
                           '__getnewargs__', '__getnewargs_ex__'):
                    raise AttributeError(attr)
                
                # Avoid infinite recursion by checking object dict directly
                try:
                    obj = object.__getattribute__(self, '_obj')
                    # Check if attribute exists in obj's type annotations
                    if hasattr(type(obj), '__annotations__') and \
                       attr in type(obj).__annotations__:
                        return ModelSX.SignalRef(
                            object.__getattribute__(self, '_prefix'), attr)
                except AttributeError:
                    pass
                raise AttributeError(f"Signal '{attr}' not found")

        self.inputs = ConnectionHelper("u", self.u)
        if output_type:
            self.outputs = ConnectionHelper("y", self.y)

    @classmethod
    def create(
        cls,
        state_type: type[TState],
        input_type: type[TInput],
        param_type: type[TParam],
        output_type: type[TOutput],
        **kwargs,
    ):
        """Create a fully-typed model instance with automatic type inference.

        This factory method ensures proper type inference for IDE autocomplete.

        Args:
            state_type: Dataclass decorated with @symbolic
            input_type: Dataclass decorated with @symbolic
            param_type: Dataclass decorated with @symbolic
            output_type: Output type dataclass decorated with @symbolic
            **kwargs: Optional types (algebraic_type, dependent_type, etc.)

        Returns:
            Fully-typed ModelSX instance

        Usage:
            model = ModelSX.create(States, Inputs, Params, Outputs)
            # IDE now knows exact types for autocomplete!

            x = model.x()  # x has full autocomplete
            u = model.u()  # u has full autocomplete
            p = model.p()  # p has full autocomplete
            y = model.y()  # y has full autocomplete
        """
        return cls(state_type, input_type, param_type, output_type=output_type, **kwargs)

    @classmethod
    def compose(
        cls,
        submodels: dict[str, "ModelSX"],
        state_type: type[TState] = None,
        input_type: type[TInput] = None,
        param_type: type[TParam] = None,
        output_type: type[TOutput] = None,
    ):
        """Compose multiple submodels into a single parent model.

        Creates a parent model with submodels already attached, ready for connections.
        Automatically composes states from all submodels if not specified.

        Args:
            submodels: Dictionary of {name: model} pairs
            state_type: Parent state type (auto-composed from submodels if None)
            input_type: Parent input type (empty if None)
            param_type: Parent parameter type (empty if None)
            output_type: Parent output type (empty if None)

        Returns:
            Parent model with submodels attached as attributes

        Example:
            >>> plant = sportcub()  # doctest: +SKIP
            >>> controller = autolevel_controller()  # doctest: +SKIP
            >>> parent = ModelSX.compose({"plant": plant, "controller": controller})  # doctest: +SKIP
            >>> parent.connect(controller.u.q, plant.x.r)  # doctest: +SKIP
        """
        # Auto-compose state type if not provided
        if state_type is None:
            state_types = [sub.state_type for sub in submodels.values()]
            state_type = compose_states(*state_types)

        # Create empty types if not provided
        if input_type is None:

            @symbolic
            class EmptyInputs:
                pass

            input_type = EmptyInputs

        if param_type is None:

            @symbolic
            class EmptyParams:
                pass

            param_type = EmptyParams

        if output_type is None:

            @symbolic
            class EmptyOutputs:
                pass

            output_type = EmptyOutputs

        # Create parent model
        parent = cls.create(state_type, input_type, param_type, output_type)

        # Add all submodels and create proxies for connection API
        for name, submodel in submodels.items():
            parent.add_submodel(name, submodel)
            # Submodel proxy is already created by add_submodel

        return parent

    @beartype
    def build(
        self,
        f_x: ca.SX | ca.MX,
        f_y: ca.SX | ca.MX | None = None,
        f_dep: ca.SX | ca.MX | None = None,
        f_alg: ca.SX | ca.MX | None = None,
        f_q: ca.SX | ca.MX | None = None,
        f_z: ca.SX | ca.MX | None = None,
        f_m: ca.SX | ca.MX | None = None,
        f_c: ca.SX | ca.MX | None = None,
        integrator: str = "rk4",
        integrator_options: dict = None,
    ):
        """Build model with all evolution functions.

        Args:
            f_x: Continuous dynamics dx/dt (required)
            f_y: Output expressions y = f_y(x, u, p)
            f_dep: Dependent variable expressions (less commonly used)
            f_alg: Algebraic constraints (for DAE)
            f_q: Quadrature dynamics dq/dt
            f_z: Discrete state update z⁺
            f_m: Discrete variable update m⁺
            f_c: Event indicator functions
            integrator: Integration method ("rk4", "euler", "idas")
            integrator_options: Integrator settings (e.g., {'N': 10})
        """
        if integrator_options is None:
            integrator_options = {}

        # Build f_x (required)
        self._build_f_x(f_x)

        # Build optional functions
        if f_y is not None:
            self._build_f_y(f_y)
        if f_dep is not None:
            self._build_f_dep(f_dep)
        if f_alg is not None:
            self._build_f_alg(f_alg)
        if f_q is not None:
            self._build_f_q(f_q)
        if f_z is not None:
            self._build_f_z(f_z)
        if f_m is not None:
            self._build_f_m(f_m)
        if f_c is not None:
            self._build_f_c(f_c)

        # Build integrator
        if integrator == "rk4":
            self._build_rk4_integrator(integrator_options)
        elif integrator == "euler":
            self._build_euler_integrator()
        elif integrator == "idas":
            self._build_idas_integrator(integrator_options)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        # Build index maps
        self._build_index_maps()

    def _build_f_x(self, f_x_expr):
        """Build continuous dynamics function.

        Note: The 'dep' input here refers to dependent variables (if present),
        not outputs. Dependent variables are computed values that feed into
        dynamics but are not integrated states.
        """
        inputs = [self.x.as_vec()]
        input_names = ["x"]

        if hasattr(self, "dep") and self.dep is not None:
            inputs.append(self.dep.as_vec())
            input_names.append("dep")
        if self.z is not None:
            inputs.append(self.z.as_vec())
            input_names.append("z")
        if self.m is not None:
            inputs.append(self.m.as_vec())
            input_names.append("m")

        inputs.append(self.u.as_vec())
        inputs.append(self.p.as_vec())
        input_names.extend(["u", "p"])

        self.f_x = ca.Function("f_x", inputs, [f_x_expr], input_names, ["dx_dt"])

    def _build_f_y(self, f_y_expr):
        """Build output function.

        Outputs (y) are observables/diagnostics computed from states.
        """
        inputs, names = self._get_standard_inputs()
        self.f_y = ca.Function("f_y", inputs, [f_y_expr], names, ["y"])

    def _build_f_dep(self, f_dep_expr):
        """Build dependent variable function."""
        inputs, names = self._get_standard_inputs(include_dep=False)
        self.f_dep = ca.Function("f_dep", inputs, [f_dep_expr], names, ["dep"])

    def _build_f_alg(self, f_alg_expr):
        """Build algebraic constraint function: 0 = g(x, z_alg, u, p)."""
        inputs = [
            self.x.as_vec(),
            self.z_alg.as_vec(),
            self.u.as_vec(),
            self.p.as_vec(),
        ]
        names = ["x", "z_alg", "u", "p"]
        self.f_alg = ca.Function("f_alg", inputs, [f_alg_expr], names, ["residual"])

    def _build_f_q(self, f_q_expr):
        """Build quadrature dynamics."""
        inputs, names = self._get_standard_inputs()
        self.f_q = ca.Function("f_q", inputs, [f_q_expr], names, ["dq_dt"])

    def _build_f_z(self, f_z_expr):
        """Build discrete state update function."""
        inputs, names = self._get_standard_inputs()
        self.f_z = ca.Function("f_z", inputs, [f_z_expr], names, ["z_plus"])

    def _build_f_m(self, f_m_expr):
        """Build discrete variable update function."""
        inputs, names = self._get_standard_inputs()
        self.f_m = ca.Function("f_m", inputs, [f_m_expr], names, ["m_plus"])

    def _build_f_c(self, f_c_expr):
        """Build event indicator function."""
        inputs, names = self._get_standard_inputs()
        self.f_c = ca.Function("f_c", inputs, [f_c_expr], names, ["indicators"])

    def _get_standard_inputs(self, include_dep=True):
        """Get standard input list for functions."""
        inputs = [self.x.as_vec()]
        names = ["x"]

        if include_dep and hasattr(self, "dep") and self.dep is not None:
            inputs.append(self.dep.as_vec())
            names.append("dep")
        if self.z is not None:
            inputs.append(self.z.as_vec())
            names.append("z")
        if self.m is not None:
            inputs.append(self.m.as_vec())
            names.append("m")

        inputs.append(self.u.as_vec())
        inputs.append(self.p.as_vec())
        names.extend(["u", "p"])

        return inputs, names

    def _build_rk4_integrator(self, options: dict):
        """Build RK4 integrator."""
        dt_sym = ca.SX.sym("dt")
        N = options.get("N", 10)

        # For composed models, use the combined state size
        if hasattr(self, "_composed") and self._composed:
            x_sym = ca.SX.sym("x", self._total_composed_states)
        else:
            x_sym = self.x.as_vec()

        u_sym = self.u.as_vec()
        p_sym = self.p.as_vec()

        # Handle discrete states/vars if present (constant during integration)
        z_sym = ca.SX.sym("z", self.z.size1()) if self.z is not None else None
        m_sym = ca.SX.sym("m", self.m.size1()) if self.m is not None else None

        # Build wrapper function for RK4 that matches f(x, u, p) signature
        def build_f_wrapper():
            x_in = ca.SX.sym("x", x_sym.size1())
            u_in = ca.SX.sym("u", u_sym.size1())
            p_in = ca.SX.sym("p", p_sym.size1())

            # Build argument list matching f_x signature
            f_x_args = [x_in]

            # Add dependent variables if present (use defaults)
            if hasattr(self, "dep") and self.dep is not None:
                f_x_args.append(self.dep0.as_vec())

            # Add discrete states/vars (use symbols that will be bound later)
            if z_sym is not None:
                f_x_args.append(z_sym)
            if m_sym is not None:
                f_x_args.append(m_sym)

            f_x_args.extend([u_in, p_in])

            # Evaluate dynamics
            dx_dt = self.f_x(*f_x_args)
            return ca.Function("f_wrapped", [x_in, u_in, p_in], [dx_dt], ["x", "u", "p"], ["dx_dt"])

        from . import integrators

        f_wrapped = build_f_wrapper()
        rk4_step = integrators.rk4(f_wrapped, dt_sym, name="rk4", N=N)

        # Build f_step with appropriate inputs
        step_inputs = [x_sym]
        step_names = ["x"]

        if z_sym is not None:
            step_inputs.append(z_sym)
            step_names.append("z")
        if m_sym is not None:
            step_inputs.append(m_sym)
            step_names.append("m")

        step_inputs.extend([u_sym, p_sym, dt_sym])
        step_names.extend(["u", "p", "dt"])

        # Call RK4 with just x, u, p (discrete states already bound in wrapper)
        self.f_step = ca.Function(
            "f_step",
            step_inputs,
            [rk4_step(x_sym, u_sym, p_sym, dt_sym)],
            step_names,
            ["x_next"],
        )

    def _build_euler_integrator(self):
        """Build Euler integrator."""
        dt_sym = ca.SX.sym("dt")

        # For composed models, use the combined state size
        if hasattr(self, "_composed") and self._composed:
            x_sym = ca.SX.sym("x", self._total_composed_states)
        else:
            x_sym = self.x.as_vec()

        u_sym = self.u.as_vec()
        p_sym = self.p.as_vec()

        # Build argument list matching f_x signature
        f_x_args = [x_sym]

        # Add dependent variables if present (not currently used in integrators)
        if hasattr(self, "dep") and self.dep is not None:
            # For now, use zero/default values for dep in integrator
            # In a full implementation, would evaluate f_dep here
            f_x_args.append(self.dep0.as_vec())

        # Add discrete states if present (constant during integration)
        if self.z is not None:
            z_sym = ca.SX.sym("z", self.z.size1())
            f_x_args.append(z_sym)
        else:
            z_sym = None

        # Add discrete variables if present (constant during integration)
        if self.m is not None:
            m_sym = ca.SX.sym("m", self.m.size1())
            f_x_args.append(m_sym)
        else:
            m_sym = None

        f_x_args.extend([u_sym, p_sym])

        # Evaluate dynamics
        dx_dt = self.f_x(*f_x_args)
        x_next = x_sym + dt_sym * dx_dt

        # Build f_step with appropriate inputs
        step_inputs = [x_sym]
        step_names = ["x"]

        if z_sym is not None:
            step_inputs.append(z_sym)
            step_names.append("z")
        if m_sym is not None:
            step_inputs.append(m_sym)
            step_names.append("m")

        step_inputs.extend([u_sym, p_sym, dt_sym])
        step_names.extend(["u", "p", "dt"])

        self.f_step = ca.Function("f_step", step_inputs, [x_next], step_names, ["x_next"])

    def _build_idas_integrator(self, options: dict):
        """Build IDAS DAE integrator using CasADi's integrator interface.

        IDAS can handle:
        - Pure ODEs: dx/dt = f(x, u, p)
        - DAEs with algebraic constraints:
            dx/dt = f(x, z_alg, u, p)
            0 = g(x, z_alg, u, p)

        Options can include:
        - abstol: absolute tolerance (default 1e-8)
        - reltol: relative tolerance (default 1e-6)
        - max_num_steps: maximum number of steps (default 10000)

        Note: Since CasADi's integrator requires numeric t0/tf at creation,
        we create the integrator for unit time [0, 1] and scale by dt.
        """
        # Get symbolic type (SX or MX)
        sym = ca.SX if self._sym == ca.SX else ca.MX

        # Create symbolic variables for the DAE
        t_sym = sym.sym("t")  # Time variable (for time-varying systems)
        x_dae = sym.sym("x", self.x.as_vec().size1())
        u_dae = sym.sym("u", self.u.as_vec().size1())
        p_dae = sym.sym("p", self.p.as_vec().size1())

        # Handle discrete states/vars (constant during continuous integration)
        # These will be parameters to the integrator
        z_disc_sym = sym.sym("z", self.z.size1()) if self.z is not None else None
        m_sym = sym.sym("m", self.m.size1()) if self.m is not None else None

        # dt will be a parameter for time scaling
        dt_param = sym.sym("dt")

        # Build combined parameter vector for integrator
        p_combined = [u_dae, p_dae]
        if z_disc_sym is not None:
            p_combined.append(z_disc_sym)
        if m_sym is not None:
            p_combined.append(m_sym)
        p_combined.append(dt_param)  # Add dt as parameter
        p_vec = ca.vertcat(*p_combined)

        # Build ODE right-hand side using original f_x function
        # We need to unpack the combined parameter vector
        idx = 0
        u_extract = p_vec[idx : idx + u_dae.size1()]
        idx += u_dae.size1()
        p_extract = p_vec[idx : idx + p_dae.size1()]
        idx += p_dae.size1()

        f_x_args = [x_dae]

        # Add dependent variables if present (use defaults for now)
        if hasattr(self, "dep") and self.dep is not None:
            f_x_args.append(self.dep0.as_vec())

        # Add discrete states/vars if present
        if z_disc_sym is not None:
            z_extract = p_vec[idx : idx + z_disc_sym.size1()]
            idx += z_disc_sym.size1()
            f_x_args.append(z_extract)
        if m_sym is not None:
            m_extract = p_vec[idx : idx + m_sym.size1()]
            idx += m_sym.size1()
            f_x_args.append(m_extract)

        # Extract dt from parameter vector
        dt_extract = p_vec[idx]

        f_x_args.extend([u_extract, p_extract])

        # Check if we have algebraic constraints
        has_alg = hasattr(self, "f_alg") and self.z_alg is not None

        if has_alg:
            # DAE mode: include algebraic variables
            z_alg_dae = sym.sym("z_alg", self.z_alg.as_vec().size1())

            # ODE: dx/dt = f(x, u, p, ...)
            ode_rhs = self.f_x(*f_x_args)

            # Algebraic equation: 0 = g(x, z_alg, u, p)
            alg_rhs = self.f_alg(x_dae, z_alg_dae, u_extract, p_extract)

            # Build DAE dictionary for CasADi integrator
            dae = {
                "t": t_sym,
                "x": x_dae,
                "z": z_alg_dae,
                "p": p_vec,
                "ode": ode_rhs,
                "alg": alg_rhs,
            }
        else:
            # Pure ODE mode: dx/dt = f(x, u, p, ...)
            ode_rhs = self.f_x(*f_x_args)

            # Build ODE dictionary for CasADi integrator
            dae = {
                "t": t_sym,
                "x": x_dae,
                "p": p_vec,
                "ode": ode_rhs,
            }

        # Set up integrator options
        integrator_opts = {
            "abstol": options.get("abstol", 1e-8),
            "reltol": options.get("reltol", 1e-6),
            "max_num_steps": options.get("max_num_steps", 10000),
        }

        # Additional IDAS-specific options
        for key in [
            "max_step_size",
            "min_step_size",
            "init_step_size",
            "exact_jacobian",
            "linear_solver",
            "max_krylov",
            "sensitivity_method",
        ]:
            if key in options:
                integrator_opts[key] = options[key]

        # Create IDAS integrator for [0, 1] interval
        # We'll scale by dt when calling it
        integrator_unit = ca.integrator("idas_unit", "idas", dae, 0.0, 1.0, integrator_opts)

        # Now build f_step that takes dt as input and scales appropriately
        x_sym = sym.sym("x", x_dae.size1())
        u_sym = sym.sym("u", u_dae.size1())
        p_sym = sym.sym("p", p_dae.size1())
        dt_sym = sym.sym("dt")

        # Rebuild parameter vector with actual symbols
        p_call = [u_sym, p_sym]
        step_inputs = [x_sym]
        step_names = ["x"]

        if z_disc_sym is not None:
            z_sym_call = sym.sym("z", z_disc_sym.size1())
            p_call.append(z_sym_call)
            step_inputs.append(z_sym_call)
            step_names.append("z")
        if m_sym is not None:
            m_sym_call = sym.sym("m", m_sym.size1())
            p_call.append(m_sym_call)
            step_inputs.append(m_sym_call)
            step_names.append("m")

        step_inputs.extend([u_sym, p_sym, dt_sym])
        step_names.extend(["u", "p", "dt"])

        # Add dt_sym to parameter vector for integrator call
        p_call.append(dt_sym)

        # Call the unit integrator but with scaled grid: [0, dt]
        # Use the grid-based form: integrator(x0=..., p=...) with implicit grid [0, 1]
        # Then we need to scale time...
        # Actually, simpler: create a new integrator for each dt value
        # But that's not efficient. Instead, use integrator map or recreate with symbolic dt

        # Alternative: Use the integrator with grid points
        # Form: integrator(..., t0, [t1, t2, ...], opts)
        # But this still requires numeric values

        # Best approach: Create scaled DAE where time is scaled by dt
        # tau = t/dt, so dt*dtau = dt, and d/dtau = dt * d/dt
        # This means ode_scaled = dt * ode_rhs

        dae_scaled = dict(dae)
        dae_scaled["ode"] = dt_extract * dae["ode"]
        if has_alg:
            # Algebraic constraints don't change with time scaling
            pass

        # Create integrator with scaled dynamics
        integrator_scaled = ca.integrator("idas_scaled", "idas", dae_scaled, 0.0, 1.0, integrator_opts)

        # Call the integrator
        if has_alg:
            z_alg_guess_sym = sym.sym("z_alg0", z_alg_dae.size1())
            integ_result = integrator_scaled(x0=x_sym, z0=z_alg_guess_sym, p=ca.vertcat(*p_call))
            step_inputs.insert(-1, z_alg_guess_sym)  # Insert before dt
            step_names.insert(-1, "z_alg0")
        else:
            integ_result = integrator_scaled(x0=x_sym, p=ca.vertcat(*p_call))

        x_next = integ_result["xf"]

        self.f_step = ca.Function(
            "f_step",
            step_inputs,
            [x_next],
            step_names,
            ["x_next"],
        )

    def _build_index_maps(self):
        """Build index maps for state/input/parameter/output access (internal use only)."""
        # State indices (private - internal use only)
        self._state_index = {}
        offset = 0
        for fname, finfo in self.state_type._field_info.items():
            dim = finfo["dim"]
            if dim == 1:
                self._state_index[fname] = offset
                offset += 1
            else:
                for i in range(dim):
                    self._state_index[f"{fname}[{i}]"] = offset + i
                offset += dim

        # Input indices (private - internal use only)
        self._input_index = {}
        offset = 0
        for fname, finfo in self.input_type._field_info.items():
            dim = finfo["dim"]
            if dim == 1:
                self._input_index[fname] = offset
                offset += 1
            else:
                for i in range(dim):
                    self._input_index[f"{fname}[{i}]"] = offset + i
                offset += dim

        # Parameter indices (private - internal use only)
        self._parameter_index = {}
        offset = 0
        for fname, finfo in self.param_type._field_info.items():
            dim = finfo["dim"]
            if dim == 1:
                self._parameter_index[fname] = offset
                offset += 1
            else:
                for i in range(dim):
                    self._parameter_index[f"{fname}[{i}]"] = offset + i
                offset += dim

        # Output indices (if outputs exist) (private - internal use only)
        if self.output_type is not None:
            self._output_index = {}
            offset = 0
            for fname, finfo in self.output_type._field_info.items():
                dim = finfo["dim"]
                self._output_index[fname] = offset
                if dim > 1:
                    for i in range(dim):
                        self._output_index[f"{fname}[{i}]"] = offset + i
                offset += dim

        # Create ordered name lists
        self.state_names = [n for n, _ in sorted(self._state_index.items(), key=lambda kv: kv[1])]
        self.input_names = (
            [n for n, _ in sorted(self._input_index.items(), key=lambda kv: kv[1])] if self._input_index else []
        )
        self.parameter_names = (
            [n for n, _ in sorted(self._parameter_index.items(), key=lambda kv: kv[1])] if self._parameter_index else []
        )
        self.output_names = (
            [n for n, _ in sorted(self._output_index.items(), key=lambda kv: kv[1])]
            if hasattr(self, "_output_index")
            else []
        )

    def _update_state_in_place(self, target_obj, new_obj):
        """Update target object fields in-place from new object.

        Preserves object identity while updating all field values.
        Used to maintain external references (e.g., adapters) during in-place simulation.
        """
        for field_name in new_obj.__dataclass_fields__:
            setattr(target_obj, field_name, getattr(new_obj, field_name))

    def simulate(
        self,
        t0: float,
        tf: float,
        dt: float,
        u_func: Callable = None,
        detect_events: bool = False,
        compute_output: bool = False,
        in_place: bool = False,
    ):
        """Simulate model from t0 to tf with event detection.

        Takes the current model state (x, p, z, m, q) and returns an updated copy
        (or updates in-place if in_place=True).

        Parameters
        ----------
        t0 : float
            Initial time
        tf : float
            Final time
        dt : float
            Timestep
        u_func : Callable, optional
            Control function (t, model) -> u_vec. If None, uses u0.
        detect_events : bool, optional
            Whether to detect and handle zero-crossings
        compute_output : bool, optional
            Whether to compute outputs during simulation. Default False for efficiency.
            If False, trajectory.y will be None.
        in_place : bool, optional
            If True, update this model's state directly instead of returning a copy.
            Useful for real-time simulation where creating copies is expensive.
            Default False for backward compatibility.

        Returns
        -------
        ModelSX
            Copy of the model with updated state after simulation (or self if in_place=True)
            
        Raises
        ------
        RuntimeError
            If NaN or Inf detected in state during simulation
        """
        if not hasattr(self, "f_step"):
            raise ValueError("Model not built. Call build() first.")

        # Create a copy unless in_place is True
        result_model = self if in_place else copy.deepcopy(self)

        # Get initial values from the model
        if hasattr(result_model, "_composed") and result_model._composed:
            x_curr = result_model.x0_composed
        else:
            x_curr = result_model.x0.as_vec()

        p_vec = result_model.p0.as_vec()
        z_curr = result_model.z0.as_vec() if result_model.z is not None else None
        m_curr = result_model.m0.as_vec() if result_model.m is not None else None
        q_curr = result_model.q0.as_vec() if result_model.q is not None else None

        t_hist = [t0]
        x_hist = [x_curr]
        if result_model.z is not None:
            z_hist = [z_curr]
        if result_model.m is not None:
            m_hist = [m_curr]
        if result_model.q is not None:
            q_hist = [q_curr]
        y_hist = [] if (compute_output and hasattr(result_model, "f_y")) else None

        # Track previous event indicator for zero-crossing detection
        c_prev = None
        if detect_events and hasattr(result_model, "f_c"):
            args = result_model._build_eval_args(x_curr, z_curr, m_curr, result_model.u0.as_vec(), p_vec)
            c_prev = float(result_model.f_c(*args))

        t = t0
        while t < tf - dt / 2:
            # Get control
            if u_func is not None:
                u_curr = u_func(t, result_model)
            else:
                u_curr = result_model.u0.as_vec()

            # Evaluate outputs before step
            if y_hist is not None:
                args = result_model._build_eval_args(x_curr, z_curr, m_curr, u_curr, p_vec)
                y_val = result_model.f_y(*args)
                y_hist.append(y_val)

            # Integration step - build arguments for f_step
            step_args = [x_curr]
            if result_model.z is not None:
                step_args.append(z_curr)
            if result_model.m is not None:
                step_args.append(m_curr)
            step_args.extend([u_curr, p_vec, dt])

            x_next = result_model.f_step(*step_args)

            # Check for invalid values (NaN/Inf)
            x_next_full = ca.DM(x_next).full().flatten()
            if np.any(np.isnan(x_next_full)) or np.any(np.isinf(x_next_full)):
                raise RuntimeError(
                    f"NaN or Inf detected in state at t={t + dt:.6f}s during simulation. "
                    f"State values: {x_next_full}"
                )

            # Check for events
            if detect_events and hasattr(result_model, "f_c"):
                args = result_model._build_eval_args(x_next, z_curr, m_curr, u_curr, p_vec)
                c_curr = float(result_model.f_c(*args))

                # Detect zero-crossing
                if c_prev is not None and c_prev > 0 and c_curr <= 0:
                    # Event occurred!
                    if hasattr(result_model, "f_z") and z_curr is not None:
                        z_curr = result_model.f_z(*args)
                    if hasattr(result_model, "f_m"):
                        # f_m can reset either discrete variables or continuous states
                        # Check output dimension to determine which
                        m_reset = result_model.f_m(*args)
                        if m_curr is not None and m_reset.size1() == m_curr.size1():
                            # Reset discrete variables
                            m_curr = m_reset
                        elif m_reset.size1() == x_next.size1():
                            # Reset continuous states
                            x_next = m_reset
                        else:
                            # Assume it's for continuous states if sizes don't match m
                            x_next = m_reset

                c_prev = c_curr

            # Integrate quadratures if present
            if result_model.q is not None and hasattr(result_model, "f_q"):
                args = result_model._build_eval_args(x_curr, z_curr, m_curr, u_curr, p_vec)
                dq_dt = result_model.f_q(*args)
                q_curr = q_curr + dt * dq_dt

            # Store
            t += dt
            t_hist.append(t)
            x_hist.append(x_next)
            if result_model.z is not None:
                z_hist.append(z_curr)
            if result_model.m is not None:
                m_hist.append(m_curr)
            if result_model.q is not None:
                q_hist.append(q_curr)

            x_curr = x_next

        # Final output
        if y_hist is not None:
            args = result_model._build_eval_args(x_curr, z_curr, m_curr, u_curr, p_vec)
            y_val = result_model.f_y(*args)
            y_hist.append(y_val)

        # Update the result model with final state
        if hasattr(result_model, "_composed") and result_model._composed:
            result_model.x0_composed = x_curr
        else:
            new_state = result_model.state_type.from_vec(x_curr)
            if in_place:
                result_model._update_state_in_place(result_model.x0, new_state)
            else:
                result_model.x0 = new_state
        
        if z_curr is not None:
            new_z = result_model.discrete_state_type.from_vec(z_curr)
            if in_place and hasattr(result_model, 'z0'):
                result_model._update_state_in_place(result_model.z0, new_z)
            else:
                result_model.z0 = new_z
                
        if m_curr is not None:
            new_m = result_model._vec_to_discrete_var(m_curr)
            if in_place and hasattr(result_model, 'm0'):
                result_model._update_state_in_place(result_model.m0, new_m)
            else:
                result_model.m0 = new_m
                
        if q_curr is not None:
            new_q = result_model._vec_to_quadrature(q_curr)
            if in_place and hasattr(result_model, 'q0'):
                result_model._update_state_in_place(result_model.q0, new_q)
            else:
                result_model.q0 = new_q

        # Store history as dict for backward compatibility
        result_model._sim_history = {
            "t": np.array(t_hist), 
            "x": ca.hcat(x_hist).full()
        }
        if result_model.z is not None:
            result_model._sim_history["z"] = ca.hcat(z_hist).full()
        if result_model.m is not None:
            result_model._sim_history["m"] = ca.hcat(m_hist).full()
        if result_model.q is not None:
            result_model._sim_history["q"] = ca.hcat(q_hist).full()
        if y_hist is not None:
            result_model._sim_history["y"] = ca.hcat(y_hist).full()
        
        # Create typed Trajectory object
        result_model._trajectory = Trajectory.from_history(result_model._sim_history, result_model)

        return result_model
    
    @property
    def trajectory(self) -> Trajectory:
        """Access simulation results as typed Trajectory object.
        
        Returns
        -------
        Trajectory
            Trajectory with typed field access for plotting and analysis
            
        Example
        -------
        >>> model = model.simulate(0, 10, 0.01)  # doctest: +SKIP
        >>> plt.plot(model.trajectory.t, model.trajectory.x.p)  # doctest: +SKIP
        """
        if not hasattr(self, '_trajectory'):
            if hasattr(self, '_sim_history'):
                self._trajectory = Trajectory.from_history(self._sim_history, self)
            else:
                raise ValueError("No simulation history available. Run simulate() first.")
        return self._trajectory

    @property
    def y_current(self):
        """Evaluate outputs at current state (x0, u0, p0).
        
        Returns
        -------
        Output dataclass instance
            Current outputs evaluated at x0, u0, p0
            
        Example
        -------
        >>> model = model.simulate(0, 10, 0.01)  # doctest: +SKIP
        >>> print(model.y_current.ail)  # Access current aileron output  # doctest: +SKIP
        """
        if not hasattr(self, 'f_y'):
            raise ValueError("Model has no output function (f_y)")
        
        # Evaluate output function at current state
        result = self.f_y(
            x=self.x0.as_vec(),
            u=self.u0.as_vec(),
            p=self.p0.as_vec()
        )
        
        # Convert output vector to dataclass
        return self.output_type.from_vec(result)

    def _build_eval_args(self, x, z, m, u, p):
        """Build argument list for function evaluation.

        This matches the signature of functions built with _get_standard_inputs(),
        which may include dependent variables (dep) if they exist.
        """
        args = [x]
        if hasattr(self, "dep") and self.dep is not None:
            # Evaluate dependent variables and add to argument list
            # f_dep has signature (x, u, p) -> dep
            if hasattr(self, "f_dep"):
                dep_val = self.f_dep(x, u, p)
                args.append(dep_val)
            else:
                # If dep exists but f_dep not built, this is an error
                raise ValueError("Model has dep but f_dep was not built")
        if z is not None:
            args.append(z)
        if m is not None:
            args.append(m)
        args.append(u)
        args.append(p)
        return args  # ============================================================================

    def saturate(self, val, lower, upper):
        """Saturate value between bounds."""
        return ca.fmin(ca.fmax(val, lower), upper)

    # Backwards compatibility properties for legacy API
    @property
    def State(self):
        """Legacy API: access to state type class."""
        return self.state_type

    @property
    def Input(self):
        """Legacy API: access to input type class."""
        return self.input_type

    @property
    def Parameters(self):
        """Legacy API: access to parameter type class."""
        return self.param_type

    @property
    def output_sizes(self):
        """Legacy API: dictionary mapping output names to their dimensions."""
        if self.output_type is None:
            return {}
        return {fname: finfo["dim"] for fname, finfo in self.output_type._field_info.items()}


@beartype
class ModelMX(ModelSX[TState, TInput, TParam]):
    """Type-safe MX model for large-scale optimization.

    Same API as ModelSX but uses MX symbolic type for better
    performance with large-scale optimization problems.

    Usage:
        model = ModelMX.create(States, Inputs, Params)

        x = model.x()
        u = model.u()
        p = model.p()

        f_x = ca.vertcat(x.v, u.thrust / p.m - p.g)
        model.build(f_x=f_x, integrator='rk4')
    """

    @classmethod
    def create(
        cls,
        state_type: type[TState],
        input_type: type[TInput],
        param_type: type[TParam],
        **kwargs,
    ):
        """Create a fully-typed MX model instance with automatic type inference."""
        return cls(state_type, input_type, param_type, **kwargs)

    def __init__(
        self,
        state_type: type[TState],
        input_type: type[TInput],
        param_type: type[TParam],
        **kwargs,
    ):
        """Initialize MX-based model."""
        # Set MX before calling parent __init__
        self._sym = ca.MX

        # Store types
        self.state_type = state_type
        self.input_type = input_type
        self.param_type = param_type
        self.output_type = kwargs.get("output_type")
        self.algebraic_type = kwargs.get("algebraic_type")
        self.dependent_type = kwargs.get("dependent_type")
        self.quadrature_type = kwargs.get("quadrature_type")
        self.discrete_state_type = kwargs.get("discrete_state_type")
        self.discrete_var_type = kwargs.get("discrete_var_type")
        self.event_indicator_type = kwargs.get("event_indicator_type")

        # Create symbolic instances with MX
        self.x = state_type.symbolic(ca.MX)
        self.u = input_type.symbolic(ca.MX)
        self.p = param_type.symbolic(ca.MX)

        # Create numeric defaults
        self.x0 = state_type.numeric()
        self.u0 = input_type.numeric()
        self.p0 = param_type.numeric()

        # Create optional types if provided
        if self.output_type:
            self.y = self.output_type.symbolic(ca.MX)
            self.y0 = self.output_type.numeric()

        if self.algebraic_type:
            self.z_alg = self.algebraic_type.symbolic(ca.MX)
            self.z_alg0 = self.algebraic_type.numeric()

        if self.dependent_type:
            self.dep = self.dependent_type.symbolic(ca.MX)
            self.dep0 = self.dependent_type.numeric()

        if self.quadrature_type:
            self.q = self.quadrature_type.symbolic(ca.MX)
            self.q0 = self.quadrature_type.numeric()

        if self.discrete_state_type:
            self.z = self.discrete_state_type.symbolic(ca.MX)
            self.z0 = self.discrete_state_type.numeric()

        if self.discrete_var_type:
            self.m = self.discrete_var_type.symbolic(ca.MX)
            self.m0 = self.discrete_var_type.numeric()

        if self.event_indicator_type:
            self.c = self.event_indicator_type.symbolic(ca.MX)

    def _build_rk4_integrator(self, options: dict):
        """Build RK4 integrator with MX."""
        from . import integrators

        dt_sym = ca.MX.sym("dt")
        N = options.get("N", 10)

        rk4_step = integrators.rk4(self.f_x, dt_sym, name="rk4", N=N)

        xin = rk4_step.mx_in(0)
        uin = rk4_step.mx_in(1)
        pin = rk4_step.mx_in(2)

        self.f_step = ca.Function(
            "f_step",
            [xin, uin, pin, dt_sym],
            [rk4_step(xin, uin, pin, dt_sym)],
            ["x", "u", "p", "dt"],
            ["x_next"],
        )

    def _build_euler_integrator(self):
        """Build Euler integrator with MX."""
        dt_sym = ca.MX.sym("dt")

        # For composed models, use the combined state size
        if hasattr(self, "_composed") and self._composed:
            x_sym = ca.MX.sym("x", self._total_composed_states)
        else:
            x_sym = self.x.as_vec()

        u_sym = self.u.as_vec()
        p_sym = self.p.as_vec()

        dx_dt = self.f_x(x_sym, u_sym, p_sym)
        x_next = x_sym + dt_sym * dx_dt

        self.f_step = ca.Function(
            "f_step",
            [x_sym, u_sym, p_sym, dt_sym],
            [x_next],
            ["x", "u", "p", "dt"],
            ["x_next"],
        )
