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

    x = model.x()
    u = model.u()
    p = model.p()

    f_x = ca.vertcat(x.v, u.thrust / p.m - p.g)
    model.build(f_x=f_x, integrator='rk4')
"""

import copy
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Generic, TypeVar, Union

import casadi as ca
import numpy as np
from beartype import beartype

from .fields import (
    state,
    param,
    input_var,
    output_var,
    algebraic_var,
    dependent_var,
    quadrature_var,
    discrete_state,
    discrete_var,
    event_indicator,
)
from .decorators import symbolic, compose_states
from .composition import CompositionMixin, SignalRef, SubmodelProxy

__all__ = [
    "ModelSX",
    "ModelMX",
]

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
        output_type: type[TOutput] | None = None,
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

        # Create optional types if provided
        if output_type:
            self.y = output_type.symbolic(ca.SX)
            self.y0 = output_type.numeric()

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
                self._prefix = prefix
                self._obj = obj

            def __getattr__(self, attr: str):
                if hasattr(self._obj, attr):
                    return ModelSX.SignalRef(self._prefix, attr)
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
        output_type: type[TOutput] = None,
        **kwargs,
    ):
        """Create a fully-typed model instance with automatic type inference.

        This factory method ensures proper type inference for IDE autocomplete.

        Args:
            state_type: Dataclass decorated with @symbolic
            input_type: Dataclass decorated with @symbolic
            param_type: Dataclass decorated with @symbolic
            output_type: Optional output type dataclass
            **kwargs: Optional types (algebraic_type, dependent_type, etc.)

        Returns:
            Fully-typed ModelSX instance

        Usage:
            model = ModelSX.create(States, Inputs, Params, Outputs)
            # IDE now knows exact types for autocomplete!

            x = model.x()  # x has full autocomplete
            u = model.u()  # u has full autocomplete
            p = model.p()  # p has full autocomplete
        """
        return cls(
            state_type, input_type, param_type, output_type=output_type, **kwargs
        )

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
            >>> plant = sportcub()
            >>> controller = autolevel_controller()
            >>> parent = ModelSX.compose({"plant": plant, "controller": controller})
            >>> parent.connect(controller.u.q, plant.x.r)
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
        from . import integrators

        dt_sym = ca.SX.sym("dt")
        N = options.get("N", 10)

        rk4_step = integrators.rk4(self.f_x, dt_sym, name="rk4", N=N)

        # Extract inputs to handle SX vs MX
        if rk4_step.is_a("SXFunction"):
            xin = rk4_step.sx_in(0)
            uin = rk4_step.sx_in(1)
            pin = rk4_step.sx_in(2)
        else:
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

        self.f_step = ca.Function(
            "f_step", step_inputs, [x_next], step_names, ["x_next"]
        )

    def _build_idas_integrator(self, options: dict):
        """Build IDAS DAE integrator."""
        # TODO: Implement IDAS for DAE systems with algebraic constraints
        print("Warning: IDAS not yet implemented, falling back to RK4")
        self._build_rk4_integrator(options)

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
        self.state_names = [
            n for n, _ in sorted(self._state_index.items(), key=lambda kv: kv[1])
        ]
        self.input_names = (
            [n for n, _ in sorted(self._input_index.items(), key=lambda kv: kv[1])]
            if self._input_index
            else []
        )
        self.parameter_names = (
            [n for n, _ in sorted(self._parameter_index.items(), key=lambda kv: kv[1])]
            if self._parameter_index
            else []
        )
        self.output_names = (
            [n for n, _ in sorted(self._output_index.items(), key=lambda kv: kv[1])]
            if hasattr(self, "_output_index")
            else []
        )

    def simulate(
        self,
        t0: float,
        tf: float,
        dt: float,
        u_func: Callable = None,
        p_vec=None,
        x0_vec=None,
        detect_events: bool = False,
    ):
        """Simulate model from t0 to tf with event detection.

        Args:
            t0: Initial time
            tf: Final time
            dt: Timestep
            u_func: Optional control function (t, x, p) -> u_vec
            p_vec: Optional parameter vector (uses p0 if None)
            x0_vec: Optional initial state vector (uses x0 if None)
            detect_events: Whether to detect and handle zero-crossings

        Returns:
            Dictionary with 't', 'x', and optionally 'z', 'm', 'q', 'out' arrays
        """
        if not hasattr(self, "f_step"):
            raise ValueError("Model not built. Call build() first.")

        # Handle composed models
        if hasattr(self, "_composed") and self._composed:
            p_vec = self.p0.as_vec() if p_vec is None else p_vec
            x_curr = self.x0_composed if x0_vec is None else x0_vec
        else:
            p_vec = self.p0.as_vec() if p_vec is None else p_vec
            x_curr = self.x0.as_vec() if x0_vec is None else x0_vec

        z_curr = self.z0.as_vec() if self.z is not None else None
        m_curr = self.m0.as_vec() if self.m is not None else None
        q_curr = self.q0.as_vec() if self.q is not None else None

        t_hist = [t0]
        x_hist = [x_curr]
        if self.z is not None:
            z_hist = [z_curr]
        if self.m is not None:
            m_hist = [m_curr]
        if self.q is not None:
            q_hist = [q_curr]
        out_hist = [] if hasattr(self, "f_y") else None

        # Track previous event indicator for zero-crossing detection
        c_prev = None
        if detect_events and hasattr(self, "f_c"):
            args = self._build_eval_args(
                x_curr, z_curr, m_curr, self.u0.as_vec(), p_vec
            )
            c_prev = float(self.f_c(*args))

        t = t0
        while t < tf - dt / 2:
            # Get control
            if u_func is not None:
                u_curr = u_func(t, x_curr, p_vec)
            else:
                u_curr = self.u0.as_vec()

            # Evaluate outputs before step
            if out_hist is not None:
                args = self._build_eval_args(x_curr, z_curr, m_curr, u_curr, p_vec)
                out_val = self.f_y(*args)
                out_hist.append(out_val)

            # Integration step - build arguments for f_step
            step_args = [x_curr]
            if self.z is not None:
                step_args.append(z_curr)
            if self.m is not None:
                step_args.append(m_curr)
            step_args.extend([u_curr, p_vec, dt])

            x_next = self.f_step(*step_args)

            # Check for events
            if detect_events and hasattr(self, "f_c"):
                args = self._build_eval_args(x_next, z_curr, m_curr, u_curr, p_vec)
                c_curr = float(self.f_c(*args))

                # Detect zero-crossing
                if c_prev is not None and c_prev > 0 and c_curr <= 0:
                    # Event occurred!
                    if hasattr(self, "f_z") and z_curr is not None:
                        z_curr = self.f_z(*args)
                    if hasattr(self, "f_m"):
                        # f_m can reset either discrete variables or continuous states
                        # Check output dimension to determine which
                        m_reset = self.f_m(*args)
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
            if self.q is not None and hasattr(self, "f_q"):
                args = self._build_eval_args(x_curr, z_curr, m_curr, u_curr, p_vec)
                dq_dt = self.f_q(*args)
                q_curr = q_curr + dt * dq_dt

            # Store
            t += dt
            t_hist.append(t)
            x_hist.append(x_next)
            if self.z is not None:
                z_hist.append(z_curr)
            if self.m is not None:
                m_hist.append(m_curr)
            if self.q is not None:
                q_hist.append(q_curr)

            x_curr = x_next

        # Final output
        if out_hist is not None:
            args = self._build_eval_args(x_curr, z_curr, m_curr, u_curr, p_vec)
            out_val = self.f_y(*args)
            out_hist.append(out_val)

        # Build result dictionary
        result = {"t": np.array(t_hist), "x": ca.hcat(x_hist).full()}
        if self.z is not None:
            result["z"] = ca.hcat(z_hist).full()
        if self.m is not None:
            result["m"] = ca.hcat(m_hist).full()
        if self.q is not None:
            result["q"] = ca.hcat(q_hist).full()
        if out_hist is not None:
            result["out"] = ca.hcat(out_hist).full()

        return result

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
        return {
            fname: finfo["dim"] for fname, finfo in self.output_type._field_info.items()
        }


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
