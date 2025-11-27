"""Unified namespace explicit ODE model.

DESIGN PRINCIPLE: Single namespace like Modelica - all variables (state, input, param, output)
in one class, accessible via model.v.

The model carries its own state internally via model.v0. After simulation,
model.v0 is automatically updated to the final state, making chained
simulations seamless.

Time is a built-in (like Modelica) - accessible as model.t in equations.

For linearization, Model auto-extracts x, u, p, y vectors from field var_type metadata.

Example:
    >>> @explicit
    >>> class MassSpring:
    >>>     # States
    >>>     x: float = state()
    >>>     v: float = state()
    >>>     # Inputs
    >>>     F: float = input_var()
    >>>     # Parameters
    >>>     m: float = param(default=1.0)
    >>>     k: float = param(default=1.0)
    >>>     c: float = param(default=0.1)
    >>>     # Outputs
    >>>     position: float = output_var()
    >>> 
    >>> model = Model(MassSpring)
    >>> 
    >>> # All variables in unified namespace:
    >>> model.v.x      # state (symbolic)
    >>> model.v.F      # input (symbolic)
    >>> model.v.m      # parameter (symbolic)
    >>> model.v.position  # output (symbolic)
    >>> 
    >>> model.t        # time (built-in)
    >>> 
    >>> # Define dynamics: dx/dt = f(x, u, p, t)
    >>> model.ode(model.v.x, model.v.v)
    >>> model.ode(model.v.v, (model.v.F - model.v.c * model.v.v - model.v.k * model.v.x) / model.v.m)
    >>> 
    >>> # Optional: define outputs
    >>> model.output(model.v.position, model.v.x)
    >>> 
    >>> model.build()
    >>> 
    >>> # Set initial conditions
    >>> model.v0.x = 1.0
    >>> 
    >>> # Simulate
    >>> t, data = model.simulate(0.0, 10.0, 0.01)
    >>> plt.plot(t, data.x)
    >>> 
    >>> # Linearization
    >>> A, B, C, D = model.linearize()
"""

import casadi as ca
import numpy as np
from typing import Optional, Type, Generic, TypeVar, Tuple, Dict, Any
from dataclasses import fields

from beartype import beartype
from cyecca.dynamics.integrators import rk4 as rk4_integrator


# Type variable for the model class
TModel = TypeVar("TModel")


class ExplicitVar:
    """Symbolic variable wrapper for explicit models.
    
    Wraps a CasADi symbolic variable with metadata about its type.
    """
    
    def __init__(self, name: str, var_type: str, shape: int = 1, sym_type=ca.SX):
        """Initialize symbolic variable.
        
        Args:
            name: Variable name
            var_type: Type ('state', 'input', 'param', 'output')
            shape: Number of elements
            sym_type: CasADi symbol type (SX or MX)
        """
        self.name = name
        self.var_type = var_type
        self.shape = shape
        self.sym = sym_type.sym(name, shape)
        
    def __repr__(self):
        return f"ExplicitVar('{self.name}', type='{self.var_type}')"
    
    # Operator overloading for CasADi operations
    def __add__(self, other):
        return self.sym + (other.sym if isinstance(other, ExplicitVar) else other)
    
    def __radd__(self, other):
        return (other.sym if isinstance(other, ExplicitVar) else other) + self.sym
    
    def __sub__(self, other):
        return self.sym - (other.sym if isinstance(other, ExplicitVar) else other)
    
    def __rsub__(self, other):
        return (other.sym if isinstance(other, ExplicitVar) else other) - self.sym
    
    def __mul__(self, other):
        return self.sym * (other.sym if isinstance(other, ExplicitVar) else other)
    
    def __rmul__(self, other):
        return (other.sym if isinstance(other, ExplicitVar) else other) * self.sym
    
    def __truediv__(self, other):
        return self.sym / (other.sym if isinstance(other, ExplicitVar) else other)
    
    def __rtruediv__(self, other):
        return (other.sym if isinstance(other, ExplicitVar) else other) / self.sym
    
    def __pow__(self, other):
        return self.sym ** (other.sym if isinstance(other, ExplicitVar) else other)
    
    def __neg__(self):
        return -self.sym
    
    def __getitem__(self, key):
        """Support indexing for vector variables."""
        return self.sym[key]
    
    def __getattr__(self, name):
        # Delegate to underlying symbol for CasADi operations
        if name in ['sym', 'name', 'var_type', 'shape']:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.sym, name)


class TypedView:
    """A view that provides access to variables filtered by type.
    
    Also supports extracting numeric vectors from dataclass instances.
    """
    
    def __init__(self, namespace: dict, field_names: list, field_info: dict):
        """Initialize the typed view.
        
        Args:
            namespace: Dict of name -> ExplicitVar for symbolic access
            field_names: List of field names in this view
            field_info: Dict of field info (from model_type._field_info)
        """
        self._namespace = namespace
        self._field_names = field_names
        self._field_info = field_info
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if name in self._namespace:
            return self._namespace[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __call__(self, instance):
        """Extract a numeric vector from a dataclass instance.
        
        Args:
            instance: A dataclass instance with the same fields
            
        Returns:
            np.ndarray: Vector of values for all fields in this view
        """
        result = []
        for name in self._field_names:
            val = getattr(instance, name)
            if isinstance(val, np.ndarray):
                result.extend(val.flatten())
            else:
                result.append(float(val))
        return np.array(result)
    
    def __repr__(self):
        return f"TypedView({self._field_names})"


@beartype
class Model(Generic[TModel]):
    """Unified namespace explicit ODE model.
    
    All variables live in a unified namespace (model.v).
    Typed views provide filtered access:
    - model.x: states only
    - model.u: inputs only
    - model.p: parameters only  
    - model.y: outputs only
    
    Time is a built-in symbolic variable (model.t).
    
    For linearization, the model auto-extracts vectors by field var_type:
    - model.x_sym: states vector
    - model.u_sym: inputs vector  
    - model.p_sym: parameters vector
    - model.y_sym: outputs vector
    
    Example:
        >>> @explicit
        >>> class MassSpring:
        >>>     x: float = state()
        >>>     v: float = state()
        >>>     F: float = input_var()
        >>>     m: float = param(default=1.0)
        >>> 
        >>> model = Model(MassSpring)
        >>> model.ode(model.v.x, model.v.v)
        >>> model.ode(model.v.v, (model.v.F - model.v.m * model.v.x))
        >>> model.build()
        >>> 
        >>> A, B = model.linearize()[:2]
    """
    
    v: TModel      # Symbolic namespace (all variables)
    x: TypedView   # States view
    u: TypedView   # Inputs view
    p: TypedView   # Parameters view
    y: TypedView   # Outputs view
    v0: TModel     # Numeric state (mutable, updated by simulate)
    t: ca.SX       # Time (built-in)
    
    def __init__(
        self,
        model_type: Type[TModel],
        name: Optional[str] = None,
        sym_type=ca.SX,
    ):
        """Initialize model with a unified type class.
        
        Args:
            model_type: Class with all fields (states, inputs, params, outputs)
                        decorated with @explicit
            name: Model name (optional)
            sym_type: CasADi symbol type (ca.SX or ca.MX)
        """
        self.name = name or model_type.__name__
        self.sym_type = sym_type
        self.model_type = model_type
        
        # Get field info from the decorated class
        if not hasattr(model_type, '_field_info'):
            raise ValueError(
                f"Class {model_type.__name__} must be decorated with @explicit"
            )
        
        self._field_info = model_type._field_info
        
        # Categorize fields by var_type
        self._state_fields = []
        self._input_fields = []
        self._param_fields = []
        self._output_fields = []
        
        for field_name, info in self._field_info.items():
            var_type = info.get("type", "")
            if var_type == 'state':
                self._state_fields.append(field_name)
            elif var_type == 'input':
                self._input_fields.append(field_name)
            elif var_type in ('param', 'parameter'):
                self._param_fields.append(field_name)
            elif var_type == 'output':
                self._output_fields.append(field_name)
        
        # ODE definitions: list of (state_name, derivative_expr)
        self._ode_defs = []
        # Output definitions: list of (output_name, expr)
        self._output_defs = []
        
        # Create symbolic namespace with all variables
        self.v = self._create_symbolic_namespace(sym_type)
        
        # Create typed views (filtered access)
        self.x = self._create_typed_view(self._state_fields)
        self.u = self._create_typed_view(self._input_fields)
        self.p = self._create_typed_view(self._param_fields)
        self.y = self._create_typed_view(self._output_fields)
        
        # Create built-in time symbol
        self.t = sym_type.sym("t")
        
        # Track current time
        self._t0 = 0.0
        
        # Create numeric state instance
        self.v0 = model_type.numeric()
        
        # Build symbolic vectors by type (for linearization)
        self._build_type_vectors()
    
    def _create_symbolic_namespace(self, sym_type):
        """Create symbolic namespace with all variables."""
        class SymbolicNamespace:
            pass
        
        ns = SymbolicNamespace()
        
        for field_name, info in self._field_info.items():
            var_type = info.get("type", "state")
            dim = info.get("dim", 1)
            var = ExplicitVar(field_name, var_type, dim, sym_type)
            setattr(ns, field_name, var)
        
        return ns
    
    def _create_typed_view(self, field_names: list) -> TypedView:
        """Create a typed view with only the specified fields."""
        namespace = {name: getattr(self.v, name) for name in field_names}
        return TypedView(namespace, field_names, self._field_info)
    
    def _extract_vector(self, instance, field_names: list) -> np.ndarray:
        """Extract a numpy vector from instance for the given fields."""
        vals = []
        for name in field_names:
            val = getattr(instance, name)
            if np.isscalar(val):
                vals.append(float(val))
            else:
                vals.extend(np.asarray(val).flatten().tolist())
        return np.array(vals) if vals else np.array([])
    
    def _build_type_vectors(self):
        """Build symbolic vectors grouped by variable type."""
        def _concat_syms(field_names):
            parts = [getattr(self.v, name).sym for name in field_names]
            return ca.vertcat(*parts) if parts else self.sym_type(0, 1)
        
        self.x_sym = _concat_syms(self._state_fields)
        self.u_sym = _concat_syms(self._input_fields)
        self.p_sym = _concat_syms(self._param_fields)
        self.y_sym = _concat_syms(self._output_fields)

    @property
    def state_names(self) -> list[str]:
        """Get expanded list of state variable names."""
        names = []
        for name in self._state_fields:
            dim = self._field_info[name]['dim']
            if dim == 1:
                names.append(name)
            else:
                for i in range(dim):
                    names.append(f'{name}[{i}]')
        return names

    @property
    def input_names(self) -> list[str]:
        """Get expanded list of input variable names."""
        names = []
        for name in self._input_fields:
            dim = self._field_info[name]['dim']
            if dim == 1:
                names.append(name)
            else:
                for i in range(dim):
                    names.append(f'{name}[{i}]')
        return names

    @property
    def param_names(self) -> list[str]:
        """Get expanded list of parameter names."""
        names = []
        for name in self._param_fields:
            dim = self._field_info[name]['dim']
            if dim == 1:
                names.append(name)
            else:
                for i in range(dim):
                    names.append(f'{name}[{i}]')
        return names

    @property
    def output_names(self) -> list[str]:
        """Get expanded list of output variable names."""
        names = []
        for name in self._output_fields:
            dim = self._field_info[name]['dim']
            if dim == 1:
                names.append(name)
            else:
                for i in range(dim):
                    names.append(f'{name}[{i}]')
        return names
    
    def ode(self, state_var: ExplicitVar, derivative_expr):
        """Define ODE for a state variable: d(state)/dt = expr.
        
        Args:
            state_var: The state variable (e.g., model.v.x)
            derivative_expr: Expression for the derivative
            
        Example:
            >>> model.ode(model.v.x, model.v.v)  # dx/dt = v
            >>> model.ode(model.v.v, -k * model.v.x)  # dv/dt = -k*x
        """
        if not isinstance(state_var, ExplicitVar):
            raise TypeError(f"First argument must be an ExplicitVar (state), got {type(state_var)}")
        if state_var.var_type != 'state':
            raise ValueError(f"Variable '{state_var.name}' is not a state (type: {state_var.var_type})")
        
        # Convert ExplicitVar to CasADi expression if needed
        if isinstance(derivative_expr, ExplicitVar):
            derivative_expr = derivative_expr.sym
            
        self._ode_defs.append((state_var.name, derivative_expr))
    
    def output(self, output_var: ExplicitVar, expr):
        """Define output equation: y = f(x, u, p, t).
        
        Args:
            output_var: The output variable (e.g., model.v.position)
            expr: Expression for the output
            
        Example:
            >>> model.output(model.v.position, model.v.x)
            >>> model.output(model.v.energy, 0.5 * model.v.m * model.v.v**2)
        """
        if not isinstance(output_var, ExplicitVar):
            raise TypeError(f"First argument must be an ExplicitVar (output), got {type(output_var)}")
        if output_var.var_type != 'output':
            raise ValueError(f"Variable '{output_var.name}' is not an output (type: {output_var.var_type})")
        
        # Convert ExplicitVar to CasADi expression if needed
        if isinstance(expr, ExplicitVar):
            expr = expr.sym
            
        self._output_defs.append((output_var.name, expr))
    
    def build(self, integrator: str = "rk4", integrator_options: dict = None):
        """Build ODE system from definitions.
        
        Args:
            integrator: Integration method ('rk4', 'cvodes', 'idas')
            integrator_options: Options for integrator. For 'rk4':
                - N: Number of sub-steps per dt (default: 1)
            
        Returns:
            self (for chaining)
        """
        if integrator_options is None:
            integrator_options = {}
        if not self._ode_defs:
            raise ValueError("No ODEs defined. Use .ode() to add state equations.")
        
        # Validate all states have ODEs
        defined_states = {name for name, _ in self._ode_defs}
        for name in self._state_fields:
            if name not in defined_states:
                raise ValueError(f"State '{name}' has no ODE defined. Use model.ode(model.v.{name}, ...)")
        
        # Build derivative vector in state order
        xdot_parts = []
        for name in self._state_fields:
            for def_name, expr in self._ode_defs:
                if def_name == name:
                    if isinstance(expr, (ca.SX, ca.MX)):
                        xdot_parts.append(expr)
                    else:
                        xdot_parts.append(ca.DM(expr))
                    break
        
        self._xdot = ca.vertcat(*xdot_parts)
        
        # Build output vector if outputs defined
        if self._output_defs:
            y_parts = []
            for name in self._output_fields:
                for def_name, expr in self._output_defs:
                    if def_name == name:
                        if isinstance(expr, (ca.SX, ca.MX)):
                            y_parts.append(expr)
                        else:
                            y_parts.append(ca.DM(expr))
                        break
                else:
                    # Output not defined, use zero
                    dim = self._field_info[name]["dim"]
                    y_parts.append(ca.DM.zeros(dim, 1))
            self._y_expr = ca.vertcat(*y_parts)
        else:
            self._y_expr = None
        
        # Store for simulate
        self._integrator_type = integrator
        self._integrator_options = integrator_options
        self._n_states = self.x_sym.shape[0]
        self._n_inputs = self.u_sym.shape[0]
        self._n_params = self.p_sym.shape[0]
        self._n_outputs = self.y_sym.shape[0] if self._y_expr is not None else 0
        
        # Create dynamics function (for ODE: f(x, u, p) -> xdot, without time for integrator)
        self._f_ode = ca.Function('f_ode', [self.x_sym, self.u_sym, self.p_sym],
                                   [self._xdot], ['x', 'u', 'p'], ['xdot'])
        # Also keep the version with time for direct access
        self._f = ca.Function('f', [self.x_sym, self.u_sym, self.p_sym, self.t],
                               [self._xdot], ['x', 'u', 'p', 't'], ['xdot'])
        
        # Create output function if outputs defined
        if self._y_expr is not None:
            self._g = ca.Function('g', [self.x_sym, self.u_sym, self.p_sym, self.t],
                                   [self._y_expr], ['x', 'u', 'p', 't'], ['y'])
        else:
            self._g = None
        
        # RK4 integrator will be built at simulate time with numeric dt
        self._rk4_step = None
        self._last_dt = None
        
        return self
    
    def simulate(
        self,
        t0: float,
        tf: float,
        dt: float,
        u_func=None,
    ) -> Tuple[np.ndarray, TModel]:
        """Simulate model from t0 to tf using internal state.
        
        Uses model.v0 as initial conditions. After simulation, model.v0 is
        updated to the final state.
        
        Args:
            t0: Initial time
            tf: Final time
            dt: Timestep
            u_func: Optional function u_func(t, model) -> input values or dict.
                    model.v0 contains current state values.
                    Can return a dict {'F': 1.0} or a vector.
                    If None, uses v0 inputs.
            
        Returns:
            Tuple of (t, data) where:
            - t: np.ndarray of time values
            - data: object with all fields as ndarray trajectories
            
        Example:
            >>> model.v0.x = 1.0
            >>> model.v0.v = 0.0
            >>> def controller(t, m):
            ...     return {'F': -0.1 * m.v0.v}  # Simple damper
            >>> t, data = model.simulate(0.0, 10.0, 0.01, u_func=controller)
            >>> plt.plot(t, data.x)
        """
        if not hasattr(self, '_f'):
            raise ValueError("Model not built. Call build() first.")
        
        # Extract vectors from v0
        x_curr = self._extract_vector(self.v0, self._state_fields)
        p_vec = self._extract_vector(self.v0, self._param_fields)
        u_default = self._extract_vector(self.v0, self._input_fields)
        
        # Build time grid
        n_steps = int((tf - t0) / dt + 0.5)
        t_grid = np.linspace(t0, tf, n_steps + 1)
        
        # Storage
        x_hist = [x_curr.copy()]
        u_hist = []
        y_hist = []
        
        # Helper to convert u_func result to vector
        def _input_to_vector(u_result):
            """Convert u_func result (dict or array) to input vector."""
            if isinstance(u_result, dict):
                u_vec = u_default.copy()
                offset = 0
                for name in self._input_fields:
                    dim = self._field_info[name]["dim"]
                    if name in u_result:
                        val = u_result[name]
                        if dim == 1:
                            u_vec[offset] = float(val)
                        else:
                            u_vec[offset:offset+dim] = np.atleast_1d(val)
                    offset += dim
                return u_vec
            else:
                return np.atleast_1d(u_result)
        
        # Build RK4 integrator if needed (lazily, with numeric dt)
        N_substeps = self._integrator_options.get('N', 1)
        if self._rk4_step is None or self._last_dt != dt:
            self._rk4_step = rk4_integrator(self._f_ode, dt, name='rk4_step', N=N_substeps)
            self._last_dt = dt
        
        # RK4 integration using cyecca.dynamics.integrators
        for i, t in enumerate(t_grid[:-1]):
            # Update v0 with current state so u_func can access it
            self._update_v0_from_vector(x_curr, 'state')
            
            # Get input at this time
            if u_func is not None:
                u_result = u_func(t, self)
                u = _input_to_vector(u_result)
            else:
                u = u_default
            u_hist.append(u.copy() if len(u) > 0 else np.array([]))
            
            # Compute output if defined
            if self._g is not None:
                y = np.array(self._g(x_curr, u, p_vec, t)).flatten()
                y_hist.append(y)
            
            # Use the RK4 integrator (with N substeps already baked in)
            x_next = self._rk4_step(x_curr, u, p_vec)
            x_curr = np.array(x_next).flatten()
            
            x_hist.append(x_curr.copy())
        
        # Final input and output
        self._update_v0_from_vector(x_curr, 'state')
        if u_func is not None:
            u_result = u_func(t_grid[-1], self)
            u_hist.append(_input_to_vector(u_result))
        else:
            u_hist.append(u_default.copy() if len(u_default) > 0 else np.array([]))
        if self._g is not None:
            y_hist.append(np.array(self._g(x_curr, u_hist[-1], p_vec, t_grid[-1])).flatten())
        
        # Convert to arrays
        x_traj = np.array(x_hist)  # (n_steps+1, n_states)
        u_traj = np.array(u_hist) if u_hist and len(u_hist[0]) > 0 else None
        y_traj = np.array(y_hist) if y_hist else None
        
        # v0 is already updated to final state
        self._t0 = tf
        
        # Create trajectory data
        data = self._create_trajectory_data(t_grid, x_traj, u_traj, y_traj, p_vec)
        data.t = t_grid  # Add time array to trajectory
        
        return t_grid, data
    
    def _update_v0_from_vector(self, vec: np.ndarray, var_type: str):
        """Update v0 fields from a vector."""
        if var_type == 'state':
            fields_list = self._state_fields
        elif var_type == 'input':
            fields_list = self._input_fields
        else:
            return
            
        offset = 0
        for name in fields_list:
            dim = self._field_info[name]["dim"]
            if dim == 1:
                setattr(self.v0, name, float(vec[offset]))
                offset += 1
            else:
                setattr(self.v0, name, vec[offset:offset+dim].copy())
                offset += dim
    
    def _create_trajectory_data(self, t_grid, x_traj, u_traj, y_traj, p_vec) -> TModel:
        """Create trajectory data instance with all fields as ndarrays."""
        traj_data = object.__new__(self.model_type)
        traj_data._field_names = list(self._field_info.keys())
        traj_data._is_trajectory = True
        
        n_steps = len(t_grid)
        
        # States
        offset = 0
        for name in self._state_fields:
            dim = self._field_info[name]["dim"]
            if dim == 1:
                setattr(traj_data, name, x_traj[:, offset])
            else:
                setattr(traj_data, name, x_traj[:, offset:offset+dim])
            offset += dim
        
        # Inputs
        if u_traj is not None:
            offset = 0
            for name in self._input_fields:
                dim = self._field_info[name]["dim"]
                if dim == 1:
                    setattr(traj_data, name, u_traj[:, offset])
                else:
                    setattr(traj_data, name, u_traj[:, offset:offset+dim])
                offset += dim
        
        # Outputs
        if y_traj is not None:
            offset = 0
            for name in self._output_fields:
                dim = self._field_info[name]["dim"]
                if dim == 1:
                    setattr(traj_data, name, y_traj[:, offset])
                else:
                    setattr(traj_data, name, y_traj[:, offset:offset+dim])
                offset += dim
        
        # Parameters (constant - replicate for consistency)
        if p_vec is not None and len(p_vec) > 0:
            offset = 0
            for name in self._param_fields:
                dim = self._field_info[name]["dim"]
                if dim == 1:
                    setattr(traj_data, name, np.full(n_steps, p_vec[offset]))
                else:
                    setattr(traj_data, name, np.tile(p_vec[offset:offset+dim], (n_steps, 1)))
                offset += dim
        
        return traj_data
    
    def linearize(
        self,
        x0: Optional[np.ndarray] = None,
        u0: Optional[np.ndarray] = None,
        p0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Linearize system at operating point.
        
        Computes state-space matrices:
            ẋ = A*x + B*u
            y = C*x + D*u
        
        Args:
            x0: State operating point (uses v0 if None)
            u0: Input operating point (uses v0 if None)
            p0: Parameters (uses v0 if None)
            
        Returns:
            (A, B, C, D) tuple of numpy arrays.
            C, D are None if no outputs defined.
            
        Example:
            >>> A, B, C, D = model.linearize()
            >>> eigenvalues = np.linalg.eigvals(A)
        """
        if not hasattr(self, '_f'):
            raise ValueError("Model not built. Call build() first.")
        
        # Get operating point from v0 if not specified
        if x0 is None:
            x0 = self._extract_vector(self.v0, self._state_fields)
        if u0 is None:
            u0 = self._extract_vector(self.v0, self._input_fields)
        if p0 is None:
            p0 = self._extract_vector(self.v0, self._param_fields)
        
        # Compute Jacobians
        # A = df/dx
        A_sym = ca.jacobian(self._xdot, self.x_sym)
        A_func = ca.Function('A', [self.x_sym, self.u_sym, self.p_sym, self.t], [A_sym])
        A = np.array(A_func(x0, u0, p0, 0.0))
        
        # B = df/du
        if self._n_inputs > 0:
            B_sym = ca.jacobian(self._xdot, self.u_sym)
            B_func = ca.Function('B', [self.x_sym, self.u_sym, self.p_sym, self.t], [B_sym])
            B = np.array(B_func(x0, u0, p0, 0.0))
        else:
            B = np.zeros((self._n_states, 0))
        
        # C, D from output function
        C = None
        D = None
        if self._g is not None and self._n_outputs > 0:
            C_sym = ca.jacobian(self._y_expr, self.x_sym)
            C_func = ca.Function('C', [self.x_sym, self.u_sym, self.p_sym, self.t], [C_sym])
            C = np.array(C_func(x0, u0, p0, 0.0))
            
            if self._n_inputs > 0:
                D_sym = ca.jacobian(self._y_expr, self.u_sym)
                D_func = ca.Function('D', [self.x_sym, self.u_sym, self.p_sym, self.t], [D_sym])
                D = np.array(D_func(x0, u0, p0, 0.0))
            else:
                D = np.zeros((self._n_outputs, 0))
        
        return A, B, C, D
    
    @property
    def n_states(self) -> int:
        """Number of state variables."""
        return sum(self._field_info[n]["dim"] for n in self._state_fields)
    
    @property
    def n_inputs(self) -> int:
        """Number of input variables."""
        return sum(self._field_info[n]["dim"] for n in self._input_fields)
    
    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return sum(self._field_info[n]["dim"] for n in self._param_fields)
    
    @property
    def n_outputs(self) -> int:
        """Number of output variables."""
        return sum(self._field_info[n]["dim"] for n in self._output_fields)
