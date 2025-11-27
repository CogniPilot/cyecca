"""Unified namespace implicit DAE model.

DESIGN PRINCIPLE: Single namespace like Modelica - all variables (state, param, alg)
in one class. Variable types inferred from usage:
- Variables with .dot() called → states
- Variables without .dot() → algebraic

The model carries its own state internally via model.v0. After simulation,
model.v0 is automatically updated to the final state, making chained
simulations seamless - just like Modelica.

Time is a built-in (like Modelica) - accessible as model.t in equations.

Example:
    >>> @implicit
    >>> class Pendulum:
    >>>     theta: float = var()   # Becomes state (has .dot() in equations)
    >>>     omega: float = var()   # Becomes state (has .dot() in equations)
    >>>     l: float = param(default=1.0)
    >>>     g: float = param(default=9.81)
    >>> 
    >>> model = Model(Pendulum)
    >>> 
    >>> # All variables in one namespace:
    >>> model.v.theta.dot()  # state derivative (symbolic) - marks theta as state!
    >>> model.v.l            # parameter (symbolic)
    >>> model.t              # time (built-in, like Modelica)
    >>> 
    >>> model.eq(model.v.theta.dot() - model.v.omega)
    >>> model.eq(model.v.omega.dot() + model.v.g/model.v.l * ca.sin(model.v.theta))
    >>> 
    >>> model.build()  # Infers: theta, omega are states (have .dot())
    >>> 
    >>> # Set initial conditions on model's internal state
    >>> model.v0.theta = 0.5
    >>> 
    >>> t, data = model.simulate(0.0, 10.0, 0.01)
    >>> plt.plot(t, data.theta)  # Full autocomplete!
    >>> 
    >>> # model.v0 is now at final state - continue seamlessly
    >>> t2, data2 = model.simulate(10.0, 20.0, 0.01)
    >>> 
    >>> # Reset to a checkpoint
    >>> model.v0 = data[50]
"""

import copy
import casadi as ca
import numpy as np
from typing import Optional, Type, Generic, TypeVar, Tuple
from beartype import beartype

from .variables import ImplicitVar, ImplicitParam


# Type variable for the model class
TModel = TypeVar("TModel")


@beartype
class Model(Generic[TModel]):
    """Unified namespace implicit DAE model (Modelica-style).
    
    All variables live in a single namespace, just like Modelica. 
    Variable types are inferred from usage:
    - Variables with .dot() called → states
    - Variables without .dot() → algebraic
    - Explicit param() declarations → parameters
    
    Time is a built-in symbolic variable (model.t), just like Modelica's `time`.
    No need to declare it in the dataclass.
    
    The model carries its own state internally via `model.v0`. After simulation,
    `model.v0` is automatically updated to the final state, making chained
    simulations seamless.
    
    Example:
        >>> @implicit
        >>> class Pendulum:
        >>>     theta: float = var()   # Becomes state (has .dot())
        >>>     omega: float = var()   # Becomes state (has .dot())
        >>>     l: float = param(default=1.0)
        >>>     g: float = param(default=9.81)
        >>> 
        >>> model = Model(Pendulum)
        >>> 
        >>> # Single namespace - no model.x vs model.p confusion:
        >>> model.v.theta.dot()  # derivative (symbolic) - marks as state
        >>> model.v.l            # parameter (symbolic)
        >>> model.t              # time (built-in)
        >>> 
        >>> model.eq(model.v.theta.dot() - model.v.omega)
        >>> model.eq(model.v.omega.dot() + model.v.g/model.v.l * ca.sin(model.v.theta))
        >>> 
        >>> model.build()  # Infers theta, omega are states
        >>> 
        >>> # Set initial conditions on model's internal state
        >>> model.v0.theta = 0.5
        >>> 
        >>> t, data = model.simulate(0.0, 10.0, 0.01)
        >>> plt.plot(t, data.theta)
        >>> 
        >>> # model.v0 is now at final state - continue seamlessly
        >>> t2, data2 = model.simulate(10.0, 20.0, 0.01)
        >>> 
        >>> # Checkpoint restore
        >>> model.v0 = data[50]
    """
    
    # Typed accessors for IDE autocomplete
    v: TModel   # Symbolic variables for equations
    v0: TModel  # Numeric state (mutable, updated by simulate)
    t: ca.SX    # Time (built-in, like Modelica)
    
    def __init__(
        self,
        model_type: Type[TModel],
        name: Optional[str] = None,
        sym_type=ca.SX,
    ):
        """Initialize model with unified namespace.
        
        Args:
            model_type: @implicit dataclass with var/param fields
            name: Model name (optional)
            sym_type: CasADi symbol type (ca.SX or ca.MX)
        """
        self.name = name or model_type.__name__
        self.sym_type = sym_type
        self.model_type = model_type
        self.equations = []
        
        # Create symbolic instance - this is the unified namespace
        self.v: TModel = model_type.symbolic(sym_type)
        
        # Create built-in time symbol (like Modelica's `time`)
        self.t = sym_type.sym("t")
        
        # Track current time (numeric, updated by simulate)
        self._t0 = 0.0
        
        # These will be populated by build() after inferring types
        self._state_fields = []
        self._alg_fields = []
        self._param_fields = []
        self._var_fields = []  # All 'var' type fields (state/alg TBD)
        
        # Categorize fields
        for field_name, info in model_type._field_info.items():
            var_type = info["type"]
            if var_type == 'var':
                self._var_fields.append(field_name)  # State/alg will be inferred from .dot()
            elif var_type in ('param', 'parameter'):
                self._param_fields.append(field_name)
            # Skip 'time' fields - time is now built-in
        
        # Create numeric state instance - this is the mutable internal state
        # Updated automatically by simulate()
        self.v0: TModel = model_type.numeric()
    
    def eq(self, expr):
        """Add an equation (or vector of equations) to the model.
        
        Equations should be residual expressions that equal zero:
        F(ẋ,x,z,p,t) = 0
        
        Args:
            expr: CasADi symbolic expression (scalar or vector)
            
        Example:
            >>> model.eq(model.v.theta.dot() - model.v.omega)
            >>> model.eq(model.v.pos.dot() - model.v.vel)  # Vector equation
        """
        if not isinstance(expr, (ca.SX, ca.MX, ca.DM)):
            raise TypeError(f"Equation must be CasADi symbolic expression, got {type(expr)}")
        
        # Handle vector equations
        if expr.shape[0] > 1 or expr.shape[1] > 1:
            expr_flat = ca.vec(expr)
            for i in range(expr_flat.shape[0]):
                self.equations.append(expr_flat[i])
        else:
            self.equations.append(expr)
    
    def build(self):
        """Build DAE system from equations.
        
        Infers which var() variables are states vs algebraic based on
        whether .dot() was called on them during equation setup.
        
        Returns:
            self (for chaining)
        """
        if not self.equations:
            raise ValueError("No equations added. Use .eq() to add equations.")
        
        # --- Infer state vs algebraic from .dot() usage ---
        # Variables with .dot() called → states
        # Variables without .dot() → algebraic
        inferred_states = []
        inferred_algs = []
        
        for name in self._var_fields:
            var = getattr(self.v, name)
            if isinstance(var, ImplicitVar) and var.has_derivative:
                inferred_states.append(name)
            else:
                inferred_algs.append(name)
        
        # Store for simulate()
        self._state_fields = inferred_states
        self._alg_fields = inferred_algs
        
        # Gather variables by type
        state_vars = []
        for name in self._state_fields:
            var = getattr(self.v, name)
            state_vars.append(var)
        
        alg_vars = []
        for name in self._alg_fields:
            var = getattr(self.v, name)
            alg_vars.append(var)
        
        param_vars = []
        for name in self._param_fields:
            var = getattr(self.v, name)
            param_vars.append(var)
        
        # Build vectors
        x_syms = []
        xdot_syms = []
        for var in state_vars:
            if var.shape == 1:
                x_syms.append(var.sym)
                xdot_syms.append(var._dot_sym)  # Created when .dot() was called
            else:
                for i in range(var.shape):
                    x_syms.append(var.sym[i])
                    xdot_syms.append(var._dot_sym[i])
        
        z_syms = []
        for var in alg_vars:
            if var.shape == 1:
                z_syms.append(var.sym)
            else:
                for i in range(var.shape):
                    z_syms.append(var.sym[i])
        
        p_syms = []
        for var in param_vars:
            if var.shape == 1:
                p_syms.append(var.sym)
            else:
                for i in range(var.shape):
                    p_syms.append(var.sym[i])
        
        x_vec = ca.vertcat(*x_syms) if x_syms else self.sym_type(0, 1)
        xdot_vec = ca.vertcat(*xdot_syms) if xdot_syms else self.sym_type(0, 1)
        z_vec = ca.vertcat(*z_syms) if z_syms else self.sym_type(0, 1)
        p_vec = ca.vertcat(*p_syms) if p_syms else self.sym_type(0, 1)
        
        # Validate equation count
        n_states = x_vec.shape[0]
        n_alg = z_vec.shape[0]
        n_total = n_states + n_alg
        n_equations = len(self.equations)
        
        if n_equations != n_total:
            raise ValueError(
                f"Equation count mismatch: {n_equations} equations for "
                f"{n_states} states + {n_alg} algebraics = {n_total} variables. "
                f"States (have .dot()): {self._state_fields}, "
                f"Algebraics (no .dot()): {self._alg_fields}"
            )
        
        # Split equations into ODE and ALG based on xdot dependency
        ode_residuals = []
        alg_residuals = []
        
        for eq in self.equations:
            eq_vars = ca.symvar(eq)
            depends_on_xdot = any(
                ca.is_equal(var, xdot_vec[i])
                for i in range(xdot_vec.shape[0])
                for var in eq_vars
            )
            
            if depends_on_xdot:
                ode_residuals.append(eq)
            else:
                alg_residuals.append(eq)
        
        # Validate ODE count
        if len(ode_residuals) != n_states:
            raise ValueError(
                f"ODE equation count mismatch: {len(ode_residuals)} ODE equations "
                f"for {n_states} states."
            )
        
        # Solve ODE residuals for xdot (assumes linear in xdot)
        xdot_exprs = []
        for i, res in enumerate(ode_residuals):
            xdot_i = xdot_vec[i]
            jac = ca.jacobian(res, xdot_i)
            res_at_zero = ca.substitute(res, xdot_i, 0)
            xdot_solution = -res_at_zero / jac
            xdot_exprs.append(xdot_solution)
        
        # Build DAE dictionary
        self._dae_dict = {
            "x": x_vec,
            "p": p_vec,
            "ode": ca.vertcat(*xdot_exprs),
        }
        
        if n_alg > 0:
            self._dae_dict["z"] = z_vec
            self._dae_dict["alg"] = ca.vertcat(*alg_residuals)
        
        self._x_vec = x_vec
        self._z_vec = z_vec if n_alg > 0 else None
        self._p_vec = p_vec
        self._n_states = n_states
        self._n_alg = n_alg
        
        return self
    
    def simulate(
        self,
        t0: float,
        tf: float,
        dt: float,
        integrator: str = "idas",
    ) -> Tuple[np.ndarray, TModel]:
        """Simulate model from t0 to tf using internal state.
        
        Uses model.v0 as initial conditions. After simulation, model.v0 is
        updated to the final state, making chained simulations seamless.
        
        Args:
            t0: Initial time
            tf: Final time
            dt: Timestep
            integrator: Integrator name ('idas', 'cvodes')
            
        Returns:
            Tuple of (t, data) where:
            - t: np.ndarray of time values
            - data: TModel instance with all fields as ndarray trajectories.
                    Supports indexing: data[-1] returns final state.
            
        Example:
            >>> model.v0.theta = 0.5  # Set initial condition
            >>> 
            >>> t, data = model.simulate(0.0, 10.0, 0.01)
            >>> plt.plot(t, data.theta)
            >>> 
            >>> # model.v0 is now at final state - continue seamlessly
            >>> t2, data2 = model.simulate(10.0, 20.0, 0.01)
            >>> 
            >>> # Or reset to a checkpoint
            >>> model.v0 = data[50]  # Reset to timestep 50
        """
        if not hasattr(self, '_dae_dict'):
            raise ValueError("Model not built. Call build() first.")
        
        # Use internal state as initial conditions
        v0 = self.v0
        
        # Extract state vector from v0
        x_vals = []
        for name in self._state_fields:
            val = getattr(v0, name)
            if np.isscalar(val):
                x_vals.append(float(val))
            else:
                x_vals.extend(val.flatten().tolist())
        x_curr = ca.DM(x_vals)
        
        # Extract algebraic vector
        z_curr = None
        has_alg = self._n_alg > 0
        if has_alg:
            z_vals = []
            for name in self._alg_fields:
                val = getattr(v0, name)
                if np.isscalar(val):
                    z_vals.append(float(val))
                else:
                    z_vals.extend(val.flatten().tolist())
            z_curr = ca.DM(z_vals)
        
        # Extract parameter vector
        p_vals = []
        for name in self._param_fields:
            val = getattr(v0, name)
            if np.isscalar(val):
                p_vals.append(float(val))
            else:
                p_vals.extend(val.flatten().tolist())
        p_vec = ca.DM(p_vals)
        
        # Build time grid for simulation
        n_steps = int((tf - t0) / dt + 0.5)
        t_grid = np.linspace(t0, tf, n_steps + 1)
        
        # Build integrator with grid (new CasADi API)
        # Pass t0 and grid as positional arguments after dae_dict
        integ = ca.integrator("integrator", integrator, self._dae_dict, t0, t_grid, {})
        
        # Call integrator once for entire trajectory
        if has_alg:
            res = integ(x0=x_curr, z0=z_curr, p=p_vec)
        else:
            res = integ(x0=x_curr, p=p_vec)
        
        # Extract results - xf is now (n_states, n_steps+1) matrix
        x_traj = np.array(res["xf"])  # Shape: (n_states, n_steps+1)
        
        # Check for invalid values
        if np.any(np.isnan(x_traj)) or np.any(np.isinf(x_traj)):
            raise RuntimeError(f"NaN/Inf in simulation results")
        
        # Build trajectory data for all timesteps
        all_hist = []
        for i in range(len(t_grid)):
            x_i = ca.DM(x_traj[:, i])
            z_i = None
            if has_alg:
                z_traj = np.array(res["zf"])
                z_i = ca.DM(z_traj[:, i])
            v_i = self._create_numeric_from_vectors(x_i, z_i, p_vec)
            all_hist.append(self._extract_all_values(v_i))
        
        # Get final state
        x_final = ca.DM(x_traj[:, -1])
        z_final = None
        if has_alg:
            z_traj = np.array(res["zf"])
            z_final = ca.DM(z_traj[:, -1])
        
        # Update internal state to final values (Modelica-like behavior)
        self.v0 = self._create_numeric_from_vectors(x_final, z_final, p_vec)
        self._t0 = t_grid[-1]  # Track current time
        
        # Create trajectory data and return as tuple (t, data)
        data = self._create_trajectory_data(all_hist)
        return t_grid, data
    
    def _extract_all_values(self, v):
        """Extract all field values as a flat dict (excluding time)."""
        values = {}
        for name, info in self.model_type._field_info.items():
            if info["type"] == "time":
                continue  # Skip time fields
            val = getattr(v, name)
            values[name] = np.atleast_1d(val).copy()
        return values
    
    def _create_numeric_from_vectors(self, x_vec, z_vec, p_vec):
        """Create numeric instance from CasADi vectors."""
        instance = self.model_type.numeric()
        
        # Unpack states
        offset = 0
        for name in self._state_fields:
            dim = self.model_type._field_info[name]["dim"]
            if dim == 1:
                setattr(instance, name, float(x_vec[offset]))
                offset += 1
            else:
                vals = np.array(x_vec[offset:offset+dim]).flatten()
                setattr(instance, name, vals)
                offset += dim
        
        # Unpack algebraics
        if z_vec is not None:
            offset = 0
            for name in self._alg_fields:
                dim = self.model_type._field_info[name]["dim"]
                if dim == 1:
                    setattr(instance, name, float(z_vec[offset]))
                    offset += 1
                else:
                    vals = np.array(z_vec[offset:offset+dim]).flatten()
                    setattr(instance, name, vals)
                    offset += dim
        
        # Unpack parameters
        offset = 0
        for name in self._param_fields:
            dim = self.model_type._field_info[name]["dim"]
            if dim == 1:
                setattr(instance, name, float(p_vec[offset]))
                offset += 1
            else:
                vals = np.array(p_vec[offset:offset+dim]).flatten()
                setattr(instance, name, vals)
                offset += dim
        
        return instance
    
    def _create_trajectory_data(self, all_hist) -> TModel:
        """Create trajectory data instance.
        
        Returns a TModel instance where all fields are ndarrays.
        Time is returned separately (not in data).
        """
        # Create a trajectory data instance
        traj_data = object.__new__(self.model_type)
        traj_data._field_names = list(self.model_type._field_info.keys())
        traj_data._is_trajectory = True
        
        n_steps = len(all_hist)
        
        for name, info in self.model_type._field_info.items():
            dim = info["dim"]
            field_type = info["type"]
            
            # Skip time fields - time is separate now
            if field_type == "time":
                continue
            
            # Stack values across timesteps
            if dim == 1:
                # Scalar: (n_steps,)
                vals = np.array([h[name][0] for h in all_hist])
            else:
                # Vector: (n_steps, dim)
                vals = np.array([h[name] for h in all_hist])
            
            setattr(traj_data, name, vals)
        
        return traj_data
    
    @property
    def dae(self) -> dict:
        """Access the CasADi DAE dictionary."""
        if not hasattr(self, '_dae_dict'):
            raise ValueError("Model not built. Call build() first.")
        return self._dae_dict
    
    def integrator(self, name: str = "idas", opts: Optional[dict] = None):
        """Create CasADi integrator for this DAE.
        
        Args:
            name: Integrator name ('idas', 'cvodes')
            opts: Integrator options
            
        Returns:
            CasADi integrator function
        """
        if not hasattr(self, '_dae_dict'):
            raise ValueError("Model not built. Call build() first.")
        if opts is None:
            opts = {}
        return ca.integrator("integrator", name, self._dae_dict, opts)
