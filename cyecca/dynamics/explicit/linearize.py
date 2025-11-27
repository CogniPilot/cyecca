"""
Linearization and trim finding utilities for explicit models.
"""

import casadi as ca
import numpy as np


def find_trim(
    model,
    v_guess=None,
    cost_fn=None,
    constraints_fn=None,
    solver_options=None,
    verbose=False,
    print_level=0,
):
    """
    Find a trim point for the model using optimization.

    The trim point is where the state derivative is approximately zero.

    Parameters
    ----------
    model : Model
        The model to find trim for.
    v_guess : dataclass instance, optional
        Initial guess for all variables (state, input, params, outputs).
        If None, uses model.v0 defaults.
    cost_fn : callable, optional
        Cost function: cost_fn(x_opt, u_opt, xdot_opt, model) -> scalar
        where x_opt, u_opt, xdot_opt are MX optimization variables.
        If None, uses sum of squared state derivatives.
    constraints_fn : callable, optional
        Constraint function: constraints_fn(opti, x_opt, u_opt, model)
        where opti is the optimizer, x_opt and u_opt are optimization variables.
        Should add constraints using opti.subject_to().
    solver_options : dict, optional
        Solver options for Ipopt.
    verbose : bool, optional
        Whether to print trim results.
    print_level : int, optional
        Ipopt print level.

    Returns
    -------
    v_trim : dataclass instance
        The trim variables as a dataclass instance with numeric values.
    stats : dict
        Solver statistics.

    Example:
        >>> from cyecca.dynamics._doctest_examples import get_built_explicit_model
        >>> import casadi as ca
        >>> model = get_built_explicit_model()
        >>> def cost_fn(x_opt, u_opt, xdot_opt, model):
        ...     return ca.sumsqr(xdot_opt)
        >>> v_trim, stats = find_trim(model, cost_fn=cost_fn)
        >>> hasattr(v_trim, 'x')
        True
    """
    opti = ca.Opti()
    model_type = model.model_type

    # Create optimization variables for state and input
    n_states = model.n_states
    n_inputs = model.n_inputs
    x_opt = opti.variable(n_states)
    u_opt = opti.variable(n_inputs)

    # Get parameter vector from model.v0
    p_vec = []
    for field_name in model._param_fields:
        val = getattr(model.v0, field_name)
        if isinstance(val, np.ndarray):
            p_vec.extend(val.flatten())
        else:
            p_vec.append(float(val))
    p_vec = np.array(p_vec)

    # Compute state derivative using model's ODE function
    # model.f_x(x, u, p, t) returns xdot (Modelica DAE standard naming)
    xdot_opt = model.f_x(x_opt, u_opt, p_vec, 0.0)

    # Default cost: minimize squared state derivatives
    if cost_fn is None:
        cost = ca.sumsqr(xdot_opt)
    else:
        cost = cost_fn(x_opt, u_opt, xdot_opt, model)

    opti.minimize(cost)

    # Apply user constraints
    if constraints_fn is not None:
        constraints_fn(opti, x_opt, u_opt, model)

    # Set initial guesses
    if v_guess is None:
        v_guess = model.v0

    # Extract state vector from guess
    x_init = []
    for field_name in model._state_fields:
        val = getattr(v_guess, field_name)
        if isinstance(val, np.ndarray):
            x_init.extend(val.flatten())
        else:
            x_init.append(float(val))
    opti.set_initial(x_opt, np.array(x_init))

    # Extract input vector from guess
    u_init = []
    for field_name in model._input_fields:
        val = getattr(v_guess, field_name)
        if isinstance(val, np.ndarray):
            u_init.extend(val.flatten())
        else:
            u_init.append(float(val))
    opti.set_initial(u_opt, np.array(u_init))

    # Set solver options
    default_options = {
        "print_time": False,
        "ipopt": {
            "print_level": print_level,
            "sb": "yes",
        },
    }
    if solver_options is not None:
        default_options.update(solver_options)

    opti.solver("ipopt", default_options)

    # Solve
    try:
        sol = opti.solve()
        x_trim = np.array(sol.value(x_opt)).flatten()
        u_trim = np.array(sol.value(u_opt)).flatten()
        stats = {
            "success": True,
            "cost": float(sol.value(cost)),
            "solver_stats": opti.stats(),
        }
    except RuntimeError as e:
        x_trim = np.array(opti.debug.value(x_opt)).flatten()
        u_trim = np.array(opti.debug.value(u_opt)).flatten()
        stats = {
            "success": False,
            "error": str(e),
            "cost": float(opti.debug.value(cost)),
            "solver_stats": opti.stats(),
        }

    # Create result dataclass instance by copying v0 and updating state/input values
    v_trim = model_type.numeric()

    # Copy all fields from model.v0 first (to get params, outputs, etc.)
    for field_name in model_type._field_info.keys():
        val = getattr(model.v0, field_name)
        if isinstance(val, np.ndarray):
            setattr(v_trim, field_name, val.copy())
        else:
            setattr(v_trim, field_name, val)

    # Update state values
    offset = 0
    for field_name in model._state_fields:
        field_info = model_type._field_info[field_name]
        dim = field_info["dim"]
        if dim == 1:
            setattr(v_trim, field_name, float(x_trim[offset]))
        else:
            setattr(v_trim, field_name, x_trim[offset : offset + dim].copy())
        offset += dim

    # Update input values
    offset = 0
    for field_name in model._input_fields:
        field_info = model_type._field_info[field_name]
        dim = field_info["dim"]
        if dim == 1:
            setattr(v_trim, field_name, float(u_trim[offset]))
        else:
            setattr(v_trim, field_name, u_trim[offset : offset + dim].copy())
        offset += dim

    if verbose:
        print("Trim solution:")
        print(f"  Success: {stats['success']}")
        print(f"  Cost: {stats['cost']:.6e}")
        print("  States:")
        for name in model._state_fields:
            print(f"    {name}: {getattr(v_trim, name)}")
        print("  Inputs:")
        for name in model._input_fields:
            print(f"    {name}: {getattr(v_trim, name)}")

    return v_trim, stats


def get_state_index(model, field_name):
    """Get the index range for a state field in the state vector."""
    offset = 0
    for name in model._state_fields:
        field_info = model.model_type._field_info[name]
        dim = field_info["dim"]
        if name == field_name:
            return offset, offset + dim
        offset += dim
    raise ValueError(f"State field '{field_name}' not found")


def get_input_index(model, field_name):
    """Get the index range for an input field in the input vector."""
    offset = 0
    for name in model._input_fields:
        field_info = model.model_type._field_info[name]
        dim = field_info["dim"]
        if name == field_name:
            return offset, offset + dim
        offset += dim
    raise ValueError(f"Input field '{field_name}' not found")


def linearize_dynamics(model, v_op=None):
    """
    Linearize the model dynamics around an operating point.

    Parameters
    ----------
    model : Model
        The model to linearize.
    v_op : dataclass instance, optional
        Operating point. If None, uses model.v0.

    Returns
    -------
    A : np.ndarray
        State matrix (n_states x n_states).
    B : np.ndarray
        Input matrix (n_states x n_inputs).
    C : np.ndarray
        Output matrix (n_outputs x n_states).
    D : np.ndarray
        Feedthrough matrix (n_outputs x n_inputs).
    """
    if v_op is None:
        v_op = model.v0

    # Extract vectors from the operating point
    x_op = _extract_vector(v_op, model._state_fields, model.model_type)
    u_op = _extract_vector(v_op, model._input_fields, model.model_type)
    p_op = _extract_vector(v_op, model._param_fields, model.model_type)

    return model.linearize(x_op, u_op, p_op)


def _extract_vector(instance, field_names, model_type):
    """Extract a vector from an instance for the given field names."""
    result = []
    for name in field_names:
        val = getattr(instance, name)
        if isinstance(val, np.ndarray):
            result.extend(val.flatten())
        else:
            result.append(float(val))
    return np.array(result)


def analyze_modes(A, names=None):
    """
    Analyze the modes of a linear system.

    Parameters
    ----------
    A : np.ndarray
        State matrix.
    names : list of str, optional
        Names for each state.

    Returns
    -------
    modes : list of dict
        Mode information including eigenvalues, damping, frequency, etc.
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)

    modes = []
    processed = set()

    for i, ev in enumerate(eigenvalues):
        if i in processed:
            continue

        mode = {
            "eigenvalue": ev,
            "real": float(np.real(ev)),
            "imag": float(np.imag(ev)),
            "eigenvector": eigenvectors[:, i],
        }

        # Compute stability metrics
        omega_n = np.abs(ev)  # Natural frequency
        if omega_n > 1e-10:
            zeta = -np.real(ev) / omega_n  # Damping ratio
            mode["omega_n"] = float(omega_n)
            mode["zeta"] = float(zeta)
            if zeta < 1 and np.abs(np.imag(ev)) > 1e-10:
                omega_d = np.abs(np.imag(ev))  # Damped frequency
                mode["omega_d"] = float(omega_d)
                mode["period"] = float(2 * np.pi / omega_d)
            else:
                mode["omega_d"] = 0.0
                mode["period"] = float("inf")
        else:
            mode["omega_n"] = 0.0
            mode["zeta"] = 0.0
            mode["omega_d"] = 0.0
            mode["period"] = float("inf")

        # Time constant
        if np.abs(np.real(ev)) > 1e-10:
            mode["time_constant"] = float(-1.0 / np.real(ev))
        else:
            mode["time_constant"] = float("inf")

        # Half-life / doubling time
        if np.real(ev) < 0:
            mode["half_life"] = float(np.log(2) / np.abs(np.real(ev)))
            mode["doubling_time"] = None
        elif np.real(ev) > 0:
            mode["half_life"] = None
            mode["doubling_time"] = float(np.log(2) / np.real(ev))
        else:
            mode["half_life"] = None
            mode["doubling_time"] = None

        # Stability classification
        if np.real(ev) < -1e-10:
            mode["stability"] = "stable"
        elif np.real(ev) > 1e-10:
            mode["stability"] = "unstable"
        else:
            mode["stability"] = "marginally stable"

        # Mode type
        if np.abs(np.imag(ev)) < 1e-10:
            mode["type"] = "real"
        else:
            mode["type"] = "complex"
            # Find conjugate pair
            for j in range(i + 1, len(eigenvalues)):
                if j not in processed and np.abs(eigenvalues[j] - np.conj(ev)) < 1e-10:
                    processed.add(j)
                    break

        # State participation
        if names is not None and len(names) == len(eigenvectors[:, i]):
            participation = np.abs(eigenvectors[:, i])
            participation = participation / np.max(participation)
            mode["participation"] = {names[k]: float(participation[k]) for k in range(len(names))}

        modes.append(mode)
        processed.add(i)

    return modes


def print_modes(modes, max_modes=None):
    """
    Print mode analysis results in a readable format.

    Parameters
    ----------
    modes : list of dict
        Mode information from analyze_modes.
    max_modes : int, optional
        Maximum number of modes to print.
    """
    if max_modes is not None:
        modes = modes[:max_modes]

    print("\nMode Analysis:")
    print("=" * 80)

    for i, mode in enumerate(modes):
        print(f"\nMode {i + 1}:")
        print(f"  Eigenvalue: {mode['eigenvalue']:.4f}")
        print(f"  Stability: {mode['stability']}")
        print(f"  Type: {mode['type']}")

        if mode["omega_n"] > 1e-10:
            print(f"  Natural frequency: {mode['omega_n']:.4f} rad/s ({mode['omega_n']/(2*np.pi):.4f} Hz)")
            print(f"  Damping ratio: {mode['zeta']:.4f}")

        if mode["type"] == "complex" and mode["omega_d"] > 0:
            print(f"  Damped frequency: {mode['omega_d']:.4f} rad/s")
            print(f"  Period: {mode['period']:.4f} s")

        if mode["time_constant"] != float("inf"):
            print(f"  Time constant: {mode['time_constant']:.4f} s")

        if mode["half_life"] is not None:
            print(f"  Half-life: {mode['half_life']:.4f} s")
        if mode["doubling_time"] is not None:
            print(f"  Doubling time: {mode['doubling_time']:.4f} s")

        if "participation" in mode:
            print("  State participation:")
            sorted_states = sorted(mode["participation"].items(), key=lambda x: -x[1])
            for name, val in sorted_states[:5]:  # Top 5 participating states
                print(f"    {name}: {val:.3f}")

    print("\n" + "=" * 80)
