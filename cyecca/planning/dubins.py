"""
Dubins Path Planner for Fixed-Wing Aircraft
============================================

Forward-only, 2D, fixed turn radius R planner.

Usage:
    >>> from cyecca.planning import derive_dubins
    >>> import casadi as ca
    >>> plan, eval_fn = derive_dubins()
    >>> p0, psi0 = ca.DM([0, 0]), 0.0  # Start position and heading
    >>> p1, psi1 = ca.DM([10, 10]), ca.pi/2  # End position and heading
    >>> R = 5.0  # Turn radius
    >>> cost, type, a1, d, a2, tp0, tp1, c0, c1 = plan(p0, psi0, p1, psi1, R)
    >>> s = 0.5  # Evaluation point along path (0 to 1)
    >>> x, y, psi = eval_fn(s, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)
    >>> float(cost) > 0  # Path cost should be positive
    True

Functions:
    - derive_dubins() -> (dubins_fixedwing, dubins_eval)
    - plot_dubins_path() -> visualization utilities
    - run_tests() -> unit tests for correctness
"""

import casadi as ca
import numpy as np

# ==============================================================================
# Core Utilities
# ==============================================================================


def casadi_min_with_cargo(costs, cargos):
    """Select minimum cost and return its cargo (branch-free CasADi)."""
    if len(costs) == 1:
        return costs[0], cargos[0]

    current_min_cost = costs[0]
    current_min_cargo = cargos[0]

    for i in range(1, len(costs)):
        is_lower = costs[i] < current_min_cost
        current_min_cost = ca.if_else(is_lower, costs[i], current_min_cost)
        current_min_cargo = ca.if_else(is_lower, cargos[i], current_min_cargo)

    return current_min_cost, current_min_cargo


class DubinsPathType:
    """Path type enumeration."""

    RSL = 0  # Right-Straight-Left
    LSR = 1  # Left-Straight-Right
    LSL = 2  # Left-Straight-Left
    RSR = 3  # Right-Straight-Right

    @staticmethod
    def name(type_id):
        names = {0: "RSL", 1: "LSR", 2: "LSL", 3: "RSR"}
        return names.get(int(type_id), "UNKNOWN")


def rotation_matrix(theta):
    """2D rotation matrix."""
    return ca.vertcat(
        ca.horzcat(ca.cos(theta), -ca.sin(theta)),
        ca.horzcat(ca.sin(theta), ca.cos(theta)),
    )


def wrap_angle(x):
    """Wrap angle to [-pi, pi]."""
    return ca.atan2(ca.sin(x), ca.cos(x))


def perp_left(v):
    """90° CCW rotation."""
    return ca.vertcat(-v[1], v[0])


def compute_turn_centers(p, psi, R):
    """Compute right and left turn centers for heading psi at point p."""
    rot = rotation_matrix(psi)
    c_right = p + rot @ ca.vertcat(0, -R)
    c_left = p + rot @ ca.vertcat(0, R)
    return c_right, c_left


# ==============================================================================
# Arc and Tangent Computations
# ==============================================================================


def compute_arc(center, p_from, p_to, turn_sign, R):
    """
    Compute oriented arc from p_from to p_to around 'center'.

    Args:
        center: Circle center
        p_from: Starting point
        p_to: Ending point
        turn_sign: +1 for left/CCW, -1 for right/CW
        R: Turn radius

    Returns:
        dtheta: Signed heading change
        arc_len: Arc length (R * |dtheta|)
    """
    ang_from = ca.atan2(p_from[1] - center[1], p_from[0] - center[0])
    ang_to = ca.atan2(p_to[1] - center[1], p_to[0] - center[0])

    dtheta = wrap_angle(ang_to - ang_from)

    # Enforce turn direction for forward motion
    # Only force the long way around if we're significantly going the wrong direction
    # Threshold of ~5 degrees prevents unnecessary loops while allowing legitimate long arcs
    eps_threshold = 0.087  # ~5 degrees in radians
    dtheta = ca.if_else(
        ca.logic_and(turn_sign > 0, dtheta < -eps_threshold),
        dtheta + 2 * ca.pi,
        dtheta,
    )
    dtheta = ca.if_else(
        ca.logic_and(turn_sign < 0, dtheta > eps_threshold),
        dtheta - 2 * ca.pi,
        dtheta,
    )

    arc_len = R * ca.fabs(dtheta)
    return dtheta, arc_len


def compute_internal_tangent_rsl(cr0, cl1, R):
    """Internal tangent for RSL (right-straight-left)."""
    v = cl1 - cr0
    d = ca.norm_2(v)
    feasible = d > 2 * R

    theta = ca.atan2(v[1], v[0])
    alpha = ca.if_else(feasible, ca.acos(2 * R / d), 0)

    angle = theta + alpha
    dir_vec = ca.vertcat(ca.cos(angle), ca.sin(angle))

    t0 = cr0 + R * dir_vec
    t1 = cl1 - R * dir_vec

    dist = ca.norm_2(t1 - t0)
    return t0, t1, dist, feasible


def compute_internal_tangent_lsr(cl0, cr1, R):
    """Internal tangent for LSR (left-straight-right)."""
    v = cr1 - cl0
    d = ca.norm_2(v)
    feasible = d > 2 * R

    theta = ca.atan2(v[1], v[0])
    alpha = ca.if_else(feasible, ca.acos(2 * R / d), 0)

    angle = theta - alpha
    dir_vec = ca.vertcat(ca.cos(angle), ca.sin(angle))

    t0 = cl0 + R * dir_vec
    t1 = cr1 - R * dir_vec

    dist = ca.norm_2(t1 - t0)
    return t0, t1, dist, feasible


def compute_external_tangent(c0, c1, R, sign):
    """
    External tangent (doesn't cross circles).

    Args:
        sign: +1 for "upper" tangent, -1 for "lower" tangent
    """
    v = c1 - c0
    d = ca.fmax(ca.norm_2(v), 1e-10)
    u = v / d
    w = perp_left(u)

    t0 = c0 + sign * R * w
    t1 = c1 + sign * R * w

    dist = ca.norm_2(t1 - t0)
    return t0, t1, dist


# ==============================================================================
# Path Type Computations
# ==============================================================================


def compute_rsl_path(p0, psi0, p1, psi1, cr0, cl1, R):
    """
    Right-Straight-Left path.
    Selects tangent where circular velocity aligns with straight direction.
    """
    v = cl1 - cr0
    d = ca.norm_2(v)
    feasible = d > 2 * R

    theta = ca.atan2(v[1], v[0])
    alpha = ca.if_else(feasible, ca.acos(2 * R / d), 0)

    # Two tangent options
    angle_up = theta + alpha
    angle_down = theta - alpha

    dir_vec_up = ca.vertcat(ca.cos(angle_up), ca.sin(angle_up))
    dir_vec_down = ca.vertcat(ca.cos(angle_down), ca.sin(angle_down))

    t0_up = cr0 + R * dir_vec_up
    t1_up = cl1 - R * dir_vec_up

    t0_down = cr0 + R * dir_vec_down
    t1_down = cl1 - R * dir_vec_down

    # Check forward motion: circular velocity at t0 should align with straight direction
    r_up = t0_up - cr0
    v_circle_up = ca.vertcat(r_up[1], -r_up[0])  # CW
    straight_dir_up = t1_up - t0_up
    dot_up = ca.dot(v_circle_up, straight_dir_up)

    r_down = t0_down - cr0
    v_circle_down = ca.vertcat(r_down[1], -r_down[0])
    straight_dir_down = t1_down - t0_down
    dot_down = ca.dot(v_circle_down, straight_dir_down)

    use_up = dot_up > dot_down

    t0 = ca.if_else(use_up, t0_up, t0_down)
    t1 = ca.if_else(use_up, t1_up, t1_down)
    dist = ca.norm_2(t1 - t0)

    a1, arc1 = compute_arc(cr0, p0, t0, turn_sign=-1, R=R)
    a2, arc2 = compute_arc(cl1, t1, p1, turn_sign=+1, R=R)

    cost = ca.if_else(feasible, arc1 + dist + arc2, ca.inf)
    return a1, dist, a2, t0, t1, cost


def compute_lsr_path(p0, psi0, p1, psi1, cl0, cr1, R):
    """
    Left-Straight-Right path.
    Selects tangent where circular velocity aligns with straight direction.
    """
    v = cr1 - cl0
    d = ca.norm_2(v)
    feasible = d > 2 * R

    theta = ca.atan2(v[1], v[0])
    alpha = ca.if_else(feasible, ca.acos(2 * R / d), 0)

    # Two tangent options
    angle_up = theta + alpha
    angle_down = theta - alpha

    dir_vec_up = ca.vertcat(ca.cos(angle_up), ca.sin(angle_up))
    dir_vec_down = ca.vertcat(ca.cos(angle_down), ca.sin(angle_down))

    t0_up = cl0 + R * dir_vec_up
    t1_up = cr1 - R * dir_vec_up

    t0_down = cl0 + R * dir_vec_down
    t1_down = cr1 - R * dir_vec_down

    # Check forward motion
    r_up = t0_up - cl0
    v_circle_up = perp_left(r_up)  # CCW
    straight_dir_up = t1_up - t0_up
    dot_up = ca.dot(v_circle_up, straight_dir_up)

    r_down = t0_down - cl0
    v_circle_down = perp_left(r_down)
    straight_dir_down = t1_down - t0_down
    dot_down = ca.dot(v_circle_down, straight_dir_down)

    use_up = dot_up > dot_down

    t0 = ca.if_else(use_up, t0_up, t0_down)
    t1 = ca.if_else(use_up, t1_up, t1_down)
    dist = ca.norm_2(t1 - t0)

    a1, arc1 = compute_arc(cl0, p0, t0, turn_sign=+1, R=R)
    a2, arc2 = compute_arc(cr1, t1, p1, turn_sign=-1, R=R)

    cost = ca.if_else(feasible, arc1 + dist + arc2, ca.inf)
    return a1, dist, a2, t0, t1, cost


def compute_lsl_path(p0, psi0, p1, psi1, cl0, cl1, R):
    """
    Left-Straight-Left path.
    Selects external tangent with forward motion.
    """
    # Compute both tangent options
    t0_up, t1_up, dist_up = compute_external_tangent(cl0, cl1, R, sign=+1)
    t0_down, t1_down, dist_down = compute_external_tangent(cl0, cl1, R, sign=-1)

    # Check forward motion
    r_up = t0_up - cl0
    v_circle_up = perp_left(r_up)
    straight_dir_up = t1_up - t0_up
    dot_up = ca.dot(v_circle_up, straight_dir_up)

    r_down = t0_down - cl0
    v_circle_down = perp_left(r_down)
    straight_dir_down = t1_down - t0_down
    dot_down = ca.dot(v_circle_down, straight_dir_down)

    use_up = dot_up > dot_down

    t0 = ca.if_else(use_up, t0_up, t0_down)
    t1 = ca.if_else(use_up, t1_up, t1_down)
    dist = ca.if_else(use_up, dist_up, dist_down)

    a1, arc1 = compute_arc(cl0, p0, t0, turn_sign=+1, R=R)
    a2, arc2 = compute_arc(cl1, t1, p1, turn_sign=+1, R=R)

    cost = arc1 + dist + arc2
    return a1, dist, a2, t0, t1, cost


def compute_rsr_path(p0, psi0, p1, psi1, cr0, cr1, R):
    """
    Right-Straight-Right path.
    Selects external tangent with forward motion.
    """
    # Compute both tangent options
    t0_up, t1_up, dist_up = compute_external_tangent(cr0, cr1, R, sign=+1)
    t0_down, t1_down, dist_down = compute_external_tangent(cr0, cr1, R, sign=-1)

    # Check forward motion
    r_up = t0_up - cr0
    v_circle_up = ca.vertcat(r_up[1], -r_up[0])  # CW
    straight_dir_up = t1_up - t0_up
    dot_up = ca.dot(v_circle_up, straight_dir_up)

    r_down = t0_down - cr0
    v_circle_down = ca.vertcat(r_down[1], -r_down[0])
    straight_dir_down = t1_down - t0_down
    dot_down = ca.dot(v_circle_down, straight_dir_down)

    use_up = dot_up > dot_down

    t0 = ca.if_else(use_up, t0_up, t0_down)
    t1 = ca.if_else(use_up, t1_up, t1_down)
    dist = ca.if_else(use_up, dist_up, dist_down)

    a1, arc1 = compute_arc(cr0, p0, t0, turn_sign=-1, R=R)
    a2, arc2 = compute_arc(cr1, t1, p1, turn_sign=-1, R=R)

    cost = arc1 + dist + arc2
    return a1, dist, a2, t0, t1, cost


# ==============================================================================
# Main API
# ==============================================================================


def derive_dubins():
    """
    Create CasADi functions for Dubins path planning and evaluation.

    Returns:
        dubins_fixedwing: Planner function
            Inputs: p0[2], psi0, p1[2], psi1, R
            Outputs: cost, type, angle1, distance, angle2,
                     tangent_start[2], tangent_goal[2], center0[2], center1[2]

        dubins_eval: Evaluator function
            Inputs: s, p0[2], psi0, angle1, distance, angle2,
                    tp0[2], tp1[2], c0[2], c1[2], R
            Outputs: x, y, psi
    """
    # --- Planner ---
    p0 = ca.SX.sym("p0", 2)
    psi0 = ca.SX.sym("psi0")
    p1 = ca.SX.sym("p1", 2)
    psi1 = ca.SX.sym("psi1")
    R = ca.SX.sym("R")

    cr0, cl0 = compute_turn_centers(p0, psi0, R)
    cr1, cl1 = compute_turn_centers(p1, psi1, R)

    # All four path candidates
    a1_rsl, d_rsl, a2_rsl, t0_rsl, t1_rsl, cost_rsl = compute_rsl_path(p0, psi0, p1, psi1, cr0, cl1, R)
    a1_lsr, d_lsr, a2_lsr, t0_lsr, t1_lsr, cost_lsr = compute_lsr_path(p0, psi0, p1, psi1, cl0, cr1, R)
    a1_lsl, d_lsl, a2_lsl, t0_lsl, t1_lsl, cost_lsl = compute_lsl_path(p0, psi0, p1, psi1, cl0, cl1, R)
    a1_rsr, d_rsr, a2_rsr, t0_rsr, t1_rsr, cost_rsr = compute_rsr_path(p0, psi0, p1, psi1, cr0, cr1, R)

    # Pack cargo
    cargo_rsl = ca.vertcat(DubinsPathType.RSL, a1_rsl, d_rsl, a2_rsl, t0_rsl, t1_rsl, cr0, cl1)
    cargo_lsr = ca.vertcat(DubinsPathType.LSR, a1_lsr, d_lsr, a2_lsr, t0_lsr, t1_lsr, cl0, cr1)
    cargo_lsl = ca.vertcat(DubinsPathType.LSL, a1_lsl, d_lsl, a2_lsl, t0_lsl, t1_lsl, cl0, cl1)
    cargo_rsr = ca.vertcat(DubinsPathType.RSR, a1_rsr, d_rsr, a2_rsr, t0_rsr, t1_rsr, cr0, cr1)

    min_cost, best_cargo = casadi_min_with_cargo(
        costs=[cost_rsl, cost_lsr, cost_lsl, cost_rsr],
        cargos=[cargo_rsl, cargo_lsr, cargo_lsl, cargo_rsr],
    )

    dubins_plan = ca.Function(
        "dubins_fixedwing",
        [p0, psi0, p1, psi1, R],
        [
            min_cost,
            best_cargo[0],
            best_cargo[1],
            best_cargo[2],
            best_cargo[3],
            best_cargo[4:6],
            best_cargo[6:8],
            best_cargo[8:10],
            best_cargo[10:12],
        ],
        ["p0", "psi0", "p1", "psi1", "R"],
        [
            "cost",
            "type",
            "angle1",
            "distance",
            "angle2",
            "tangent_start",
            "tangent_goal",
            "center0",
            "center1",
        ],
    )

    # --- Evaluator ---
    s = ca.SX.sym("s")
    angle1 = ca.SX.sym("angle1")
    dist = ca.SX.sym("distance")
    angle2 = ca.SX.sym("angle2")
    tp0 = ca.SX.sym("tp0", 2)
    tp1 = ca.SX.sym("tp1", 2)
    c0 = ca.SX.sym("c0", 2)
    c1 = ca.SX.sym("c1", 2)
    p0_e = ca.SX.sym("p0_e", 2)
    psi0_e = ca.SX.sym("psi0_e")
    R_e = ca.SX.sym("R_e")

    arc1_len = R_e * ca.fabs(angle1)
    arc2_len = R_e * ca.fabs(angle2)
    straight_len = dist

    total_len = arc1_len + straight_len + arc2_len
    path_dist = s * total_len

    # Arc 1
    ang0 = ca.atan2(p0_e[1] - c0[1], p0_e[0] - c0[0])
    s1 = ca.if_else(arc1_len > 0, path_dist / arc1_len, 0)
    theta1 = s1 * angle1

    x1 = c0[0] + R_e * ca.cos(ang0 + theta1)
    y1 = c0[1] + R_e * ca.sin(ang0 + theta1)
    psi1_out = psi0_e + theta1

    # Straight
    straight_heading = ca.atan2(tp1[1] - tp0[1], tp1[0] - tp0[0])

    s2 = ca.if_else(straight_len > 0, (path_dist - arc1_len) / straight_len, 0)
    x2 = tp0[0] + s2 * (tp1[0] - tp0[0])
    y2 = tp0[1] + s2 * (tp1[1] - tp0[1])
    psi2_out = straight_heading

    # Arc 2
    s3 = ca.if_else(arc2_len > 0, (path_dist - arc1_len - straight_len) / arc2_len, 0)
    theta2 = s3 * angle2

    ang2_start = ca.atan2(tp1[1] - c1[1], tp1[0] - c1[0])
    x3 = c1[0] + R_e * ca.cos(ang2_start + theta2)
    y3 = c1[1] + R_e * ca.sin(ang2_start + theta2)
    psi3_out = straight_heading + theta2

    # Select segment
    in_arc1 = path_dist <= arc1_len
    in_straight = ca.logic_and(path_dist > arc1_len, path_dist <= arc1_len + straight_len)

    x_out = ca.if_else(in_arc1, x1, ca.if_else(in_straight, x2, x3))
    y_out = ca.if_else(in_arc1, y1, ca.if_else(in_straight, y2, y3))
    psi_out = ca.if_else(in_arc1, psi1_out, ca.if_else(in_straight, psi2_out, psi3_out))

    dubins_eval = ca.Function(
        "dubins_eval",
        [s, p0_e, psi0_e, angle1, dist, angle2, tp0, tp1, c0, c1, R_e],
        [x_out, y_out, psi_out],
        [
            "s",
            "p0",
            "psi0",
            "angle1",
            "distance",
            "angle2",
            "tp0",
            "tp1",
            "c0",
            "c1",
            "R",
        ],
        ["x", "y", "psi"],
    )

    return dubins_plan, dubins_eval


# ==============================================================================
# Plotting Utilities
# ==============================================================================


def plot_dubins_path(p0, psi0, p1, psi1, R, plan, eval_fn, ax=None, n_points=200):
    """
    Plot a Dubins path.

    Args:
        p0, psi0: Start position and heading
        p1, psi1: Goal position and heading
        R: Turn radius
        plan: Planner function from derive_dubins()
        eval_fn: Evaluator function from derive_dubins()
        ax: Matplotlib axis (creates new if None)
        n_points: Number of points to evaluate

    Returns:
        ax: Matplotlib axis (or None if matplotlib not available)
        path_data: Dict with path information
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, plotting skipped")
        # Still compute path data for return
        cost, ct, a1, d, a2, tp0, tp1, c0, c1 = plan(p0, psi0, p1, psi1, R)

        # Evaluate path
        s_vals = np.linspace(0, 1, n_points)
        path_x, path_y, path_psi = [], [], []
        for s in s_vals:
            x, y, psi = eval_fn(s, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)
            path_x.append(float(x))
            path_y.append(float(y))
            path_psi.append(float(psi))

        path_data = {
            "cost": float(cost),
            "type": DubinsPathType.name(ct),
            "angle1": float(a1),
            "distance": float(d),
            "angle2": float(a2),
            "x": path_x,
            "y": path_y,
            "psi": path_psi,
        }
        return None, path_data

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plan
    cost, ct, a1, d, a2, tp0, tp1, c0, c1 = plan(p0, psi0, p1, psi1, R)

    tp0 = np.array(tp0).flatten()
    tp1 = np.array(tp1).flatten()
    c0 = np.array(c0).flatten()
    c1 = np.array(c1).flatten()

    # Draw headings
    ax.arrow(
        p0[0],
        p0[1],
        0.3 * R * np.cos(psi0),
        0.3 * R * np.sin(psi0),
        color="green",
        width=0.02 * R,
        head_width=0.15 * R,
        alpha=0.7,
    )
    ax.arrow(
        p1[0],
        p1[1],
        0.3 * R * np.cos(psi1),
        0.3 * R * np.sin(psi1),
        color="red",
        width=0.02 * R,
        head_width=0.15 * R,
        alpha=0.7,
    )

    # Draw circles
    for c in [c0, c1]:
        circ = plt.Circle((c[0], c[1]), R, fill=False, color="gray", alpha=0.6, linestyle="--")
        ax.add_patch(circ)

    # Evaluate path
    s_vals = np.linspace(0, 1, n_points)
    path_x, path_y, path_psi = [], [], []
    for s in s_vals:
        x, y, psi = eval_fn(s, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)
        path_x.append(float(x))
        path_y.append(float(y))
        path_psi.append(float(psi))

    # Plot path
    ax.plot(path_x, path_y, "b", linewidth=2, alpha=0.8)
    ax.plot(tp0[0], tp0[1], "ko", markersize=4)
    ax.plot(tp1[0], tp1[1], "ko", markersize=4)

    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    path_data = {
        "cost": float(cost),
        "type": DubinsPathType.name(ct),
        "angle1": float(a1),
        "distance": float(d),
        "angle2": float(a2),
        "x": path_x,
        "y": path_y,
        "psi": path_psi,
    }

    return ax, path_data


# ==============================================================================
# Unit Tests
# ==============================================================================


def check_continuity(path_data, pos_tol=0.1, heading_tol_deg=5):
    """
    Check position and heading continuity.

    Returns:
        dict: Test results with 'passed', 'max_pos_jump', 'max_heading_jump'
    """
    x = np.array(path_data["x"])
    y = np.array(path_data["y"])
    psi = np.array(path_data["psi"])

    dx = np.diff(x)
    dy = np.diff(y)
    dpsi = np.diff(psi)

    # Wrap heading differences
    dpsi = np.arctan2(np.sin(dpsi), np.cos(dpsi))

    max_pos_jump = np.max(np.sqrt(dx**2 + dy**2))
    max_heading_jump = np.rad2deg(np.max(np.abs(dpsi)))

    pos_ok = max_pos_jump < pos_tol
    heading_ok = max_heading_jump < heading_tol_deg

    return {
        "passed": pos_ok and heading_ok,
        "max_pos_jump": max_pos_jump,
        "max_heading_jump_deg": max_heading_jump,
        "pos_ok": pos_ok,
        "heading_ok": heading_ok,
    }


def check_forward_motion(path_data, plan_inputs, eval_fn):
    """
    Check that vehicle always moves forward.

    Returns:
        dict: Test results with 'passed', 'min_forward_velocity'
    """
    p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R = plan_inputs

    s_vals = np.linspace(0, 1, 200)
    min_forward_vel = float("inf")

    for i in range(len(s_vals) - 1):
        s = s_vals[i]
        s_next = s_vals[i + 1]

        x1, y1, psi1 = eval_fn(s, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)
        x2, y2, psi2 = eval_fn(s_next, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)

        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)

        # Velocity in heading direction
        heading = float(psi1)
        forward_vel = dx * np.cos(heading) + dy * np.sin(heading)
        min_forward_vel = min(min_forward_vel, forward_vel)

    return {
        "passed": min_forward_vel >= -1e-6,  # Allow small numerical error
        "min_forward_velocity": min_forward_vel,
    }


def run_tests(n_random=10, verbose=True):
    """
    Run unit tests on Dubins planner.

    Args:
        n_random: Number of random test cases
        verbose: Print detailed results

    Returns:
        dict: Test summary with pass/fail counts
    """
    plan, eval_fn = derive_dubins()

    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "failures": [],
    }

    if verbose:
        print("Running Dubins Path Planner Tests")
        print("=" * 60)

    np.random.seed(42)

    for i in range(n_random):
        # Random start/goal
        p0 = 10 * np.random.rand(2)
        p1 = 10 * np.random.rand(2)
        psi0 = np.random.uniform(-np.pi, np.pi)
        psi1 = np.random.uniform(-np.pi, np.pi)
        R = 1.0

        # Plan
        cost, ct, a1, d, a2, tp0, tp1, c0, c1 = plan(p0, psi0, p1, psi1, R)

        # Evaluate
        s_vals = np.linspace(0, 1, 200)
        path_x, path_y, path_psi = [], [], []
        for s in s_vals:
            x, y, psi = eval_fn(s, p0, psi0, a1, d, a2, tp0, tp1, c0, c1, R)
            path_x.append(float(x))
            path_y.append(float(y))
            path_psi.append(float(psi))

        path_data = {"x": path_x, "y": path_y, "psi": path_psi}

        # Check continuity
        cont_result = check_continuity(path_data)

        # Check forward motion
        tp0_arr = np.array(tp0).flatten()
        tp1_arr = np.array(tp1).flatten()
        c0_arr = np.array(c0).flatten()
        c1_arr = np.array(c1).flatten()
        plan_inputs = (
            p0,
            psi0,
            float(a1),
            float(d),
            float(a2),
            tp0_arr,
            tp1_arr,
            c0_arr,
            c1_arr,
            R,
        )
        fwd_result = check_forward_motion(path_data, plan_inputs, eval_fn)

        results["total"] += 1
        test_passed = cont_result["passed"] and fwd_result["passed"]

        if test_passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append(
                {
                    "test_id": i,
                    "type": DubinsPathType.name(ct),
                    "continuity": cont_result,
                    "forward_motion": fwd_result,
                }
            )

        if verbose:
            status = "✓ PASS" if test_passed else "✗ FAIL"
            print(f"\nTest {i+1}: {status}")
            print(f"  Type: {DubinsPathType.name(ct)}")
            print(
                f"  Continuity: pos={cont_result['max_pos_jump']:.4f}, "
                f"heading={cont_result['max_heading_jump_deg']:.2f}°"
            )
            print(f"  Forward motion: min_vel={fwd_result['min_forward_velocity']:.6f}")

    if verbose:
        print("\n" + "=" * 60)
        print(f"Results: {results['passed']}/{results['total']} passed")
        if results["failed"] > 0:
            print(f"FAILED: {results['failed']} tests")

    return results
