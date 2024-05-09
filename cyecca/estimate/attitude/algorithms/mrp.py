from cyecca.estimate.attitude.algorithms.common import *

"""
A right invariant extended kalman filter parameterized with
modified rodrigues parameters
:return: dict of equations
"""

# x, state (6)
# -----------
# mrp (3)
# b, gyro bias (3)
G = SO3Mrp * R3
x = ca.SX.sym("x", G.n_param)
X = G.elem(x)
r, b_gyro = G.sub_elems(X)
q = SO3Quat.from_Mrp(r)
C_nb = r.to_Matrix()

# e, error state (6)
# ----------------
# er, so(3) lie algebra rotation error
# eb, R(3) lie algebra rotation error
n_e = 6
eta = ca.SX.sym("eta", G.algebra.n_param)  # (right inv error)
W = ca.SX.sym("W", ca.Sparsity_lower(n_e))


def get_state(**kwargs):
    return ca.Function(
        "get_state", [x], [q.param, r.param, b_gyro.param], ["x"], ["q", "r", "b_gyro"]
    )


def constants(**kwargs):
    x0 = ca.DM.zeros(6)
    return ca.Function("constants", [], [x0, W0], [], ["x0", "W0"])


def initialize(**kwargs):
    g_b = ca.SX.sym("g_b", 3)
    B_b = ca.SX.sym("B_b", 3)

    C_nm = (
        so3.elem(mag_decl * e3).exp(SO3Mrp) * so3.elem(-mag_incl * e2).exp(SO3Mrp)
    ).to_Matrix()

    B_n = C_nm @ e1

    g_norm = ca.norm_2(g_b)
    B_norm = ca.norm_2(B_b)

    n3_b = -g_b / g_norm
    Bh_b = B_b / B_norm

    n2_dir = ca.cross(n3_b, Bh_b)
    n2_dir_norm = ca.norm_2(n2_dir)
    theta = ca.asin(n2_dir_norm)

    # require
    # * g_norm > 5
    # * B_norm > 0
    # * 10 degrees between grav accel and mag vector
    init_ret = ca.if_else(
        ca.fabs(g_norm - 9.8) > 1,
        1,
        ca.if_else(B_norm <= 0, 2, ca.if_else(theta < 10 * deg2rad, 3, 0)),
    )

    n2_b = n2_dir / n2_dir_norm

    # correct based on declination to true east
    n2_b = so3.elem(-mag_decl * n3_b).exp(SO3Mrp).to_Matrix() @ n2_b

    tmp = ca.cross(n2_b, n3_b)
    n1_b = tmp / ca.norm_2(tmp)

    R0 = ca.SX(3, 3)
    R0[0, :] = n1_b
    R0[1, :] = n2_b
    R0[2, :] = n3_b

    r0 = SO3Mrp.from_Matrix(R0)
    SO3Mrp.shadow_if_necessary(r0)
    b0 = ca.SX(3, 1)  # initial bias, zero
    x0 = ca.if_else(init_ret == 0, ca.vertcat(r0.param, b0), ca.SX(G.n_param, 1))
    return ca.Function(
        "init",
        [g_b, B_b, mag_decl],
        [x0, init_ret],
        ["g_b", "B_b", "decl"],
        ["x0", "error_code"],
    )


def predict(**kwargs):
    # state derivative
    xdot = ca.vertcat(
        SO3Mrp.right_jacobian(r) @ (omega_m - b_gyro.param), std_gyro_rw * w_gyro_rw
    )
    f_xdot = ca.Function(
        "xdot",
        [t, x, omega_m, std_gyro, sn_gyro_rw, w_gyro, w_gyro_rw, dt],
        [xdot],
        ["t", "x", "omega_m", "std_gyro", "sn_gyro_rw", "w_gyro", "w_gyro_rw", "dt"],
        ["xdot"],
    )

    # state prop w/o noise
    x1 = util.rk4(
        lambda t, x: f_xdot(t, x, omega_m, 0, 0, ca.DM.zeros(3), ca.DM.zeros(3), dt),
        t,
        x,
        dt,
    )
    r1 = SO3Mrp.elem(x1[:3])
    SO3Mrp.shadow_if_necessary(r1)
    x1[:3] = r1.param

    # error dynamics
    f = ca.Function(
        "f",
        [omega_m, eta, x, w_gyro_rw],
        [ca.vertcat(r @ (-eta[3:6]), w_gyro_rw)],
    )

    # linearized error dynamics
    F = ca.sparsify(
        ca.substitute(
            ca.jacobian(f(omega_m, eta, x, w_gyro_rw), eta), eta, ca.SX.zeros(n_e)
        )
    )

    if "results_dir" in kwargs.keys():
        os.makedirs(kwargs["results_dir"], exist_ok=True)
        g = graph.dotgraph(F)
        g.set("dpi", 180)
        g.write_png(os.path.join(kwargs["results_dir"], "casadi_graph_F.png"))

    # covariance propagation
    f_W_dot_lt = ca.Function(
        "W_dot_lt",
        [x, W, omega_m, std_gyro, sn_gyro_rw, dt],
        [ca.tril(util.sqrt_covariance_predict(W, F, Q))],
    )
    W1 = util.rk4(
        lambda t, y: f_W_dot_lt(x, y, omega_m, std_gyro, sn_gyro_rw, dt), t, W, dt
    )

    # combined prediction function
    return ca.Function(
        "predict",
        [t, x, W, omega_m, std_gyro, sn_gyro_rw, dt],
        [x1, W1],
        ["t", "x", "W", "omega_m", "std_gyro", "sn_gyro_rw", "dt"],
        ["x1", "W1"],
    )


def correct_mag(**kwargs):
    C_nm = SO3EulerB321.elem(ca.vertcat(mag_decl, -mag_incl, 0)).to_Matrix()
    B_n = mag_str * ca.mtimes(C_nm, ca.SX([1, 0, 0]))
    h_mag = ca.Function(
        "h_mag",
        [x, mag_str, mag_decl, mag_incl, std_mag, w_mag],
        [ca.mtimes(C_nb.T, B_n) + w_mag * std_mag],
        ["x", "mag_str", "mag_decl", "mag_incl", "std_mag", "w_mag"],
        ["y"],
    )

    yh_mag = h_mag(x, 1, mag_decl, 0, 0, 0)
    gamma = ca.acos(yh_mag[2] / ca.norm_2(yh_mag))
    h = ca.fmax(ca.sin(gamma), 1e-3)

    y_mag = ca.SX.sym("y_mag", 3, 1)
    y_n = ca.mtimes(C_nb, y_mag)

    H_mag = ca.SX(1, 6)
    H_mag[0, 2] = 1

    std_rot = std_mag + 0.2 * ca.norm_2(
        ca.diag(W)[0:2]
    )  # roll/pitch and mag uncertainty contrib. to projection uncertainty
    arg = std_rot / (2 * h)
    Rs_mag = 8 * ca.if_else(ca.norm_2(arg) < 1, 2 * ca.asin(arg), std_rot)

    W_mag, K_mag, Ss_mag = util.sqrt_correct(Rs_mag, H_mag, W)
    S_mag = ca.mtimes(Ss_mag, Ss_mag.T)
    r_std_mag = ca.diag(Ss_mag)
    r_mag = -ca.atan2(y_n[1], y_n[0]) + mag_decl
    x_mag = G.product(G.exp(G.algebra.elem(ca.mtimes(K_mag, r_mag))), G.elem(x)).param
    x_mag[3] = x[3]  # keep shadow state the same
    beta_mag = ca.mtimes([r_mag.T, ca.inv(S_mag), r_mag]) / beta_mag_c

    # ignore correction when near singular point
    mag_ret = ca.if_else(
        std_rot / 2 > h,  # too close to vertical
        1,
        ca.if_else(ca.norm_2(ca.diag(W)[0:2]) > 0.1, 2, 0),  # too much roll/pitch noise
    )
    x_mag = ca.if_else(mag_ret == 0, x_mag, x)
    W_mag = ca.if_else(mag_ret == 0, W_mag, W)

    return ca.Function(
        "correct_mag",
        [x, W, y_mag, mag_decl, std_mag, beta_mag_c],
        [x_mag, W_mag, beta_mag, r_mag, r_std_mag, mag_ret],
        ["x", "W", "y_b", "decl", "std_mag", "beta_mag_c"],
        ["x_mag", "W_mag", "beta_mag", "r_mag", "r_std_mag", "error_code"],
    )


def correct_accel(**kwargs):
    H_accel = ca.SX(2, 6)
    H_accel[0, 0] = 1
    H_accel[1, 1] = 1

    f_measure_accel = ca.Function(
        "measure_accel",
        [x, g],
        [g * ca.mtimes(C_nb.T, ca.SX([0, 0, -1]))],
        ["x", "g"],
        ["y"],
    )
    yh_accel = f_measure_accel(x, g)
    y_b = ca.SX.sym("y_b", 3)
    n3 = ca.SX([0, 0, 1])
    y_n = ca.mtimes(C_nb, -y_b)
    v_n = ca.cross(y_n, n3) / ca.norm_2(y_b)
    norm_v = ca.norm_2(v_n)
    vh_n = v_n / norm_v
    omega_c_accel_n = ca.if_else(
        ca.logic_and(norm_v > 0, ca.fabs(norm_v) < 1),
        ca.asin(norm_v) * vh_n,
        ca.SX([0, 0, 0]),
    )

    Rs_accel = ca.SX.eye(2) * (std_accel + ca.norm_2(omega_m) ** 2 * std_accel_omega)

    W_accel, K_accel, Ss_accel = util.sqrt_correct(Rs_accel, H_accel, W)
    S_accel = ca.mtimes(Ss_accel, Ss_accel.T)
    r_accel = omega_c_accel_n[0:2]
    r_std_accel = ca.diag(Ss_accel)
    beta_accel = ca.mtimes([r_accel.T, ca.inv(S_accel), r_accel]) / beta_accel_c
    x_accel = G.product(
        G.exp(G.algebra.elem(ca.mtimes(K_accel, r_accel))), G.elem(x)
    ).param
    x_accel[3] = x[3]  # keep shadow state the same
    x_accel = ca.sparsify(x_accel)

    # return status
    accel_ret = ca.if_else(
        ca.fabs(ca.norm_2(y_b) - g) > 1.0, 1, 0  # accel magnitude not close to g,
    )

    x_accel = ca.if_else(accel_ret == 0, x_accel, x)
    W_accel = ca.if_else(accel_ret == 0, W_accel, W)

    return ca.Function(
        "correct_accel",
        [x, W, y_b, g, omega_m, std_accel, std_accel_omega, beta_accel_c],
        [x_accel, W_accel, beta_accel, r_accel, r_std_accel, accel_ret],
        [
            "x",
            "W",
            "y_b",
            "g",
            "omega_b",
            "std_accel",
            "std_accel_omega",
            "beta_accel_c",
        ],
        ["x_accel", "W_accel", "beta_accel", "r_accel", "r_std_accel", "error_code"],
    )


def eqs(**kwargs):
    return {
        "initialize": initialize(**kwargs),
        "predict": predict(**kwargs),
        "correct_mag": correct_mag(**kwargs),
        "correct_accel": correct_accel(**kwargs),
        "get_state": get_state(**kwargs),
        "constants": constants(**kwargs),
    }
