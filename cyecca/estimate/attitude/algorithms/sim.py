from .common import *

# x, state (7)
# -----------
# r, mrp (3)
# b, gyro bias (3)
x = ca.SX.sym("x", 6)
r = SO3Mrp.elem(x[0:3])  # last state is shadow state
b_gyro = x[3:6]
q = SO3Quat.from_Mrp(r)
C_nb = SO3Dcm.from_Mrp(r)


def get_state(**kwargs):
    return ca.Function(
        "get_state", [x], [q.param, r.param, b_gyro], ["x"], ["q", "r", "b_gyro"]
    )


def simulate(**kwargs):
    # state derivative
    xdot = ca.vertcat(SO3Mrp.right_jacobian(r) @ omega_t, std_gyro_rw * w_gyro_rw)
    f_xdot = ca.Function(
        "xdot",
        [t, x, omega_t, sn_gyro_rw, w_gyro_rw, dt],
        [xdot],
        ["t", "x", "omega_t", "sn_gyro_rw", "w_gyro_rw", "dt"],
        ["xdot"],
    )

    # state prop with noise
    x1_sim = util.rk4(
        lambda t, x: f_xdot(t, x, omega_t, sn_gyro_rw, w_gyro_rw, dt), t, x, dt
    )
    r1 = SO3Mrp.elem(x1_sim[:3])
    SO3Mrp.shadow_if_necessary(r1)
    x1_sim[:3] = r1.param
    return ca.Function(
        "simulate",
        [t, x, omega_t, sn_gyro_rw, w_gyro_rw, dt],
        [x1_sim],
        ["t", "x", "omega_t", "sn_gyro_rw", "w_gyro_rw", "dt"],
        ["x1"],
    )


def measure_gyro(**kwargs):
    return ca.Function(
        "measure_gyro",
        [x, omega_t, std_gyro, w_gyro],
        [omega_t + b_gyro + w_gyro * std_gyro],
        ["x", "omega_t", "std_gyro", "w_gyro"],
        ["y"],
    )


def measure_mag(**kwargs):
    C_nm = SO3Dcm.exp(so3.elem(mag_decl * e3)) * SO3Dcm.exp(so3.elem(-mag_incl * e2))
    B_n = C_nm @ (mag_str * e1)
    return ca.Function(
        "measure_mag",
        [x, mag_str, mag_decl, mag_incl, std_mag, w_mag],
        [C_nb.inverse() @ B_n + w_mag * std_mag],
        ["x", "mag_str", "mag_decl", "mag_incl", "std_mag", "w_mag"],
        ["y"],
    )


def measure_accel(**kwargs):
    return ca.Function(
        "measure_accel",
        [x, g, std_accel, w_accel],
        [C_nb.inverse() @ (-g * e3) + w_accel * std_accel],
        ["x", "g", "std_accel", "w_accel"],
        ["y"],
    )


def constants(**kwargs):
    x0 = ca.DM([0.1, 0.2, 0.3, 0, 0, 0.01])
    return ca.Function("constants", [], [x0], [], ["x0"])


def rotation_error(**kwargs):
    q1 = SO3Quat.elem(ca.SX.sym("q1", 4))
    q2 = SO3Quat.elem(ca.SX.sym("q2", 4))
    dq = q1.inverse() * q2
    xi = dq.log()
    return ca.Function(
        "rotation_error", [q1.param, q2.param], [xi.param], ["q1", "q2"], ["xi"]
    )


def eqs(**kwargs):
    return {
        "simulate": simulate(**kwargs),
        "measure_gyro": measure_gyro(**kwargs),
        "measure_accel": measure_accel(**kwargs),
        "measure_mag": measure_mag(**kwargs),
        "constants": constants(**kwargs),
        "get_state": get_state(**kwargs),
        "rotation_error": rotation_error(**kwargs),
    }
