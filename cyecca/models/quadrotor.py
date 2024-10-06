import casadi as ca
import cyecca
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def derive_model():
    n_motor = 4
    g0 = 9.8

    # p, parameters
    tau_up = ca.SX.sym("tau_up")
    tau_down = ca.SX.sym("tau_down")
    dir_motor = ca.SX.sym("dir_motor", n_motor)
    l_motor = ca.SX.sym("l_motor", n_motor)
    theta_motor = ca.SX.sym("theta_motor", n_motor)
    CT = ca.SX.sym("CT")
    CM = ca.SX.sym("CM")
    Cl_p = ca.SX.sym("Cl_p")
    Cm_q = ca.SX.sym("Cm_q")
    Cn_r = ca.SX.sym("Cn_r")
    CD0 = ca.SX.sym("CD0")
    S = ca.SX.sym("S")
    rho = ca.SX.sym("rho")
    g = ca.SX.sym("g")
    m = ca.SX.sym("m")
    Jx = ca.SX.sym("Jx")
    Jy = ca.SX.sym("Jy")
    Jz = ca.SX.sym("Jz")
    J = ca.diag(ca.vertcat(Jx, Jy, Jz))
    noise_power_sqrt_a_b = ca.SX.sym("noise_power_sqrt_a_b", 3)
    noise_power_sqrt_omega_wb_b = ca.SX.sym("noise_power_sqrt_omega_wb_b", 3)
    noise_power_sqrt_mag_b = ca.SX.sym("noise_power_sqrt_mag_b", 3)
    noise_power_sqrt_gps_pos = ca.SX.sym("noise_power_sqrt_gps_pos", 3)
    p = ca.vertcat(
        tau_up,
        tau_down,
        dir_motor,
        l_motor,
        theta_motor,
        CT,
        CM,
        Cl_p,
        Cm_q,
        Cn_r,
        CD0,
        S,
        rho,
        g,
        m,
        Jx,
        Jy,
        Jz,
        noise_power_sqrt_a_b,
        noise_power_sqrt_omega_wb_b,
        noise_power_sqrt_mag_b,
        noise_power_sqrt_gps_pos,
    )
    p_defaults = {
        "tau_up": 0.0125,  # time to spin up motors
        "tau_down": 0.025,  # time to spin down motors
        "dir_motor_0": 1,  # diretion of motor 0 (1 CCW, -1 CW)
        "dir_motor_1": 1,
        "dir_motor_2": -1,
        "dir_motor_3": -1,
        "l_motor_0": 0.25,  # arm length of motor 0
        "l_motor_1": 0.25,
        "l_motor_2": 0.25,
        "l_motor_3": 0.25,
        "theta_motor_0": -np.pi / 4,  # angle of arm 0
        "theta_motor_1": 3 * np.pi / 4,
        "theta_motor_2": np.pi / 4,
        "theta_motor_3": -3 * np.pi / 4,
        "CT": 8.54858e-06,  # thrust coefficient
        "CM": 0.016,  # moment coefficient
        "Cl_p": 0,
        "Cm_q": 0,
        "Cn_r": 0,
        "CD0": 0,
        "S": 1e-1,  # aerodynamic reference area
        "rho": 1.225,  # air density
        "g": 9.8,
        "m": 2.0,
        "Jx": 0.02166666666666667,
        "Jy": 0.02166666666666667,
        "Jz": 0.04000000000000001,
        "noise_power_sqrt_a_b_0": 70e-6 * g0,  # micro-g/sqrt(hz)
        "noise_power_sqrt_a_b_1": 70e-6 * g0,
        "noise_power_sqrt_a_b_2": 70e-6 * g0,
        "noise_power_sqrt_omega_wb_b_0": np.deg2rad(2.8e-3),  # 2.8 milli-dpgs/sqrt(hz)
        "noise_power_sqrt_omega_wb_b_1": np.deg2rad(2.8e-3),
        "noise_power_sqrt_omega_wb_b_2": np.deg2rad(2.8e-3),
        "noise_power_sqrt_mag_b_0": 0,
        "noise_power_sqrt_mag_b_1": 0,
        "noise_power_sqrt_mag_b_2": 0,
        "noise_power_sqrt_gps_pos_0": 0,
        "noise_power_sqrt_gps_pos_1": 0,
        "noise_power_sqrt_gps_pos_2": 0,
    }

    # x, state
    omega_motor = ca.SX.sym("omega_motor", n_motor)
    omega_wb_b = ca.SX.sym("omega_wb_b", 3)
    quaternion_wb = ca.SX.sym("quaternion_wb", 4)
    velocity_w_p_b = ca.SX.sym("velocity_w_p_b", 3)
    position_op_w = ca.SX.sym("position_op_w", 3)

    x = ca.vertcat(
        position_op_w,
        velocity_w_p_b,
        quaternion_wb,
        omega_wb_b,
        omega_motor,
    )

    x0_defaults = {
        "position_op_w_0": 0,
        "position_op_w_1": 0,
        "position_op_w_2": 0,
        "velocity_w_p_b_0": 0,
        "velocity_w_p_b_1": 0,
        "velocity_w_p_b_2": 0,
        "quaternion_wb_0": 1,
        "quaternion_wb_1": 0,
        "quaternion_wb_2": 0,
        "quaternion_wb_3": 0,
        "omega_wb_b_0": 0,
        "omega_wb_b_1": 0,
        "omega_wb_b_2": 0,
        "omega_motor_0": 0,
        "omega_motor_1": 0,
        "omega_motor_2": 0,
        "omega_motor_3": 0,
    }

    # u, input
    omega_motor_cmd = ca.SX.sym("omega_motor_cmd", n_motor)
    u = ca.vertcat(
        omega_motor_cmd,
    )

    # motor first order model
    tau_inv = ca.SX.zeros(4, 1)

    for i in range(n_motor):
        tau_inv[i] = ca.if_else(
            omega_motor_cmd[i] - omega_motor[i] > 0, 1.0 / tau_up, 1.0 / tau_down
        )
    derivative_omega_motor = tau_inv * (omega_motor_cmd - omega_motor)

    # sum of forces and moments
    xAxis = ca.vertcat(1, 0, 0)
    yAxis = ca.vertcat(0, 1, 0)
    zAxis = ca.vertcat(0, 0, 1)
    q_wb = cyecca.lie.SO3Quat.elem(quaternion_wb)
    q_bw = q_wb.inverse()
    V = ca.norm_2(velocity_w_p_b)
    wX = ca.if_else(ca.fabs(V) > 1e-5, velocity_w_p_b / V, ca.vertcat(1, 0, 0))
    qbar = 0.5 * rho * V**2
    P = omega_wb_b[0]
    Q = omega_wb_b[1]
    R = omega_wb_b[2]

    # aerodynamic coefficients
    CD = CD0  # drag
    Cl = Cl_p * P  # rolling moment
    Cm = Cm_q * Q  # pitching moment
    Cn = Cn_r * R  # yawing moment

    velocity_w_p_w = q_wb @ velocity_w_p_b

    # attempt for multiple leg collision forces
    # position_oa_w = position_op_w + q_bw @ ca.vertcat(0.17, 0.17, -0.1)
    # position_ob_w = position_op_w + q_bw @ ca.vertcat(0.17, -0.17, -0.1)
    # position_oc_w = position_op_w + q_bw @ ca.vertcat(-0.17, -0.17, -0.1)
    # position_od_w = position_op_w + q_bw @ ca.vertcat(-0.17, 0.17, -0.1)

    # Fa_w = ca.if_else(position_oa_w[2] < 0, -1000*position_oa_w[2] * zAxis - 100 * velocity_w_p_w, ca.vertcat(0, 0, 0))
    # Fb_w = ca.if_else(position_ob_w[2] < 0, -1000*position_ob_w[2] * zAxis - 100 * velocity_w_p_w, ca.vertcat(0, 0, 0))
    # Fc_w = ca.if_else(position_oc_w[2] < 0, -1000*position_oc_w[2] * zAxis - 100 * velocity_w_p_w, ca.vertcat(0, 0, 0))
    # Fd_w = ca.if_else(position_od_w[2] < 0, -1000*position_od_w[2] * zAxis - 100 * velocity_w_p_w, ca.vertcat(0, 0, 0))

    F_w = ca.if_else(  # ground
        position_op_w[2] < 0,
        -1000 * position_op_w[2] * zAxis - 1000 * velocity_w_p_w,
        ca.vertcat(0, 0, 0),
    )

    F_b = q_bw @ F_w - CD * qbar * S * wX  # drag

    M_b = ca.vertcat(0, 0, 0)
    for i in range(n_motor):
        thrust = CT * omega_motor[i] ** 2
        Fi_b = thrust * zAxis
        ri_b = l_motor[i] * ca.vertcat(
            ca.cos(theta_motor[i]), ca.sin(theta_motor[i]), 0
        )
        Mi_b = (
            ca.cross(ri_b, Fi_b)  # moment due to thrust
            - CM * dir_motor[i] * thrust * zAxis  # moment due prop torque
            + ca.vertcat(Cl, Cm, Cn) * S * l_motor[i]  # aerodynamic moment
        )
        F_b += Fi_b
        M_b += Mi_b

    # accelerometer zero in freefall (doesn't measure gravity)
    a_b = F_b / m

    F_b += q_bw @ (-m * g * zAxis)  # gravity

    # kinematics
    derivative_omega_wb_b = ca.inv(J) @ (M_b - ca.cross(omega_wb_b, J @ omega_wb_b))
    derivative_quaternion_wb = q_wb.right_jacobian() @ omega_wb_b
    derivative_position_op_w = q_wb @ velocity_w_p_b
    derivative_velocity_w_p_b = F_b / m - ca.cross(omega_wb_b, velocity_w_p_b)

    # state derivative vector
    x_dot = ca.vertcat(
        derivative_position_op_w,
        derivative_velocity_w_p_b,
        derivative_quaternion_wb,
        derivative_omega_wb_b,
        derivative_omega_motor,
    )
    f = ca.Function("f", [x, u, p], [x_dot], ["x", "u", "p"], ["x_dot"])

    # algebraic (these algebraic expressions are used during the simulation)
    z = ca.vertcat()
    alg = z

    # -----------------------------------------
    # measurements
    # -----------------------------------------

    # measurement noise
    dt = ca.SX.sym("dt")  # sample time
    w3 = ca.SX.sym(
        "w", 3
    )  # 3 dim noise (std dev = 1, mean = 0), scaling via noise power occurs in function from params

    # cyecca.lie.SO3EulerB321.from_Quat(q_wb).param
    g_accel = ca.Function(
        "g_accel",
        [x, u, p, w3, dt],
        [a_b + w3 * noise_power_sqrt_a_b * np.sqrt(dt)],
        ["x", "u", "p", "w", "dt"],
        ["y"],
    )
    g_gyro = ca.Function(
        "g_gyro",
        [x, u, p, w3, dt],
        [omega_wb_b + w3 * noise_power_sqrt_omega_wb_b * np.sqrt(dt)],
        ["x", "u", "p", "w", "dt"],
        ["y"],
    )
    g_mag = ca.Function(
        "g_mag",
        [x, u, p, w3, dt],
        [a_b + w3 * noise_power_sqrt_mag_b * np.sqrt(dt)],
        ["x", "u", "p", "w", "dt"],
        ["y"],
    )
    g_gps_pos = ca.Function(
        "g_gps_pos",
        [x, u, p, w3, dt],
        [position_op_w + w3 * np.sqrt(noise_power_sqrt_gps_pos / dt)],
        ["x", "u", "p", "w", "dt"],
        ["y"],
    )

    # setup integrator
    dae = {"x": x, "ode": f(x, u, p), "p": p, "u": u, "z": z, "alg": alg}

    p_index = {p[i].name(): i for i in range(p.shape[0])}
    x_index = {x[i].name(): i for i in range(x.shape[0])}
    u_index = {u[i].name(): i for i in range(u.shape[0])}
    z_index = {z[i].name(): i for i in range(z.shape[0])}

    return locals()


def sim(model, t, u, x0=None, p=None, plot=True):
    x0_dict = model["x0_defaults"]
    if x0 is not None:
        for k in x0.keys():
            if not k in x0_dict.keys():
                raise KeyError(k)
            x0_dict[k] = x0[k]
    p_dict = model["p_defaults"]
    if p is not None:
        for k in p.keys():
            if not k in p_dict.keys():
                raise KeyError(k)
            p_dict[k] = p[k]
    dae = model["dae"]
    f_int = ca.integrator("test", "idas", dae, t[0], t)
    res = f_int(x0=x0_dict.values(), z0=0, p=p_dict.values(), u=u)
    res["p"] = p_dict
    res["yf"] = model["g"](res["xf"], u, p_dict.values())

    for k in ["xf", "yf", "zf"]:
        res[k] = np.array(res[k])
    return res
