import casadi as ca
import cyecca
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def derive_model():
    n_motor = 4

    # p, parameters
    tau_up = ca.SX.sym("tau_up")
    tau_down = ca.SX.sym("tau_down")
    kv = ca.SX.sym("kv")
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
    lbar = ca.SX.sym("lbar")
    g = ca.SX.sym("g")
    J_x = ca.SX.sym("J_x")
    J_y = ca.SX.sym("J_y")
    J_z = ca.SX.sym("J_z")
    m = ca.SX.sym("m")
    J = ca.diag(ca.vertcat(J_x, J_y, J_z))  ## assuming symmetrical
    p = ca.vertcat(
        tau_up, tau_down, kv, dir_motor, l_motor, theta_motor,
        CT, CM, Cl_p, Cm_q, Cn_r, CD0, S, rho, lbar, g, m,
        J_x, J_y, J_z,
    )
    p_defaults = {
        "tau_up": 0.1,
        "tau_down": 0.5,
        "kv": 850,
        "dir_motor_0": 1,
        "dir_motor_1": 1,
        "dir_motor_2": -1,
        "dir_motor_3": -1,
        "l_motor_0": 1,
        "l_motor_1": 1,
        "l_motor_2": 1,
        "l_motor_3": 1,
        "theta_motor_0": np.pi / 4,
        "theta_motor_1": -3 * np.pi / 4,
        "theta_motor_2": -np.pi / 4,
        "theta_motor_3": 3 * np.pi / 4,
        "CT": 1e-5,
        "CM": 1e-8,
        "Cl_p": -1e-2,
        "Cm_q": -1e-2,
        "Cn_r": -1e-2,
        "CD0" : 1e-1,
        "S": 1e-1,  # aerodynamic reference area
        "rho": 1.225, # air density
        "lbar": 1, # reference length for aerodynamic moments (should be distance from center to motor)
        "g": 9.8,
        "m": 1.0,
        "J_x": 1,
        "J_y": 1,
        "J_z": 1,
    }

    # x, state
    normalized_motor = ca.SX.sym("normalized_motor", n_motor)
    omega_wb_b = ca.SX.sym("omega_wb_b", 3)
    quaternion_wb = ca.SX.sym("quaternion_wb", 4)
    velocity_w_p_b = ca.SX.sym("velocity_w_p_b", 3)
    position_op_w = ca.SX.sym("position_op_w", 3)

    x = ca.vertcat(
        normalized_motor,
        omega_wb_b,
        quaternion_wb,
        velocity_w_p_b,
        position_op_w,
    )

    x0_defaults = {
        "normalized_motor_0": 0,
        "normalized_motor_1": 1,
        "normalized_motor_2": 2,
        "normalized_motor_3": 3,
        "omega_wb_b_0": 0,
        "omega_wb_b_1": 0,
        "omega_wb_b_2": 0,
        "quaternion_wb_0": 1,
        "quaternion_wb_1": 0,
        "quaternion_wb_2": 0,
        "quaternion_wb_3": 0,
        "velocity_w_p_b_0": 0,
        "velocity_w_p_b_1": 0,
        "velocity_w_p_b_2": 0,
        "position_op_w_0": 0,
        "position_op_w_1": 0,
        "position_op_w_2": 0,
    }

    # u, input
    command_normalized_motors = ca.SX.sym("command_normalized_motors", n_motor)
    u = ca.vertcat(command_normalized_motors)

    # motor first order model
    tau = ca.if_else(
        command_normalized_motors - normalized_motor > 0, tau_up, tau_down
    )
    derivative_normalized_motors = (
        -1 / tau * (normalized_motor - command_normalized_motors)
    )
    state_omega_motors = kv * normalized_motor

    # sum of forces and moments
    xAxis = ca.vertcat(1, 0, 0)
    yAxis = ca.vertcat(0, 1, 0)
    zAxis = ca.vertcat(0, 0, 1)
    q_wb = cyecca.lie.SO3Quat.elem(quaternion_wb)
    V = ca.norm_2(velocity_w_p_b)
    wX = ca.if_else(ca.fabs(V) > 1e-5, velocity_w_p_b/V, ca.vertcat(1, 0, 0))
    qbar = 0.5*rho*V**2
    P = omega_wb_b[0]
    Q = omega_wb_b[1]
    R = omega_wb_b[2]

    # aerodynamic coefficients
    CD = CD0  # drag
    Cl = Cl_p*P  # rolling moment
    Cm = Cm_q*Q  # pitching moment
    Cn = Cn_r*R  # yawing moment
        
    F_b = (
        q_wb @ (-m * g * zAxis) # gravity
        - CD*qbar*S*wX # drag
    )
    
    M_b = ca.vertcat(0, 0, 0)
    for i in range(n_motor):
        Fi_b = CT * state_omega_motors[i] ** 2 * zAxis
        ri_b = l_motor[i] * ca.vertcat(
            ca.cos(theta_motor[i]), ca.sin(theta_motor[i]), 0
        )
        Mi_b = (
            ca.cross(ri_b, Fi_b) # moment due to thrust
            + CM * dir_motor[i] * state_omega_motors[i] ** 2 * zAxis  # moment due prop torque
            + ca.vertcat(Cl, Cm, Cn)*S*lbar # aerodynamic moment
        )
        F_b += Fi_b
        M_b += Mi_b

    # kinematics
    derivative_omega_wb_b = ca.inv(J) @ (
        M_b - ca.cross(omega_wb_b, J @ omega_wb_b)
    )
    derivative_quaternion_wb = (
        q_wb.right_jacobian() @ omega_wb_b
    )
    derivative_position_op_w = q_wb @ velocity_w_p_b
    derivative_velocity_w_p_b = F_b / m - ca.cross(
        omega_wb_b, velocity_w_p_b
    )

    # state derivative vector
    x_dot = ca.vertcat(
        derivative_normalized_motors,
        derivative_omega_wb_b,
        derivative_quaternion_wb,
        derivative_velocity_w_p_b,
        derivative_position_op_w,
    )
    f = ca.Function("f", [x, u, p], [x_dot], ["x", "u", "p"], ["x_dot"])
    
    # algebraic (these algebraic expressions are used during the simulation)
    z = ca.vertcat()
    alg = z
    
    # output  (these happen at end of the simulation)
    q_norm = ca.norm_2(quaternion_wb)
    output_M_b = ca.SX.sym("M_b", 3)
    output_F_b = ca.SX.sym("F_b", 3)
    output_q_norm = ca.SX.sym("q_norm")
    output_euler = ca.SX.sym("euler", 3)
    y = ca.vertcat(
        output_M_b,
        output_F_b,
        output_q_norm,
        output_euler
    )
    y_expressions = ca.vertcat(M_b, F_b, q_norm, cyecca.lie.SO3EulerB321.from_Quat(q_wb).param)
    g = ca.Function("g", [x, u, p], [y_expressions], ["x", "u", "p"], ["y"])

    # setup integrator
    dae = {"x": x, "ode": f(x, u, p), "p": p, "u": u, "z": z, "alg": alg}

    p_index = {p[i].name(): i for i in range(p.shape[0])}
    x_index = {x[i].name(): i for i in range(x.shape[0])}
    y_index = {y[i].name(): i for i in range(y.shape[0])}
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

def plotting(model: dict, t: np.array, res: dict, path: str):
    path = Path(path)
    if not path.exists():
        path.mkdir()

    plt.figure()
    plt.title("motor normalized rpm")
    plt.plot(t, res["xf"].T[:, model["x_index"]["normalized_motor_0"]], label="0")
    plt.plot(t, res["xf"].T[:, model["x_index"]["normalized_motor_1"]], label="1")
    plt.plot(t, res["xf"].T[:, model["x_index"]["normalized_motor_2"]], label="2")
    plt.plot(t, res["xf"].T[:, model["x_index"]["normalized_motor_3"]], label="3")
    plt.xlabel("t [sec]")
    plt.ylabel("normalized rpm []")
    plt.grid()
    plt.legend(ncols=4)
    plt.savefig(path / "motor_normalized_rpm")
    plt.close()

    plt.figure()
    plt.title("moment in body frame")
    plt.plot(t, res["yf"].T[:, model["y_index"]["M_b_0"]], label="x")
    plt.plot(t, res["yf"].T[:, model["y_index"]["M_b_1"]], label="y")
    plt.plot(t, res["yf"].T[:, model["y_index"]["M_b_2"]], label="z")
    plt.xlabel("t [sec]")
    plt.ylabel("moment [N-m]")
    plt.grid()
    plt.legend(ncols=3)
    plt.savefig(path / "moment_in_body_frame")
    plt.close()
    
    plt.figure()
    plt.title("force in body frame")
    plt.plot(t, res["yf"].T[:, model["y_index"]["F_b_0"]], label="x")
    plt.plot(t, res["yf"].T[:, model["y_index"]["F_b_1"]], label="y")
    plt.plot(t, res["yf"].T[:, model["y_index"]["F_b_2"]], label="z")
    plt.xlabel("t [sec]")
    plt.ylabel("force [N]")
    plt.grid()
    plt.legend(ncols=3)
    plt.savefig(path / "force_in_body_frame")
    plt.close()

    plt.figure()
    plt.title("position in world frame")
    plt.plot(t, res["xf"].T[:, model["x_index"]["position_op_w_0"]], label="x")
    plt.plot(t, res["xf"].T[:, model["x_index"]["position_op_w_1"]], label="y")
    plt.plot(t, res["xf"].T[:, model["x_index"]["position_op_w_2"]], label="z")
    plt.xlabel("t [sec]")
    plt.ylabel("position [m]")
    plt.grid()
    plt.legend(ncols=3)
    plt.savefig(path / "position_in_world_frame")
    plt.close()

    plt.figure()
    plt.title("velocity in body frame")
    plt.plot(t, res["xf"].T[:, model["x_index"]["velocity_w_p_b_0"]], label="x")
    plt.plot(t, res["xf"].T[:, model["x_index"]["velocity_w_p_b_1"]], label="y")
    plt.plot(t, res["xf"].T[:, model["x_index"]["velocity_w_p_b_2"]], label="z")
    plt.xlabel("t [sec]")
    plt.ylabel("velocity [m/s]")
    plt.grid()
    plt.legend(ncols=3)
    plt.savefig(path / "velocity_in_body_frame")
    plt.close()

    plt.figure()
    plt.title("quaternion from world to body frame")
    plt.plot(t, res["xf"].T[:, model["x_index"]["quaternion_wb_0"]], label="w")
    plt.plot(t, res["xf"].T[:, model["x_index"]["quaternion_wb_1"]], label="x")
    plt.plot(t, res["xf"].T[:, model["x_index"]["quaternion_wb_2"]], label="y")
    plt.plot(t, res["xf"].T[:, model["x_index"]["quaternion_wb_3"]], label="z")
    plt.xlabel("t [sec]")
    plt.ylabel("component")
    plt.grid()
    plt.savefig(path / "quaternion_from_world_to_body_frame")
    plt.close()

    plt.figure()
    plt.title("quaternion normal error")
    plt.plot(t, res["yf"].T[:, model["y_index"]["q_norm"]] - 1, label="norm")
    plt.xlabel("t [sec]")
    plt.ylabel("error []")
    plt.grid()
    plt.legend(ncols=5)
    plt.savefig(path / "quaternion_normal_error")
    plt.close()

    plt.figure()
    plt.title("euler angles from world to body")
    plt.plot(t, np.rad2deg(res["yf"].T[:, model["y_index"]["euler_0"]]), label="yaw")
    plt.plot(t, np.rad2deg(res["yf"].T[:, model["y_index"]["euler_1"]]), label="pitch")
    plt.plot(t, np.rad2deg(res["yf"].T[:, model["y_index"]["euler_2"]]), label="roll")
    plt.grid()
    plt.xlabel("t [sec]")
    plt.ylabel("angle [deg]")
    plt.legend(ncols=3)
    plt.close()

    plt.figure()
    plt.title("angular velocity in body frame")
    plt.plot(t, np.rad2deg(res["xf"].T[:, model["x_index"]["omega_wb_b_0"]]), label="x")
    plt.plot(t, np.rad2deg(res["xf"].T[:, model["x_index"]["omega_wb_b_1"]]), label="y")
    plt.plot(t, np.rad2deg(res["xf"].T[:, model["x_index"]["omega_wb_b_2"]]), label="z")
    plt.grid()
    plt.xlabel("t [sec]")
    plt.ylabel("angular velocity [deg/s]")
    plt.legend(ncols=3)
    plt.savefig(path / "angular_velocity_in_body_frame")
    plt.close()