import casadi as ca
import cyecca
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cyecca.lie.group_so3 import SO3Quat, SO3EulerB321
import cyecca.lie as lie
# from tf_transformations import euler_from_quaternion


def derive_model():
    # p, parameters
    thr_max = ca.SX.sym("thr_max")
    m = ca.SX.sym("m")
    cl = ca.SX.sym("cl")
    cd = ca.SX.sym("cd")
    S = ca.SX.sym("S")
    rho = ca.SX.sym("rho")
    g = ca.SX.sym("g")
    Jx = ca.SX.sym("Jx")
    Jy = ca.SX.sym("Jy")
    Jz = ca.SX.sym("Jz")
    J = ca.diag(ca.vertcat(Jx, Jy, Jz))
    Cl_p = ca.SX.sym("Cl_p")
    Cm_q = ca.SX.sym("Cm_q")
    Cn_r = ca.SX.sym("Cn_r")
    p= ca.vertcat(
        thr_max,
        m,
        cl,
        cd,
        S,
        rho,
        g,
        Jx,
        Jy,
        Jz,
        Cl_p,
        Cm_q,
        Cn_r,
    )
    p_defaults = {
        "thr_max" : 1,
        "m" : 0.2,
        "cl": 6.28,
        "cd" : 0.0,
        "S":1.0,
        "rho": 1.225,
        "g": 9.8,
        'Jx': 0.0217,
        'Jy': 0.0217,
        "Jz" : 0.04,
        "Cl_p": 0.02,
        "Cm_q": 0.02,
        "Cn_r": 0.02,
    }

    # states
    position_w = ca.SX.sym("position_w",3) # w = world frame
    velocity_b = ca.SX.sym("velocity_b",3)
    quat_wb = ca.SX.sym("quat_wb",4) # Quaternion world - body frame
    omega_wb_b = ca.SX.sym("omega_wb_b",3) # world-body

    x = ca.vertcat(
        position_w,
        velocity_b,
        quat_wb,
        omega_wb_b,
        )
    x0_defaults = {
        "position_w_0" : 0,
        "position_w_1" : 0,
        "position_w_2" : 0,
        "velocity_b_0" : 1e-10,
        "velocity_b_1" : 1e-10,
        "velocity_b_2" : 1e-10,
        "quat_wb_0" : 1,
        "quat_wb_1" : 0,
        "quat_wb_2" : 0,
        "quat_wb_3" : 0,
        "omega_wb_b_0" : 0,
        "omega_wb_b_1" : 0,
        "omega_wb_b_2" : 0,
    }

    # input
    throttle_cmd = ca.SX.sym("throttle_cmd")
    ail_cmd = ca.SX.sym("ail_cmd")
    elev_cmd = ca.SX.sym("elev_cmd")
    rud_cmd = ca.SX.sym("rud_cmd")

    u = ca.vertcat(throttle_cmd, ail_cmd, elev_cmd, rud_cmd)

    xAxis = ca.vertcat(1, 0, 0)
    yAxis = ca.vertcat(0, 1, 0)
    zAxis = ca.vertcat(0, 0, 1)

    V_b = ca.norm_2(velocity_b)
    alpha = ca.if_else(ca.fabs(velocity_b[0]) > 1e-1, ca.atan(velocity_b[2]/velocity_b[0]), ca.SX(0))
    beta = ca.if_else(ca.fabs(V_b) > 1e-5, ca.asin(velocity_b[1]/V_b), ca.SX(0))
    euler_n = lie.SO3EulerB321.elem(ca.vertcat(beta, -alpha, 0)) # Euler elements for wind frame
    quat_bn = lie.SO3Quat.from_Euler(euler_n)

    quat_wb = lie.SO3Quat.elem(quat_wb)
    quat_bw = quat_wb.inverse()
    P = omega_wb_b[0]
    Q = omega_wb_b[1]
    R = omega_wb_b[2]

    velocity_w_w = quat_wb @ velocity_b #Velocity in Wind frame

    # force and moment
    # qbar = 0.5 * rho * velocity_b[0]**2  # qbar in terms of body vel_x
    qbar = 0.5 * rho * ca.norm_2(velocity_w_w)**2 # TODO Recheck wind frame velocity --> qbar in terms of wind velocity

    ground = ca.if_else(position_w[2]<0,
        -position_w[2] * 150 *zAxis - velocity_w_w * 150,
        ca.vertcat(0,0,0))
    
    D = cd * qbar * S # Drag force -- wind frame
    L = cl * qbar * S # Lift force -- wind frame
    Fs = 0 #side force

    F_n = ca.vertcat(-D, Fs, L) #force in wind frame (n)
    F_b = quat_bn @ ca.SX(F_n) # Aerodynamic force from wind in body frame

    F_b += quat_bw @ ground *zAxis # TODO current method does not consider wheel friction on ground
    # F_b += quat_bw@ground

    # TODO RECHECK Gravity --> frame seems reverse
    F_b += (-m*g*zAxis) #gravity default in world frame
    # F_b += quat_bw @ (-m * g * zAxis) # gravity

    fx_b = (thr_max*u[0]-velocity_b[0])*xAxis # Longitudinal Force assume thrust is directly on the x axis
    F_b += fx_b #force due to thrust

    # Moment
    M_b = ca.vertcat(0, 0, 0)
    # moment_Cl = Cl_p * P  # rolling moment
    # moment_Cm = Cm_q * Q  # pitching moment
    # moment_Cn = Cn_r * R  # yawing moment
    # Mi_b = ca.vertcat(moment_Cl, moment_Cm, moment_Cn) * S # aerodynamic moment in body frame
    # M_b += Mi_b

    M_b += (u[1]-omega_wb_b[0]) * Cl_p *xAxis #moment due to roll
    M_b += (u[2]-omega_wb_b[1]) * Cm_q *yAxis #moment due to elevator
    M_b += (u[3]-omega_wb_b[2]) * Cn_r *zAxis #moment due to rudder

    # # kinematics
    derivative_omega_wb_b = ca.inv(J) @ (M_b - ca.cross(omega_wb_b, J @ omega_wb_b))
    derivative_quaternion_wb = quat_wb.right_jacobian() @ omega_wb_b
    derivative_position_w = quat_wb @ velocity_b
    derivative_velocity_b = F_b / m - ca.cross(omega_wb_b, velocity_b)

    # state derivative vector
    xdot = ca.vertcat(
        derivative_position_w,
        derivative_velocity_b,
        derivative_quaternion_wb,
        derivative_omega_wb_b,
    )


    # algebraic (these algebraic expressions are used during the simulation)
    z = ca.vertcat()
    alg = z


    f = ca.Function("f", [x, u, p], [xdot], ["x", "u", "p"], ["xdot"])

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
