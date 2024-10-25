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
        "thr_max" : 1.0,
        "m" : 0.2,
        "cl": 6.28,
        "cd" : 0.0,
        "S":1.0,
        "rho": 1.225,
        "g": 9.8,
        'Jx': 0.0217,
        'Jy': 0.0217,
        "Jz" : 0.04,
        "Cl_p": 0,
        "Cm_q": 0,
        "Cn_r": 0,
    }

    # states
    # # x, state
    # posx = ca.SX.sym("posx")
    # posy = ca.SX.sym("posy")
    # posz = ca.SX.sym("posz")
    # velx = ca.SX.sym("velx")
    # vely = ca.SX.sym("vely")
    # velz = ca.SX.sym("velz")

    position_w = ca.SX.sym("position_w",3) # w = world frame
    velocity_b = ca.SX.sym("velocity_b",3)
    quat_wb = ca.SX.sym("quat_b",4) # Quaternion world - body frame
    # quad_w = ca.SX.sym("quad_w",4)
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
    elev_cmd = ca.SX.sym("elev_cmd")

    u = ca.vertcat(throttle_cmd, elev_cmd)


    # Defining frames
    # code this, recheck in stewen and lewis
   # alpha = atan(vel_b_z/vel_b_x)
   # beta = y/atan(posy/posx)
   # test =lie.SO3EulerB321.elem(ca.vertcat(beta,-alpha,0))
    # q_bn = lie.SO3Quat.from_Euler(test)  #body to wind
    # fn = ca.vert(L ,sideforce,drag) #n is wind frame
    # fb= q_bn@fn #force in body frame

    # idea:calculate alpha and beta based on stephen and lewis
    # quat --> world to body
    # Quat --> body to wind
    xAxis = ca.vertcat(1, 0, 0)
    yAxis = ca.vertcat(0, 1, 0)
    zAxis = ca.vertcat(0, 0, 1)


    # VT = ca.norm_2(ca.vertcat(velocity_b[0],velocity_b[1],velocity_b[2]))
    V_b = ca.norm_2(velocity_b)
    alpha = ca.atan(velocity_b[2]/velocity_b[0])
    V_b = ca.if_else(V_b ==0, 1e-10,V_b)
    beta = ca.asin(velocity_b[1]/V_b)
    euler_n = lie.SO3EulerB321.elem(ca.vertcat(-beta, -alpha, 0)) # Euler elements for wind frame
    quat_bn = lie.SO3Quat.from_Euler(euler_n)

    quat_wb = lie.SO3Quat.elem(quat_wb)
    quat_bw = quat_wb.inverse()
    P = omega_wb_b[0]
    Q = omega_wb_b[1]
    R = omega_wb_b[2]

    velocity_w_w = quat_wb @ velocity_b #Velocity in Wind frame


    # force and moment
    qbar = 0.5 * rho * velocity_b[0]**2 # TODO velocity should be in wind frame
    # qbar = 0.5 * rho * velocity_w_w**2 # TODO velocity should be in wind frame

    # ground = ca.if_else(position_w[2]<0,
    #                     -position_w[2] * 150 - velocity_b[2] * 150,
    #                     0)
    ground = ca.if_else(position_w[2]<0,
        -position_w[2] * 50 *zAxis - velocity_b[2] * 50,
        ca.vertcat(0,0,0))
    D = cd * qbar * S 
    L = cl * qbar * S

    # Thrust Control
    fx_b = (thr_max*u[0]-velocity_b[0]) # Longitudinal Force assume thrust is directly on the x axis
    Fs = 0 #side force

    F_n = ca.vertcat(L, Fs, D) #force in wind frame (n)
    F_b = quat_bn @ F_n # Aerodynamic force from wind

    # F_b += (L - m *g + ground) * zAxis # Vertical Component TODO m*g is in body frame, L is in wind frame
    F_b = quat_bw @ ground #ground
    F_b += quat_bw @ (-m * g * zAxis) # gravity

    # ax_b = fx_b/m
    # az_b = F_b[2]/m

    # Moment
    M_b = ca.vertcat(0, 0, 0)
    moment_Cl = Cl_p * P  # rolling moment
    moment_Cm = Cm_q * Q  # pitching moment
    moment_Cn = Cn_r * R  # yawing moment
    Fi_b = fx_b * xAxis #thrust
    Mi_b = ca.vertcat(moment_Cl, moment_Cm, moment_Cn) * S # aerodynamic moment in body frame
    F_b += Fi_b
    M_b += Mi_b

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
