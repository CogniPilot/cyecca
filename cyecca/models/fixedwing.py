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
    # cl = ca.SX.sym("cl")
    cd = ca.SX.sym("cd")
    S = ca.SX.sym("S")
    rho = ca.SX.sym("rho")
    g = ca.SX.sym("g")
    Jx = ca.SX.sym("Jx")
    Jy = ca.SX.sym("Jy")
    Jz = ca.SX.sym("Jz")
    J = ca.diag(ca.vertcat(Jx, Jy, Jz))
    Cm_p = ca.SX.sym("Cm_p") #moment coefficient for roll
    Cm_q = ca.SX.sym("Cm_q") #moment coefficient for pitch
    Cm_r = ca.SX.sym("Cm_r") #moment coefficent for yaw
    cbar = ca.SX.sym("cbar") #mean chord (m)
    span = ca.SX.sym("span") #wing span (m)
    cla = ca.SX.sym("cla") # Cl_alpha per alpha
    cl0 = ca.SX.sym("cl0")
    p= ca.vertcat(
        thr_max,
        m,
        # cl,
        cd,
        S,
        rho,
        g,
        Jx,
        Jy,
        Jz,
        Cm_p,
        Cm_q,
        Cm_r,
        cbar,
        span,
        cla,
        cl0
    )
    p_defaults = {
        "thr_max" : 4.50,
        "m" : 0.5,
        # "cl": 6.28,
        "cd" : 0.3, #0.1,
        "S": 1.0,
        "rho": 1.225,
        "g": 9.8,
        'Jx': 0.02166666666666667,
        'Jy': 0.02166666666666667,
        "Jz" : 0.02166666666666667,
        "Cm_p": 0.1,
        "Cm_q": 0.3,
        "Cm_r": 0.1,
        "cbar" : 0.075,
        "span" : 0.30,
        "cla" : 6.28,
        "cl0" : 2.25,
    }

    # states
    position_w = ca.SX.sym("position_w", 3) # w = world frame
    velocity_b = ca.SX.sym("velocity_b", 3)
    quat_wb = ca.SX.sym("quat_wb", 4) # Quaternion world - body frame
    omega_wb_b = ca.SX.sym("omega_wb_b",3) # world-body

    x = ca.vertcat(
        position_w,
        velocity_b,
        quat_wb,
        omega_wb_b,
        )
    x0_defaults = {
        "position_w_0" : 0.0,
        "position_w_1" : 0.0,
        "position_w_2" : 0.0,
        "velocity_b_0" : 1e-5,
        "velocity_b_1" : 1e-5,
        "velocity_b_2" : 1e-5,
        "quat_wb_0" : 1.0,
        "quat_wb_1" : 0.0,
        "quat_wb_2" : 0.0,
        "quat_wb_3" : 0.0,
        "omega_wb_b_0" : 0.0,
        "omega_wb_b_1" : 0.0,
        "omega_wb_b_2" : 0.0,
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
    # frac_alp = ca.if_else(ca.fabs(velocity_b[2]/velocity_b[0])> 0.1,  (velocity_b[2]/velocity_b[0]), ca.SX(0))
    # alpha = ca.if_else(ca.fabs(velocity_b[0]) > 1e-4, ca.atan(frac_alp), ca.SX(0))

    alpha = ca.if_else(ca.fabs(velocity_b[0]) > 1e-3, ca.atan(velocity_b[2]/velocity_b[0]), ca.SX(0))
    vxvb = ca.if_else(ca.fabs(velocity_b[1]/V_b) < 1, (velocity_b[1]/V_b), ca.sign(velocity_b[1]/V_b))
    beta = ca.asin(vxvb)
    euler_n = lie.SO3EulerB321.elem(ca.vertcat(beta, -alpha, 0.0)) # Euler elements for wind frame
    quat_bn = lie.SO3Quat.from_Euler(euler_n)

    quat_wb = lie.SO3Quat.elem(quat_wb)
    quat_bw = quat_wb.inverse()
    P = omega_wb_b[0]
    Q = omega_wb_b[1]
    R = omega_wb_b[2]

    velocity_w_w = quat_wb @ velocity_b #Velocity in Wind frame

    ##############################################################################################
    # Force and Moment Model
    cl = cl0 + cla*(-1*alpha) # Lift Coefficient
    
    # qbar = 0.5 * rho * velocity_b[0]**2  # qbar in terms of body vel_x
    qbar = 0.5 * rho * ca.norm_2(velocity_w_w)**2 # TODO Recheck wind frame velocity --> qbar in terms of wind velocity

    ground = ca.if_else(position_w[2]< 0.0,
        -position_w[2] * 1500 *zAxis - velocity_b * 1,
        ca.vertcat(0,0,0))
    
    D = cd * qbar * S # Drag force -- wind frame
    L = cl * qbar * S # Lift force -- wind frame
    Fs = -4*velocity_b[1] #crosswind force to counteract slip


    F_b = ca.vertcat(0,0,0)
    F_n = ca.vertcat(-D, Fs, L) #force in wind frame (n)
    F_b += quat_bn @ ca.SX(F_n) # Aerodynamic force from wind in body frame
    F_b += quat_bw @ ground # TODO current method does not consider wheel friction on ground
    F_b += quat_bw @ (-m * g * zAxis) # gravity converted to body frame

    throttle = ca.if_else(u[0] > 1e-6, u[0], 1e-6)
    fx_b = (thr_max*throttle-velocity_b[0])*xAxis # Longitudinal Force assume thrust is directly on the x axis
    F_b += fx_b #force due to thrust

    # Moment
    M_b = ca.vertcat(0, 0, 0)

    # Direct Command Rotation
    # M_b += (u[1]-omega_wb_b[0]) * Cm_p *xAxis #moment due to roll angle command
    # M_b += (u[2]-omega_wb_b[1]) * 0.1 *yAxis #moment due to pitch angle command
    # M_b += (u[3]-omega_wb_b[2]) * 0.1 *zAxis #moment due to yaw angle command

    # Rotation by aerodynamic moment
    M_b += Cm_p * qbar * S * span *(u[1]) * xAxis- 0.1*omega_wb_b*xAxis#roll moment due to aileron
    M_b += Cm_q * qbar * S * cbar * (u[2]) * yAxis - 0.1*omega_wb_b*yAxis #pitch moment due to elevator deflection
    M_b += Cm_r * qbar * S * span *(u[3]) * zAxis - 0.05*omega_wb_b*zAxis #yaw moment due to rudder
    Cm_yr = 0.01 # Moment coefficent for yaw due to aileron rolling
    M_b += Cm_yr * qbar * S * span *(-u[1]-omega_wb_b[2]) * zAxis #yaw moment due to aileron
    # Cm_thr = 0.001 # Moment coefficient for pitch due to throttle
    # M_b += -1 *(Cm_thr * u[0] - omega_wb_b[0]) * yAxis # moment due to throttle


    ##############################################################################################
    
    # # kinematics
    derivative_omega_wb_b = ca.inv(J) @ (M_b - ca.cross(omega_wb_b, J @ omega_wb_b))
    derivative_quaternion_wb = quat_wb.right_jacobian() @ omega_wb_b
    derivative_position_w = quat_wb @ velocity_b
    derivative_velocity_b = F_b / m - ca.cross(omega_wb_b, (velocity_b- F_b/m))

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

    dae = {"x": x, "ode": f(x, u, p), "p": p, "u": u, "z": z, "alg": alg} # set up dae

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
