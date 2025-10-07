"""
Fixed-Wing Vehicle Dynamics for HH Sport Cub S2
4-channel input control: [Throttle, Aileron, Elevator, Rudder]
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cyecca.lie.group_so3 import SO3Quat, SO3EulerB321
import cyecca.lie as lie


# check steven lewis p184
def saturate(x, min_val, max_val):
    """
    A casadi function for saturation.
    """
    return ca.if_else(x < min_val, min_val, ca.if_else(x > max_val, max_val, x))


def derive_model():

    ##############################################################################################
    # Parameters

    # p, parameters
    thr_max = ca.SX.sym("thr_max")  # maximum thrust
    m = ca.SX.sym("m")  # mass
    XCG = ca.SX.sym("XCG")  # Center of gravity on longitudinal plane
    XAC = ca.SX.sym("XAC")  # Aerodynamic Center on longitudinal plane
    S = ca.SX.sym("S")  # Wing Surface area
    rho = ca.SX.sym("rho")  # Air Density
    g = ca.SX.sym("g")  # Gravitational Acceleration (m/s^2)
    Jx = ca.SX.sym("Jx")  # Moment of Inertia in x direction
    Jy = ca.SX.sym("Jy")  # Moment of Inertia in y direction
    Jz = ca.SX.sym("Jz")  # Moment of Inertia in z direction
    Jxz = ca.SX.sym("Jxz")  # Product of Inertia x and z

    J = ca.SX.zeros(3, 3)
    J[0, 0] = Jx
    J[1, 1] = Jy
    J[2, 2] = Jz
    J[0, 2] = Jxz
    J[2, 0] = Jxz

    cbar = ca.SX.sym("cbar")  # mean chord (m)
    span = ca.SX.sym("span")  # wing span (m)

    # Control Moment
    Cm0 = ca.SX.sym("Cm0")  # Coefficient of Moment
    Clda = ca.SX.sym("Clda")  # roll moment coefficient based on aileron
    Cldr = ca.SX.sym("Cldr")  # roll moment coefficient based on rudder
    Cmde = ca.SX.sym("Cmde")  # pitch moment coefficient based on elevator
    Cndr = ca.SX.sym("Cndr")  # yaw moment coefficient based on rudder
    Cnda = ca.SX.sym("Cnda")  # yaw moment coefficient based on aileron
    CYda = ca.SX.sym("CYda")  # Sideforce due to aileron
    CYdr = ca.SX.sym("CYdr")  # Sideforce due to rudder

    # Longitudinal Stability Coefficients
    CL0 = ca.SX.sym("CL0")  # Coefficient of lift at zero alpha
    CLa = ca.SX.sym("CLa")  # Cl_alpha per alpha
    Cma = ca.SX.sym("Cma")  # Coefficient of Moment due to angle of attack
    Cmq = ca.SX.sym("Cmq")  # Pitch Damping Derivative Coefficient
    CD0 = ca.SX.sym("CD0")  # drag coefficient
    CDCLS = ca.SX.sym("CDCLS")  # Lift-induced drag coefficient

    # Lateral-Directional Coefficient
    Cnb = ca.SX.sym("Cnb")  # Cn_beta for yaw stiffness
    Clp = ca.SX.sym("Clp")  # Roll Damping Derivative Coefficient
    Cnr = ca.SX.sym(
        "Cnr"
    )  # Yaw Damping Derivative Coefficient wrt to yaw rate (magnitude)
    Cnp = ca.SX.sym("Cnp")  # Yaw Damping Derivative Coefficient wrt to roll rate
    Clr = ca.SX.sym("Clr")  # Roll Damping Derivative Coefficient wrt to yaw rate
    CYb = ca.SX.sym("CYb")  # Sideforce due to sideslip
    CYr = ca.SX.sym("CYr")  # Sideforce due to yaw rate
    CYp = ca.SX.sym("CYp")  # Side force due to roll rate

    p = ca.vertcat(
        thr_max,
        m,
        XCG,
        XAC,
        S,
        rho,
        g,
        Jx,
        Jy,
        Jz,
        Jxz,
        cbar,
        span,
        Cm0,
        Clda,
        Cldr,
        Cmde,
        Cndr,
        Cnda,
        CYda,
        CYdr,
        CL0,
        CLa,
        Cma,
        Cmq,
        CD0,
        CDCLS,
        Cnb,
        Clp,
        Cnr,
        Cnp,
        Clr,
        CYb,
        CYr,
        CYp,
    )

    p_defaults = {
        "thr_max": 0.56,  # 0.38 # Maximum thrust (N)
        "m": 0.057,  # Mass (kg)
        "XCG": 0.25,  # Center of gravity, nondimensional along-chord
        "XAC": 0.25,  # Aerodynamic Center
        "S": 0.05553,  # Wing surface area (m^2)
        "rho": 1.225,  # Air density at sea level (kg/m^3)
        "g": 9.81,  # Gravitational acceleration (m/s^2)
        # Moments of Inertia
        "Jx": 2.0e-4,  # Roll moment of inertia (kg·m²)
        "Jy": 2.6e-4,  # Pitch moment of inertia (kg·m²)
        "Jz": 3.2e-4,  # Yaw moment of inertia (kg·m²)
        "Jxz": 0.0e-4,  # Product of inertia (kg·m²)
        "cbar": 0.09,  # Mean aerodynamic chord (m)
        "span": 0.617,  # Wingspan (m)
        # Control Effectiveness (Converted to radians)
        "Cm0": 0.0314,  # Zero-lift pitching moment coefficient
        "Clda": 0.10,  # 0.16,  # Aileron control effectiveness in roll(per rad)
        "Cldr": 0.05,  # Rudder Control effectiveness in roll
        "Cmde": 0.9,  # Elevator control effectiveness in pitch(per rad)
        "Cndr": 0.12,  # Rudder control effectiveness in yaw (per rad)
        "Cnda": 0.03,  # Aileron control effectiveness in yaw (per rad)
        "CYda": 0.02,  # Sideforce due to aileron defelction (per rad)
        "CYdr": -0.12,  # Side force due to rudder deflection (per rad)
        # Longitudinal Stability
        "CL0": 0.20,  # Adjusted lift coefficient at zero AoA
        "CLa": 5.2,  # Lift slope (per rad)
        "Cma": -0.60,  # Pitching moment due to AoA (per rad) (equilibrium at AoA = ~3.0deg)
        "Cmq": -18.0,  # Pitch damping (per rad/s)
        "CD0": 0.09,  # Parasitic drag coefficient
        "CDCLS": 0.062,  # Lift-induced drag coefficient
        # Lateral-Directional Stability
        "Cnb": 0.10,  # Yaw stiffness (per rad)
        "Clp": -1.30,  # Roll damping per rad/s
        "Cnr": -0.12,  # Yaw damping per rad/s
        "Cnp": -0.10,  # Yaw damping due to roll rate
        "Clr": 0.10,  # Roll damping due to yaw rate
        "CYb": -0.65,  # Sideforce due to sideslip (per rad)
        "CYr": 0.25,  # Sideforce due to yaw rate (per rad/s)
        "CYp": 0.15,  # Side force due to roll rate (per rad/s)
    }

    ##############################################################################################
    # Constants

    DEG2RAD = ca.pi / 180
    max_defl_ail = 30  # maximum aileron deflection in deg
    max_defl_elev = 24  # maximum elevator deflection in deg
    max_defl_rud = 20  # maximum rudder deflection in deg
    alpha_stall = 20 * DEG2RAD  # stall angle of attack in rad
    tol_v = 1e-3  # 1e-1 # Aerodynamic Tolerance for Velocity
    xAxis = ca.vertcat(1, 0, 0)
    yAxis = ca.vertcat(0, 1, 0)
    zAxis = ca.vertcat(0, 0, 1)

    ##############################################################################################
    # States

    # frames
    # w: world frame (NED, local tangent plane, assumed inertial)
    # b: body frame (fixed to vehicle)
    # n: wind frame (aligned with relative wind, x axis aligned with velocity)

    # states
    position_w = ca.SX.sym("position_w", 3)  # position in world frame
    velocity_b = ca.SX.sym("velocity_b", 3)  # velocity in body frame
    quat_wb = ca.SX.sym("quat_wb", 4)  # Quaternion world - body frame
    omega_wb_b = ca.SX.sym("omega_wb_b", 3)  # world-body

    x = ca.vertcat(
        position_w,
        velocity_b,
        quat_wb,
        omega_wb_b,
    )
    x0_defaults = {
        "position_w_0": 20.0,
        "position_w_1": -5.0,
        "position_w_2": 0.0,
        "velocity_b_0": 0.0,
        "velocity_b_1": 0.0,
        "velocity_b_2": 0.0,
        "quat_wb_0": 0.0,
        "quat_wb_1": 0.09,
        "quat_wb_2": 0.0,
        "quat_wb_3": 1.0,
        "omega_wb_b_0": 0.0,
        "omega_wb_b_1": 0.0,
        "omega_wb_b_2": 0.0,
    }

    ##############################################################################################
    # Input
    throttle_cmd = ca.SX.sym("throttle_cmd")
    ail_cmd = ca.SX.sym("ail_cmd")
    elev_cmd = ca.SX.sym("elev_cmd")
    rud_cmd = ca.SX.sym("rud_cmd")
    u = ca.vertcat(throttle_cmd, ail_cmd, elev_cmd, rud_cmd)

    ##############################################################################################
    # Velocities and Aerodynamic Angles
    V_b = ca.norm_2(velocity_b)  # true airspeed
    V_b = ca.if_else(ca.fabs(V_b) < tol_v, tol_v, V_b)  # sideslip angle
    alpha = ca.atan2(-velocity_b[2], velocity_b[0])  # normalized velocity componenet
    beta = ca.asin(velocity_b[1] / V_b)  # sideslip angle

    # rotation from body to wind frame
    euler_bn = lie.SO3EulerB321.elem(
        ca.vertcat(beta, -alpha, 0.0)
    )  # Euler elements for wind frame
    euler_nb = lie.SO3EulerB321.elem(ca.vertcat(-beta, alpha, 0.0))
    q_nb = lie.SO3Quat.from_Euler(euler_nb)
    q_bn = q_nb.inverse()

    # rotation from world to body frame
    # note quat_wb is components of quaternion, while
    # q_wb is a SO3Quat element
    q_wb = lie.SO3Quat.elem(quat_wb)
    q_bw = q_wb.inverse()

    # Euler elements for body frame
    P = omega_wb_b[0]
    Q = omega_wb_b[1]
    R = omega_wb_b[2]

    velocity_w = q_wb @ velocity_b  # velocity in world frame
    velocity_n = q_nb @ velocity_b  # velocity in wind frame

    ##############################################################################################
    # Control Surface Defelction

    ail_rad = max_defl_ail * DEG2RAD * u[1] * 1  # mapped for HH Sport Cub 2
    elev_rad = max_defl_elev * DEG2RAD * u[2] * 1  # mapped for HH Sport Cub 2
    rud_rad = max_defl_rud * DEG2RAD * u[3] * -1  # mapped for HH Sport Cub 2

    ##############################################################################################
    # Aerodynamic Forces and Moments

    CL = CL0 + CLa * alpha  # Lift Coefficient
    CL = ca.if_else(
        ca.fabs(alpha) < alpha_stall, CL, 0
    )  # Stall Model --> set CL to CL0 after stall

    CD = CD0 + CDCLS * CL * CL  # Drag Polar (must be positive)

    # (Steven pg 91 eqn 2.3-17a) and (Steven Pg 79 eqn 2.3-8b)
    CC = (
        -CYb * beta
        + CYda * ail_rad / (max_defl_ail * DEG2RAD)
        + CYdr * rud_rad / (max_defl_rud * DEG2RAD)
    )  # Crosswind Force Coefficient
    CC += (
        CYp * span / (2 * V_b) * P
    )  # Sideforce due to roll rate  #(Steven pg 91 eqn 2.3-17a)
    CC += (
        CYr * span / (2 * V_b) * R
    )  # Sideforce due to yaw rate #(Steven pg 91 eqn 2.3-17a)

    ### Using Equation to calculate rotational moment coefficent
    Cl = Clda * ail_rad + (-1) * Cldr * rud_rad  # roll moment coefficient
    Cm = (
        Cm0 + Cma * alpha + Cmde * elev_rad + (XAC - XCG) * CL
    )  # pitch moment coefficient
    Cn = Cnb * beta + Cndr * rud_rad + (-1) * Cnda * ail_rad  # yaw moment coefficient

    # Aerodynamic Forces and Moments
    qbar = 0.5 * rho * V_b * V_b  # calculated using airspeed
    D = CD * qbar * S  # Drag force -- wind frame
    L = CL * qbar * S  # Lift force -- wind frame
    Fs = CC * qbar * S  # Crosswind side Force (Steven pg 80, eqn. 2.3-8b)

    # Forces in body frame
    v_hat = (
        velocity_b / V_b
    )  # defining drag this way based on velocity is more numerically stable
    D_b = -D * v_hat  # Drag opposite to velocity direction
    L_b = q_bn @ (L * zAxis)
    C_b = q_bn @ (Fs * yAxis)
    FA_b = D_b + L_b + C_b

    # Aerodynamic Moments (Steven pg.79 eqn 2.3-8b)
    MA_b = ca.vertcat(qbar * S * span * Cl, qbar * S * cbar * Cm, qbar * S * span * Cn)

    # (Steven pg 81 eqn 2.3-9a)
    # Aerodynamic Damping Moment
    MA_b += (Clp * span / (2 * V_b) * P) * xAxis  # Roll Damping
    MA_b += (Clr * span / (2 * V_b) * R) * xAxis  # Roll damping due to yaw rate
    MA_b += (Cmq * cbar / (2 * V_b) * Q) * yAxis  # Pitch Damping
    MA_b += (Cnp * span / (2 * V_b) * P) * zAxis  # # Yaw damping due to roll rate
    MA_b += (Cnr * span / (2 * V_b) * R) * zAxis  # Yaw Damping

    ###############################################################################################
    # Ground Forces

    # Wheel position wrt center of gravity
    left_wheel_b = ca.SX([0.1, 0.1, -0.1])
    right_wheel_b = ca.SX([0.1, -0.1, -0.1])
    tail_wheel_b = ca.SX([-0.4, 0.0, 0.0])

    wheel_b_list = [left_wheel_b, right_wheel_b, tail_wheel_b]
    wheel_w_list = [q_wb @ pos for pos in wheel_b_list]
    ground_force_w_list = []
    MG_b = ca.SX.zeros(3)  # ground moment in body frame

    for wheel_w, wheel_b in zip(wheel_w_list, wheel_b_list):
        pos_wheel_w = position_w + wheel_w
        vel_wheel_b = velocity_b + ca.cross(omega_wb_b, wheel_b)
        vel_wheel_w = q_wb @ vel_wheel_b
        force_w = ca.if_else(
            pos_wheel_w[2] < 0.0,
            saturate(-(pos_wheel_w[2]) * 10, -100, 100) * zAxis
            - vel_wheel_w[2] * 01.10 * zAxis  # Vertical Component Damping
            - vel_wheel_w[0] * 0.001 * xAxis  # ground damping
            - vel_wheel_w[1] * 0.001 * yAxis,  # ground damping
            ca.vertcat(0, 0, 0),  # Airborne (no ground force effect)
        )
        force_b = q_bw @ force_w
        ground_force_w_list.append(force_w)
        MG_b += ca.cross(wheel_b, force_b)

    FG_b = q_bw @ ca.sum2(ca.horzcat(*ground_force_w_list))  # Ground force

    ###############################################################################################
    # Thrust Force
    throttle = ca.if_else(u[0] > 1e-3, u[0], 1e-3)
    FT_b = (
        thr_max * throttle
    ) * xAxis  # Longitudinal Force assume thrust is directly on the x axis
    MT_b = ca.SX.zeros(3)  # No thrust moment

    ###############################################################################################
    # Gravity Force
    FW_b = q_bw @ (-m * g * zAxis)  # Gravitational Force components on body frame
    MW_b = ca.SX.zeros(3)  # No gravitational moment

    ###############################################################################################
    # Sum of forces in Body Frame
    F_b = FA_b + FG_b + FT_b + FW_b
    M_b = MA_b + MG_b + MT_b + MW_b

    ##############################################################################################
    # Kinematics

    derivative_position_w = q_wb @ velocity_b
    derivative_velocity_b = F_b / m - ca.cross(omega_wb_b, velocity_b)  # 2.5-9
    derivative_quaternion_wb = q_wb.right_jacobian() @ omega_wb_b
    derivative_omega_wb_b = ca.inv(J) @ (M_b - ca.cross(omega_wb_b, J @ omega_wb_b))

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

    Info = ca.Function(
        "Info",
        [x, u, p],
        [
            L_b,
            D_b,
            C_b,
            FW_b,
            FT_b,
            velocity_b,
            velocity_n,
            CL,
            CD,
            alpha,
            qbar,
            beta,
            ail_rad,
            elev_rad,
            rud_rad,
            Cndr,
        ],
        ["x", "u", "p"],
        [
            "L_b",
            "D_b",
            "C_b",
            "FW_b",
            "FT_b",
            "v_b",
            "v_n",
            "CL",
            "CD",
            "alpha",
            "qbar",
            "beta",
            "ail",
            "elev",
            "rud",
            "Cndr",
        ],
    )

    f = ca.Function("f", [x, u, p], [xdot], ["x", "u", "p"], ["xdot"])

    dae = {"x": x, "ode": f(x, u, p), "p": p, "u": u, "z": z, "alg": alg}  # set up dae

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
