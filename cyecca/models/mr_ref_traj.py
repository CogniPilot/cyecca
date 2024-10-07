import casadi as ca
import cyecca.lie as lie


def derive_mr_ref_traj():
    # based on:
    # Minimum snap trajectory generation and control for quadrotors
    # see: https://ieeexplore.ieee.org/abstract/document/5980409
    
    # Symbols and Parameters

    tol = 1e-6  # tolerance for singularities

    # flat output (input variables from trajectory planner)
    p_w = ca.SX.sym("p_w", 3)  # position
    v_w = ca.SX.sym("v_w", 3)  # velocity
    a_w = ca.SX.sym("a_w", 3)  # accel
    j_w = ca.SX.sym("j_w", 3)  # jerk
    s_w = ca.SX.sym("s_w", 3)  # snap

    psi = ca.SX.sym("psi")  # desired heading direction
    psi_dot = ca.SX.sym("psi_dot")  # derivative of desired heading
    psi_ddot = ca.SX.sym("psi_ddot")  # second derivative of desired heading

    # constants
    m = ca.SX.sym("m")  # mass
    g = ca.SX.sym("g")  # accel of gravity

    # unit vectors
    xh = ca.SX([1, 0, 0])
    yh = ca.SX([0, 1, 0])
    zh = ca.SX([0, 0, 1])

    # Rotational Moment of Inertia
    J_x = ca.SX.sym("J_x")
    J_y = ca.SX.sym("J_y")
    J_z = ca.SX.sym("J_z")
    J_xz = ca.SX.sym("J_xz")

    # Solve for C_be

    # acceleration
    thrust_w = m * (g * zh - a_w)  # grav accel + desired accel

    T = ca.norm_2(thrust_w)  # magnitude of thrust vector
    T = ca.if_else(T > tol, T, tol)  # can have singularity when T = 0, this prevents it

    zb_w = thrust_w / T  # direction of thrust vector is body z expressed in world frame

    # desired heading direction
    xc_w = ca.cos(psi) * xh + ca.sin(psi) * yh  # unit vector in direction of camera vector

    yb_w = ca.cross(zb_w, xc_w)  # use cross product to find orthogonal direction (yb)
    N_yb_w = ca.norm_2(yb_w)  # zb_w and xc_w are not orthogonal, so won't be unit vector yet
    yb_w = ca.if_else(  # make yb_w into a unit vector
        N_yb_w > tol, yb_w / N_yb_w, yh
    )  # normalize y_b, can have singularity when z_b and x_c aligned
    xb_w = ca.cross(yb_w, zb_w)  # now using yb and zb, can find xb using cross product

    # now we can construct a direction cosine matrix from orthogonal unit vectors
    C_bw = lie.SO3Dcm.from_Matrix(ca.hcat([xb_w, yb_w, zb_w]))
    C_wb = C_bw.inverse()  # is just transpose for SO(3)

    # now we can find Body 321 euler angles (psi, theta, phi) from DCM
    euler = lie.SO3EulerB321.from_Dcm(C_wb)
    psi = euler.param[0]
    theta = euler.param[1]
    phi = euler.param[2]

    # Solve for omega_eb_b

    # note h_omega z_b component can be ignored with dot product below
    # original paper subtracted z component then took dot product with
    # x and y component
    t2_w = m / T * j_w
    p = -ca.dot(t2_w, yb_w)
    q = ca.dot(t2_w, xb_w)

    omega_wc_w = psi_dot * zh
    r = ca.dot(omega_wc_w, zb_w)

    omega_wb_b = p * xh +  q * yh + r * zh

    # Solve for omega_dot_eb_b

    omega_wb_b_cross_zh = ca.cross(omega_wb_b, zh)

    T_dot = -ca.dot(m * j_w, zb_w)
    coriolis_b = 2 * T_dot / T * omega_wb_b_cross_zh
    centrip_b = ca.cross(omega_wb_b, omega_wb_b_cross_zh)

    q_dot = -m / T * ca.dot(s_w, xb_w) - ca.dot(coriolis_b, xh) - ca.dot(centrip_b, xh)
    p_dot = m / T * ca.dot(s_w, yb_w) + ca.dot(coriolis_b, yh) + ca.dot(centrip_b, yh)

    omega_wb_w = C_bw @ omega_wb_b
    omega_wc_w = psi_dot * zh

    theta_dot = (q - ca.sin(phi) * ca.cos(theta) * psi_dot) / ca.cos(phi)
    phi_dot = p + ca.sin(theta) * psi_dot

    zc_w = zh  # c frame rotates about ze so zc_c = zc_e = zh
    yc_w = ca.cross(zc_w, xc_w)
    T1 = ca.inv(ca.horzcat(xb_w, yc_w, zh))
    A = T1 @ C_bw
    b = -T1 @ (
        ca.cross(omega_wb_w, phi_dot * xb_w) + ca.cross(omega_wc_w, theta_dot * yc_w)
    )
    r_dot = (psi_ddot - A[2, 0] * p_dot - A[2, 1] * q_dot - b[2]) / A[2, 2]

    omega_dot_wb_b = p_dot * xh + q_dot * yh + r_dot * zh

    # Solve for Inputs

    J = ca.SX(3, 3)
    J[0, 0] = J_x
    J[1, 1] = J_y
    J[2, 2] = J_z
    J[0, 2] = J[2, 0] = J_xz

    M_b = J @ omega_dot_wb_b + ca.cross(omega_wb_b, J @ omega_wb_b)

    q_wb = lie.SO3Quat.from_Matrix(C_wb)

    # Code Generation

    v_b = C_bw @ v_w
    f_ref = ca.Function(
        "mr_ref_traj",
        [psi, psi_dot, psi_ddot, v_w, a_w, j_w, s_w, m, g, J_x, J_y, J_z, J_xz],
        [v_b, q_wb.param, omega_wb_b, omega_dot_wb_b, M_b, T],
        [
            "psi",
            "psi_dot",
            "psi_ddot",
            "v_w",
            "a_w",
            "j_w",
            "s_w",
            "m",
            "g",
            "J_x",
            "J_y",
            "J_z",
            "J_xz",
        ],
        ["v_b", "q_wb", "omega_wb_b", "omega_dot_wb_b", "M_b", "T"],
    )

    return {"mr_ref_traj": f_ref}
