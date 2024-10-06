import casadi as ca


def derive_mr_ref_traj():
    # Symbols and Parameters

    tol = 1e-6  # tolerance for singularities

    # flat output (input variables from trajectory planner)
    p_e = ca.SX.sym("p_e", 3)  # position
    v_e = ca.SX.sym("v_e", 3)  # velocity
    a_e = ca.SX.sym("a_e", 3)  # accel
    j_e = ca.SX.sym("j_e", 3)  # jerk
    s_e = ca.SX.sym("s_e", 3)  # snap

    psi = ca.SX.sym("psi")  # desired heading direction
    psi_dot = ca.SX.sym("psi_dot")  # derivative of desired heading
    psi_ddot = ca.SX.sym("psi_ddot")  # second derivative of desired heading

    # constants
    m = ca.SX.sym("m")  # mass
    g = ca.SX.sym("g")  # accel of gravity

    # unit vectors xh = xb_b = xe_e etc.
    xh = ca.SX([1, 0, 0])
    yh = ca.SX([0, 1, 0])
    zh = ca.SX([0, 0, 1])

    # Rotational Moment of Inertia
    J_xx = ca.SX.sym("J_x")
    J_yy = ca.SX.sym("J_y")
    J_zz = ca.SX.sym("J_z")
    J_xz = ca.SX.sym("J_xz")

    # Solve for C_be

    # acceleration
    thrust_e = m * (g * zh - a_e)

    T = ca.norm_2(thrust_e)
    T = ca.if_else(T > tol, T, tol)  # can have singularity when T = 0, this prevents it

    zb_e = thrust_e / T

    # desired heading direction
    xc_e = ca.cos(psi) * xh + ca.sin(psi) * yh

    yb_e = ca.cross(zb_e, xc_e)
    N_yb_e = ca.norm_2(yb_e)
    yb_e = ca.if_else(
        N_yb_e > tol, yb_e / N_yb_e, yh
    )  # normalize y_b, can have singularity when z_b and x_c aligned
    xb_e = ca.cross(yb_e, zb_e)

    # T_dot = ca.dot(m*s_e, zb_e)
    C_be = ca.hcat([xb_e, yb_e, zb_e])

    # Solve for omega_eb_b

    # note h_omega z_b component can be ignored with dot product below
    t2_e = m / T * j_e
    p = ca.dot(t2_e, yb_e)
    q = -ca.dot(t2_e, xb_e)

    C_eb = C_be.T

    # solve for euler angles based on DCM
    theta = ca.asin(-C_eb[2, 0])  # check if transpose
    phi = ca.if_else(
        ca.fabs(ca.fabs(theta) - ca.pi / 2) < tol, 0, ca.atan2(C_eb[2, 1], C_eb[2, 2])
    )

    # solve for r
    cos_phi = ca.cos(phi)
    cos_phi = ca.if_else(ca.fabs(cos_phi) > tol, cos_phi, 0)
    r = (
        -q * ca.tan(phi) + ca.cos(theta) * psi_dot / cos_phi
    )  # from R_solve below, singularity at phi=pi

    T_dot = -ca.dot(m * j_e, zb_e)

    # Mellinger approach
    # yc_e = ca.cross(xc_e, zh)
    # R_sol = ca.inv(ca.horzcat(xb_e, yc_e, zh))@C_be
    # R_sol[2, 0]*p + R_sol[2, 1]*q + R_sol[2, 0]*r = psi_dot
    # r2 = (psi_dot - R_sol[2, 0]*p + R_sol[2, 1]*q)/R_sol[2, 0]

    omega_eb_b = p * xh + q * yh + r * zh

    # Solve for omega_dot_eb_b

    omega_eb_b_cross_zh = ca.cross(omega_eb_b, zh)

    coriolis_b = 2 * T_dot / T * omega_eb_b_cross_zh
    centrip_b = ca.cross(omega_eb_b, omega_eb_b_cross_zh)

    q_dot = -m / T * ca.dot(s_e, xb_e) - ca.dot(coriolis_b, xh) - ca.dot(centrip_b, xh)
    p_dot = m / T * ca.dot(s_e, yb_e) + ca.dot(coriolis_b, yh) + ca.dot(centrip_b, yh)

    omega_eb_e = C_be @ omega_eb_b
    omega_ec_e = psi_dot * zh

    theta_dot = (q - ca.sin(phi) * ca.cos(theta) * psi_dot) / ca.cos(phi)
    phi_dot = p + ca.sin(theta) * psi_dot

    zc_e = zh  # c frame rotates about ze so zc_c = zc_e = zh
    yc_e = ca.cross(zc_e, xc_e)
    T1 = ca.inv(ca.horzcat(xb_e, yc_e, zh))
    A = T1 @ C_be
    b = -T1 @ (
        ca.cross(omega_eb_e, phi_dot * xb_e) + ca.cross(omega_ec_e, theta_dot * yc_e)
    )
    r_dot = (psi_ddot - A[2, 0] * p_dot - A[2, 1] * q_dot - b[2]) / A[2, 2]

    omega_dot_eb_b = p_dot * xh + q_dot * yh + r_dot * zh

    # Solve for Inputs

    J = ca.SX(3, 3)
    J[0, 0] = J_xx
    J[1, 1] = J_yy
    J[2, 2] = J_zz
    J[0, 2] = J[2, 0] = J_xz

    M_b = J @ omega_dot_eb_b + ca.cross(omega_eb_b, J @ omega_eb_b)

    # Code Generation

    v_b = C_eb @ v_e
    f_ref = ca.Function(
        "mr_ref_traj",
        [psi, psi_dot, psi_ddot, v_e, a_e, j_e, s_e, m, g, J_xx, J_yy, J_zz, J_xz],
        [v_b, C_be, omega_eb_b, omega_dot_eb_b, M_b, T],
        [
            "psi",
            "psi_dot",
            "psi_ddot",
            "v_e",
            "a_e",
            "j_e",
            "s_e",
            "m",
            "g",
            "J_xx",
            "J_yy",
            "J_zz",
            "J_xz",
        ],
        ["v_b", "C_be", "omega_eb_b", "omega_dot_eb_b", "M_b", "T"],
    )

    return {"mr_ref_traj": f_ref}
