import casadi as ca


def derive_ekf_predict(f):
    """
    x1 = f(x, u, p, dt)
    """
    dt = ca.SX.sym("dt")

    n = f.numel_in(0)  # number of states
    m = f.numel_in(1)  # number of inputs
    l = f.numel_in(2)  # number of parameters

    Q = ca.SX.sym("Q", n, n)
    M = ca.SX.sym("M", m, m)

    x = ca.SX.sym("x", n)
    u = ca.SX.sym("u", m)

    x_0 = ca.SX.sym("x_0", n)
    u_0 = ca.SX.sym("u_0", m)
    p_0 = ca.SX.sym("p_0", l)
    P_0 = ca.SX.sym("P_0", n, n)

    F = ca.sparsify(ca.jacobian(f(x, u_0, p_0, dt), x))
    F = ca.substitute(F, x, x_0)

    B = ca.jacobian(f(x_0, u, p_0, dt), u)
    B = ca.substitute(B, u, u_0)

    P_1 = F @ P_0 @ F.T + B @ M @ B.T + Q
    x_1 = f(x_0, u_0, p_0, dt)
    return ca.Function(
        "predict",
        [x_0, u_0, p_0, P_0, Q, M, dt],
        [x_1, P_1],
        ["x_0", "u", "p", "P_0", "Q", "M", "dt"],
        ["x_1", "P_1"],
    )


def derive_ekf_correct(g, joseph=True):
    """
    y = g(x, u, p)
    """
    n = g.numel_in(0)  # number of states
    m = g.numel_in(1)  # number of inputs
    l = g.numel_in(2)  # number of parameters
    o = g.numel_out(0)  # number of outputs

    y = ca.SX.sym("y", o)
    R = ca.SX.sym("R", o, o)

    x = ca.SX.sym("x", n)
    u = ca.SX.sym("u", m)

    x_0 = ca.SX.sym("x_0", n)
    u_0 = ca.SX.sym("u_0", m)
    p = ca.SX.sym("p", l)
    P_0 = ca.SX.sym("P_0", n, n)

    H = ca.jacobian(g(x, u_0, p), x)
    H = ca.substitute(H, x, x_0)

    I = ca.SX.eye(n)
    S = H @ P_0 @ H.T + R
    S_I = ca.inv(S)
    K = P_0 @ H.T @ S_I

    # Non-Joseph Form
    if joseph:
        e1 = I - K @ H
        P_1 = e1 @ P_0 @ e1.T + K @ R @ K.T
    else:
        P_1 = (I - K @ H) @ P_0

    y_h = g(x_0, u_0, p)
    x_1 = x_0 + K @ (y - y_h)
    return ca.Function(
        "predict",
        [x_0, u_0, p, y, P_0, R],
        [x_1, P_1],
        ["x_0", "u_0", "p", "y", "P_0", "R"],
        ["x_1", "P_1"],
    )
