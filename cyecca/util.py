from beartype import beartype
from beartype.typing import Tuple, Union, Callable


import casadi as ca


@beartype
def rk4(f: Callable, t: Union[ca.SX, float, int, ca.DM], y: ca.SX, h: ca.SX) -> ca.SX:
    """Runge Kuta 4th order integrator"""
    k1 = h * f(t, y)
    k2 = h * f(t + h / 2, y + k1 / 2)
    k3 = h * f(t + h / 2, y + k2 / 2)
    k4 = h * f(t + h, y + k3)
    return ca.simplify(y + (k1 + 2 * k2 + 2 * k3 + k4) / 6)


@beartype
def sqrt_covariance_predict(W: ca.SX, F: ca.SX, Q: ca.SX) -> ca.SX:
    """
    Finds a sqrt factorization of the continuous time covariance
    propagation equations. Requires solving a linear system of equations
    to keep the sqrt lower triangular.

    'A Square Root Formulation of the Kalman Covariance Equations', Andrews 68

    https://arc.aiaa.org/doi/10.2514/3.4696

    W: sqrt P, symbolic, with sparsity lower triangulr
    F: dynamics matrix
    Q: process noise matrix

    returns:
    W_dot_sol: sqrt of P deriative, lower triangular
    """
    n_x = F.shape[0]
    XL = ca.SX.sym("X", ca.Sparsity_lower(n_x))
    X = XL - XL.T
    for i in range(n_x):
        X[i, i] = 0
    W_dot = ca.mtimes(F, W) + ca.mtimes(Q / 2 + X, ca.inv(W).T)

    # solve for XI that keeps W dot lower triangular
    y = ca.vertcat(*ca.triu(W_dot, False).nonzeros())
    x_dep = []
    for i, xi in enumerate(XL.nonzeros()):
        if ca.depends_on(y, xi):
            x_dep += [xi]
    x_dep = ca.vertcat(*x_dep)
    A = ca.jacobian(y, x_dep)
    for i, xi in enumerate(XL.nonzeros()):
        assert not ca.depends_on(A, xi)
    b = -ca.substitute(y, x_dep, 0)
    x_sol = ca.solve(A, b)

    X_sol = ca.SX(X)
    for i in range(x_dep.shape[0]):
        X_sol = ca.substitute(X_sol, x_dep[i], x_sol[i])
    X_sol = ca.sparsify(X_sol)
    W_dot_sol = ca.mtimes(F, W) + ca.mtimes(Q / 2 + X_sol, ca.inv(W).T)

    return W_dot_sol


@beartype
def sqrt_correct(Rs: ca.SX, H: ca.SX, W: ca.SX) -> Tuple[ca.SX, ca.SX, ca.SX]:
    """
    source: Fast Stable Kalman Filter Algorithms Utilising the Square Root, Steward 98
    Rs: sqrt(R)
    H: measurement matrix
    W: sqrt(P)

    https://doi.org/10.1109/ICASSP.1990.115844

    @return:
        Wp: sqrt(P+) = sqrt((I - KH)P)
        K: Kalman gain
        Ss: Innovation variance

    """
    n_x = H.shape[1]
    n_y = H.shape[0]
    B = ca.sparsify(ca.blockcat(Rs, ca.mtimes(H, W), ca.SX.zeros(n_x, n_y), W))
    # qr  by default is upper triangular, so we transpose inputs and outputs
    B_Q, B_R = ca.qr(B.T)  # B_Q orthogonal, B_R, lower triangular
    B_Q = B_Q.T
    B_R = B_R.T
    Wp = B_R[n_y:, n_y:]
    Ss = B_R[:n_y, :n_y]
    P_HT_SsInv = B_R[n_y:, :n_y]
    K = ca.mtimes(P_HT_SsInv, ca.inv(Ss))
    return Wp, K, Ss


def ldl_symmetric_decomposition(P: ca.SX) -> Tuple[ca.SX, ca.SX]:
    """
    @param P: Symmetric positive definite matrix
    @return:
        L: Lower triangular, unit diagonal
        D: Diagonal
    """
    n = P.shape[0]
    D = ca.SX.zeros(ca.Sparsity_diag(n))
    L = ca.SX.zeros(ca.Sparsity_lower(n))
    for j in range(n):
        D[j, j] = P[j, j]
        L[j, j] = 1
        for k in range(0, j):
            D[j, j] -= L[j, k] ** 2 * D[k, k]
        for i in range(j + 1, n):
            T = P[i, j]
            for k in range(0, j):
                T -= L[i, k] * L[j, k] * D[k, k]
            L[i, j] = T / D[j, j]
    return L, D


def udu_symmetric_decomposition(P: ca.SX) -> Tuple[ca.SX, ca.SX]:
    """
    @param P: Symmetric positive definite matrix
    @return:
        U: Upper triangular, unit diagonal
        D: Diagonal
    """
    n = P.shape[0]
    P2 = ca.SX(P)
    D = ca.SX.zeros(ca.Sparsity_diag(n))
    U = ca.SX.zeros(ca.Sparsity_upper(n))
    for j in range(n - 1, 0, -1):
        D[j, j] = P2[j, j]
        U[j, j] = 1
        for k in range(j):
            U[k, j] = P2[k, j] / D[j, j]
            for i in range(j):
                P2[i, k] -= P2[k, j] * U[i, j]
    U[0, 0] = 1
    D[0, 0] = P2[0, 0]
    return U, D
