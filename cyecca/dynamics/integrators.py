"""
Custom explicit Runge–Kutta integrators in CasADi.

This module provides:
- build_rk_integrator: generic explicit RK step builder from a Butcher tableau
- rk4: classic 4th-order RK one-step integrator (fixed step)
- rk8: scaffold for 8th-order RK (requires a tableau; see notes)
- integrate_n_steps: helper to roll a one-step integrator for N steps

All functions operate on an ODE of the form x_dot = f(x, u, p), where
- x: state vector (SX/DM)
- u: input vector (SX/DM) — can be empty
- p: parameter vector (SX/DM)

Returns CasADi Functions that take (x, u, p) and return xf after one step
of size h.

Note on RK8: You must provide a valid (A, b, c) Butcher tableau for an
8th-order method (e.g., Verner's 8(7) or Dormand–Prince 8(7)13). Due to the
length of those tables and potential licensing concerns for verbatim tables
in code comments, this module exposes a hook to pass your own tableau.
"""

from __future__ import annotations

from typing import Dict, Sequence

import casadi as ca


def build_rk_integrator(
    f: ca.Function,
    h: float,
    tableau: Dict[str, Sequence],
    name: str = "rk_step",
) -> ca.Function:
    """
    Build a one-step explicit Runge–Kutta integrator from a Butcher tableau.

    Parameters
    ----------
    f : ca.Function
        Dynamics function f(x, u, p) -> x_dot of shape (nx, 1)
    h : float
        Step size
    tableau : dict
        Dictionary with keys 'A', 'b', 'c':
          - A: list of lists (s x s) lower-triangular coefficients
          - b: list of length s (weights)
          - c: list of length s (nodes)
    name : str
        Name of the resulting CasADi function

    Returns
    -------
    ca.Function
        Function F(x, u, p) -> xf applying one RK step of size h
    """
    A = tableau["A"]
    b = tableau["b"]
    c = tableau["c"]
    s = len(b)
    assert len(A) == s and all(len(row) == s for row in A), "Invalid A size"
    assert len(c) == s, "Invalid c size"

    x = ca.SX.sym("x", f.size_out(0))  # state symbol
    u = ca.SX.sym("u", f.size_in(1))  # input symbol (can be 0-length)
    p = ca.SX.sym("p", f.size_in(2))  # param symbol (can be 0-length)

    # Stage storage
    K = [None] * s

    # Compute stages
    for i in range(s):
        # Sum_{j=0}^{i-1} a_ij * K_j
        inc = 0
        if i > 0:
            for j in range(i):
                a_ij = A[i][j]
                if a_ij != 0:
                    inc = inc + a_ij * K[j]
        x_i = x + h * inc
        K[i] = f(x_i, u, p)

    # Combine stages
    x_next = x
    for i in range(s):
        b_i = b[i]
        if b_i != 0:
            x_next = x_next + h * b_i * K[i]

    F = ca.Function(name, [x, u, p], [x_next], ["x", "u", "p"], ["xf"])
    return F


def rk4(f: ca.Function, h: float | ca.SX, name: str = "rk4_step", N: int = 1) -> ca.Function:
    """
    Classic 4th-order Runge–Kutta (RK4) one-step integrator builder.

    Uses 4 stages with the standard Butcher tableau::

        0   |
        1/2 | 1/2
        1/2 | 0     1/2
        1   | 0     0     1
        ----------------------
              1/6   1/3   1/3   1/6

    Parameters
    ----------
    f : ca.Function
        Dynamics function f(x, u, p) -> x_dot
    h : float or ca.SX
        Step size (can be symbolic)
    name : str
        Name of the resulting function
    N : int
        Number of substeps to divide the step into (default=1).
        If N > 1, the step h is divided into N smaller steps of size h/N,
        which are then applied sequentially for improved accuracy.

    Returns
    -------
    ca.Function
        Function F(x, u, p) -> xf applying RK4 integration over step h
    """
    A = [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    b = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
    c = [0.0, 0.5, 0.5, 1.0]

    if N == 1:
        # Single step
        return build_rk_integrator(f, h, {"A": A, "b": b, "c": c}, name=name)
    else:
        # Multiple substeps for improved accuracy
        # For symbolic h, we need to build the integrator with h_sub symbolically
        is_symbolic = isinstance(h, ca.SX) or isinstance(h, ca.MX)

        if is_symbolic:
            # Select appropriate symbolic type matching h (avoid MX/SX mixing)
            if isinstance(h, ca.MX):
                sym = ca.MX
            else:
                sym = ca.SX
            # Build integrator with symbolic step size divided by N
            h_sub = h / N
            # Create symbols for inputs using chosen type
            x = sym.sym("x", f.size_out(0))
            u = sym.sym("u", f.size_in(1))
            p = sym.sym("p", f.size_in(2))

            # Manually unroll RK4 substeps with h_sub
            xk = x
            for _ in range(N):
                # Single RK4 step
                k1 = f(xk, u, p)
                k2 = f(xk + h_sub / 2 * k1, u, p)
                k3 = f(xk + h_sub / 2 * k2, u, p)
                k4 = f(xk + h_sub * k3, u, p)
                xk = xk + h_sub / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Include h as an input since it's symbolic
            return ca.Function(name, [x, u, p, h], [xk], ["x", "u", "p", "h"], ["xf"])
        else:
            # Numeric step size - can use build_rk_integrator
            h_sub = h / N
            f_substep = build_rk_integrator(f, h_sub, {"A": A, "b": b, "c": c}, name=f"{name}_substep")

            # Unroll N substeps
            # Use SX for numeric h path (f may be SX or MX; SX vars acceptable when h numeric)
            x = ca.SX.sym("x", f.size_out(0))
            u = ca.SX.sym("u", f.size_in(1))
            p = ca.SX.sym("p", f.size_in(2))

            xk = x
            for _ in range(N):
                xk = f_substep(xk, u, p)

            return ca.Function(name, [x, u, p], [xk], ["x", "u", "p"], ["xf"])


def rk8(
    f: ca.Function,
    h: float | ca.SX,
    name: str = "rk8_step",
    tableau: Dict[str, Sequence] | None = None,
    N: int = 1,
) -> ca.Function:
    """
    8th-order explicit Runge–Kutta one-step integrator builder.

    You must provide a valid Butcher tableau for an 8th-order method via
    the `tableau` argument. Examples from the literature include Verner's
    8(7) or Dormand–Prince 8(7)13. The tableau dict must have keys 'A','b','c'.

    Parameters
    ----------
    f : ca.Function
        Dynamics function f(x, u, p) -> x_dot
    h : float or ca.SX
        Step size (can be symbolic)
    name : str
        Name of the resulting function
    tableau : dict, optional
        Butcher tableau for the 8th-order method. If None, uses DOP853.
    N : int
        Number of substeps to divide the step into (default=1).
        If N > 1, the step h is divided into N smaller steps of size h/N.

    Returns
    -------
    ca.Function
        Function F(x, u, p) -> xf applying RK8 integration over step h

    Example usage with a user-supplied tableau:
        rk8_F = rk8(f, h, tableau=my_verner_87_tableau())

    Raises
    ------
    NotImplementedError
        If no tableau is provided.
    """
    if tableau is None:
        tableau = dop853_tableau()

    if N == 1:
        # Single step
        return build_rk_integrator(f, h, tableau, name=name)
    else:
        # Multiple substeps for improved accuracy
        is_symbolic = isinstance(h, ca.SX) or isinstance(h, ca.MX)
        h_sub = h / N

        if is_symbolic:
            # For symbolic h, manually unroll the RK8 steps
            A = tableau["A"]
            b = tableau["b"]
            c = tableau["c"]
            s = len(b)

            x = ca.SX.sym("x", f.size_out(0))
            u = ca.SX.sym("u", f.size_in(1))
            p = ca.SX.sym("p", f.size_in(2))

            xk = x
            for _ in range(N):
                # Single RK8 step with h_sub
                K = [None] * s
                for i in range(s):
                    inc = 0
                    if i > 0:
                        for j in range(i):
                            a_ij = A[i][j]
                            if a_ij != 0:
                                inc = inc + a_ij * K[j]
                    x_i = xk + h_sub * inc
                    K[i] = f(x_i, u, p)

                # Combine stages
                x_next = xk
                for i in range(s):
                    b_i = b[i]
                    if b_i != 0:
                        x_next = x_next + h_sub * b_i * K[i]
                xk = x_next

            # Include h as an input since it's symbolic
            return ca.Function(name, [x, u, p, h], [xk], ["x", "u", "p", "h"], ["xf"])
        else:
            # Numeric step size - can use build_rk_integrator
            f_substep = build_rk_integrator(f, h_sub, tableau, name=f"{name}_substep")

            # Unroll N substeps
            x = ca.SX.sym("x", f.size_out(0))
            u = ca.SX.sym("u", f.size_in(1))
            p = ca.SX.sym("p", f.size_in(2))

            xk = x
            for _ in range(N):
                xk = f_substep(xk, u, p)

            return ca.Function(name, [x, u, p], [xk], ["x", "u", "p"], ["xf"])


def dop853_tableau() -> Dict[str, Sequence]:
    """
    Default Butcher tableau for Dormand–Prince 8(5,3) (DOP853).

    Source: Hairer & Wanner, Solving Ordinary Differential Equations I (Nonstiff Problems),
    and the reference implementation DOP853 (publicly available). Coefficients are
    numerical constants (facts) and provided here to construct the explicit RK tableau.
    """
    # Nodes (c): 12 stages for the main method; c1 = 0, c12 = 1
    c = [
        0.0,
        5.26001519587677318785587544488e-2,
        7.89002279381515978178381316732e-2,
        1.18350341907227396726757197510e-1,
        2.81649658092772603273242802490e-1,
        1.0 / 3.0,
        1.0 / 4.0,
        3.07692307692307692307692307692e-1,
        6.51282051282051282051282051282e-1,
        6.0e-1,
        6.0 / 7.0,
        1.0,
    ]

    # Weights (b): combination for y_{n+1} from stages
    b = [
        5.42937341165687622380535766363e-2,  # b1
        0.0,  # b2
        0.0,  # b3
        0.0,  # b4
        0.0,  # b5
        4.45031289275240888144113950566e0,  # b6
        1.89151789931450038304281599044e0,  # b7
        -5.8012039600105847814672114227e0,  # b8
        3.1116436695781989440891606237e-1,  # b9
        -1.52160949662516078556178806805e-1,  # b10
        2.01365400804030348374776537501e-1,  # b11
        4.47106157277725905176885569043e-2,  # b12
    ]

    # Coefficients matrix A (lower-triangular), 12x12
    # Initialize all zeros then fill known nonzeros
    A = [[0.0 for _ in range(12)] for _ in range(12)]

    def setA(i: int, j: int, val: float):
        A[i][j] = val

    # Stage 2
    setA(1, 0, 5.26001519587677318785587544488e-2)  # a21
    # Stage 3
    setA(2, 0, 1.97250569845378994544595329183e-2)  # a31
    setA(2, 1, 5.91751709536136983633785987549e-2)  # a32
    # Stage 4
    setA(3, 0, 2.95875854768068491816892993775e-2)  # a41
    setA(3, 2, 8.87627564304205475450678981324e-2)  # a43
    # Stage 5
    setA(4, 0, 2.41365134159266685502369798665e-1)  # a51
    setA(4, 2, -8.84549479328286085344864962717e-1)  # a53
    setA(4, 3, 9.24834003261792003115737966543e-1)  # a54
    # Stage 6
    setA(5, 0, 3.7037037037037037037037037037e-2)  # a61
    setA(5, 3, 1.70828608729473871279604482173e-1)  # a64
    setA(5, 4, 1.25467687566822425016691814123e-1)  # a65
    # Stage 7
    setA(6, 0, 3.7109375e-2)  # a71
    setA(6, 3, 1.70252211019544039314978060272e-1)  # a74
    setA(6, 4, 6.02165389804559606850219397283e-2)  # a75
    setA(6, 5, -1.7578125e-2)  # a76
    # Stage 8
    setA(7, 0, 3.70920001185047927108779319836e-2)  # a81
    setA(7, 3, 1.70383925712239993810214054705e-1)  # a84
    setA(7, 4, 1.07262030446373284651809199168e-1)  # a85
    setA(7, 5, -1.53194377486244017527936158236e-2)  # a86
    setA(7, 6, 8.27378916381402288758473766002e-3)  # a87
    # Stage 9
    setA(8, 0, 6.24110958716075717114429577812e-1)  # a91
    setA(8, 3, -3.36089262944694129406857109825e0)  # a94
    setA(8, 4, -8.68219346841726006818189891453e-1)  # a95
    setA(8, 5, 2.75920996994467083049415600797e1)  # a96
    setA(8, 6, 2.01540675504778934086186788979e1)  # a97
    setA(8, 7, -4.34898841810699588477366255144e1)  # a98
    # Stage 10
    setA(9, 0, 4.77662536438264365890433908527e-1)  # a101
    setA(9, 3, -2.48811461997166764192642586468e0)  # a104
    setA(9, 4, -5.90290826836842996371446475743e-1)  # a105
    setA(9, 5, 2.12300514481811942347288949897e1)  # a106
    setA(9, 6, 1.52792336328824235832596922938e1)  # a107
    setA(9, 7, -3.32882109689848629194453265587e1)  # a108
    setA(9, 8, -2.03312017085086261358222928593e-2)  # a109
    # Stage 11
    setA(10, 0, -9.3714243008598732571704021658e-1)  # a111
    setA(10, 3, 5.18637242884406370830023853209e0)  # a114
    setA(10, 4, 1.09143734899672957818500254654e0)  # a115
    setA(10, 5, -8.14978701074692612513997267357e0)  # a116
    setA(10, 6, -1.85200656599969598641566180701e1)  # a117
    setA(10, 7, 2.27394870993505042818970056734e1)  # a118
    setA(10, 8, 2.49360555267965238987089396762e0)  # a119
    setA(10, 9, -3.0467644718982195003823669022e0)  # a1110
    # Stage 12
    setA(11, 0, 2.27331014751653820792359768449e0)  # a121
    setA(11, 3, -1.05344954667372501984066689879e1)  # a124
    setA(11, 4, -2.00087205822486249909675718444e0)  # a125
    setA(11, 5, -1.79589318631187989172765950534e1)  # a126
    setA(11, 6, 2.79488845294199600508499808837e1)  # a127
    setA(11, 7, -2.85899827713502369474065508674e0)  # a128
    setA(11, 8, -8.87285693353062954433549289258e0)  # a129
    setA(11, 9, 1.23605671757943030647266201528e1)  # a1210
    setA(11, 10, 6.43392746015763530355970484046e-1)  # a1211

    return {"A": A, "b": b, "c": c}


def integrate_n_steps(
    F_step: ca.Function,
    x0: ca.DM | ca.SX,
    u: ca.DM | ca.SX,
    p: ca.DM | ca.SX,
    N: int,
    name: str = "rollout",
) -> ca.Function:
    """
    Roll a one-step integrator for N steps (fixed input, params) symbolically.

    Parameters
    ----------
    F_step : ca.Function
        One-step integrator Function with signature (x,u,p)->xf
    x0 : DM|SX
        Initial state symbol/value
    u : DM|SX
        Input (held constant across steps)
    p : DM|SX
        Parameters (held constant across steps)
    N : int
        Number of steps to perform
    name : str
        Name of the resulting function

    Returns
    -------
    ca.Function
        Function taking (x0,u,p) and returning (xf)
    """
    x_sym = (
        ca.SX.sym("x0", x0.shape[0])
        if isinstance(x0, ca.SX) or isinstance(x0, ca.MX)
        else ca.SX.sym("x0", int(x0.numel()))
    )
    u_sym = ca.SX.sym("u", F_step.size_in(1))
    p_sym = ca.SX.sym("p", F_step.size_in(2))

    xk = x_sym
    for _ in range(N):
        xk = F_step(xk, u_sym, p_sym)

    return ca.Function(name, [x_sym, u_sym, p_sym], [xk], ["x0", "u", "p"], ["xf"])
