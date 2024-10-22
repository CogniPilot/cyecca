import casadi as ca
import cyecca
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from tf_transformations import euler_from_quaternion


def derive_model(dt):
    # p, parameters
    thr_max = ca.SX.sym("thr_max")
    m = ca.SX.sym("m")
    cl = ca.SX.sym("cl")
    S = ca.SX.sym("S")
    rho = ca.SX.sym("rho")
    g = ca.SX.sym("g")

    p= ca.vertcat(
        thr_max,
        m,
        cl,
        S,
        rho,
        g
    )

    # states
    # x, state
    posx = ca.SX.sym("posx")
    velx = ca.SX.sym("velx")
    x = ca.vertcat(posx,velx)
    x0_defaults = {
        "posx" : 0,
        "velx" : 0
    }

    # input
    throttle_cmd = ca.SX.sym("throttle_cmd")
    u = ca.vertcat(throttle_cmd)


    # force and moment
    fx_b = thr_max*u[0]-velx
    ax = fx_b/m
    velx = (ax)*dt

    # states derivative
    posx_dot = velx
    velx_dot = ax
    xdot = ca.vertcat(posx_dot,velx_dot)


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
