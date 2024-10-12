import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

import cyecca.lie as lie
from cyecca.graph import draw_casadi
from cyecca.util import rk4, count_ops
from pathlib import Path


def derive_strapdown_ins_propagation():
    dt = ca.SX.sym("dt")
    X0 = lie.SE23Mrp.elem(ca.SX.sym("X0", 9))
    a_b = ca.SX.sym("a_b", 3)
    g = ca.SX.sym("g")
    omega_b = ca.SX.sym("omega_b", 3)
    l = lie.se23.elem(ca.vertcat(0, 0, 0, a_b, omega_b))
    r = lie.se23.elem(ca.vertcat(0, 0, 0, 0, 0, g, 0, 0, 0))
    B = ca.sparsify(ca.SX([[0, 1], [0, 0]]))
    X1 = lie.SE23Mrp.exp_mixed(X0, l * dt, r * dt, B * dt)
    r1 = lie.SO3Mrp.elem(X1.param[6:9])
    lie.SO3Mrp.shadow_if_necessary(r1)
    X1.param[6:9] = r1.param
    return ca.Function(
        "strapdown_ins_propagate",
        [X0.param, a_b, omega_b, g, dt],
        [X1.param],
        ["x0", "a_b", "omega_b", "g", "dt"],
        ["x1"],
    )


f_strapdown_ins_propagation = derive_strapdown_ins_propagation()

r = lie.SO3Mrp.elem(ca.SX.sym("r", 3))
euler = lie.SO3EulerB321.from_Mrp(r)
f_r_to_eulerB321 = ca.Function("r_to_eulerB321", [r.param], [euler.param])


def derive_rk4_kinematics():
    x = ca.SX.sym("x0", 9)
    a_b = ca.SX.sym("a_b", 3)
    omega_b = ca.SX.sym("omega_b", 3)
    g = ca.SX.sym("g")
    p, v, r = x[:3], x[3:6], x[6:9]
    R = lie.SO3Mrp.elem(r).to_Matrix()
    p_dot = v
    v_dot = R @ a_b + ca.vertcat(0, 0, g)

    X = lie.so3.elem(r).to_Matrix()
    n_sq = ca.dot(r, r)
    B = 0.25 * ((1 - n_sq) * ca.SX.eye(3) + 2 * X + 2 * r @ r.T)
    r_dot = B @ omega_b
    return ca.Function(
        "ode_dynamics",
        [x, a_b, omega_b, g],
        [ca.vertcat(p_dot, v_dot, r_dot)],
        ["x", "a_b", "omega_b", "g"],
        ["x_dot"],
    )


def derive_integrate_rk4():
    x0 = ca.SX.sym("x0", 9)
    dt = ca.SX.sym("dt")
    a_b = ca.SX.sym("a_b", 3)
    g = ca.SX.sym("g")
    omega_b = ca.SX.sym("omega_b", 3)
    f_rk4_kinematics = derive_rk4_kinematics()
    x1 = rk4(lambda t, x: f_rk4_kinematics(x, a_b, omega_b, g), 0, x0, dt)
    r1 = lie.SO3Mrp.elem(x1[6:9])
    lie.SO3Mrp.shadow_if_necessary(r1)
    x1[6:9] = r1.param
    return ca.Function(
        "integrate_rk4",
        [x0, a_b, omega_b, g, dt],
        [x1[:, 0]],
        ["x0", "a_b", "omega_b", "g", "dt"],
        ["x1"],
    )


f_integrate_rk4 = derive_integrate_rk4()
f_integrate_rk4([0, 0, 0, 0, 0, 0, 0, 0, 0], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 9.8, 0.1)


def integrate_rk4(x0, a_b, omega_b, g, dt, tf):
    t_list = np.arange(0, tf, dt)
    data = {"t": t_list, "x": np.zeros((len(t_list), len(x0)), dtype=float)}
    x = data["x"]
    x[0, :] = x0
    for i in range(1, len(t_list)):
        x[i, :] = np.array(f_integrate_rk4(x[i - 1, :], a_b, omega_b, g, dt)).reshape(
            -1
        )
    return data


def integrate_mixed_invariant(x0, a_b, omega_b, g, dt, tf):
    t_list = np.arange(0, tf, dt)
    data = {"t": t_list, "x": np.zeros((len(t_list), len(x0)), dtype=float)}
    x = data["x"]

    x[0, :] = x0
    for i in range(1, len(t_list)):
        x[i, :] = np.array(
            f_strapdown_ins_propagation(x0, a_b, omega_b, g, t_list[i])
        ).reshape(-1)
    return data


def derive_f_quat_from_mrp():
    r = lie.SO3Mrp.elem(ca.SX.sym("r", 3))
    q = lie.SO3Quat.from_Mrp(r)
    return ca.Function("quat_from_mrp", [r.param], [q.param], ["r"], ["q"])


f_quat_from_mrp = derive_f_quat_from_mrp()


def generate_code():
    import cyecca.codegen

    cyecca.codegen.generate_code(
        eqs={
            "strapdown_ins": {
                "predict": f_strapdown_ins_propagation,
                "quat_from_mrp": f_quat_from_mrp,
            },
        },
        dest_dir="gen",
    )


def comparison(name: str, a_b: np.array, omega_b: np.array):
    fig_path = Path("fig")
    fig_path.mkdir(exist_ok=True)
    dt = 0.005
    tf = 10
    g = 9.8

    # pos, vel, rot
    x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    data_rk4 = integrate_rk4(x0=x0, a_b=a_b, omega_b=omega_b, g=g, dt=dt, tf=tf)

    data_mixed = integrate_mixed_invariant(
        x0=x0, a_b=a_b, omega_b=omega_b, g=g, dt=dt, tf=tf
    )

    if True:
        plt.figure()
        plt.title("Modified Rodrigues Parameters Comparison")
        h_rk4 = plt.plot(data_rk4["t"], data_rk4["x"][:, 7:10], "r-", linewidth=5)[0]
        h_mixed = plt.plot(data_mixed["t"], data_mixed["x"][:, 7:10], "y-")[0]
        plt.legend([h_rk4, h_mixed], ["rk4", "mixed"])
        plt.xlabel("t, sec")
        plt.ylabel("mrp components")
        plt.grid()
        plt.savefig(f"fig/mrp_{name:s}.png")

    if True:
        plt.figure()
        plt.title("Mrp Error in RK4 Method")
        plt.plot(
            data_rk4["t"],
            data_rk4["x"][:, 7:10] - data_mixed["x"][:, 7:10],
            "r-",
            linewidth=5,
            label="rk4",
        )
        plt.xlabel("t, sec")
        plt.ylabel("mrp error")
        plt.grid()
        plt.savefig(f"fig/mrp_error_{name:s}.png")

    if False:
        plt.figure()
        plt.title("positon")
        h_rk4 = plt.plot(data_rk4["t"], data_rk4["x"][:, 0], "r.-", linewidth=10)[0]
        h_mixed = plt.plot(data_mixed["t"], data_mixed["x"][:, 0], "y.-")[0]
        plt.legend([h_rk4, h_mixed], ["rk4", "mixed"])
        plt.xlabel("t, sec")
        plt.ylabel("position")
        plt.grid()
        plt.savefig(f"fig/pos_{name:s}.png")

    if False:
        plt.figure()
        plt.title("position error")
        plt.plot(data_rk4["t"], data_rk4["x"][:, 0] - data_mixed["x"][:, 0], "r.-")
        plt.xlabel("t, sec")
        plt.grid()
        plt.savefig(f"fig/pos_error_{name:s}.png")

    if True:
        plt.figure()
        plt.title("Trajectory Comparison in XZ-plane")
        h_rk4 = plt.plot(
            data_rk4["x"][:, 0], -data_rk4["x"][:, 2], "r-", linewidth=5, label="rk4"
        )[0]
        h_mixed = plt.plot(
            data_mixed["x"][:, 0], -data_mixed["x"][:, 2], "y-", label="mixed"
        )[0]
        plt.xlabel("t, sec")
        plt.legend([h_rk4, h_mixed], ["rk4", "mixed"])
        plt.grid()
        plt.savefig(f"fig/traj_xz_{name:s}.png")


def draw_graphs():
    x0 = ca.SX.sym("x0", 9)
    a_b = ca.SX.sym("a_b", 3)
    omega_b = ca.SX.sym("omega_b", 3)
    g = ca.SX.sym("g")
    dt = ca.SX.sym("dt")
    draw_casadi(
        f_strapdown_ins_propagation(x0, a_b, omega_b, g, dt),
        filename="fig/f_mixed.png",
        width=400,
    )
    draw_casadi(
        f_integrate_rk4(x0, a_b, omega_b, g, dt), filename="fig/f_rk4.png", width=400
    )


def find_flops():
    x0 = ca.SX.sym("x0", 9)
    a_b = ca.SX.sym("a_b", 3)
    omega_b = ca.SX.sym("omega_b", 3)
    g = ca.SX.sym("g")
    dt = ca.SX.sym("dt")

    op_dict_mixed = count_ops(f_strapdown_ins_propagation(x0, a_b, omega_b, g, dt))
    op_dict_rk4 = count_ops(f_integrate_rk4(x0, a_b, omega_b, g, dt))

    def find_flops(op_dict):
        flops = 0
        for k, v in op_dict.items():
            if k not in ["OP_PARAMETER", "OP_CONST"]:
                flops += v
        return flops

    flops_mixed = find_flops(op_dict_mixed)
    flops_rk4 = find_flops(op_dict_rk4)

    plt.title("floating point operations")
    plt.bar(x=["rk4", "mixed"], height=[flops_rk4, flops_mixed], color=["r", "g"])
    plt.ylabel("floating point operations")
    Path("fig").mkdir(exist_ok=True)
    plt.savefig("fig/ops.png")


def comparison_tolerance(name: str, x0: np.array, a_b: np.array, omega_b: np.array):
    plt.close()
    fig_path = Path("fig")
    fig_path.mkdir(exist_ok=True)
    tf = 10
    g = 9.8

    rk4_steps = [10 ** (-n) for n in range(1, 4)]

    plt.ioff()

    plt.figure(1)
    # plt.title(
    #    f"{name:s} Error in Runge-Kutta 4th Order Method"
    # )
    plt.ylabel("$log_{10} |e|$")
    plt.xlabel("t, sec")
    plt.grid()

    plt.figure(2)
    # plt.title(f"Trajectory")
    plt.xlabel("x, m")
    plt.ylabel("y, m")
    plt.grid()
    plt.axis("equal")

    plt.figure(3)
    plt.title("mrp")

    plt.figure(4)
    plt.title("velocity")

    plt.figure(5)
    plt.title("position")

    for step in rk4_steps:
        data_rk4 = integrate_rk4(x0=x0, a_b=a_b, omega_b=omega_b, g=g, dt=step, tf=tf)
        data_mixed = integrate_mixed_invariant(
            x0=x0, a_b=a_b, omega_b=omega_b, g=g, dt=step, tf=tf
        )
        e = np.linalg.norm(data_rk4["x"] - data_mixed["x"], axis=1)
        e = np.where(e == 0, 1e-15, e)
        log_e = np.log10(np.abs(e))

        # plt.subplot(121)
        # plt.plot(data_rk4[step]['t'], e)
        # plt.title(f'step {step:f}')
        # plt.ylabel('e')
        # plt.xlabel('t, s')

        # plt.subplot(122)
        plt.figure(1)
        plt.plot(data_rk4["t"], log_e, label=f"step size: {step:10g}")

        plt.figure(2)
        plt.plot(
            data_rk4["x"][:, 0], data_rk4["x"][:, 1], label=f"step size: {step:10g}"
        )

        plt.figure(3)
        plt.subplot(321)
        plt.plot(
            data_rk4["t"], data_rk4["x"][:, 6], label=f"roll rk4 step size: {step:10g}"
        )
        plt.subplot(323)
        plt.plot(
            data_rk4["t"], data_rk4["x"][:, 7], label=f"pitch rk4 step size: {step:10g}"
        )
        plt.subplot(325)
        plt.plot(
            data_rk4["t"], data_rk4["x"][:, 8], label=f"yaw rk4 step size: {step:10g}"
        )
        plt.subplot(322)
        plt.plot(
            data_mixed["t"],
            data_mixed["x"][:, 6],
            label=f"roll mixed step size: {step:10g}",
        )
        plt.subplot(324)
        plt.plot(
            data_mixed["t"],
            data_mixed["x"][:, 7],
            label=f"pitch mixed step size: {step:10g}",
        )
        plt.subplot(326)
        plt.plot(
            data_mixed["t"],
            data_mixed["x"][:, 8],
            label=f"yaw mixed step size: {step:10g}",
        )

        plt.figure(4)
        plt.plot(
            data_rk4["t"], data_rk4["x"][:, 3], label=f"vx rk4 step size: {step:10g}"
        )
        plt.plot(
            data_rk4["t"], data_rk4["x"][:, 4], label=f"vy rk4 step size: {step:10g}"
        )
        plt.plot(
            data_mixed["t"],
            data_mixed["x"][:, 3],
            label=f"vx mixed step size: {step:10g}",
        )
        plt.plot(
            data_mixed["t"],
            data_mixed["x"][:, 4],
            label=f"vy mixed step size: {step:10g}",
        )

        plt.figure(5)
        plt.plot(
            data_rk4["t"], data_rk4["x"][:, 0], label=f"px rk4 step size: {step:10g}"
        )
        plt.plot(
            data_rk4["t"], data_rk4["x"][:, 0], label=f"py rk4 step size: {step:10g}"
        )
        plt.plot(
            data_mixed["t"],
            data_mixed["x"][:, 0],
            label=f"px mixed step size: {step:10g}",
        )
        plt.plot(
            data_mixed["t"],
            data_mixed["x"][:, 0],
            label=f"py mixed step size: {step:10g}",
        )

    plt.figure(1)
    plt.legend(loc="best")
    plt.savefig(f"fig/{name:s}_tol.png")
    # plt.close(1)

    plt.figure(2)
    plt.legend(loc="best")
    plt.savefig(f"fig/{name:s}_traj.png")
    # plt.close(2)

    plt.figure(3)
    plt.legend(loc="best")
    plt.savefig(f"fig/{name:s}_mrp.png")
    plt.close(3)

    plt.figure(4)
    plt.legend(loc="best")
    plt.savefig(f"fig/{name:s}_4.png")
    plt.close(4)

    plt.figure(5)
    plt.legend(loc="best")
    plt.savefig(f"fig/{name:s}_5.png")
    plt.close(5)

    plt.show()
    plt.close()
