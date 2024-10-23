import cyecca.lie as lie
import casadi as ca
import numpy as np


def derive_dynamics_linearization():
    # input varibles
    P = ca.SX.sym("P", 3, 3)  # covariance matrix
    # x = lie.so3.elem(ca.SX.sym("x", 3))  # state
    omega = lie.so3.elem(ca.SX.sym("omega", 3))  # angular velocity, from gyroscope
    q = lie.SO3Quat.elem(ca.SX.sym("q", 4))  # quaternion
    qr = lie.SO3Quat.elem(ca.SX.sym("qr", 4))  # reference quaternion

    # input parameters
    dt = ca.SX.sym("dt")  # prediction period of kalman filter
    gyro_sqrt_noise_power = ca.SX.sym("gyro_sqrt_noise_power")

    # computed values
    Q = ca.SX.eye(3) * gyro_sqrt_noise_power**2 * dt
    # qr = lie.SO3Quat.elem(ca.SX.sym("qr", 4))

    # here we assume that noise can be viewed as constant during this
    # sampling period, this is valid if the noise power is constant
    # and dt is used to sample noise with proper std. dev.

    # A = -(omega*dt).ad()
    B = ca.SX.eye(3)  # lie.so3.right_jacobian_inv(x)
    # exp(ad(-omega*dt)) = Ad(exp(-omega*dt))
    eta = (-omega * dt).exp(lie.SO3Quat)
    F = eta.Ad()

    # assumptions
    # * sampling noise as constant over interval of discretization
    # * normally distributed, mean zero noise
    # * neglecting nonlinear terms in Jr_inv, approximating at identity
    # P1 = ca.sparsify(F @ (P + Q) @ F.T)
    P1 = F @ P @ F.T + F @ Q @ F.T

    euler_error = lie.SO3EulerB321.from_Quat(q.inverse() * qr).param
    f_euler_error = ca.Function(
        "eulerB321_error", [q.param, qr.param], [euler_error], ["q", "qr"], ["euler"]
    )

    q1 = q * (omega * dt).exp(lie.SO3Quat)

    f_exp_quat = ca.Function(
        "exp_quat", [q.param, omega.param, dt], [q1.param], ["q", "omega", "dt"], ["q1"]
    )

    f_kalman_predict = ca.Function(
        "kalman_predict",
        [q.param, ca.reshape(P, 9, 1), gyro_sqrt_noise_power, dt, omega.param],
        [q1.param.T, ca.reshape(P1, 9, 1)],
        ["q", "P", "gyro_sqrt_noise_power", "dt", "omega"],
        ["q1", "P1"],
    )

    return {
        #'f_log_error_quat': f_log_error_quat,
        "exp_quat": f_exp_quat,
        "kalman_predict": f_kalman_predict,
        "euler_error": f_euler_error,
    }


def derive_accel_measurement():
    v = ca.SX.sym("v", 3)  # measurement noise
    x = lie.so3.elem(ca.SX.sym("x", 3))  # state (lie algebra)
    zh = ca.vertcat(0, 0, 1)  # z unit vector
    accel_noise_str = ca.SX.sym("accel_noise_str")
    g = ca.SX.sym("g")
    qr = lie.SO3Quat.elem(ca.SX.sym("qr", 4))
    qe = x.exp(lie.SO3Quat)

    # parameterize measurements in exponential coordinates
    y_accel = (qe * qr).inverse() @ (g * zh) + v * accel_noise_str

    def sub_est_zero_error(expr):
        expr = ca.substitute(expr, x.param, [0, 0, 0])
        expr = ca.substitute(expr, v, [0, 0, 0])
        return expr

    f_accel_g = ca.Function(
        "accel_g",
        [qr.param, v, g, accel_noise_str],
        [sub_est_zero_error(y_accel)],
        ["qr", "v", "g", "accel_noise_str"],
        ["y"],
    )

    # jacobian of measurement wrt state at zero error
    H_accel = ca.jacobian(y_accel, x.param)
    H_accel = sub_est_zero_error(H_accel)

    # jacobian of measurement wrt to noise at zero error
    N_accel = ca.jacobian(y_accel, v)
    N_accel = sub_est_zero_error(N_accel)
    R_accel = N_accel @ N_accel.T

    # kalman update
    y_accel_meas = ca.SX.sym("y_accel", 3)
    P = ca.SX.sym("P", 3, 3)
    S = H_accel @ P @ H_accel.T + R_accel
    K = P @ H_accel.T @ ca.inv(S)
    P1 = (ca.SX.eye(3) - K @ H_accel) @ P

    y_accel_est = f_accel_g(qr.param, [0, 0, 0], g, accel_noise_str)
    q1 = (lie.so3.elem(K @ (y_accel_meas - y_accel_est))).exp(lie.SO3Quat) * qr

    f_accel_kalman_update = ca.Function(
        "accel_kalman_update",
        [y_accel_meas, qr.param, ca.reshape(P, 9, 1), g, accel_noise_str],
        [q1.param, ca.reshape(P1, 9, 1), K, H_accel, R_accel],
        ["y_accel", "q0", "P0", "g", "accel_noise_str"],
        ["q1", "P1", "K", "H", "R"],
    )

    return {
        "accel_g": f_accel_g,
        "accel_kalman_update": f_accel_kalman_update,
    }


def derive_mag_measurement():
    # parameters
    mag_decl = ca.SX.sym("decl")
    mag_incl = ca.SX.sym("incl")
    mag_str = ca.SX.sym("mag_str")
    mag_noise_str = ca.SX.sym("mag_noise_str")

    # inputs
    v = ca.SX.sym("v", 3)  # measurement noise
    x = lie.so3.elem(ca.SX.sym("x", 3))  # state (lie algebra)
    q_we = x.exp(lie.SO3Quat)  # right inv error quaternion
    q_eb = lie.SO3Quat.elem(ca.SX.sym("q_eb", 4))  # reference quaterion

    # constants
    xh = ca.vertcat(1, 0, 0)  # x unit vector

    # quaterions SO3 Lie group
    q_wb = q_we * q_eb

    R_mag = lie.SO3EulerB321.elem(ca.vertcat(mag_decl, mag_incl, 0))
    B_w = R_mag @ (mag_str * xh)  # magnetic field vector in world frame
    B_w[2] = 0

    # mag measurement in right inv error frame
    y_mag_e = q_we.inverse() @ B_w + v * mag_noise_str  # noise is rotationally invar.
    y_mag_e = y_mag_e[:2]

    # mag measurement in body frame
    y_mag_b = q_wb.inverse() @ B_w + v * mag_noise_str  # noise is rotationally invar.

    def sub_est_zero_error(expr):
        expr = ca.substitute(expr, x.param, [0, 0, 0])
        expr = ca.substitute(expr, v, [0, 0, 0])
        return expr

    H_mag = sub_est_zero_error(ca.jacobian(y_mag_e, x.param))
    N = sub_est_zero_error(ca.jacobian(y_mag_e, v))
    R_mag = N @ N.T

    # define functions
    f_g_mag = ca.Function(
        "g_mag",
        [q_eb.param, v, mag_decl, mag_incl, mag_str, mag_noise_str],
        [ca.substitute(y_mag_b, x.param, [0, 0, 0])],
        ["q_eb", "v", "mag_decl", "mag_incl", "mag_str", "mag_noise_str"],
        ["y_mag_b"],
    )

    # kalman update
    y_mag_meas_b = ca.SX.sym("y_mag_b", 3)
    y_mag_meas_e = (q_eb @ y_mag_meas_b)[:2]

    P = ca.SX.sym("P", 3, 3)
    S = H_mag @ P @ H_mag.T + R_mag
    K = P @ H_mag.T @ ca.inv(S)
    P1 = (ca.SX.eye(3) - K @ H_mag) @ P

    y_mag_est_e = sub_est_zero_error(y_mag_e)[:2]

    delta_x = K @ (y_mag_meas_e - y_mag_est_e)
    delta_x[0] = 0  # should be zero, planar rotation in z after xy proj.
    delta_x[1] = 0  # should be zero, planar rotation in z after xy proj.

    q1 = lie.so3.elem(delta_x).exp(lie.SO3Quat) * q_eb

    f_mag_kalman_update = ca.Function(
        "mag_kalman_update",
        [
            y_mag_meas_b,
            q_eb.param,
            ca.reshape(P, 9, 1),
            mag_decl,
            mag_incl,
            mag_str,
            mag_noise_str,
        ],
        [q1.param, ca.reshape(P1, 9, 1), K, H_mag, delta_x],
        ["y_mag", "q0", "P0", "mag_decl", "mag_incl", "mag_str", "mag_noise_str"],
        ["q1", "P1", "K", "H_mag", "delta_x"],
    )

    return {
        "mag_g": f_g_mag,
        "mag_kalman_update": f_mag_kalman_update,
    }


def simulate():
    # parameters
    enable_mag = True
    enable_accel = False
    tf = 5  # final time
    dt = 0.001  # period for gyro prediction
    dt_mag = 1.0 / 10  # period for mag update
    dt_accel = 1.0 / 10  # period for accel update
    g = 9.8  # gravitational accel

    # earth's magnetic field in Laf/ Indiana
    mag_incl = np.deg2rad(67.37)  # magnetic inclination
    mag_decl = np.deg2rad(-4.51)  # magnetic declination
    mag_str = 0.522  # magnetic field strength [gauss]

    mag_sqrt_noise_power = 1e-4  # [gauss/sqrt(hz)]
    accel_sqrt_noise_power = g * 70e-6  # sqrt of accel noise power [(m/s^2)/sqrt(hz)]
    gyro_sqrt_noise_power = np.deg2rad(
        2.8e-3
    )  # sqrt of gyro noise power  [rad/sqrt(hz)]
    omega_b = np.array([1, 2, 3])  # angular velocity of body during simulation
    euler0 = np.deg2rad(np.array([0.0, 0.0, 0.0]))  # initial true euler angles
    euler_est_0 = np.deg2rad(np.array([100, 0, 0]))  # initial estimated euler angles
    P0 = (1.0 * np.eye(3)).reshape(
        -1
    )  # initial attitude covariance (right invariant lie algebra)

    # equations
    eqs = {}
    eqs.update(derive_accel_measurement())
    eqs.update(derive_mag_measurement())
    eqs.update(derive_dynamics_linearization())

    def ca2np(vect: ca.SX):
        """function to turn casadi vectors into numpy vectors"""
        return np.array(ca.DM(vect)).reshape(-1)

    # initialize loops
    gyro_noise_std_dev = gyro_sqrt_noise_power / ca.sqrt(dt)
    accel_noise_std_dev = accel_sqrt_noise_power / ca.sqrt(dt)
    mag_noise_std_dev = mag_sqrt_noise_power / ca.sqrt(dt)

    x = ca2np(ca.DM(lie.SO3Quat.from_Euler(lie.SO3EulerB321.elem(ca.DM(euler0))).param))
    x_est = ca2np(
        ca.DM(lie.SO3Quat.from_Euler(lie.SO3EulerB321.elem(ca.DM(euler_est_0))).param)
    )
    P = P0
    t_mag_last = -2 * dt_mag
    t_accel_last = -2 * dt_accel

    def sim_mag(x):
        return ca2np(
            eqs["mag_g"](
                x, np.random.randn(3), mag_decl, mag_incl, mag_str, mag_noise_std_dev
            )
        )

    def sim_accel(x):
        return ca2np(eqs["accel_g"](x, np.random.randn(3), g, accel_noise_std_dev))

    def sim_gyro(omega_b):
        return omega_b + gyro_noise_std_dev * np.random.randn(3)

    def kalman_predict(x_est, P):
        x_est, P = eqs["kalman_predict"](x_est, P, gyro_sqrt_noise_power, dt, y_gyro)
        x_est = ca2np(x_est)
        P = ca2np(P)
        assert np.all(np.isfinite(x_est))
        assert np.all(np.isfinite(P))
        return x_est, P

    def kalman_correct_mag(x_est, P, y_mag):
        [x_est, P, K, H, delta_x] = eqs["mag_kalman_update"](
            y_mag,
            x_est,
            P,
            mag_decl,
            mag_incl,
            mag_str,
            mag_noise_std_dev,
        )
        x_est = ca2np(x_est)
        P = ca2np(P)
        assert np.all(np.isfinite(K))
        assert np.all(np.isfinite(H))
        assert np.all(np.isfinite(x_est))
        assert np.all(np.isfinite(P))
        return x_est, P

    def kalman_correct_accel(x_est, P, y_accel):
        [x_est, P, K, H, R] = eqs["accel_kalman_update"](
            y_accel,
            x_est,
            P,
            g,
            accel_noise_std_dev,
        )
        x_est = ca2np(x_est)
        P = ca2np(P)
        assert np.all(np.isfinite(K))
        assert np.all(np.isfinite(H))
        assert np.all(np.isfinite(R))
        assert np.all(np.isfinite(x_est))
        assert np.all(np.isfinite(P))
        return x_est, P

    y_mag = sim_mag(x)
    y_accel = sim_accel(x)
    y_gyro = sim_gyro(omega_b)

    # data dictionary for simulation
    data = {
        "t": [],
        "x": [],
        "x_est": [],
        "y_mag": [],
        "y_accel": [],
        "y_gyro": [],
        "P": [],
        "euler_error": [],
    }

    # main simulation loop
    for ti in np.arange(0, tf, dt):
        euler_error = ca2np(eqs["euler_error"](x_est, x))

        # store data
        data["t"].append(ti)
        data["x"].append(x)
        data["x_est"].append(x_est)
        data["y_mag"].append(y_mag)
        data["y_accel"].append(y_accel)
        data["y_gyro"].append(y_gyro)
        data["P"].append(P)
        data["euler_error"].append(euler_error)

        # simulation
        x = ca2np(eqs["exp_quat"](x, omega_b, dt))
        y_gyro = sim_gyro(omega_b)

        # kalman prediction
        x_est, P = kalman_predict(x_est, P)

        # magnetometer measurement
        if enable_mag and ti - t_mag_last > dt_mag:
            t_mag_last = ti
            y_mag = sim_mag(x)
            x_est, P = kalman_correct_mag(x_est, P, y_mag)

        # accelerometer measurement
        if enable_accel and ti - t_accel_last > dt_accel:
            t_accel_last = ti
            y_accel = sim_accel(x)
            x_est, P = kalman_correct_accel(x_est, P, y_accel)

    # turn all of lists into numpy arrays
    for k in data.keys():
        data[k] = np.array(data[k])

    return data
