import time

import numpy as np

import cyecca.sim.msgs as msgs
import cyecca.sim.uros as uros


class AttitudeEstimator:
    """
    An attitude estimator node for uros
    """

    def __init__(self, core, name, eqs, initialize):
        self.core = core
        self.name = name

        # subscriptions
        self.sub_imu = uros.Subscriber(core, "imu", msgs.Imu, self.imu_callback)
        self.sub_mag = uros.Subscriber(core, "mag", msgs.Mag, self.mag_callback)

        # publications
        self.pub_est = uros.Publisher(core, name + "_status", msgs.EstimatorStatus)
        self.pub_att = uros.Publisher(core, name + "_attitude", msgs.Attitude)

        self.msg_est_status = msgs.EstimatorStatus()
        self.msg_att = msgs.Attitude()

        self.sub_params = uros.Subscriber(
            core, "params", msgs.Params, self.params_callback
        )

        # parameters
        self.param_list = []

        def add_param(name, value, type):
            p = uros.Param(self.core, self.name + "/" + name, value, type)
            self.param_list.append(p)
            return p

        self.std_mag = add_param("std_mag", 2.5e-3, "f8")
        self.std_accel = add_param("std_accel", 35.0e-3, "f8")
        self.std_accel_omega = add_param("std_accel_omega", 0, "f8")
        self.std_gyro = add_param("std_gyro", 1e-3, "f8")
        self.sn_gyro_rw = add_param("sn_gyro_rw", 1e-5, "f8")
        self.mag_decl = add_param("mag_decl", 0, "f8")
        self.beta_mag_c = add_param("beta_mag_c", 6.6, "f8")  # 99% for n=1
        self.beta_accel_c = add_param("beta_accel_c", 9.2, "f8")  # 99% for n=2
        self.dt_min_accel = add_param("dt_min_accel", 1.0 / 200, "f8")
        self.dt_min_mag = add_param("dt_min_mag", 1.0 / 200, "f8")

        self.g = add_param("g", 9.8, "f8")

        # misc
        self.x = eqs["constants"]()["x0"]
        self.W = eqs["constants"]()["W0"]
        self.n_x = self.x.shape[0]
        self.n_e = self.W.shape[0]
        self.t_last_imu = 0
        self.t_last_accel = 0
        self.t_last_mag = 0
        self.eqs = eqs
        if initialize:
            self.initialized = False
        else:
            self.initialized = True
        self.last_mag = None
        self.last_imu = None
        self.time_eps = 1e-3  # small period of time to prevent missing pub

    def params_callback(self, msg):
        for p in self.param_list:
            p.update()

    def mag_callback(self, msg):
        t = msg.data["time"]
        self.last_mag = msg  # must always set, since used for init

        if not self.initialized or t - self.t_last_mag < (
            self.dt_min_mag.get() - self.time_eps
        ):
            return

        self.t_last_mag = t
        y = msg.data["mag"]
        # out: ['x_mag', 'W_mag', 'beta_mag', 'r_mag', 'r_std_mag', 'error_code'])
        # in: ['x', 'W', 'y_b', 'decl', 'std_mag', 'beta_mag_c']
        start = time.thread_time()
        self.x, self.W, beta_mag, r_mag, r_std_mag, mag_ret = self.eqs["correct_mag"](
            self.x,
            self.W,
            y,
            self.mag_decl.get(),
            self.std_mag.get(),
            self.beta_mag_c.get(),
        )
        cpu_mag = time.thread_time() - start

        uros.check_nan(
            locals(),
            "{:s} mag correction".format(self.name),
            t,
            ["self.x", "self.W", "beta_mag", "r_mag", "r_std_mag", "mag_ret"],
        )

        self.msg_est_status.data["beta_mag"] = beta_mag
        self.msg_est_status.data["r_mag"][: r_mag.shape[0]] = np.array(r_mag).T
        self.msg_est_status.data["r_std_mag"][: r_std_mag.shape[0]] = np.array(
            r_std_mag
        ).T
        self.msg_est_status.data["mag_ret"] = mag_ret
        self.msg_est_status.data["cpu_mag"] = cpu_mag

    def imu_callback(self, msg):
        # compute dt
        t = msg.data["time"]
        dt = t - self.t_last_imu
        self.t_last_imu = t
        self.last_imu = msg

        # initialize
        if not self.initialized:
            if self.last_imu is not None and self.last_mag is not None:
                # in: ['g_b', 'B_b', 'decl'],
                # out: ['x0', 'error_code']
                x0, ret = self.eqs["initialize"](
                    self.last_imu.data["accel"],
                    self.last_mag.data["mag"],
                    self.mag_decl.get(),
                )

                if ret != 0:
                    print("initialization failed with error code", ret)
                else:
                    print("initialized at time ", self.core.now, x0, ret)
                    self.x = x0
                    self.initialized = True
                    if np.any((np.isnan(np.array(self.x)))):
                        s = "nan in estimator {:s} @ initialization, x = {:s}".format(
                            self.name, str(self.x)
                        )
                        raise ValueError(s)
            return

        assert self.initialized

        if dt <= 0:
            return

        # estimate state
        omega = msg.data["gyro"]
        start = time.thread_time()
        # in: ['t', 'x', 'W', 'omega_m', 'std_gyro', 'sn_gyro_rw', 'dt']
        # out ['x1', 'W1']
        self.x, self.W = self.eqs["predict"](
            t, self.x, self.W, omega, self.std_gyro.get(), self.sn_gyro_rw.get(), dt
        )
        q, r, b_g = self.eqs["get_state"](self.x)
        cpu_predict = time.thread_time() - start
        uros.check_nan(
            locals(),
            "{:s} prediction".format(self.name),
            t,
            ["self.x", "self.W", "q", "r", "b_g"],
        )

        if t - self.t_last_accel >= (self.dt_min_accel.get() - self.time_eps):
            # correct accel
            # out: ['x_accel', 'W_accel', 'beta_accel', 'r_accel', 'r_std_accel', 'error_code'])
            # in: ['x', 'W', 'y_b', 'g', 'omega_b', 'std_accel', 'std_accel_omega', 'beta_accel_c']
            start = time.thread_time()
            self.x, self.W, beta_accel, r_accel, r_std_accel, accel_ret = self.eqs[
                "correct_accel"
            ](
                self.x,
                self.W,
                msg.data["accel"],
                self.g.get(),
                omega,
                self.std_accel.get(),
                self.std_accel_omega.get(),
                self.beta_accel_c.get(),
            )
            cpu_accel = time.thread_time() - start

            uros.check_nan(
                locals(),
                "{:s} accel correction".format(self.name),
                t,
                [
                    "self.x",
                    "self.W",
                    "beta_accel",
                    "r_accel",
                    "r_std_accel",
                    "accel_ret",
                ],
            )

            self.msg_est_status.data["beta_accel"] = beta_accel
            self.msg_est_status.data["r_accel"][: r_accel.shape[0]] = np.array(
                r_accel
            ).T
            self.msg_est_status.data["r_std_accel"][: r_accel.shape[0]] = np.array(
                r_std_accel
            ).T
            self.msg_est_status.data["accel_ret"] = accel_ret
            self.msg_est_status.data["cpu_accel"] = cpu_accel
            self.t_last_accel = t

        # publish vehicle state
        self.msg_att.data["time"] = t
        self.msg_att.data["q"] = np.array(q).T
        self.msg_att.data["r"] = np.array(r).T
        self.msg_att.data["b"] = np.array(b_g).T
        self.msg_att.data["omega"] = np.array(omega).T
        self.pub_att.publish(self.msg_att)

        # publish estimator status
        self.msg_est_status.data["time"] = t
        self.msg_est_status.data["n_x"] = self.n_x
        self.msg_est_status.data["x"][: self.n_x] = np.array(self.x).T
        W_vect = np.reshape(np.array(self.W)[np.diag_indices(self.n_e)], -1)
        self.msg_est_status.data["W"][: len(W_vect)] = W_vect
        self.msg_est_status.data["cpu_predict"] = cpu_predict
        self.pub_est.publish(self.msg_est_status)
