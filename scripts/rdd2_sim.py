#!/usr/bin/env python3

from cyecca.models import quadrotor
from cyecca.models import rdd2, rdd2_loglinear, mr_ref_traj, bezier

import casadi as ca
import numpy as np

import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, PoseStamped
from geometry_msgs.msg import TwistWithCovarianceStamped, TwistStamped
from synapse_msgs.msg import BezierTrajectory
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Joy, Imu
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


class Simulator(Node):
    def __init__(self, x0=None, p=None):
        # ----------------------------------------------
        # ROS2 node setup
        # ----------------------------------------------
        param_list = [Parameter("use_sim_time", Parameter.Type.BOOL, True)]
        super().__init__("simulator", parameter_overrides=param_list)

        # ----------------------------------------------
        # publications
        # ----------------------------------------------
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, "pose", 1)
        self.pub_pose_sp = self.create_publisher(PoseStamped, "pose_sp", 1)
        self.pub_clock = self.create_publisher(Clock, "clock", 1)
        self.pub_odom = self.create_publisher(Odometry, "odom", 1)
        self.pub_twist_cov = self.create_publisher(
            TwistWithCovarianceStamped, "twist_cov", 1
        )
        self.pub_twist = self.create_publisher(TwistStamped, "twist", 1)
        self.pub_path = self.create_publisher(Path, "path", 1)
        self.pub_imu = self.create_publisher(Imu, "imu", 1)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # ----------------------------------------------
        # subscriptions
        # ----------------------------------------------
        self.sub_joy = self.create_subscription(Joy, "/joy", self.joy_callback, 1)
        self.sub_bezier = self.create_subscription(
            BezierTrajectory, "/cerebri/in/bezier_trajectory", self.bezier_callback, 1
        )

        # ----------------------------------------------
        # dynamics
        # ----------------------------------------------
        dynamics = quadrotor
        self.model = dynamics.derive_model()
        self.x0_dict = self.model["x0_defaults"]
        if x0 is not None:
            for k in x0.keys():
                if not k in self.x0_dict.keys():
                    raise KeyError(k)
                self.x0_dict[k] = x0[k]
        self.p_dict = self.model["p_defaults"]
        if p is not None:
            for k in p.keys():
                if not k in self.p_dict.keys():
                    raise KeyError(k)
                self.p_dict[k] = p[k]

        # init state (x), param(p), and input(u)
        self.x = np.array(list(self.x0_dict.values()), dtype=float)
        self.est_x = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=float)
        self.p = np.array(list(self.p_dict.values()), dtype=float)
        self.u = np.zeros(4, dtype=float)

        # ----------------------------------------------
        # casadi control/ estimation algorithms
        # ----------------------------------------------
        self.eqs = {}
        self.eqs.update(rdd2.derive_attitude_rate_control())
        self.eqs.update(rdd2.derive_attitude_control())
        self.eqs.update(rdd2.derive_velocity_control())
        self.eqs.update(rdd2.derive_position_control())
        self.eqs.update(rdd2.derive_input_auto_level())
        self.eqs.update(rdd2.derive_input_velocity())
        self.eqs.update(rdd2.derive_strapdown_ins_propagation())
        self.eqs.update(rdd2.derive_control_allocation())
        self.eqs.update(rdd2.derive_attitude_estimator())
        self.eqs.update(rdd2.derive_position_correction())
        self.eqs.update(rdd2.derive_common())
        self.eqs.update(rdd2_loglinear.derive_se23_error())
        self.eqs.update(rdd2_loglinear.derive_so3_attitude_control())
        self.eqs.update(rdd2_loglinear.derive_outerloop_control())
        self.eqs.update(mr_ref_traj.derive_mr_ref_traj())
        self.eqs.update(bezier.derive_multirotor())
        self.eqs.update(bezier.derive_ref())
        self.eqs.update(bezier.derive_eulerB321_to_quat())

        # ----------------------------------------------
        # sim state data
        # ----------------------------------------------
        self.path_len = 30
        self.t = 0.0
        self.dt = 1.0 / 100
        self.real_time_factor = 1.0

        self.pose_list = []
        self.motor_pose = np.zeros(4, dtype=float)
        self.msg_path = Path()
        self.input_aetr = np.zeros(4, dtype=float)
        self.input_mode = "velocity"
        self.control_mode = "mellinger"
        self.i0 = 0.0  # integrators for attitude rate loop
        self.e0 = np.zeros(3, dtype=float)  # error for attitude rate loop
        self.de0 = np.zeros(3, dtype=float)  # deriv of att error (for lowpass)

        # estimator data
        self.use_estimator = True  # if false, will use sim state instead for control
        self.P = 1e-2 * np.array([1, 0, 0, 1, 0, 1], dtype=float)  # state covariance
        self.Q = 1e-9 * np.array([1, 0, 0, 1, 0, 1], dtype=float)  # process noise
        self.P_temp = 1e-2 * np.eye(6, dtype=float)

        # velocity control data
        self.psi_sp = 0.0  # world yaw orientation set point
        self.psi_vel_sp = 0.0  # " velocity
        self.psi_acc_sp = 0.0  # " accel

        # control state (from estimator if use_estimator = True, else from sim)
        self.vb = np.zeros(3, dtype=float)  # velocity in body frame
        self.vw = np.zeros(3, dtype=float)  # velocity in world frame
        self.q = np.array([1, 0, 0, 0], dtype=float)  # quaternion

        # diff flat trajectory points
        self.pw_sp = np.zeros(3, dtype=float)  # pos in world sp
        self.vw_sp = np.zeros(3, dtype=float)  # vel "
        self.aw_sp = np.zeros(3, dtype=float)  # accel "
        self.jw_sp = np.zeros(3, dtype=float)  # jerk "
        self.sw_sp = np.zeros(3, dtype=float)  # snap "

        # setpoints
        self.q_sp = np.array([1, 0, 0, 0], dtype=float)  # quaternion setpoint
        self.qc_sp = np.array(
            [1, 0, 0, 0], dtype=float
        )  # quaternion camera setpoint (based on psi)
        self.z_i = 0  # z error integrator

        # start main loop on timer
        self.system_clock = rclpy.clock.Clock(
            clock_type=rclpy.clock.ClockType.SYSTEM_TIME
        )
        self.sim_timer = self.create_timer(
            timer_period_sec=self.dt / self.real_time_factor,
            callback=self.timer_callback,
            clock=self.system_clock,
        )

        # bezier
        self.bezier_msg = None
        self.PX = np.zeros(8)
        self.PY = np.zeros(8)
        self.PZ = np.zeros(8)
        self.Ppsi = np.zeros(4)

    def update_estimator(self):
        """
        Update the estimator
        """
        res = self.eqs["strapdown_ins_propagate"](
            self.est_x, self.y_accel, self.y_gyro, self.get_param_by_name("g"), self.dt
        )
        self.est_x = np.array(res, dtype=float).reshape(-1)

        X1, P1 = self.eqs["position_correction"](
            self.est_x, self.y_gps_pos, self.dt, self.P_temp
        )

        # print(X1)
        # print(P1)

        self.est_x = np.array(X1, dtype=float).reshape(-1)
        self.P_temp = np.array(P1, dtype=float)

        # self.P = np.array(
        #     self.eqs["attitude_covariance_propagation"](
        #         self.P, self.Q, self.y_gyro, self.dt
        #     )
        # ).reshape(-1)

        DECLANATION = 0
        # IDK what this is

        # new code
        temp_q = np.array(
            self.eqs["attitude_estimator"](
                self.q, self.y_mag, DECLANATION, self.y_gyro, self.y_accel, self.dt
            ),
            dtype=float,
        )

        self.est_x[6] = temp_q[0]
        self.est_x[7] = temp_q[1]
        self.est_x[8] = temp_q[2]
        self.est_x[9] = temp_q[3]
        self.q = np.array(
            [self.est_x[6], self.est_x[7], self.est_x[8], self.est_x[9]],
            dtype=float,
        )

        self.omega = self.y_gyro
        self.pw = np.array([self.est_x[0], self.est_x[1], self.est_x[2]], dtype=float)
        self.vw = np.array([self.est_x[3], self.est_x[4], self.est_x[5]], dtype=float)
        self.vb = np.array(
            self.eqs["rotate_vector_w_to_b"](self.q, self.vw), dtype=float
        ).reshape(-1)

    def update_controller(self):
        # ------------------------------------
        # control constants
        # ------------------------------------
        m = self.get_param_by_name("m")
        g = self.get_param_by_name("g")
        thrust_delta = 0.5 * m * g
        thrust_trim = m * g
        # TODO move to constant section
        F_max = 20
        l = self.get_param_by_name("l_motor_0")  # assuming all the same
        CM = self.get_param_by_name("CM")
        CT = self.get_param_by_name("CT")

        k_p_att = np.array([5, 5, 2], dtype=float)

        # attitude rate
        kp = np.array([0.3, 0.3, 0.05], dtype=float)
        ki = np.array([0, 0, 0], dtype=float)
        kd = np.array([0.1, 0.1, 0], dtype=float)
        f_cut = 10.0
        i_max = np.array([0, 0, 0], dtype=float)

        # ---------------------------------------------------------------------
        # differential flatness (reference trajectory -> reference signals)
        # ---------------------------------------------------------------------
        # [psi, psi_dot, psi_ddot, v_e, a_e, j_e, s_e, m, g, J_xx, J_yy, J_zz, J_xz],
        # [v_b, C_be, omega_eb_b, omega_dot_eb_b, M_b, T],
        self.eqs["mr_ref_traj"](
            self.psi_sp,
            self.psi_vel_sp,
            self.psi_acc_sp,
            self.vw_sp,
            self.aw_sp,
            self.jw_sp,
            self.sw_sp,
            m,
            g,
            self.get_param_by_name("Jx"),
            self.get_param_by_name("Jy"),
            self.get_param_by_name("Jz"),
            0,
        )

        # ---------------------------------------------------------------------
        # mode handling
        # ---------------------------------------------------------------------
        if self.input_mode == "acro":
            [omega_sp, thrust] = self.eqs["input_acro"](
                thrust_trim, thrust_delta, self.input_aetr
            )

        elif self.input_mode == "auto_level":
            [self.q_sp, thrust] = self.eqs["input_auto_level"](
                thrust_trim,
                thrust_delta,
                self.input_aetr,
                self.q,
            )
            omega_sp = self.eqs["attitude_control"](k_p_att, self.q, self.q_sp)

        elif self.input_mode == "velocity":

            vb_sp, self.psi_vel_sp = self.eqs["input_velocity"](self.input_aetr)

            reset_position = False
            [
                self.psi_sp,
                self.pw_sp,
                self.vw_sp,
                self.aw_sp,
                self.qc_sp,
            ] = self.eqs["velocity_control"](
                self.dt,
                self.psi_sp,
                self.pw_sp,
                self.pw,
                vb_sp,
                self.psi_vel_sp,
                reset_position,
            )
            if self.control_mode == "mellinger":
                [thrust, self.q_sp, self.z_i] = self.eqs["position_control"](
                    thrust_trim,
                    self.pw_sp,
                    self.vw_sp,
                    self.aw_sp,
                    self.qc_sp,
                    self.pw,
                    self.vw,
                    self.z_i,
                    self.dt,
                )
                omega_sp = self.eqs["attitude_control"](k_p_att, self.q, self.q_sp)
            elif self.control_mode == "loglinear":
                zeta = self.eqs["se23_error"](
                    self.pw,
                    self.vw,
                    self.q,
                    self.pw_sp,
                    self.vw_sp,
                    self.qc_sp,
                )
                # position control: world frame
                [thrust, self.q_sp, self.z_i] = self.eqs["se23_position_control"](
                    thrust_trim,
                    k_p_att,
                    zeta,
                    self.aw_sp,
                    self.qc_sp,
                    self.z_i,
                    self.dt,
                )
                # attitude control: q_br
                omega_sp = self.eqs["so3_attitude_control"](k_p_att, self.q, self.q_sp)
        elif self.input_mode == "bezier":
            time_start_nsec = (
                self.bezier_msg.time_start.sec * 1e9
                + self.bezier_msg.time_start.nanosec
            )
            time_stop_nsec = time_start_nsec
            now = self.get_clock().now()
            sec, nanosec = now.seconds_nanoseconds()
            time_nsec = sec * 1e9 + nanosec
            curve_idx = 0

            while True:
                curve = self.bezier_msg.curves[curve_idx]
                curve_prev = self.bezier_msg.curves[curve_idx - 1]
                curve_stop_nsec = curve.time_stop.sec * 1e9 + curve.time_stop.nanosec
                curve_prev_stop_nsec = (
                    curve_prev.time_stop.sec * 1e9 + curve_prev.time_stop.nanosec
                )
                if time_nsec < curve_stop_nsec:
                    time_stop_nsec = curve_stop_nsec
                    if curve_idx > 0:
                        time_start_nsec = curve_prev_stop_nsec
                    break
                curve_idx += 1
            if curve_idx < len(self.bezier_msg.curves):
                T = (time_stop_nsec - time_start_nsec) * 1e-9
                t = (time_nsec - time_start_nsec) * 1e-9
                for i in range(8):
                    self.PX[i] = self.bezier_msg.curves[curve_idx].x[i]
                    self.PY[i] = self.bezier_msg.curves[curve_idx].y[i]
                    self.PZ[i] = self.bezier_msg.curves[curve_idx].z[i]
                for i in range(4):
                    self.Ppsi[i] = self.bezier_msg.curves[curve_idx].yaw[i]
                [x, y, z, psi, dpsi, ddpsi, self.vw_sp, a, j, s] = self.eqs[
                    "bezier_multirotor"
                ](t, T, self.PX, self.PY, self.PZ, self.Ppsi)
                [_, q_att, self.omega, _, M, _] = self.eqs["f_ref"](
                    psi, dpsi, ddpsi, self.vw_sp, a, j, s
                )
                self.qc_sp = self.eqs["eulerB321_to_quat"](psi, 0, 0)
                self.pw_sp = np.array([x, y, z]).reshape(-1)
                print(
                    thrust_trim,
                    self.pw_sp,
                    self.vw_sp,
                    self.aw_sp,
                    self.qc_sp,
                    self.pw,
                    self.vw,
                    self.z_i,
                    self.dt,
                )
                [thrust, self.q_sp, self.z_i] = self.eqs["position_control"](
                    thrust_trim,
                    self.pw_sp,
                    self.vw_sp,
                    self.aw_sp,
                    self.qc_sp,
                    self.pw,
                    self.vw,
                    self.z_i,
                    self.dt,
                )
                omega_sp = self.eqs["attitude_control"](k_p_att, self.q, self.q_sp)

        else:
            self.get_logger().info("unhandled mode: %s" % self.input_mode)
            omega_sp = np.zeros(3, dtype=float)
            thrust = 0

        # ---------------------------------------------------------------------
        # attitude rate control
        # ---------------------------------------------------------------------
        # print(omega_sp, self.i0, self.e0, self.de0)
        M, i1, e1, de1, alpha = self.eqs["attitude_rate_control"](
            kp,
            ki,
            kd,
            f_cut,
            i_max,
            self.omega,
            omega_sp,
            self.i0,
            self.e0,
            self.de0,
            self.dt,
        )
        self.i0 = i1
        # self.get_logger().info('i0: %s' % self.i0)
        # self.get_logger().info('e0: %s' % self.e0)
        # self.get_logger().info('de0: %s' % self.de0)
        self.e0 = e1
        self.de0 = de1

        # ---------------------------------------------------------------------
        # control allocation
        # ---------------------------------------------------------------------
        self.u, Fp, Fm, Ft, Msat = self.eqs["f_alloc"](F_max, l, CM, CT, thrust, M)
        # self.get_logger().info("Ct: %s" % self.u)
        # self.get_logger().info('u: %s' % self.u)

    def joy_callback(self, msg: Joy):
        self.input_aetr = ca.vertcat(
            -msg.axes[3],  # aileron
            msg.axes[4],  # elevator
            msg.axes[1],  # thrust
            msg.axes[0],  # rudder
        )
        new_mode = self.input_mode
        new_control_mode = self.control_mode
        if msg.buttons[0] == 1:
            new_mode = "auto_level"
        elif msg.buttons[1] == 1:
            new_mode = "velocity"
        elif msg.buttons[2] == 1:
            # self.get_logger().info(
            #     "bezier mode not yet supported, reverted to %s" % self.input_mode
            # )
            new_mode = "bezier"
        if new_mode != self.input_mode:
            self.get_logger().info(
                "mode changed from: %s to %s" % (self.input_mode, new_mode)
            )
            self.input_mode = new_mode
        if msg.buttons[4] == 1:
            new_control_mode = "loglinear"
        elif msg.buttons[5] == 1:
            new_control_mode = "mellinger"
        if new_control_mode != self.control_mode:
            self.get_logger().info(
                "control mode changed from: %s to %s"
                % (self.control_mode, new_control_mode)
            )
            self.control_mode = new_control_mode

    def bezier_callback(self, msg: BezierTrajectory):
        self.bezier_msg = msg

    def publish_static_transforms(self):
        msg_clock = self.clock_as_msg()

        tf = TransformStamped()
        tf.header.frame_id = "base_link"
        tf.child_frame_id = "base_footprint"
        tf.header.stamp = msg_clock.clock
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = -0.22
        tf.transform.rotation.w = 1.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        self.tf_static_broadcaster.sendTransform(tf)

        tf = TransformStamped()
        tf.header.frame_id = "base_link"
        tf.child_frame_id = "camera_link"
        tf.header.stamp = msg_clock.clock
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
        tf.transform.rotation.w = 1.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        self.tf_static_broadcaster.sendTransform(tf)

        tf = TransformStamped()
        tf.header.frame_id = "camera_link"
        tf.child_frame_id = "camera_link_optical"
        tf.header.stamp = msg_clock.clock
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
        tf.transform.rotation.w = 1.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        self.tf_static_broadcaster.sendTransform(tf)

        tf = TransformStamped()
        tf.header.frame_id = "base_link"
        tf.child_frame_id = "lidar_link"
        tf.header.stamp = msg_clock.clock
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
        tf.transform.rotation.w = 1.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        self.tf_static_broadcaster.sendTransform(tf)

    def clock_as_msg(self):
        msg = Clock()
        msg.clock.sec = int(self.t)
        msg.clock.nanosec = int(1e9 * (self.t - msg.clock.sec))
        return msg

    def integrate_simulation(self):
        """
        Integrate the simulation one step and calculate measurements
        """
        try:
            # opts = {"abstol": 1e-9,"reltol":1e-9,"fsens_err_con": True,"calc_ic":True,"calc_icB":True}
            f_int = ca.integrator(
                "test", "cvodes", self.model["dae"], self.t, self.t + self.dt
            )
            res = f_int(x0=self.x, z0=0, p=self.p, u=self.u)
        except RuntimeError as e:
            print(e)
            xdot = self.model["f"](x=self.x, u=self.u, p=self.p)
            print(xdot, self.x, self.u, self.p)
            raise e

        x1 = np.array(res["xf"]).reshape(-1)
        if not np.all(np.isfinite(x1)):
            print("integration not finite")
            raise RuntimeError("nan in integration")

        # ---------------------------------------------------------------------
        # store states and measurements
        # ---------------------------------------------------------------------
        self.x = np.array(res["xf"]).reshape(-1)
        res["yf_gyro"] = self.model["g_gyro"](
            res["xf"], self.u, self.p, np.random.randn(3), self.dt
        )
        res["yf_accel"] = self.model["g_accel"](
            res["xf"], self.u, self.p, np.random.randn(3), self.dt
        )
        res["yf_mag"] = self.model["g_mag"](
            res["xf"], self.u, self.p, np.random.randn(3), self.dt
        )
        res["yf_gps_pos"] = self.model["g_gps_pos"](
            res["xf"], self.u, self.p, np.random.randn(3), self.dt
        )
        self.y_gyro = np.array(res["yf_gyro"]).reshape(-1)
        self.y_mag = np.array(res["yf_mag"]).reshape(-1)
        self.y_accel = np.array(res["yf_accel"]).reshape(-1)
        self.y_gps_pos = np.array(res["yf_gps_pos"]).reshape(-1)
        self.publish_state()

    def update_fake_estimator(self):
        # if not using estimator, use true states from sim
        self.q = np.array(
            [
                self.get_state_by_name("quaternion_wb_0"),
                self.get_state_by_name("quaternion_wb_1"),
                self.get_state_by_name("quaternion_wb_2"),
                self.get_state_by_name("quaternion_wb_3"),
            ]
        )
        self.omega = np.array(
            [
                self.get_state_by_name("omega_wb_b_0"),
                self.get_state_by_name("omega_wb_b_1"),
                self.get_state_by_name("omega_wb_b_2"),
            ],
            dtype=float,
        )
        self.pw = np.array(
            [
                self.get_state_by_name("position_op_w_0"),
                self.get_state_by_name("position_op_w_1"),
                self.get_state_by_name("position_op_w_2"),
            ],
            dtype=float,
        )
        self.vb = np.array(
            [
                self.get_state_by_name("velocity_w_p_b_0"),
                self.get_state_by_name("velocity_w_p_b_1"),
                self.get_state_by_name("velocity_w_p_b_2"),
            ],
            dtype=float,
        )
        self.vw = self.eqs["rotate_vector_b_to_w"](self.q, self.vb)

    def timer_callback(self):
        self.integrate_simulation()
        self.publish_state()
        if self.use_estimator:
            # print("updating")
            self.update_estimator()
        else:
            self.update_fake_estimator()
            # print("wrong")
        self.update_controller()

    def get_state_by_name(self, name):
        return self.x[self.model["x_index"][name]]

    def get_param_by_name(self, name):
        return self.p[self.model["p_index"][name]]

    def publish_state(self):
        x = self.get_state_by_name("position_op_w_0")
        y = self.get_state_by_name("position_op_w_1")
        z = self.get_state_by_name("position_op_w_2")

        wx = self.get_state_by_name("omega_wb_b_0")
        wy = self.get_state_by_name("omega_wb_b_1")
        wz = self.get_state_by_name("omega_wb_b_2")

        vx = self.get_state_by_name("velocity_w_p_b_0")
        vy = self.get_state_by_name("velocity_w_p_b_1")
        vz = self.get_state_by_name("velocity_w_p_b_2")

        qw = self.get_state_by_name("quaternion_wb_0")
        qx = self.get_state_by_name("quaternion_wb_1")
        qy = self.get_state_by_name("quaternion_wb_2")
        qz = self.get_state_by_name("quaternion_wb_3")

        m0 = self.get_state_by_name("omega_motor_0")
        m1 = self.get_state_by_name("omega_motor_1")
        m2 = self.get_state_by_name("omega_motor_2")
        m3 = self.get_state_by_name("omega_motor_3")
        motors = np.array([m0, m1, m2, m3])

        P_full = np.array(
            [
                [self.P[0], self.P[1], self.P[2]],
                [self.P[1], self.P[3], self.P[4]],
                [self.P[2], self.P[4], self.P[5]],
            ]
        )
        P_pose_full = np.block(
            [[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), P_full]]
        )

        # ------------------------------------
        # publish simulation clock
        # ------------------------------------
        self.t += self.dt
        msg_clock = self.clock_as_msg()
        self.pub_clock.publish(msg_clock)

        # ------------------------------------
        # publish tf2 transform
        # ------------------------------------
        tf = TransformStamped()
        tf.header.frame_id = "map"
        tf.child_frame_id = "base_link"
        tf.header.stamp = msg_clock.clock
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
        tf.transform.rotation.w = qw
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        self.tf_broadcaster.sendTransform(tf)

        # publish motor tf2 transforms to see spin
        for i in range(self.model["n_motor"]):
            theta = self.get_param_by_name("theta_motor_" + str(i))
            r = self.get_param_by_name("l_motor_" + str(i))
            dir = self.get_param_by_name("dir_motor_" + str(i))
            tf = TransformStamped()
            tf.header.frame_id = "base_link"
            tf.child_frame_id = "motor_{:d}".format(i)
            tf.header.stamp = msg_clock.clock
            tf.transform.translation.x = r * np.cos(theta)
            tf.transform.translation.y = r * np.sin(theta)
            tf.transform.translation.z = 0.02
            self.motor_pose[i] += motors[i] * self.dt
            tf.transform.rotation.w = np.cos(self.motor_pose[i] / 2)
            tf.transform.rotation.x = 0.0
            tf.transform.rotation.y = 0.0
            tf.transform.rotation.z = dir * np.sin(self.motor_pose[i] / 2)
            self.tf_broadcaster.sendTransform(tf)

        tf = TransformStamped()
        tf.header.frame_id = "map"
        tf.child_frame_id = "base_link_est"
        tf.header.stamp = msg_clock.clock
        tf.transform.translation.x = self.est_x[0]
        tf.transform.translation.y = self.est_x[1]
        tf.transform.translation.z = self.est_x[2]
        tf.transform.rotation.w = self.est_x[6]
        tf.transform.rotation.x = self.est_x[7]
        tf.transform.rotation.y = self.est_x[8]
        tf.transform.rotation.z = self.est_x[9]
        self.tf_broadcaster.sendTransform(tf)

        # ------------------------------------
        # publish imu
        # ------------------------------------
        msg_imu = Imu()
        msg_imu.header.frame_id = "base_link"
        msg_imu.header.stamp = msg_clock.clock
        msg_imu.angular_velocity.x = self.y_gyro[0]
        msg_imu.angular_velocity.y = self.y_gyro[1]
        msg_imu.angular_velocity.z = self.y_gyro[2]
        msg_imu.angular_velocity_covariance = np.eye(3).reshape(-1)
        msg_imu.linear_acceleration.x = self.y_accel[0]
        msg_imu.linear_acceleration.y = self.y_accel[1]
        msg_imu.linear_acceleration.z = self.y_accel[2]
        msg_imu.linear_acceleration_covariance = np.eye(3).reshape(-1)
        self.pub_imu.publish(msg_imu)

        # ------------------------------------
        # publish pose with covariance stamped
        # ------------------------------------
        msg_pose = PoseWithCovarianceStamped()
        msg_pose.header.stamp = msg_clock.clock
        msg_pose.header.frame_id = "map"
        msg_pose.pose.covariance = P_pose_full.reshape(-1)
        msg_pose.pose.pose.position.x = x
        msg_pose.pose.pose.position.y = y
        msg_pose.pose.pose.position.z = z
        msg_pose.pose.pose.orientation.w = qw
        msg_pose.pose.pose.orientation.x = qx
        msg_pose.pose.pose.orientation.y = qy
        msg_pose.pose.pose.orientation.z = qz
        self.pub_pose.publish(msg_pose)

        # ------------------------------------
        # publish pose sp
        # ------------------------------------
        msg_pose_sp = PoseStamped()
        msg_pose_sp.header.stamp = msg_clock.clock
        msg_pose_sp.header.frame_id = "map"
        msg_pose_sp.pose.position.x = float(self.pw_sp[0])
        msg_pose_sp.pose.position.y = float(self.pw_sp[1])
        msg_pose_sp.pose.position.z = float(self.pw_sp[2])
        msg_pose_sp.pose.orientation.w = float(self.qc_sp[0])
        msg_pose_sp.pose.orientation.x = float(self.qc_sp[1])
        msg_pose_sp.pose.orientation.y = float(self.qc_sp[2])
        msg_pose_sp.pose.orientation.z = float(self.qc_sp[3])
        self.pub_pose_sp.publish(msg_pose_sp)

        # ------------------------------------
        # publish odometry
        # ------------------------------------
        msg_odom = Odometry()
        msg_odom.header.stamp = msg_clock.clock
        msg_odom.header.frame_id = "map"
        msg_odom.child_frame_id = "base_link"
        msg_odom.pose.covariance = P_full.reshape(-1)
        msg_odom.pose.pose.position.x = self.est_x[0]
        msg_odom.pose.pose.position.y = self.est_x[1]
        msg_odom.pose.pose.position.z = self.est_x[2]
        msg_odom.twist.twist.linear.x = self.vb[0]
        msg_odom.twist.twist.linear.y = self.vb[1]
        msg_odom.twist.twist.linear.z = self.vb[2]
        msg_odom.pose.pose.orientation.w = self.est_x[6]
        msg_odom.pose.pose.orientation.x = self.est_x[7]
        msg_odom.pose.pose.orientation.y = self.est_x[8]
        msg_odom.pose.pose.orientation.z = self.est_x[9]
        msg_odom.twist.covariance = np.eye(6).reshape(-1)
        msg_odom.twist.twist.angular.x = self.y_gyro[0]
        msg_odom.twist.twist.angular.y = self.y_gyro[1]
        msg_odom.twist.twist.angular.z = self.y_gyro[2]

        self.pub_odom.publish(msg_odom)

        # ------------------------------------
        # publish twist with covariance stamped
        # ------------------------------------
        msg_twist_cov = TwistWithCovarianceStamped()
        msg_twist_cov.header.stamp = msg_clock.clock
        msg_twist_cov.header.frame_id = "base_link"
        msg_twist_cov.twist.covariance = np.eye(6).reshape(-1)
        msg_twist_cov.twist.twist.angular.x = wx
        msg_twist_cov.twist.twist.angular.y = wy
        msg_twist_cov.twist.twist.angular.z = wz
        msg_twist_cov.twist.twist.linear.x = vx
        msg_twist_cov.twist.twist.linear.y = vy
        msg_twist_cov.twist.twist.linear.z = vz
        self.pub_twist_cov.publish(msg_twist_cov)

        # ------------------------------------
        # publish twist
        # ------------------------------------
        msg_twist = TwistStamped()
        msg_twist.header.stamp = msg_clock.clock
        msg_twist.header.frame_id = "base_link"
        msg_twist.twist.angular.x = wx
        msg_twist.twist.angular.y = wy
        msg_twist.twist.angular.z = wz
        msg_twist.twist.linear.x = vx
        msg_twist.twist.linear.y = vy
        msg_twist.twist.linear.z = vz
        self.pub_twist.publish(msg_twist)

        # ------------------------------------
        # publish path message of previous poses
        # ------------------------------------
        self.msg_path.header.stamp = msg_clock.clock
        self.msg_path.header.frame_id = "map"
        pose = PoseStamped()
        pose.pose = msg_pose.pose.pose
        pose.header = msg_pose.header
        self.msg_path.poses.append(pose)
        while len(self.msg_path.poses) > self.path_len:
            self.msg_path.poses.pop(0)
        self.pub_path.publish(self.msg_path)


def main(args=None):
    try:
        rclpy.init(args=args)
        sim = Simulator()
        rclpy.spin(sim)
    except KeyboardInterrupt as e:
        print(e)


if __name__ == "__main__":
    main()
