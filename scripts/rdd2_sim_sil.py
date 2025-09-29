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

    def update_sil(self):
        pass

    def timer_callback(self):
        self.integrate_simulation()
        self.publish_state()
        self.update_sil()

    def get_state_by_name(self, name):
        return self.x[self.model["x_index"][name]]

    def get_param_by_name(self, name):
        return self.p[self.model["p_index"][name]]

    def publish_state(self):
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


def main(args=None):
    try:
        rclpy.init(args=args)
        sim = Simulator()
        rclpy.spin(sim)
    except KeyboardInterrupt as e:
        print(e)


if __name__ == "__main__":
    main()
