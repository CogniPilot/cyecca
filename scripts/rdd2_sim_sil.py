#!/usr/bin/env python3

from cyecca.models import quadrotor
from cyecca.models import rdd2, rdd2_loglinear, mr_ref_traj, bezier

import casadi as ca
import numpy as np

import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import TransformStamped, PoseStamped, TwistStamped
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

import synapse_pb


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
        self.pub_clock = self.create_publisher(Clock, "clock", 1)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pub_pose = self.create_publisher(PoseStamped, "pose", 1)
        self.pub_twist = self.create_publisher(TwistStamped, "twist", 1)

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

        # start main loop on timer
        self.system_clock = rclpy.clock.Clock(
            clock_type=rclpy.clock.ClockType.SYSTEM_TIME
        )
        self.sim_timer = self.create_timer(
            timer_period_sec=self.dt / self.real_time_factor,
            callback=self.timer_callback,
            clock=self.system_clock,
        )

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
        # publish pose with covariance stamped
        # ------------------------------------
        msg_pose = PoseStamped()
        msg_pose.header.stamp = msg_clock.clock
        msg_pose.header.frame_id = "map"
        msg_pose.pose.position.x = x
        msg_pose.pose.position.y = y
        msg_pose.pose.position.z = z
        msg_pose.pose.orientation.w = qw
        msg_pose.pose.orientation.x = qx
        msg_pose.pose.orientation.y = qy
        msg_pose.pose.orientation.z = qz
        self.pub_pose.publish(msg_pose)

        # ------------------------------------
        # publish twist with covariance stamped
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


def main(args=None):
    try:
        rclpy.init(args=args)
        sim = Simulator()
        rclpy.spin(sim)
    except KeyboardInterrupt as e:
        print(e)


if __name__ == "__main__":
    main()
