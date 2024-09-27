#!/usr/bin/env cyecca_python
import quadrotor

import casadi as ca
import numpy as np

import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, PoseStamped
from geometry_msgs.msg import TwistWithCovarianceStamped, TwistStamped
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Joy
from tf2_ros import TransformBroadcaster

class Simulator(Node):

    def __init__(self, x0=None, p=None):
        param_list = [
            Parameter('use_sim_time', Parameter.Type.BOOL, True)
        ]
        super().__init__('simulator', parameter_overrides=param_list)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, 'pose', 1)
        self.pub_clock = self.create_publisher(Clock, 'clock', 1)
        self.pub_odom = self.create_publisher(Odometry, 'odom', 1)
        self.pub_twist_cov = self.create_publisher(TwistWithCovarianceStamped, 'twist_cov', 1)
        self.pub_twist = self.create_publisher(TwistStamped, 'twist', 1)
        self.pub_path = self.create_publisher(Path, 'path', 1)
        self.sub_joy = self.create_subscription(Joy, '/joy', self.joy_callback, 1)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.path_len = 30

        self.t = 0
        self.dt = 1.0/100
        self.real_time_factor = 1
        self.system_clock = rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.SYSTEM_TIME)
        self.sim_timer = self.create_timer(timer_period_sec=self.dt/self.real_time_factor, callback=self.timer_callback, clock=self.system_clock)
        self.pose_list = []

        self.msg_path = Path()

        # dynamics
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

        self.x = self.x0_dict.values()
        print(self.x)
        self.p = self.p_dict.values()
        self.u = np.array([0, 0, 0, 0])

    def clock_as_msg(self):
        msg = Clock()
        msg.clock.sec = int(self.t)
        msg.clock.nanosec = int(1e9*(self.t - msg.clock.sec))
        return msg

    def joy_callback(self, msg: Joy):
        rudder = msg.axes[0]
        throttle = msg.axes[1]
        aileron = -msg.axes[3]
        elevator = msg.axes[4]
        print('throttle: ', throttle)
        print('aileron: ', aileron)

        k_ail = 0.1
        k_elv = 0.1
        k_thr = 0.1
        k_rdr = 0.1
        mix_ail = k_ail * aileron
        mix_elv = k_elv * elevator
        mix_rdr = k_rdr * elevator
        mix_thr = k_thr * throttle + 0.57

        print(np.array([mix_ail, mix_elv, mix_thr]))
        self.u  = np.array([
            mix_thr + mix_ail - mix_elv,
            mix_thr - mix_ail + mix_elv,
            mix_thr - mix_ail - mix_elv,
            mix_thr + mix_ail + mix_elv,
        ])

    def timer_callback(self):
        T_trim = 0.7
        t = [self.t, self.t + self.dt]
        f_int = ca.integrator("test", "idas", self.model['dae'], self.t, self.t + self.dt)
        res = f_int(x0=self.x, z0=0, p=self.p, u=self.u)
        res["yf"] = self.model["g"](res["xf"], self.u, self.p)
        for k in ["xf", "yf", "zf"]:
            res[k] = np.array(res[k])

        x = res["xf"][self.model["x_index"]["position_op_w_0"], 0]
        y = res["xf"][self.model["x_index"]["position_op_w_1"], 0]
        z = res["xf"][self.model["x_index"]["position_op_w_2"], 0]
        wx = res["xf"][self.model["x_index"]["omega_wb_b_0"], 0]
        wy = res["xf"][self.model["x_index"]["omega_wb_b_1"], 0]
        wz = res["xf"][self.model["x_index"]["omega_wb_b_2"], 0]
        vx = res["xf"][self.model["x_index"]["velocity_w_p_b_0"], 0]
        vy = res["xf"][self.model["x_index"]["velocity_w_p_b_1"], 0]
        vz = res["xf"][self.model["x_index"]["velocity_w_p_b_2"], 0]
        qw = res["xf"][self.model["x_index"]["quaternion_wb_0"], 0]
        qx = res["xf"][self.model["x_index"]["quaternion_wb_1"], 0]
        qy = res["xf"][self.model["x_index"]["quaternion_wb_2"], 0]
        qz = res["xf"][self.model["x_index"]["quaternion_wb_3"], 0]

        m0 = res["xf"][self.model["x_index"]["normalized_motor_0"], 0]
        m1 = res["xf"][self.model["x_index"]["normalized_motor_1"], 0]
        m2 = res["xf"][self.model["x_index"]["normalized_motor_2"], 0]
        m3 = res["xf"][self.model["x_index"]["normalized_motor_3"], 0]

        # print('m', np.array([m0, m1, m2, m3]))

        self.x = res["xf"]

        # publish clock
        self.t += self.dt
        sec = int(self.t)
        nanosec = int(1e9*(self.t - sec))
        msg_clock = self.clock_as_msg()
        self.pub_clock.publish(msg_clock)

        tf = TransformStamped()
        tf.header.frame_id = 'map'
        tf.child_frame_id = 'base_link'
        tf.header.stamp = msg_clock.clock
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
        tf.transform.rotation.w = qw
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        self.tf_broadcaster.sendTransform(tf)

        msg_pose = PoseWithCovarianceStamped()
        msg_pose.header.stamp = msg_clock.clock
        msg_pose.header.frame_id = 'map';
        msg_pose.pose.covariance = np.eye(6).reshape(-1)
        msg_pose.pose.pose.position.x = x
        msg_pose.pose.pose.position.y = y
        msg_pose.pose.pose.position.z = z
        msg_pose.pose.pose.orientation.w = qw
        msg_pose.pose.pose.orientation.x = qx
        msg_pose.pose.pose.orientation.y = qy
        msg_pose.pose.pose.orientation.z = qz
        self.pub_pose.publish(msg_pose)

        msg_odom = Odometry()
        msg_odom.header.stamp = msg_clock.clock
        msg_odom.header.frame_id = 'map'
        msg_odom.child_frame_id = 'base_link'
        msg_odom.pose.covariance = np.eye(6).reshape(-1)
        msg_odom.pose.pose.position.x = x
        msg_odom.pose.pose.position.y = y
        msg_odom.pose.pose.position.z = z
        msg_odom.pose.pose.orientation.w = qw
        msg_odom.pose.pose.orientation.x = qx
        msg_odom.pose.pose.orientation.y = qy
        msg_odom.pose.pose.orientation.z = qz
        msg_odom.twist.covariance = np.eye(6).reshape(-1)
        msg_odom.twist.twist.angular.x = wx
        msg_odom.twist.twist.angular.y = wy
        msg_odom.twist.twist.angular.z = wz
        msg_odom.twist.twist.linear.x = vx
        msg_odom.twist.twist.linear.y = vy
        msg_odom.twist.twist.linear.z = vz
        self.pub_odom.publish(msg_odom)

        msg_twist_cov = TwistWithCovarianceStamped()
        msg_twist_cov.header.stamp = msg_clock.clock
        msg_twist_cov.header.frame_id = 'base_link'
        msg_twist_cov.twist.covariance = np.eye(6).reshape(-1)
        msg_twist_cov.twist.twist.angular.x = wx
        msg_twist_cov.twist.twist.angular.y = wy
        msg_twist_cov.twist.twist.angular.z = wz
        msg_twist_cov.twist.twist.linear.x = vx
        msg_twist_cov.twist.twist.linear.y = vy
        msg_twist_cov.twist.twist.linear.z = vz
        self.pub_twist_cov.publish(msg_twist_cov)
  
        msg_twist = TwistStamped()
        msg_twist.header.stamp = msg_clock.clock
        msg_twist.header.frame_id = 'base_link'
        msg_twist.twist.angular.x = wx
        msg_twist.twist.angular.y = wy
        msg_twist.twist.angular.z = wz
        msg_twist.twist.linear.x = vx
        msg_twist.twist.linear.y = vy
        msg_twist.twist.linear.z = vz
        self.pub_twist.publish(msg_twist)

        self.msg_path.header.stamp = msg_clock.clock
        self.msg_path.header.frame_id = 'map'
        pose = PoseStamped()
        pose.pose = msg_pose.pose.pose
        pose.header = msg_pose.header
        self.msg_path.poses.append(pose)
        while len(self.msg_path.poses) > self.path_len:
            self.msg_path.poses.pop(0)
        self.pub_path.publish(self.msg_path)


def main(args=None):
    rclpy.init(args=args)
    sim = Simulator()
    rclpy.spin(sim)

if __name__ == '__main__':
    main()