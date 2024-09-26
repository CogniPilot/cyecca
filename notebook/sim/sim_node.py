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
from tf2_ros import TransformBroadcaster

class Simulator(Node):

    def __init__(self, x0=None, p=None):
        param_list = [
            Parameter('use_sim_time', Parameter.Type.BOOL, True)
        ]
        super().__init__('simulator', parameter_overrides=param_list)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, 'pose', 10)
        self.pub_clock = self.create_publisher(Clock, 'clock', 10)
        self.pub_odom = self.create_publisher(Odometry, 'odom', 10)
        self.pub_twist_cov = self.create_publisher(TwistWithCovarianceStamped, 'twist_cov', 10)
        self.pub_twist = self.create_publisher(TwistStamped, 'twist', 10)
        self.pub_path = self.create_publisher(Path, 'path', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.path_len = 30

        self.t = 0
        self.dt = 0.001
        self.real_time_factor = 2
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

    def clock_as_msg(self):
        msg = Clock()
        msg.clock.sec = int(self.t)
        msg.clock.nanosec = int(1e9*(self.t - msg.clock.sec))
        return msg

    def timer_callback(self):

        T_trim = 0.582324
        u = np.array([T_trim, T_trim, T_trim, T_trim])

        t = [self.t, self.t + self.dt]
        print(u)
        print(self.x)
        f_int = ca.integrator("test", "idas", self.model['dae'], t[0], t)
        res = f_int(x0=self.x, z0=0, p=self.p_dict.values(), u=u)
        res["p"] = self.p_dict
        res["yf"] = self.model["g"](res["xf"], u, self.p_dict.values())
        for k in ["xf", "yf", "zf"]:
            res[k] = np.array(res[k])

        x = res["xf"][self.model["x_index"]["position_op_w_0"], 1]
        y = res["xf"][self.model["x_index"]["position_op_w_1"], 1]
        z = res["xf"][self.model["x_index"]["position_op_w_2"], 1]
        wx = res["xf"][self.model["x_index"]["omega_wb_b_0"], 1]
        wy = res["xf"][self.model["x_index"]["omega_wb_b_1"], 1]
        wz = res["xf"][self.model["x_index"]["omega_wb_b_2"], 1]
        vx = res["xf"][self.model["x_index"]["velocity_w_p_b_0"], 1]
        vy = res["xf"][self.model["x_index"]["velocity_w_p_b_1"], 1]
        vz = res["xf"][self.model["x_index"]["velocity_w_p_b_2"], 1]
        qw = res["xf"][self.model["x_index"]["quaternion_wb_0"], 1]
        qx = res["xf"][self.model["x_index"]["quaternion_wb_1"], 1]
        qy = res["xf"][self.model["x_index"]["quaternion_wb_2"], 1]
        qz = res["xf"][self.model["x_index"]["quaternion_wb_3"], 1]
        print(qw, qx, qy, qz)
        self.x = res["xf"][:, 1]

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