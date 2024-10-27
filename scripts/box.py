#!/usr/bin/env python3
from cyecca.models import fixedwing

import casadi as ca
import numpy as np

import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped

from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Joy
from tf2_ros import TransformBroadcaster


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
        self.pub_clock = self.create_publisher(Clock, "clock", 1)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ----------------------------------------------
        # subscriptions
        # ----------------------------------------------
        self.sub_joy = self.create_subscription(Joy, "/joy", self.joy_callback, 1)

        self.input_aetr = ca.vertcat(0.0, 0.0, 0.0, 0.0)
        self.t = 0.0
        self.dt = 0.01
        self.real_time_factor = 1.0
        self.x = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vz = 0.0

        self.m = 0.1
        self.rho = 1.225
        self.g = 9.8

        # start main loop on timer
        self.system_clock = rclpy.clock.Clock(
            clock_type=rclpy.clock.ClockType.SYSTEM_TIME
        )
        self.sim_timer = self.create_timer(
            timer_period_sec=self.dt / self.real_time_factor,
            callback=self.timer_callback,
            clock=self.system_clock,
        )



    def joy_callback(self, msg: Joy):
        self.input_aetr = ca.vertcat(
            -msg.axes[3],  # aileron
            msg.axes[4],  # elevator
            msg.axes[1],  # thrust
            msg.axes[0],  # rudder
        )

    def clock_as_msg(self):
        msg = Clock()
        msg.clock.sec = int(self.t)
        msg.clock.nanosec = int(1e9 * (self.t - msg.clock.sec))
        return msg

    def timer_callback(self):
        # self.integrate_simulation()
        # self.get_logger().info("in timer")
        fx = float(self.input_aetr[2]-self.vx)
        self.vx += (fx/self.m)*self.dt
        self.x += self.vx*self.dt

        q = 0.5 * self.rho * self.vx**2
        S = 1
        cl = 1.8
        if self.z < 0:
            ground = -self.z * 10 - self.vz * 10
        else:
            ground = 0
        L = cl * q * S 
        fz = L - self.m * self.g + ground
        self.vz += (fz/self.m)*self.dt
        self.z += self.vz*self.dt

        self.get_logger().info(f"x: {self.x:0.2f}, z: {self.z:0.2f}, vx: {self.vx:0.2f}, vz: {self.vz:0.2f}")
        self.get_logger().info(f"fx: {fx:0.2f}, fz: {fz:0.2f}")


        self.publish_state()


    def publish_state(self):
        x= self.x
        y= 0.0
        z= self.z
        qx=0.0
        qy=0.0
        qz=0.0
        qw=1.0

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


        # ------------------------------------
        # publish pose with covariance stamped
        # ------------------------------------
        msg_pose = PoseWithCovarianceStamped()
        msg_pose.header.stamp = msg_clock.clock
        msg_pose.header.frame_id = "map"
        # msg_pose.pose.covariance = P_pose_full.reshape(-1)
        msg_pose.pose.pose.position.x = x
        msg_pose.pose.pose.position.y = y
        msg_pose.pose.pose.position.z = z
        msg_pose.pose.pose.orientation.w = qw
        msg_pose.pose.pose.orientation.x = qx
        msg_pose.pose.pose.orientation.y = qy
        msg_pose.pose.pose.orientation.z = qz
        self.pub_pose.publish(msg_pose)

def main(args=None):
    try:
        rclpy.init(args=args)
        sim = Simulator()
        rclpy.spin(sim)
    except KeyboardInterrupt as e:
        print(e)


if __name__ == "__main__":
    main()
