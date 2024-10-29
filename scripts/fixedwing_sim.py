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

        # -------------------------------------------------------
        # Dynamics
        # ----------------------------------------------
        dynamics = fixedwing
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
        self.state = np.array(list(self.x0_dict.values()), dtype=float)
        self.p = np.array(list(self.p_dict.values()), dtype=float)
        # self.get_logger().info(f"p: {self.p}")
        self.u = np.zeros(1, dtype=float)


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
    

    def integrate_simulation(self):
        """
        Integrate the simulation one step and calculate measurements
        """
        self.u = ca.vertcat(
            float(self.input_aetr[2]),
            float(self.input_aetr[0]),
            float(self.input_aetr[1]),
            float(self.input_aetr[3])
            )
        try:
            # opts = {"abstol": 1e-9,"reltol":1e-9,"fsens_err_con": True,"calc_ic":True,"calc_icB":True}
            f_int = ca.integrator(
                "test", "cvodes", self.model["dae"], self.t, self.t + self.dt
            )
            res = f_int(x0=self.state, z0=0.0, p=self.p, u=self.u)
        except RuntimeError as e:
            print(e)
            raise e

        x1 = np.array(res["xf"]).reshape(-1)
        if not np.all(np.isfinite(x1)):
            print("integration not finite")
            raise RuntimeError("nan in integration")
        
        # ---------------------------------------------------------------------
        # store states and measurements
        # ---------------------------------------------------------------------
        self.state = np.array(res["xf"]).reshape(-1)
        # self.x = self.state[0] 
        # self.y = self.state[1]
        # self.z = self.state[2]
        # self.vx = self.state[3]
        # self.vy = self.state[4]
        # self.vz = self.state[5]

        # self.get_logger().info(f"x: {self.x:0.2f}, vx: {self.vx:0.2f}")

        # self.get_logger().info(f"F_b: {self.state[1]}")


        self.publish_state()

    def timer_callback(self):
        self.integrate_simulation()
        # self.get_logger().info(f"fx: {fx:0.2f}, fz: {fz:0.2f}")
        self.publish_state()

    def get_state_by_name(self, name):
        return self.state[self.model["x_index"][name]]

    def publish_state(self):
        x = self.get_state_by_name("position_w_0")
        y = self.get_state_by_name("position_w_1")
        z = self.get_state_by_name("position_w_2")

        wx = self.get_state_by_name("omega_wb_b_0")
        wy = self.get_state_by_name("omega_wb_b_1")
        wz = self.get_state_by_name("omega_wb_b_2")

        vx = self.get_state_by_name("velocity_b_0")
        vy = self.get_state_by_name("velocity_b_1")
        vz = self.get_state_by_name("velocity_b_2")

        qw = self.get_state_by_name("quat_wb_0")
        qx = self.get_state_by_name("quat_wb_1")
        qy = self.get_state_by_name("quat_wb_2")
        qz = self.get_state_by_name("quat_wb_3")

        # Logger output
        # alpha = -1*float(ca.if_else(ca.fabs(vx) > 1e-1, ca.atan(vz/vx), ca.SX(0)))
        # self.get_logger().info(f"alpha: {alpha:0.3f}")

        self.get_logger().info(f"x: {x:0.2f}, y: {y:0.2f}, z: {z:0.2f},\n vx: {vx:0.2f},  vy: {vy:0.2} vz: {vz:0.2f}")

        # self.get_logger().info(f"x: {x:0.2f}, y: {y:0.2f}, z: {z:0.2f},\n vx: {vx:0.2f},  vy: {vy:0.2} vz: {vz:0.2f}")
        # self.get_logger().info(f"wx: {wx:0.2f} wy: {wy:0.2f} wz: {wz:0.2f}")
        # self.get_logger().info(f"qw: {qw:0.2f} qx: {qx:0.2f} qy: {qy:0.2f} qz{qz:0.2f}")

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
