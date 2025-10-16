#!/usr/bin/env python3
from cyecca.models import fixedwing_4ch

import casadi as ca
import numpy as np

import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Point

from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Time


class Simulator(Node):
    def __init__(self, x0=None, p=None):
        # ----------------------------------------------
        # ROS2 node setup
        # ----------------------------------------------
        param_list = [Parameter("use_sim_time", Parameter.Type.BOOL, True)]
        super().__init__("simulator", parameter_overrides=param_list)

        # ----------------------------------------------
        # parameters
        # ----------------------------------------------
        self.declare_parameter("mocap_vehicle_id", "/sim")
        self.declare_parameter("frame_id", "/map")

        # ----------------------------------------------
        # publications
        # ----------------------------------------------
        self.pub_pose = self.create_publisher(
            PoseWithCovarianceStamped,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/pose",
            1,
        )
        self.pub_clock = self.create_publisher(
            Clock,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/clock",
            1,
        )
        self.pub_odom = self.create_publisher(
            Odometry,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/odom",
            1,
        )
        self.pub_lift = self.create_publisher(
            Marker,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/lift",
            1,
        )
        self.pub_drag = self.create_publisher(
            Marker,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/drag",
            1,
        )
        self.pub_weight = self.create_publisher(
            Marker,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/weight",
            1,
        )
        self.pub_thrust = self.create_publisher(
            Marker,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/thrust",
            1,
        )
        self.pub_side_force = self.create_publisher(
            Marker,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/side_force",
            1,
        )
        self.pub_vel_b = self.create_publisher(
            Marker,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/vel_b",
            1,
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # ----------------------------------------------
        # subscriptions
        # ----------------------------------------------
        self.sub_joy = self.create_subscription(
            Joy,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/joy",
            self.joy_callback,
            1,
        )
        self.sub_auto_joy = self.create_subscription(
            Joy,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/auto_joy",
            self.auto_joy_callback,
            1,
        )

        self.input_aetr = ca.DM.zeros(4)  # aileron, elevator, thrust, rudder
        self.input_auto = ca.DM.zeros(4)  # aileron, elevator, thrust, rudder

        self.t = 0.0
        self.dt = 0.01
        self.real_time_factor = 1.0

        # -------------------------------------------------------
        # mode handling
        # ----------------------------------------------
        self.input_mode = "auto"
        self.pub_mode = self.create_publisher(
            String,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/flight_mode",
            1,
        )
        self.pub_mode_marker = self.create_publisher(
            Marker,
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
            + "/mode_marker",
            1,
        )

        # -------------------------------------------------------
        # Dynamics
        # ----------------------------------------------
        self.model = fixedwing_4ch.derive_model()
        self.publish_static_wheel_frames()
        self.x0_dict = self.model["x0_defaults"]
        if x0 is not None:
            for k in x0.keys():
                if not k in self.x0_dict.keys():
                    raise KeyError(k)
                self.x0_dict[k] = x0[k]
        self.p_dict = self.model["p_defaults"]
        # print(self.p_dict)
        if p is not None:
            for k in p.keys():
                if not k in self.p_dict.keys():
                    raise KeyError(k)
                self.p_dict[k] = p[k]

        # init state (x), param(p), and input(u)
        self.state = np.array(list(self.x0_dict.values()), dtype=float)
        self.p = np.array(
            [
                self.p_dict[str(self.model["p"][i])]
                for i in range(self.model["p"].shape[0])
            ],
            dtype=float,
        )
        self.u = np.zeros(4, dtype=float)
        # start main loop on timer
        self.system_clock = rclpy.clock.Clock(
            clock_type=rclpy.clock.ClockType.SYSTEM_TIME
        )
        self.sim_timer = self.create_timer(
            timer_period_sec=self.dt / self.real_time_factor,
            callback=self.timer_callback,
            clock=self.system_clock,
        )
        self.Info = None

    # Manual Joy AETR
    def joy_callback(self, msg: Joy):
        self.input_aetr = ca.vertcat(
            -msg.axes[3],  # aileron
            msg.axes[4],  # elevator
            msg.axes[1],  # thrust
            msg.axes[0],  # rudder
        )

        new_mode = self.input_mode
        if msg.buttons[0] == 1:
            new_mode = "auto"
        elif msg.buttons[1] == 1:
            new_mode = "manual"

        if new_mode != self.input_mode:
            self.get_logger().info(
                "mode changed from: %s to %s" % (self.input_mode, new_mode)
            )
            self.input_mode = new_mode

    # Auto joy commands accepts AETR
    def auto_joy_callback(self, msg: Joy):
        self.input_auto = ca.vertcat(
            msg.axes[0],  # aileron
            -msg.axes[1],  # elevator
            msg.axes[2],  # thrust
            msg.axes[3],  # rudder
        )

    def publish_flight_mode(self):
        self.pub_mode.publish(String(data=self.input_mode))

        m = Marker()
        m.header.frame_id = (
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
        )
        m.header.stamp = Time()
        m.ns = "flight_mode"
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.text = self.input_mode.upper()
        m.scale.z = 0.12
        m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 1.0, 1.0, 0.95)
        m.pose.orientation.w = 1.0
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.25
        self.pub_mode_marker.publish(m)

    def clock_as_msg(self):
        msg = Clock()
        msg.clock.sec = int(self.t)
        msg.clock.nanosec = int(1e9 * (self.t - msg.clock.sec))
        return msg

    def update_controller(self):
        # ---------------------------------------------------------------------
        # mode handling
        # ---------------------------------------------------------------------

        # 4 Channel Airplane

        if self.input_mode == "manual":
            input = self.input_aetr
        elif self.input_mode == "auto":
            input = self.input_auto
        else:
            self.get_logger().info("unhandled mode: %s" % self.input_mode)
            input = ca.vertcat(float(0), float(0), float(0), float(0))

        self.u = ca.vertcat(  # alocate AETR mode
            float(input[0]),
            float(input[1]),
            float(input[2]),
            float(input[3]),
        )

    def integrate_simulation(self):
        """
        Integrate the simulation one step and calculate measurements
        """
        RAD2DEG = 180 / ca.pi

        try:

            opts = {}
            xdot = self.model["f"](self.state, self.u, self.p)

            method = "cvodes"

            if method == "rk":
                opts = {}
            elif method == "idas":
                opts = {
                    "abstol": 1e-9,
                    "reltol": 1e-9,
                    "fsens_err_con": True,
                    "calc_ic": True,
                    "calc_icB": True,
                }
            elif method == "cvodes":
                opts = {"abstol": 1e-2, "reltol": 1e-6, "fsens_err_con": True}
            else:
                raise ValueError("unknown integration method: %s" % method)

            f_int = ca.integrator(
                "test", method, self.model["dae"], self.t, self.t + self.dt, opts
            )
            res = f_int(x0=self.state, z0=0.0, p=self.p, u=self.u)

        except RuntimeError as e:
            print(e)
            xdot = self.model["f"](x=self.state, u=self.u, p=self.p)
            print(xdot)
            raise e

        x1 = np.array(res["xf"]).reshape(-1)
        if not np.all(np.isfinite(x1)):
            print("integration not finite")
            raise RuntimeError("nan in integration")

        # ---------------------------------------------------------------------
        # store states and measurements
        # ---------------------------------------------------------------------
        self.state = np.array(res["xf"]).reshape(-1)
        self.Info = self.model["Info"](x=self.state, u=self.u, p=self.p)

        # self.publish_state()

    def timer_callback(self):
        self.update_controller()  # Controller
        self.integrate_simulation()  # Integrator
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
        tf.header.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )
        tf.child_frame_id = (
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
        )
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
        # publish tf2 transform vector lift
        # ------------------------------------
        def vector(v, name, color, scale):
            marker = Marker()
            marker.header.frame_id = (
                self.get_parameter("mocap_vehicle_id")
                .get_parameter_value()
                .string_value
            )
            marker.ns = name
            marker.id = 0
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.header.stamp = Time()  # msg_clock.clock
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.scale.x = 0.05
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            p1 = Point()
            p1.x = 0.0
            p1.y = 0.0
            p1.z = 0.0
            p2 = Point()
            p2.x = float(v[0] * scale)
            p2.y = float(v[1] * scale)
            p2.z = float(v[2] * scale)
            marker.points.append(p1)
            marker.points.append(p2)
            return marker

        self.pub_lift.publish(
            vector(self.Info["L_b"], "lift", [1.0, 0.0, 0.0, 1.0], 1.0)
        )
        self.pub_drag.publish(
            vector(self.Info["D_b"], "drag", [0.0, 0.0, 1.0, 1.0], 1.0)
        )
        self.pub_weight.publish(
            vector(self.Info["FW_b"], "weight", [0.0, 1.0, 0.0, 1.0], 1.0)
        )
        self.pub_thrust.publish(
            vector(self.Info["FT_b"], "thrust", [1.0, 0.0, 1.0, 1.0], 1.0)
        )
        self.pub_side_force.publish(
            vector(self.Info["C_b"], "side_force", [1.0, 1.0, 0.0, 1.0], 1.0)
        )

        self.pub_vel_b.publish(
            vector(self.Info["v_b"], "vel_b", [0.0, 1.0, 1.0, 1.0], 0.1)
        )
        # print("alpha", self.Info["alpha"], "CL", self.Info["CL"], "CD", self.Info["CD"])
        # ------------------------------------
        # publish pose with covariance stamped
        # ------------------------------------
        msg_pose = PoseWithCovarianceStamped()
        msg_pose.header.stamp = msg_clock.clock
        msg_pose.header.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )
        # msg_pose.pose.covariance = P_pose_full.reshape(-1)
        msg_pose.pose.pose.position.x = x
        msg_pose.pose.pose.position.y = y
        msg_pose.pose.pose.position.z = z
        msg_pose.pose.pose.orientation.w = qw
        msg_pose.pose.pose.orientation.x = qx
        msg_pose.pose.pose.orientation.y = qy
        msg_pose.pose.pose.orientation.z = qz
        self.pub_pose.publish(msg_pose)

        # ------------------------------------
        # publish odom
        # ------------------------------------
        msg_odom = Odometry()
        msg_odom.header.stamp = msg_clock.clock
        msg_odom.header.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )
        msg_odom.child_frame_id = (
            self.get_parameter("mocap_vehicle_id").get_parameter_value().string_value
        )
        # msg_pose.pose.covariance = P_pose_full.reshape(-1)
        msg_odom.pose.pose.position.x = x
        msg_odom.pose.pose.position.y = y
        msg_odom.pose.pose.position.z = z
        msg_odom.pose.pose.orientation.w = qw
        msg_odom.pose.pose.orientation.x = qx
        msg_odom.pose.pose.orientation.y = qy
        msg_odom.pose.pose.orientation.z = qz
        self.pub_odom.publish(msg_odom)

        self.publish_flight_mode()

    def publish_static_wheel_frames(self):
        """
        Publish static TFs for three-wheel configuration relative to body frame. Wheel positions are defined in the dynamics model as offsets from body-frame base.
        """

        def sx_to_xyz(sx_vec):
            return [float(sx_vec[0]), float(sx_vec[1]), float(sx_vec[2])]

        wheels_pos = [
            ("left_main_wheel", self.model["left_wheel_b"]),
            ("right_main_wheel", self.model["right_wheel_b"]),
            ("tail_wheel", self.model["tail_wheel_b"]),
        ]

        tfs = []
        for child, pos_b in wheels_pos:
            x, y, z = sx_to_xyz(pos_b)
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = (
                self.get_parameter("mocap_vehicle_id")
                .get_parameter_value()
                .string_value
            )
            tf.child_frame_id = child
            tf.transform.translation.x = x
            tf.transform.translation.y = y
            tf.transform.translation.z = z
            tf.transform.rotation.w = 1.0
            tf.transform.rotation.x = 0.0
            tf.transform.rotation.y = 0.0
            tf.transform.rotation.z = 0.0
            tfs.append(tf)

        self.static_tf_broadcaster.sendTransform(tfs)


def main(args=None):
    try:
        rclpy.init(args=args)
        sim = Simulator()
        rclpy.spin(sim)
    except KeyboardInterrupt as e:
        print(e)


if __name__ == "__main__":
    main()
