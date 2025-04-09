#!/usr/bin/env python3
from cyecca.models import fixedwing, lookupTableFixedwing

# from cyecca.models.lookupTableFixedwing import build_tables
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
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker


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
        self.pub_odom = self.create_publisher(Odometry, "odom", 1)
        self.pub_lift = self.create_publisher(Marker, "lift", 1)
        self.pub_drag = self.create_publisher(Marker, "drag", 1)
        self.pub_weight = self.create_publisher(Marker, "weight", 1)
        self.pub_thrust = self.create_publisher(Marker, "thrust", 1)
        self.pub_side_force = self.create_publisher(Marker, "side_force", 1)
        self.pub_vel_b = self.create_publisher(Marker, "vel_b", 1)

        self.tf_broadcaster = TransformBroadcaster(self)

        # ----------------------------------------------
        # subscriptions
        # ----------------------------------------------
        self.sub_joy = self.create_subscription(Joy, "/joy", self.joy_callback, 1)
        self.sub_auto_joy = self.create_subscription(
            Joy, "/auto_joy", self.auto_joy_callback, 1
        )

        self.input_aetr = ca.vertcat(0.0, 0.0, 0.0)
        self.input_auto = ca.vertcat(0.0, 0.0, 0.0)

        self.t = 0.0
        self.dt = 0.01
        self.real_time_factor = 1.0

        # -------------------------------------------------------
        # mode handling
        # ----------------------------------------------
        self.input_mode = "manual"

        # -------------------------------------------------------
        #  Lookup Table
        # -------------------------------------------------------

        self.lookup_tab = lookupTableFixedwing.build_tables()
        self.coeff_data = {
            "CL": 0.0,
            "CD": 0.0,
            "Cl": 0.0,
            "Cm": 0.0,
            "Cn": 0.0,
            "Cmdr": 0.0,
            "Cmda": 0.0,
        }
        # -------------------------------------------------------
        # Dynamics
        # ----------------------------------------------
        dynamics = fixedwing
        self.model = dynamics.derive_model(coeff_data=self.coeff_data)
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
        # new_control_mode = self.control_mode
        if msg.buttons[0] == 1:
            new_mode = "auto"
        elif msg.buttons[1] == 1:
            new_mode = "manual"

        if new_mode != self.input_mode:
            self.get_logger().info(
                "mode changed from: %s to %s" % (self.input_mode, new_mode)
            )
            self.input_mode = new_mode

    # Auto Joy TAER
    def auto_joy_callback(self, msg: Joy):
        self.input_auto = ca.vertcat(
            msg.axes[0],  # thrust
            # msg.axes[1],  # aileron
            -msg.axes[2],  # elevator -- positive = elevator down (pitch up)
            msg.axes[1],  # rudder
        )  # TER

    def clock_as_msg(self):
        msg = Clock()
        msg.clock.sec = int(self.t)
        msg.clock.nanosec = int(1e9 * (self.t - msg.clock.sec))
        return msg

    def update_controller(self):
        # ---------------------------------------------------------------------
        # mode handling
        # ---------------------------------------------------------------------

        # # 4 Channel Airplane
        # if self.input_mode == "manual": #logitech f310
        #     self.u = ca.vertcat(  # TAER mode
        #         float(self.input_aetr[2]),
        #         float(self.input_aetr[0]), # Sets aileron into auto or manual
        #         float(self.input_aetr[1]),
        #         float(self.input_aetr[3]),
        #     )

        # NVP Mapped with Roll Control
        if self.input_mode == "manual":  # logitech f310
            self.u = ca.vertcat(  # TER mode 3-Channel
                float(self.input_aetr[2]),
                float(self.input_aetr[0]),
                float(self.input_aetr[1]),
                -1 * float(self.input_aetr[0]),
            )
            print("manual input", self.u)
        elif self.input_mode == "auto":
            self.u = ca.vertcat(  # TER mode 3-Channel
                float(self.input_auto[0]),
                float(self.input_auto[1]),
                float(self.input_auto[2]),
                float(self.input_auto[3]),  # Rudder directly affects yaw for nvp
            )
            print("auto input", self.u)
        else:
            self.get_logger().info("unhandled mode: %s" % self.input_mode)
            self.u = ca.vertcat(float(0), float(0), float(0), float(0))

    def integrate_simulation(self):
        """
        Integrate the simulation one step and calculate measurements
        """
        RAD2DEG = 180 / ca.pi

        try:
            # Grab from Look Up Table
            if self.Info == None:
                self.Info = {}
                self.Info["alpha"] = 0.0
                self.Info["beta"] = 0.0
                # self.Info["ail"] = 0.0
                self.Info["elev"] = 0.0
                self.Info["rud"] = 0.0

            else:
                self.coeff_data["CD"] = -1 * self.lookup_tab["Cx"](
                    self.Info["alpha"] * RAD2DEG, self.Info["elev"] * RAD2DEG
                )
                self.coeff_data["Cmdr"] = -1 * self.lookup_tab["DnDr"](
                    self.Info["alpha"] * RAD2DEG, self.Info["beta"] * RAD2DEG / 2
                )
                self.coeff_data["Cmda"] = -1 * self.lookup_tab["DlDa"](
                    self.Info["alpha"] * RAD2DEG, self.Info["beta"] * RAD2DEG / 2
                )
                self.coeff_data["CL"] = self.lookup_tab["Cz"](
                    self.Info["alpha"] * RAD2DEG,
                    self.Info["beta"] * RAD2DEG,
                    self.Info["elev"] * RAD2DEG,
                )
                self.coeff_data["Cl"] = -1 * self.lookup_tab["Cl"](
                    self.Info["alpha"] * RAD2DEG, self.Info["beta"] * RAD2DEG
                )
                self.coeff_data["Cm"] = -1 * self.lookup_tab["Cm"](
                    self.Info["alpha"] * RAD2DEG, self.Info["elev"] * RAD2DEG
                )
                self.coeff_data["Cn"] = -1 * self.lookup_tab["Cn"](
                    self.Info["alpha"] * RAD2DEG, self.Info["beta"] * RAD2DEG
                )

                dynamics = fixedwing
                # print("Cm", self.coeff_data["Cm"])
                self.model = dynamics.derive_model(coeff_data=self.coeff_data)

            # opts = {"abstol": 1e-9,"reltol":1e-9,"fsens_err_con": True,"calc_ic":True,"calc_icB":True}
            opts = {"abstol": 1e-2, "reltol": 1e-6, "fsens_err_con": True}
            xdot = self.model["f"](self.state, self.u, self.p)

            # RK4
            # f_int = ca.integrator(
            #     "test", "rk", self.model["dae"], {"t0": self.t, "tf": self.t + self.dt}
            # )
            # res = f_int(x0=self.state, z0=ca.DM.zeros(self.model["z"].size1()), p=self.p, u=self.u)

            # IDAS
            # f_int = ca.integrator(
            #     "test", "idas", self.model["dae"], {"t0": self.t, "tf": self.t + self.dt, **opts}
            # )
            # res = f_int(x0=self.state, z0=ca.DM.zeros(self.model["z"].size1()), p=self.p, u=self.u)

            # #CVODES
            f_int = ca.integrator(
                "test", "cvodes", self.model["dae"], self.t, self.t + self.dt, opts
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

        self.publish_state()

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
        # publish tf2 transform vector lift
        # ------------------------------------
        def vector(v, name, color, scale):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.ns = name
            marker.id = 0
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.header.stamp = msg_clock.clock
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
            vector(self.Info["W_b"], "weight", [0.0, 1.0, 0.0, 1.0], 1.0)
        )
        self.pub_thrust.publish(
            vector(self.Info["T_b"], "thrust", [1.0, 0.0, 1.0, 1.0], 1.0)
        )
        self.pub_side_force.publish(
            vector(self.Info["S_b"], "side_force", [1.0, 1.0, 0.0, 1.0], 1.0)
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

        # ------------------------------------
        # publish odom
        # ------------------------------------
        msg_odom = Odometry()
        msg_odom.header.stamp = msg_clock.clock
        msg_odom.header.frame_id = "map"
        msg_odom.child_frame_id = "base_link"
        # msg_pose.pose.covariance = P_pose_full.reshape(-1)
        msg_odom.pose.pose.position.x = x
        msg_odom.pose.pose.position.y = y
        msg_odom.pose.pose.position.z = z
        msg_odom.pose.pose.orientation.w = qw
        msg_odom.pose.pose.orientation.x = qx
        msg_odom.pose.pose.orientation.y = qy
        msg_odom.pose.pose.orientation.z = qz
        self.pub_odom.publish(msg_odom)


def main(args=None):
    try:
        rclpy.init(args=args)
        sim = Simulator()
        rclpy.spin(sim)
    except KeyboardInterrupt as e:
        print(e)


if __name__ == "__main__":
    main()
