from dataclasses import dataclass
from typing import List

from beartype import beartype

import numpy as np
import pyulog
import simpy

from cyecca.sim import msgs, uros
from cyecca.sim.uros import Publisher


@beartype
class LogEvent:
    def __init__(self, timestamp: np.uint64, index: int, topic: pyulog.ULog.Data):
        self.timestamp = timestamp
        self.index = index
        self.topic = topic

    def get(self, name):
        data = np.nan
        try:
            data = self.topic.data[name][self.index]
        except KeyError as e:
            raise KeyError(e, "valid fields:", self.topic.data.keys())
        except Exception as e:
            print(e)
        return data

    def get_array(self, name, n):
        data = np.zeros(n)
        data.fill(np.nan)
        try:
            data = np.array(
                [
                    self.topic.data["{:s}[{:d}]".format(name, i)][self.index]
                    for i in range(n)
                ]
            )
        except KeyError as e:
            raise KeyError(e, "valid fields", self.topic.data.keys())
        except Exception as e:
            print(e)
        return data


@beartype
class ULogReplay:
    def __init__(self, core: uros.Core, ulog_file: str):
        with open(ulog_file, "rb") as f:
            ulog = pyulog.ULog(f)

        # annotate types for python editors
        topic_list = ulog.data_list  # type: List[pyulog.ULog.Data]
        event_list = []

        # sort every publication up front by timestamp, might be a faster way to do this, but
        # very quick during actual sim
        for topic in topic_list:
            for i, timestamp in enumerate(topic.data["timestamp"]):
                event_list.append(LogEvent(timestamp=timestamp, index=i, topic=topic))
        event_list.sort(key=lambda x: x.timestamp)

        self.ignored_events = [
            "vehicle_air_data",
            "vehicle_rates_setpoint",
            "vehicle_attitude_setpoint",
            "rate_ctrl_status",
            "rate_ctrl_status",
            "actuator_controls_0",
            "vehicle_local_position",
            "vehicle_local_position_groundtruth",
            "vehicle_global_position",
            "vehicle_global_position_groundtruth",
            "vehicle_actuator_outputs",
            "vehicle_gps_position",
            "vehicle_local_position_setpoint",
            "actuator_outputs",
            "battery_status",
            "manual_control_setpoint",
            "vehicle_land_detected",
            "telemetry_status",
            "vehicle_status_flags",
            "vehicle_status",
            "sensor_preflight",
            "vehicle_command",
            "commander_state",
            "actuator_armed",
            "sensor_selection",
            "input_rc",
            "ekf2_innovations",
            "system_power",
            "radio_status",
            "cpuload",
            "ekf_gps_drift",
            "home_position",
            "mission_result",
            "position_setpoint_triplet",
            "ekf2_timestamps",
        ]

        self.pubs = {}
        for topic in topic_list:
            if topic.name in self.ignored_events:
                pass
            elif topic.name == "sensor_combined":
                self.pubs[topic.name] = Publisher(core, "imu", msgs.Imu)
            elif topic.name == "vehicle_magnetometer":
                self.pubs[topic.name] = Publisher(core, "mag", msgs.Mag)
            elif topic.name == "estimator_status":
                self.pubs[topic.name] = Publisher(
                    core, "log_status", msgs.EstimatorStatus
                )
            elif topic.name == "vehicle_attitude_groundtruth":
                self.pubs[topic.name] = Publisher(
                    core, "ground_truth_attitude", msgs.Attitude
                )
                print("ground truth attitude is published")
            elif topic.name == "vehicle_attitude":
                self.pubs[topic.name] = Publisher(core, "log_attitude", msgs.Attitude)
            else:
                print("unhandled init for event", topic.name)

        # class members
        self.core = core  # type: simpy.Environment
        self.event_list = event_list  # type: List[LogEvent]

        # start process
        self.core.process(self.run())

    def run(self):
        index = 0
        t0 = self.event_list[0].timestamp / 1e6
        while index < len(self.event_list):
            event = self.event_list[index]  # type: LogEvent
            t = event.timestamp / 1.0e6 - t0

            wait = t - self.core.now
            assert wait >= 0
            yield self.core.timeout(wait)

            # data message
            m = None
            name = event.topic.name

            if name in self.ignored_events:
                pass
            elif name == "sensor_combined":
                m = msgs.Imu()
                m.data["time"] = t
                m.data["gyro"] = event.get_array("gyro_rad", 3)
                m.data["accel"] = event.get_array("accelerometer_m_s2", 3)
            elif name == "vehicle_magnetometer":
                m = msgs.Mag()
                m.data["time"] = t
                m.data["mag"] = event.get_array("magnetometer_ga", 3)
            elif name in ["vehicle_attitude", "vehicle_attitude_groundtruth"]:
                m = msgs.Attitude()
                m.data["time"] = t
                m.data["q"] = event.get_array("q", 4)
                m.data["omega"][0] = event.get("rollspeed")
                m.data["omega"][1] = event.get("pitchspeed")
                m.data["omega"][2] = event.get("yawspeed")
            elif name == "estimator_status":
                m = msgs.EstimatorStatus()
                m.data["time"] = t
                n_states = event.get("n_states")
                m.data["x"][:n_states] = event.get_array("states", n_states)
                m.data["W"][:n_states] = event.get_array("covariances", n_states)
                for i in range(n_states):
                    if m.data["W"][i] > 0:
                        m.data["W"][i] = np.sqrt(m.data["W"][i])
                m.data["beta_mag"] = event.get("mag_test_ratio")
            else:
                print("unhandled pub for event", name)

            if m is not None:
                pub = self.pubs[name]
                pub.publish(m)
                # print('publishing:', log.topic.name, 'to:', pub.topic, 'data:', m)

            index += 1
