# ROS 2 Launch Files

This directory contains ROS 2 launch configurations for running cyecca simulations with visualization and teleoperation support.

## Available Launch Files

### `rdd2_sim.xml`
Complete quadrotor simulation with RDD2 controller
- **Nodes:**
  - `rdd2_sim.py` - Quadrotor dynamics + RDD2 cascaded controller
  - `joy_node` - Joystick input for manual control
  - `static_transform_publisher` - TF tree (base_link → camera_link)
- **Topics:**
  - `/pose` - Quadrotor pose with covariance
  - `/odom` - Odometry (position, velocity, orientation)
  - `/imu` - Simulated IMU data
  - `/joy` - Joystick commands
  - `/cerebri/in/bezier_trajectory` - Trajectory setpoints
- **Features:**
  - Configurable log levels per node
  - Simulation time support
  - Trajectory tracking via Bezier curves
- **Usage:**
  ```bash
  ros2 launch cyecca rdd2_sim.xml
  ros2 launch cyecca rdd2_sim.xml sim_log_level:=DEBUG
  ```

### `fixedwing_sim.xml`
Fixed-wing aircraft simulation (HH Sport Cub S2)
- **Nodes:**
  - `fixedwing_sim.py` - 4-channel fixed-wing dynamics
  - `joy_node` - RC transmitter input
  - `rviz2` - 3D visualization
- **Topics:**
  - `/{vehicle}/pose` - Aircraft pose
  - `/{vehicle}/odom` - Odometry
  - `/{vehicle}/lift`, `/drag`, `/thrust`, `/weight`, `/side_force` - Force visualization markers
  - `/{vehicle}/clock` - Simulation clock
- **Parameters:**
  - `mocap_vehicle_id` - Namespace (default: "sim")
  - `frame_id` - Global frame (default: "map")
  - `rviz_config` - Path to RViz config
- **Usage:**
  ```bash
  ros2 launch cyecca fixedwing_sim.xml
  ros2 launch cyecca fixedwing_sim.xml mocap_vehicle_id:=cub01 log_level:=INFO
  ```

### `viewer.xml`
Standalone RViz2 viewer for RDD2 simulation
- **Nodes:**
  - `robot_state_publisher` - Publishes robot URDF/TF
  - `rviz2` - Visualization with custom config
- **Features:**
  - Loads `rdd2.urdf` from `rdd2_description` package
  - Minimal TF buffering for low latency (0.01s buffer, 0.0s tolerance)
  - Supports simulation and real-time modes
- **Usage:**
  ```bash
  ros2 launch cyecca viewer.xml
  ros2 launch cyecca viewer.xml use_sim_time:=false
  ros2 launch cyecca viewer.xml rviz_config:=/path/to/custom.rviz
  ```

### `viewer_nvp.xml`
Lightweight RViz2 viewer for Night Vapor (NVP) aircraft
- **Nodes:**
  - `rviz2` - Visualization only
- **Features:**
  - Zero TF interpolation delay (tf_buffer_length:=0.0)
  - Custom config for fixed-wing visualization
- **Usage:**
  ```bash
  ros2 launch cyecca viewer_nvp.xml
  ```

## ROS 2 Simulation Scripts

### `scripts/rdd2_sim.py`
Quadrotor simulation node with integrated RDD2 controller
- **Dynamics:** Uses `cyecca.models.quadrotor`
- **Controllers:**
  - Position control → Velocity control → Attitude control → Attitude rate control
  - Auto-level mode and velocity mode
  - Strapdown INS propagation for state estimation
- **Inputs:**
  - Joystick (`/joy`) - Manual flight control
  - Bezier trajectory (`/cerebri/in/bezier_trajectory`) - Autonomous waypoint tracking
- **Outputs:**
  - Pose, odometry, twist (velocity)
  - IMU sensor data
  - Pose history path
  - TF transforms (map → base_link)
- **Integration:** RK4 integrator with configurable time step
- **State:** 10-state (position, velocity, quaternion)

### `scripts/fixedwing_sim.py`
Fixed-wing aircraft simulation node
- **Dynamics:** Uses `cyecca.models.fixedwing_4ch` (4-channel control)
- **Inputs:**
  - Joystick axes mapped to [throttle, aileron, elevator, rudder]
- **Outputs:**
  - Pose and odometry
  - Force vector visualization (lift, drag, thrust, weight, sideforce)
  - Body-frame velocity vectors
  - TF transforms
- **Features:**
  - Aerodynamic force visualization as RViz markers
  - Configurable vehicle namespace
  - Real-time physics simulation

### `scripts/bag_play.py`
Minimal ROS 2 publisher/subscriber example (testing/template)

## Common Launch Arguments

All launch files support:
- `use_sim_time` - Enable ROS simulation time (default: true)
- `log_level` - Global or per-node logging (DEBUG, INFO, WARN, ERROR, FATAL)
- `rviz_config` - Path to custom RViz configuration file

## Integration with CogniPilot

These launch files are designed to work with the CogniPilot ecosystem:
- **synapse_msgs** - Bezier trajectory messages
- **rdd2_description** - URDF models for RDD2 quadrotor
- **cerebri** - High-level mission planning and trajectory generation

## Visualization

RViz2 configurations visualize:
- Robot model (URDF meshes)
- TF tree (coordinate frames)
- Odometry paths
- Force vectors (fixed-wing only)
- Camera frames
- IMU orientation

## Development

To add new simulations:
1. Create dynamics model in `cyecca/models/`
2. Create simulation script in `scripts/` with ROS 2 publishers/subscribers
3. Add launch file in `launch/` following existing patterns
4. Create or modify RViz config in `config/`
5. Update this README with node descriptions and usage
