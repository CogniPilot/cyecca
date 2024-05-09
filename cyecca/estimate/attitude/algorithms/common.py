import os

import casadi as ca
from casadi.tools.graph import graph

import cyecca.util as util
from cyecca.lie.group_so3 import *
from cyecca.lie.group_rn import *
from cyecca.lie.direct_product import *

# misc variables
omega_t = ca.SX.sym("omega_t", 3, 1)  # angular velocity in body frame, true
omega_m = ca.SX.sym("omega_m", 3, 1)  # angular velocity in body frame, measured
t = ca.SX.sym("t")  # time
dt = ca.SX.sym("dt")  # integration time step
std_mag = ca.SX.sym("std_mag")
std_gyro = ca.SX.sym("std_gyro")
std_accel = ca.SX.sym("std_accel")
sn_gyro_rw = ca.SX.sym("sn_gyro_rw")
std_accel_omega = ca.SX.sym("std_accel_omega")

# constants
mag_decl = ca.SX.sym("mag_decl")
mag_incl = ca.SX.sym("mag_incl")  # only useful for sim, neglected in correction
mag_str = ca.SX.sym("mag_str")  # mag field strength
beta_mag_c = ca.SX.sym(
    "beta_mag_c"
)  # normalizes beta mag so that 1 represents exceeding thresh
beta_accel_c = ca.SX.sym(
    "beta_accel_c"
)  # normalizes beta mag so that 1 represents exceeding thresh
g = ca.SX.sym("g")
deg2rad = ca.pi / 180

# noise, mean zero, variance 1
w_mag = ca.SX.sym("w_mag", 3, 1)
w_gyro = ca.SX.sym("w_gyro", 3, 1)
w_gyro_rw = ca.SX.sym("w_gyro_rw", 3, 1)
w_accel = ca.SX.sym("w_accel", 3, 1)

std_gyro_rw = sn_gyro_rw / ca.sqrt(dt)
Q = ca.diag(
    ca.vertcat(std_gyro, std_gyro, std_gyro, std_gyro_rw, std_gyro_rw, std_gyro_rw) ** 2
)
W0 = ca.diag([1, 1, 1, 5e-2, 5e-2, 5e-2])

e1 = ca.SX([1, 0, 0])
e2 = ca.SX([0, 1, 0])
e3 = ca.SX([0, 0, 1])
