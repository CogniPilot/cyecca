""" 
Based off from Casadi f16 model
https://github.com/jgoppert/casadi_f16/blob/master/f16.py
"""
import dataclasses
import numpy as np

import casadi as ca
import control

INTERP_DEFAULT = 'linear'
#INTERP_DEFAULT = 'bspline'
TABLE_CHECK_TOL = 1e-9  # need to increase if using bspline


def saturate(x, min_val, max_val):
    """
    A casadi function for saturation.
    """
    return ca.if_else(x < min_val, min_val, ca.if_else(x > max_val, max_val, x))

# tables, func = build_tables()


class CasadiDataClass:
    """
    A base class for dataclasses with casadi.
    """

    def __post_init__(self):
        self.__name_to_index = {}
        self.__index_to_name = {}
        for i, field in enumerate(self.fields()):
            self.__name_to_index[field.name] = i
            self.__index_to_name[i] = field.name

    @classmethod
    def fields(cls):
        return dataclasses.fields(cls)

    def to_casadi(self):
        return ca.vertcat(*self.to_tuple())

    def to_tuple(self):
        return dataclasses.astuple(self)

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_casadi(cls, v):
        return cls(*[v[i] for i in range(v.shape[0])])

    @classmethod
    def sym(cls, name):
        v = ca.MX.sym(name, len(cls.fields()))
        return cls(*[v[i] for i in range(v.shape[0])])

    def name_to_index(self, name):
        return self.__name_to_index[name]

    def index_to_name(self, index):
        return self.__index_to_name[index]


@dataclasses.dataclass
class State(CasadiDataClass):
    """The vehicle state."""
    VT: float = 0  # true velocity, (ft/s)
    alpha: float = 0  # angle of attack, (rad)
    beta: float = 0  # sideslip angle, (rad)
    phi: float = 0  # B321 roll angle, (rad)
    theta: float = 0  # B321 pitch angle, (rad)
    psi: float = 0  # B321 yaw angle, (rad)
    P: float = 0  # body roll rate, (rad/s)
    Q: float = 0  # body pitch rate, (rad/s)
    R: float = 0  # body yaw rate, (rad/s)
    p_N: float = 0  # north position, (m)
    p_E: float = 0  # east position, (m)
    alt: float = 0  # altitude, (m)
    power: float = 0  # power, (0-1)
    ail_deg: float = 0  # aileron position, (deg)
    elv_deg: float = 0  # elevator position, (deg)
    rdr_deg: float = 0  # rudder position, (deg)


@dataclasses.dataclass
class StateDot(CasadiDataClass):
    """The derivative of the vehicle state."""
    VT_dot: float = ca.MX(0)  # true velocity derivative, (ft/s^2)
    alpha_dot: float = ca.MX(0)  # angle of attack rate, (rad/s)
    beta_dot: float = ca.MX(0)  # sideslip rate, (rad/s)
    phi_dot: float = ca.MX(0)  # B321 roll rate, (rad/s)
    theta_dot: float = ca.MX(0)  # B321 pitch rate, (rad/s)
    psi_dot: float = ca.MX(0)  # B321 yaw rate, (rad/s)
    P_dot: float = ca.MX(0)  # body roll accel, (rad/s^2)
    Q_dot: float = ca.MX(0)  # body pitch accel, (rad/s^2)
    R_dot: float = ca.MX(0)  # body yaw accel, (rad/s^2)
    V_N: float = ca.MX(0)  # north velocity, (m/s)
    V_E: float = ca.MX(0)  # east velocity, (m/s)
    alt_dot: float = ca.MX(0)  # climb rate, (m/s)
    power_dot: float = ca.MX(0)  # power rate, (NA)
    ail_rate_dps: float = ca.MX(0)  # aileron rate, (deg/s)
    elv_rate_dps: float = ca.MX(0)  # elevator rate, (deg/s)
    rdr_rate_dps: float = ca.MX(0)  # rudder rate, (deg/s)


@dataclasses.dataclass
class Control(CasadiDataClass):
    """The control input."""
    thtl: float = 0  # throttle (0-1)
    ail_cmd_deg: float = 0  # aileron command, (deg)
    elv_cmd_deg: float = 0  # elevator command, (deg)
    rdr_cmd_deg: float = 0  # rudder command, (deg)


@dataclasses.dataclass
class Parameters(CasadiDataClass):
    """The constant parameters."""
    s: float = ca.MX(300.00)  # reference area, ft^2
    b: float = ca.MX(30.0)  # wing span, ft
    cbar: float = ca.MX(11.32)  # mean chord, ft
    xcgr: float = ca.MX(0.35)  # reference cg, %chord
    xcg: float = ca.MX(0.35)  # actual cg, %chord
    hx: float = ca.MX(160.0)
    g: float = ca.MX(32.17)  # acceleration of gravity, ft/s^2
    weight: float = ca.MX(20490.446)  # weight, slugs
    axx: float = ca.MX(9496.0)  # moment of inertia about x
    ayy: float = ca.MX(55814.0)  # moment of inertia about y
    azz: float = ca.MX(63100.0)  # moment of inertia about z
    axz: float = ca.MX(982.0)  # xz moment of inertia

@dataclasses.dataclass
class AeroProp(CasadiDataClass):
    """Aerodynamic and Propulsion Parameters"""
    Cl: float = ca.MX(-0.008) #Rolling moment coefficient
    Cm: float = ca.MX(0.186) #pitching moment coefficient
    Cn: float = ca.MX(0.018) #Yawing moment coefficient
    DlDa: float = ca.MX(0) # Rolling moment due to ailerons
    DlDr: float = ca.MX(0) # Rolling moment due to rudder
    DnDa: float = ca.MX(0) # Yawing moment due to ailerons
    DnDr: float = ca.MX(0) # Yawing moment due to rudder
    damp: float = ca.MX(0) # Damping Coefficient
    Cx: float = ca.MX(0) #x-axis aerodynamic force coefficient
    Cy: float = ca.MX(0) #y-axis aerodynamic force coefficient (sideforce)
    Cz: float = ca.MX(0) #z-axis aerodynamic force coefficient
    thrust_idl: float = ca.MX(1060) # idle thrust
    thrust_mil: float = ca.MX(12680) # thrust at "military" level
    thrust_max: float = ca.MX(20000) # maximum trust

@dataclasses.dataclass
class Damp(CasadiDataClass):
    """Damping Coefficient"""
    CXq: float = ca.MX(0.308) #CXq
    CYr: float = ca.MX(0.876) #CYr
    CYp: float = ca.MX(-0.188) #CYp
    CZq: float = ca.MX(-28.9)
    Clr: float = ca.MX(0.063)
    Clp: float = ca.MX(-0.443)
    Cmq: float = ca.MX(-5.23)
    Cnr: float = ca.MX(-0.378)
    Cnp: float = ca.MX(0.052)

def atmosphere(vt, alt):
    """Atmospheric Model"""
    ft2m = 0.3048 #feet to meter
    R0 = 2.377e-3
    temp_fac = 1.0 - 0.703e-5*alt/ft2m
    temp = ca.if_else(alt>35e3*ft2m, 390, 519.0*temp_fac)
    rho = R0*temp_fac**4.14
    mach = vt/ca.sqrt(1.4*1716.3*temp)
    qbar = 0.5*rho*vt**2
    return rho, mach, qbar# ca.MX(rho), ca.MX(mach), ca.MX(qbar)

def propulsion():
        dp = ca.MX.sym('dp')
        thtl = ca.MX.sym('thtl')
        power = ca.MX.sym('power')
        power_cmd = ca.MX.sym('power_cmd')

        # reciprocal of time constant
        rtau = ca.Function('rtau', [dp], [ca.if_else(dp < 25, 1, ca.if_else(dp > 50, 0.1, 1.9 - 0.036*dp))])

        # power command vs. throttle relationship
        tgear = ca.Function('tgear', [thtl],
                            [ca.if_else(thtl < 0.77, 64.94*thtl, 217.38*thtl - 117.38)],
                            ['thtl'], ['pow'])

        # rate of change of power
        pdot = ca.Function('pdot', [power, power_cmd], [
            ca.if_else(power_cmd > 50,
                       ca.if_else(power > 50, 5*(power_cmd - power), rtau(60 - power)*(60 - power)),
                       ca.if_else(power > 50, 5*(40 - power), rtau(power_cmd - power)*(power_cmd - power))
                       )
        ], ['power', 'power_cmd'], ['pdot'])
        # func['tgear'] = tgear
        # func['pdot'] = pdot
        return tgear, pdot
tgear, pdot = propulsion()


def force_moment(x: State, u: Control, p: Parameters):
    """
    The function computes the forces and moments acting on the aircraft.
    It is important to separate this from the dynamics as the Gazebo
    simulator will be used to simulate extra forces and moments
    from collision.
    """

    # functions
    cos = ca.cos
    sin = ca.sin

    # parameters
    weight = p.weight
    g = p.g
    hx = p.hx
    b = p.b
    cbar = p.cbar
    s = p.s
    xcg = p.xcg
    xcgr = p.xcgr

    # state
    VT = x.VT
    alpha = x.alpha
    beta = x.beta
    phi = x.phi
    theta = x.theta
    P = x.P
    Q = x.Q
    R = x.R
    alt = x.alt
    power = x.power
    ail_deg = x.ail_deg
    elv_deg = x.elv_deg
    rdr_deg = x.rdr_deg

    # mass properties
    mass = weight/g

    # air data computer and engine model
    rho, amach, qbar = atmosphere(VT, alt)
    # thrust = tables['thrust'](power, alt, amach)
    thrust = ca.if_else(power < 50,
        AeroProp.thrust_idl + (AeroProp.thrust_mil - AeroProp.thrust_idl)*power*0.02,
        AeroProp.thrust_mil + (AeroProp.thrust_max - AeroProp.thrust_mil)*(power-50)*0.02)

    # force component buildup
    rad2deg = 180/np.pi
    alpha_deg = rad2deg*alpha
    beta_deg = rad2deg*beta
    dail = ail_deg/20.0
    drdr = rdr_deg/30.0

    cxt = AeroProp.Cx
    cyt = AeroProp.Cy
    czt = AeroProp.Cz

    clt = ca.sign(beta_deg)*AeroProp.Cl \
        + AeroProp.DlDa*dail \
        + AeroProp.DlDr*drdr
    cmt = AeroProp.Cm
    cnt = ca.sign(beta_deg)*AeroProp.Cn \
        + AeroProp.DnDa*dail \
        + AeroProp.DnDr*drdr

    # damping
    tvt = 0.5/VT
    b2v = b*tvt
    cq = cbar*Q*tvt
    cxt += cq* Damp.CXq
    cyt += b2v*(Damp.CYr*R + Damp.CYp*P)
    czt += cq*Damp.CZq
    clt += b2v*(Damp.Clr*R + Damp.Clp*P)
    cmt += cq*Damp.Cmq + czt*(xcgr - xcg)
    cnt += b2v*(Damp.Cnr*R + Damp.Cnp*P) - cyt*(xcgr - xcg)*cbar/b

    # get ready for state equations
    sth = sin(theta)
    cth = cos(theta)
    sph = sin(phi)
    cph = cos(phi)
    qs = qbar*s
    qsb = qs*b
    rmqs = qs/mass
    gcth = g*cth
    ay = rmqs*cyt
    az = rmqs*czt
    qhx = Q*hx

    # force
    Fx = -mass*g*sth + qs*cxt + thrust
    Fy = mass*(gcth*sph + ay)
    Fz = mass*(gcth*cph + az)

    # moment
    Mx = qsb*clt  # roll
    My = qs*cbar*cmt - R*hx  # pitch
    Mz = qsb*cnt + qhx  # yaw

    return ca.vertcat(Fx, Fy, Fz), ca.vertcat(Mx, My, Mz)


def dynamics(x: State, u: Control, p: Parameters):
    """
    This function implements wind frame kinematics tied to the force and moment model.
    It does not take into account any collision forces.
    """

    Fb, Mb = force_moment(x, u, p)

    dx = StateDot()

    # functions
    cos = ca.cos
    sin = ca.sin

    # parameters
    weight = p.weight
    g = p.g
    axz = p.axz
    axzs = axz*axz
    axx = p.axx
    ayy = p.ayy
    azz = p.azz

    # state
    VT = x.VT
    alpha = x.alpha
    beta = x.beta
    phi = x.phi
    theta = x.theta
    psi = x.psi
    P = x.P
    Q = x.Q
    R = x.R
    power = x.power
    ail_deg = x.ail_deg
    rdr_deg = x.rdr_deg
    elv_deg = x.elv_deg

    # mass properties
    mass = weight/g
    xqr = azz*(azz - ayy) + axzs
    xpq = axz*(axx - ayy + azz)
    zpq = (axx - ayy)*axx + axzs
    gam = axx*azz - axzs
    ypr = azz - axx

    # get ready for state equations
    cbta = cos(beta)
    U = VT*cos(alpha)*cbta
    V = VT*sin(beta)
    W = VT*sin(alpha)*cbta
    sth = sin(theta)
    cth = cos(theta)
    sph = sin(phi)
    cph = cos(phi)
    spsi = sin(psi)
    cpsi = cos(psi)
    qsph = Q*sph

    pq = P*Q
    qr = Q*R

    power_cmd = tgear(u.thtl)
    dx.power_dot = pdot(power, power_cmd)

    # kinematics
    dx.phi_dot = P + (sth/cth)*(qsph + R*cph)
    dx.theta_dot = Q*cph - R*sph
    dx.psi_dot = (qsph + R*cph)/cth

    # force equations
    U_dot = R*V - Q*W + Fb[0]/mass
    V_dot = P*W - R*U + Fb[1]/mass
    W_dot = Q*U - P*V + Fb[2]/mass
    dum = U**2 + W**2

    dx.VT_dot = (U*U_dot + V*V_dot + W*W_dot)/VT
    dx.alpha_dot = (U*W_dot - W*U_dot) / dum
    dx.beta_dot = (VT*V_dot - V*dx.VT_dot)*cbta/dum

    dx.P_dot = (xpq*pq - xqr*qr + azz*Mb[0] + axz*Mb[2]) / gam
    dx.Q_dot = (ypr*P*R - axz*(P**2 - R**2) + Mb[1]) / ayy
    dx.R_dot = (zpq*pq - xpq*qr + axz*Mb[0] + axx*Mb[2]) / gam

    # navigation
    t1 = sph*cpsi
    t2 = cph*sth
    t3 = sph*spsi
    s1 = cth*cpsi
    s2 = cth*spsi
    s3 = t1*sth - cph*spsi
    s4 = t3*sth + cph*cpsi
    s5 = sph*cth
    s6 = t2*cpsi + t3
    s7 = t2*spsi - t1
    s8 = cph*cth

    dx.V_N = U*s1 + V*s3 + W*s6
    dx.V_E = U*s2 + V*s4 + W*s7
    dx.alt_dot = U*sth - V*s5 - W*s8

    # actuators
    ail_deg = saturate(x.ail_deg, -21.5, 21.5)
    elv_deg = saturate(x.elv_deg, -25.0, 25.0)
    rdr_deg = saturate(x.rdr_deg, -30.0, 30.0)

    def actuator_model(cmd, pos, rate_limit, pos_limit):
        rate = saturate(20.202*(cmd - pos), -rate_limit, rate_limit)
        return ca.if_else(rate < 0,
                          ca.if_else(pos < -pos_limit, 0, rate),
                          ca.if_else(pos > pos_limit, 0, rate))

    dx.ail_rate_dps = actuator_model(u.ail_cmd_deg, ail_deg, 60, 21.5)
    dx.elv_rate_dps = actuator_model(u.elv_cmd_deg, elv_deg, 60, 25.0)
    dx.rdr_rate_dps = actuator_model(u.rdr_cmd_deg, rdr_deg, 60, 30.0)

    return dx


def trim_actuators(x, u):
    """
    This function sets the actuator output to the actuator command.
    """
    x.power = tgear(u.thtl)
    x.ail_deg = u.ail_cmd_deg
    x.elv_deg = u.elv_cmd_deg
    x.rdr_deg = u.rdr_cmd_deg
    return x


def trim_cost(dx: StateDot):
    """
    Computes the trim cost based on the state derivative.
    """
    return dx.VT_dot**2 + \
        100*(dx.alpha_dot**2 + dx.beta_dot**2) + \
        10*(dx.P_dot**2 + dx.Q_dot**2 + dx.R_dot**2)


class StateSpace:
    """
    A convenience class for create state space representations
    easily and for creating subsystems based on the state names.
    The class keeps track of the state, input, and output vector
    component names.
    """

    def __init__(self, A, B, C, D, x, u, y=None, dt=None):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)
        self.dt = dt
        self.x = {xi: i for i, xi in enumerate(x)}
        self.u = {ui: i for i, ui in enumerate(u)}
        if y is None:
            y = x
        self.y = {yi: i for i, yi in enumerate(y)}

    def sub_system(self, x, u, y=None):
        xi = np.array([self.x[state] for state in x])
        ui = np.array([self.u[inp] for inp in u])
        if y is None:
            y = x
        yi = np.array([self.y[out] for out in y])

        A = self.A[xi].T[xi].T
        B = self.B[xi].T[ui].T
        C = self.C[yi].T[xi].T
        D = self.D[yi].T[ui].T
        return StateSpace(A, B, C, D, x, u, y, self.dt)

    def to_control(self):
        if self.dt is None:
            return control.ss(self.A, self.B, self.C, self.D)
        else:
            return control.ss(self.A, self.B, self.C, self.D, self.dt)

    def __str__(self):
        return 'A:\n{:s}\nB:\n{:s}\nC:\n{:s}\nD:\n{:s}\ndt:{:s}\nx:{:s}\nu:{:s}\ny:{:s}'.format(
            str(self.A), str(self.B), str(self.C), str(self.D),
            str(self.dt), str(self.x), str(self.u), str(self.y))

    __repr__ = __str__


def linearize(x0, u0, p0):
    """
    A function to perform linearizatoin of the f16 model

    Parameters:
    x0: state
    u0: input
    p0: parameters

    Returns:
    StateSpace: linearized system
    """
    x0 = x0.to_casadi()
    u0 = u0.to_casadi()  # Plot the compensated openloop bode plot

    x_sym = ca.MX.sym('x', x0.shape[0])
    u_sym = ca.MX.sym('u', u0.shape[0])
    x = State.from_casadi(x_sym)
    u = Control.from_casadi(u_sym)
    dx = dynamics(x, u, p0)
    A = ca.jacobian(dx.to_casadi(), x_sym)
    B = ca.jacobian(dx.to_casadi(), u_sym)
    f_A = ca.Function('A', [x_sym, u_sym], [A])
    f_B = ca.Function('B', [x_sym, u_sym], [B])
    A = f_A(x0, u0)
    B = f_B(x0, u0)
    n = A.shape[0]
    p = B.shape[1]
    C = np.eye(n)
    D = np.zeros((n, p))
    return StateSpace(A=A, B=B, C=C, D=D,
                      x=[f.name for f in x.fields()],
                      u=[f.name for f in u.fields()],
                      y=[f.name for f in x.fields()])


def trim(x: State, p: Parameters,
         phi_dot: float, theta_dot: float, psi_dot: float, gam: float, s0: np.array = None):
    """
    Trims the aircraft at the given conditions.

    Parameters:
    x: vehicle state
    p: parameters
    phi_dot: Body321 roll rate
    theta_dot: Body321 pitch rate
    psi_dot: Body321 yaw rate
    s0: the initial guess for the trim design vector

    Returns:
    State: x0
    Control: u0
    """
    if s0 is None:
        s0 = np.zeros(6)

    def constrain(x, s):
        u = Control(thtl=s[0], elv_cmd_deg=s[1], ail_cmd_deg=s[2], rdr_cmd_deg=s[3])
        alpha = s[4]
        beta = s[5]

        x = trim_actuators(x, u)

        x.alpha = alpha
        x.beta = beta

        cos = ca.cos
        sin = ca.sin
        tan = ca.tan
        atan = ca.arctan
        sqrt = ca.sqrt

        VT = x.VT
        g = p.g
        G = psi_dot*VT/g

        a = 1 - G*tan(alpha)*sin(beta)
        b = sin(gam)/cos(beta)
        c = 1 + G**2*cos(beta)**2

        # coordinated turn constraint pg. 188
        phi = atan(G*cos(beta)/cos(alpha) *
                   ((a - b**2) + b*tan(alpha)*sqrt(c*(1 - b**2) + G**2*sin(beta)**2))
                   / (a**2 - b**2*(1 + c*tan(alpha)**2)))
        x.phi = phi

        # rate of climb constraint pg. 187
        a = cos(alpha)*cos(beta)
        b = sin(phi)*sin(beta) + cos(phi)*sin(alpha)*cos(beta)
        theta = (a*b + sin(gam)*sqrt(a**2 - sin(gam)**2 + b**2)) \
            / (a**2 - sin(gam)**2)
        x.theta = theta

        # kinematics pg. 20
        x.P = phi_dot - sin(theta)*psi_dot
        x.Q = cos(phi)*phi_dot + sin(phi)*cos(theta)*psi_dot
        x.R = -sin(phi)*theta_dot + cos(phi)*cos(theta)*psi_dot

        return x, u

    s = ca.MX.sym('s', 6)
    x, u = constrain(x, s)
    f = trim_cost(dynamics(x, u, p))
    nlp = {'x': s, 'f': f}
    S = ca.nlpsol('S', 'ipopt', nlp, {
        'print_time': 0,
        'ipopt': {
            'sb': 'yes',
            'print_level': 0,
        }
    })
    r = S(x0=s0, lbg=0, ubg=0)
    s_opt = r['x']
    x, u = constrain(x, s_opt)
    return x, u


def simulate(x0: State, f_control, p: Parameters, t0: float, tf: float, dt: float):
    """
    Simulate the aircraft for a given control function and initial state.

    Parameters:
    x0: initial state (see State)
    f_control: A function of the form f(t, x), which returns the control u
    p: Aircraft parameters
    t0: initial time
    tf: fintal time
    dt: The discrete sampling time of the controller.
    """
    xs = ca.MX.sym('x', 16)
    x = State.from_casadi(xs)
    us = ca.MX.sym('u', 4)
    u = Control.from_casadi(us)
    dae = {'x': xs, 'p': us, 'ode': dynamics(x, u, p).to_casadi()}
    F = ca.integrator('F', 'idas', dae, {'t0': 0, 'tf': dt, 'jit': True})
    x = np.array(x0.to_casadi()).reshape(-1)
    u0 = f_control(t0, x0)
    u = np.array(u0.to_casadi()).reshape(-1)
    data = {
        't': [0],
        'x': [x]
    }
    t_vect = np.arange(t0, tf, dt)
    for t in t_vect:
        u0 = f_control(t, x)
        u = np.array(u0.to_casadi()).reshape(-1)
        x = np.array(F(x0=x, p=u)['xf']).reshape(-1)
        data['t'].append(t)
        data['x'].append(x)
    for k in data.keys():
        data[k] = np.array(data[k])
    return data