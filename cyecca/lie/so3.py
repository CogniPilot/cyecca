import casadi as ca

from .base import LieAlgebra, LieGroup, EPS


class LieAlgebraSO3(LieAlgebra):
    """
    The so(3) Lie Algebra
    """

    @property
    def x(self):
        return self.param[0]

    @property
    def y(self):
        return self.param[1]

    @property
    def z(self):
        return self.param[2]
 
    def neg(self):
        return LieAlgebraSO3(-self.param)

    def add(self, other):
        return LieAlgebraSO3(self.param + other.param)

    def rmul(self, scalar):
        return LieAlgebraSO3(scalar*self.param)

    def wedge(self):
        X = ca.SX(3, 3)
        X[0, 1] = -self.z
        X[0, 2] = self.y
        X[1, 0] = self.z
        X[1, 2] = -self.x
        X[2, 0] = -self.y
        X[2, 1] = self.x
        return X

    def vee(self):
        return self.param


class LieGroupSO3Quat(LieGroup):
    """
    The SO(3) Lie Group, parameterized by Quaternions
    """

    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (4, 1)

    @property
    def v(self):
        """return vector part of quaternion"""
        return self.param[:3]

    @property
    def x(self):
        """return vector x component of quternion"""
        return self.param[0]

    @property
    def y(self):
        """return vector y component of quternion"""
        return self.param[1]

    @property
    def z(self):
        """return vector w component of quternion"""
        return self.param[2]

    @property
    def w(self):
        """return scalar component of quaternion"""
        return self.param[3]

    @staticmethod
    def identity():
        return LieGroupSO3Quat(ca.vertcat(0, 0, 0, 1))

    def inv(self):
        return LieGroupSO3Quat(ca.vertcat(-self.v, self.w))

    def log(self):
        theta = 2 * ca.acos(self.w)
        c = ca.sin(theta / 2)
        v = ca.vertcat(self.x, self.y, self.z)
        return LieAlgebraSO3(ca.if_else(
            ca.fabs(c) > EPS,
            theta * v / c,
            ca.vertcat(0, 0, 0)
        ))

    def product(self, other : 'LieGroupSO3Quat'):
        assert isinstance(self, LieGroupSO3Quat)
        assert isinstance(other, LieGroupSO3Quat)
        w = self.w*other.w - ca.dot(self.v, other.v)
        v = self.w*other.v + other.w*self.v + ca.cross(self.v, other.v)
        return LieGroupSO3Quat(ca.vertcat(v, w))

    def to_matrix_lie_group(self):
        a = self.param[3]
        b = self.param[0]
        c = self.param[1]
        d = self.param[2]
        aa = a * a
        ab = a * b
        ac = a * c
        ad = a * d
        bb = b * b
        bc = b * c
        bd = b * d
        cc = c * c
        cd = c * d
        dd = d * d
        R = ca.SX.sym(3, 3)
        R[0, 0] = aa + bb - cc - dd
        R[0, 1] = 2 * (bc - ad)
        R[0, 2] = 2 * (bd + ac)
        R[1, 0] = 2 * (bc + ad)
        R[1, 1] = aa + cc - bb - dd
        R[1, 2] = 2 * (cd - ab)
        R[2, 0] = 2 * (bd - ac)
        R[2, 1] = 2 * (cd + ab)
        R[2, 2] = aa + dd - bb - cc
        return R
    
    @classmethod
    def exp(cls, g : LieAlgebraSO3):
        theta = ca.norm_2(g.param)
        w = ca.cos(theta / 2)
        c = ca.sin(theta / 2)
        v = c * g.param / theta
        return LieGroupSO3Quat(ca.vertcat(v, w))
