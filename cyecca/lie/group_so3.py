from __future__ import annotations

from enum import Enum

import casadi as ca

from beartype import beartype
from beartype.typing import List

from .base import *

from .util import series_dict

__all__ = [
    "so3",
    "SO3EulerLieGroup",
    "SO3EulerB321",
    "Axis",
    "EulerType",
    "SO3Quat",
    "SO3Mrp",
    "SO3Dcm",
    "SO3LieGroup",
]


@beartype
class SO3LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=3, matrix_shape=(3, 3))

    def bracket(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        c = left.to_Matrix() @ right.to_Matrix() - right.to_Matrix() @ left.to_Matrix()
        return self.elem(param=ca.vertcat(c[2, 1], c[0, 2], c[1, 0]))

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.elem(param=left.param + right.param)

    def scalar_multipication(
        self, left: SCALAR_TYPE, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == right.algebra
        return self.elem(param=left * right.param)

    def adjoint(self, left: LieAlgebraElement) -> ca.SX:
        assert self == left.algebra
        return left.to_Matrix()

    def to_Matrix(self, left: LieAlgebraElement) -> ca.SX:
        assert self == left.algebra
        M = ca.SX(3, 3)
        M[0, 1] = -left.param[2, 0]
        M[1, 0] = left.param[2, 0]
        M[0, 2] = left.param[1, 0]
        M[2, 0] = -left.param[1, 0]
        M[1, 2] = -left.param[0, 0]
        M[2, 1] = left.param[0, 0]
        return ca.sparsify(M)

    def wedge(self, left: PARAM_TYPE) -> LieAlgebraElement:
        self = SO3LieAlgebra()
        return self.elem(param=left)

    def vee(self, left: LieAlgebraElement) -> ca.SX:
        assert self == left.algebra
        return left.param


class Axis(Enum):
    x = 1
    y = 2
    z = 3


class EulerType(Enum):
    body_fixed = 1
    space_fixed = 2


def rotation_matrix(axis: Axis, angle: Real):
    if axis == Axis.x:
        R = ca.SX_eye(3)
        R[1, 1] = ca.cos(angle)
        R[1, 2] = -ca.sin(angle)
        R[2, 1] = ca.sin(angle)
        R[2, 2] = ca.cos(angle)
        return R
    elif axis == Axis.y:
        R = ca.SX_eye(3)
        R[0, 0] = ca.cos(angle)
        R[2, 0] = -ca.sin(angle)
        R[0, 2] = ca.sin(angle)
        R[2, 2] = ca.cos(angle)
        return R
    elif axis == Axis.z:
        R = ca.SX_eye(3)
        R[0, 0] = ca.cos(angle)
        R[0, 1] = -ca.sin(angle)
        R[1, 0] = ca.sin(angle)
        R[1, 1] = ca.cos(angle)
        return R
    else:
        raise ValueError("unknown axis")


so3 = SO3LieAlgebra()


@beartype
class SO3LieGroup(LieGroup):
    pass


@beartype
class SO3DcmLieGroup(SO3LieGroup):
    def __init__(self):
        super().__init__(algebra=so3, n_param=9, matrix_shape=(3, 3))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        return self.elem(param=left.param + right.param)

    def inverse(self, arg: LieGroupElement) -> LieGroupElement:
        assert self == arg.group
        return self.from_matrix(param=arg.to_Matrix().T())

    def identity(self) -> LieGroupElement:
        return self.elem(param=ca.SX(self.n_param, 1))

    def adjoint(self, arg: LieGroupElement):
        assert self == arg.group

    def exp(self, arg: LieAlgebraElement) -> LieGroupElement:
        theta = ca.norm_2(arg.param)
        X = arg.to_Matrix()
        A = series_dict["sin(x)/x"]
        B = series_dict["(1 - cos(x))/x^2"]
        return self.from_matrix(ca.SX.eye(3) + A(theta) * X + B(theta) * X @ X)

    def log(self, arg: LieGroupElement) -> LieAlgebraElement:
        R = self.to_Matrix()
        theta = ca.arccos((ca.trace(R) - 1) / 2)
        A = series_dict["sin(x)/x"]
        return self.algebra.from_matrix((R - R.T) / (A(theta) * 2))

    def from_SO3Quat(self, arg: LieGroupElement):
        assert isinstance(arg.group, SO3QuatLieGroup)
        R = ca.SX(3, 3)
        a = arg.param[0]
        b = arg.param[1]
        c = arg.param[2]
        d = arg.param[3]
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
        R[0, 0] = aa + bb - cc - dd
        R[0, 1] = 2 * (bc - ad)
        R[0, 2] = 2 * (bd + ac)
        R[1, 0] = 2 * (bc + ad)
        R[1, 1] = aa + cc - bb - dd
        R[1, 2] = 2 * (cd - ab)
        R[2, 0] = 2 * (bd - ac)
        R[2, 1] = 2 * (cd + ab)
        R[2, 2] = aa + dd - bb - cc
        return self.from_matrix(arg=R)

    def from_SO3Mrp(self, arg: LieGroupElement):
        assert isinstance(arg.group, SO3MrpLieGroup)
        a = arg.param[:3]
        X = self.to_Matrix(a)
        n_sq = ca.dot(a, a)
        X_sq = X @ X
        R = ca.SX.eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
        # return transpose, due to convention difference in book
        return self.from_matrix(R.T)

    def to_Matrix(self, arg: LieGroupElement) -> ca.SX:
        return arg.param.reshape((3, 3))

    def from_matrix(self, arg: ca.SX) -> LieGroupElement:
        return SO3Dcm.elem(param=arg.reshape((9, 1)))


SO3Dcm = SO3DcmLieGroup()


@beartype
class SO3EulerLieGroup(SO3LieGroup):
    def __init__(self, euler_type: EulerType, sequence: List[Axis]):
        super().__init__(algebra=so3, n_param=3, matrix_shape=(3, 3))
        self.euler_type = euler_type
        assert len(sequence) == 3
        self.sequence = sequence

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        return self.elem(param=left.param + right.param)

    def inverse(self, arg: LieGroupElement) -> LieGroupElement:
        assert self == arg.group
        return self.elem(param=-arg.param)

    def identity(self) -> LieGroupElement:
        return self.elem(param=ca.SX(self.n_param, 1))

    def adjoint(self, arg: LieGroupElement):
        assert self == arg.group
        return arg.to_Matrix()

    def exp(self, arg: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == arg.algebra
        omega = arg.param
        omega_x = arg.to_Matrix()
        omega_n = ca.norm_2(omega)
        A = ca.if_else(
            ca.fabs(omega_n) < 1e-7,
            1 - omega_n**2 / 6 + omega_n**4 / 120,
            ca.sin(omega_n) / omega_n,
        )
        B = ca.if_else(
            ca.fabs(omega_n) < 1e-7,
            0.5 - omega_n**2 / 24 + omega_n**4 / 720,
            (1 - ca.cos(omega_n)) / omega_n**2,
        )
        R = ca.SX_eye(3) + A * omega_x + B * (omega_x @ omega_x)

        if self.euler_type == EulerType.body_fixed:
            theta = ca.asin(-R[2, 0])
            if ca.fabs(theta - ca.pi / 2) < 1e-7:
                phi = 0
                psi = ca.atan2(R[1, 2], R[0, 2])
            elif ca.fabs(theta + ca.pi / 2) < 1e-7:
                psi = 0
                phi = ca.atan2(-R[0, 1], R[1, 1])
            else:
                phi = ca.atan2(R[2, 1], R[2, 2])
                psi = ca.atan2(R[1, 0], R[0, 0])
            angle = ca.vertcat(psi, theta, phi)
        elif self.euler_type == EulerType.space_fixed:
            theta = ca.asin(R[0, 2])
            if ca.fabs(theta - ca.pi / 2) < 1e-7:
                phi = 0
                psi = ca.atan2(-R[1, 0], R[2, 0])
            elif ca.fabs(theta + ca.pi / 2) < 1e-7:
                psi = 0
                phi = ca.atan2(-R[1, 0], R[1, 1])
            else:
                phi = ca.atan2(-R[1, 2], R[2, 2])
                psi = ca.atan2(-R[0, 1], R[0, 0])
            angle = ca.vertcat(psi, theta, phi)
        else:
            raise ValueError("euler_type must be body_fixed or space_fixed")
        return self.elem(param=angle)

    def log(self, arg: LieGroupElement) -> LieAlgebraElement:
        assert self == arg.group
        R = arg.to_Matrix()
        theta = ca.acos((ca.trace(R) - 1) / 2)
        A = ca.if_else(
            ca.fabs(theta) < 1e-7,
            1 - theta**2 / 6 + theta**4 / 120,
            ca.sin(theta) / theta,
        )
        r_matrix = (R - R.T) / (A * 2)  # matrix of so3 in np.array
        r = ca.vertcat(r_matrix[2, 1], r_matrix[0, 2], r_matrix[1, 0])  # vector of so3
        return self.algebra.elem(param=r)

    def to_Matrix(self, arg: LieGroupElement) -> ca.SX:
        assert self == arg.group
        m = ca.SX_eye(3)
        for axis, angle in zip(self.sequence, ca.vertsplit(arg.param)):
            if self.euler_type == EulerType.body_fixed:
                m = m @ rotation_matrix(axis=axis, angle=angle)
            elif self.euler_type == EulerType.space_fixed:
                m = rotation_matrix(axis=axis, angle=angle) @ m
            else:
                raise ValueError("euler_type must be body_fixed or space_fixed")
        return m


SO3EulerB321 = SO3EulerLieGroup(
    euler_type=EulerType.body_fixed, sequence=[Axis.z, Axis.y, Axis.x]
)


@beartype
class SO3QuatLieGroup(SO3LieGroup):
    def __init__(self):
        super().__init__(algebra=so3, n_param=4, matrix_shape=(3, 3))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        q = left.param
        p = right.param
        return self.elem(
            param=ca.vertcat(
                q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3],
                q[1] * p[0] + q[0] * p[1] - q[3] * p[2] + q[2] * p[3],
                q[2] * p[0] + q[3] * p[1] + q[0] * p[2] - q[1] * p[3],
                q[3] * p[0] - q[2] * p[1] + q[1] * p[2] + q[0] * p[3],
            )
        )

    def inverse(self, arg: LieGroupElement) -> LieGroupElement:
        assert self == arg.group
        q = arg.param
        return self.elem(param=ca.vertcat(q[0], -q[1], -q[2], -q[3]))

    def identity(self) -> LieGroupElement:
        return self.elem(param=ca.SX([1, 0, 0, 0]))

    def adjoint(self, arg: LieGroupElement):
        assert self == arg.group
        return arg.to_Matrix()

    def exp(self, arg: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == arg.algebra
        v = arg.param
        theta = ca.norm_2(v)
        c = ca.sin(theta / 2)
        q = ca.vertcat(
            ca.cos(theta / 2), c * v[0] / theta, c * v[1] / theta, c * v[2] / theta
        )
        return self.elem(
            param=ca.if_else(ca.fabs(theta) > 1e-7, q, ca.SX([1, 0, 0, 0]))
        )

    def log(self, arg: LieGroupElement) -> LieAlgebraElement:
        assert self == arg.group
        q = arg.param
        theta = 2 * ca.acos(q[0])
        c = ca.sin(theta / 2)
        v = ca.vertcat(theta * q[1] / c, theta * q[2] / c, theta * q[3] / c)
        return self.algebra.elem(
            param=ca.if_else(ca.fabs(c) > 1e-7, v, ca.SX([0, 0, 0]))
        )

    def to_Matrix(self, arg: LieGroupElement) -> ca.SX:
        assert self == arg.group
        q = arg.param
        R = ca.SX(3, 3)
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
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


SO3Quat = SO3QuatLieGroup()


@beartype
class SO3MrpLieGroup(SO3LieGroup):
    def __init__(self):
        super().__init__(algebra=so3, n_param=3, matrix_shape=(3, 3))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        a = left.param[:3]
        b = right.param[:3]
        na_sq = ca.dot(a, a)
        nb_sq = ca.dot(b, b)
        den = 1 + na_sq * nb_sq - 2 * ca.dot(b, a)
        res = ((1 - na_sq) * b + (1 - nb_sq) * a - 2 * ca.cross(b, a)) / den
        return self.elem(param=res)

    def inverse(self, arg: LieGroupElement) -> LieGroupElement:
        assert self == arg.group
        return self.elem(param=-arg.param)

    def identity(self) -> LieGroupElement:
        return self.elem(param=ca.SX([0, 0, 0]))

    def shadow_if_necessary(self, arg: LieGroupElement):
        assert self == arg.group
        param = arg.param
        assert param.shape == (3, 1)
        n_sq = ca.dot(param, param)
        shadow_param = -param / n_sq
        param = ca.if_else(ca.norm_2(param) > 1, shadow_param, param)
        arg.param = param

    def adjoint(self, arg: LieGroupElement):
        assert self == arg.group
        return arg.to_Matrix()

    def exp(self, arg: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == arg.algebra
        v = arg.param
        angle = ca.norm_2(v)
        res = ca.tan(angle / 4) * v / angle
        p = ca.if_else(angle > 1e-7, res, ca.SX([0, 0, 0]))
        V = self.elem(param=p)
        self.shadow_if_necessary(arg=V)
        return V

    def log(self, arg: LieGroupElement) -> LieAlgebraElement:
        assert self == arg.group
        r = arg.param
        n = ca.norm_2(r[:3])
        v = 4 * ca.atan(n) * r[:3] / n
        return self.algebra.elem(param=ca.if_else(n > 1e-7, v, ca.SX([0, 0, 0])))

    def to_Matrix(self, arg: LieGroupElement) -> ca.SX:
        assert self == arg.group
        r = arg.param
        a = r[:3]
        X = so3.elem(param=a).to_Matrix()
        n_sq = ca.dot(a, a)
        X_sq = X @ X
        R = ca.SX_eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
        # return transpose, due to convention difference in book
        return R.T

    def from_SO3Quat(self, q: LieGroupElement) -> LieGroupElement:
        assert q.group == SO3Quat
        x = ca.SX(3, 1)
        den = 1 + q.param[0]
        x[0] = q.param[1] / den
        x[1] = q.param[2] / den
        x[2] = q.param[3] / den
        self.shadow_if_necessary(arg=self.elem(param=x))
        return arg


SO3Mrp = SO3MrpLieGroup()
