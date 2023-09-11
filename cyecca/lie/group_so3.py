from __future__ import annotations

from enum import Enum

import casadi as ca

from beartype import beartype
from beartype.typing import List
from cyecca.lie.base import *

from cyecca.symbolic import SERIES

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

    def elem(self, param: PARAM_TYPE) -> SO3LieAlgebraElement:
        return SO3LieAlgebraElement(algebra=self, param=param)

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

    def scalar_multiplication(
        self, left: SCALAR_TYPE, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == right.algebra
        return self.elem(param=left * right.param)

    def adjoint(self, left: LieAlgebraElement) -> ca.SX:
        assert self == left.algebra
        return left.to_Matrix()

    def wedge(self, left: PARAM_TYPE) -> LieAlgebraElement:
        return self.elem(param=left)

    def vee(self, left: LieAlgebraElement) -> ca.SX:
        assert self == left.algebra
        return left.param

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

    def from_Matrix(self, arg: ca.SX) -> LieAlgebraElement:
        assert arg.shape == (3, 3)
        return self.elem(ca.vertcat(arg[2, 1], arg[0, 2], arg[1, 0]))


@beartype
class SO3LieAlgebraElement(LieAlgebraElement):
    """
    This is an SO3 Lie algebra elem
    """

    def __init__(self, algebra: SO3LieAlgebra, param: PARAM_TYPE):
        super().__init__(algebra, param)


@beartype
class Axis(Enum):
    x = 1
    y = 2
    z = 3


@beartype
class EulerType(Enum):
    body_fixed = 1
    space_fixed = 2


@beartype
def rotation_matrix(axis: Axis, angle: SCALAR_TYPE):
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

    def elem(self, param: PARAM_TYPE) -> SO3DcmLieGroupElement:
        return SO3DcmLieGroupElement(group=self, param=param)

    def product(self, left: LieGroupElement, right: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        assert self == right.group
        return self.elem(param=left.param + right.param)

    def inverse(self, arg: LieGroupElement) -> LieGroupElement:
        assert self == arg.group
        return self.from_Matrix(param=arg.to_Matrix().T())

    def identity(self) -> LieGroupElement:
        return self.elem(param=ca.SX(self.n_param, 1))

    def adjoint(self, arg: LieGroupElement) -> ca.SX:
        return self.to_Matrix()

    def exp(self, arg: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == arg.algebra
        theta = ca.norm_2(arg.param)
        X = arg.to_Matrix()
        A = SERIES["sin(x)/x"]
        B = SERIES["(1 - cos(x))/x^2"]
        return self.from_Matrix(ca.SX.eye(3) + A(theta) * X + B(theta) * X @ X)

    def log(self, arg: LieGroupElement) -> LieAlgebraElement:
        R = self.to_Matrix(arg)
        theta = ca.arccos((ca.trace(R) - 1) / 2)
        A = SERIES["sin(x)/x"]
        return self.algebra.from_Matrix((R - R.T) / (A(theta) * 2))

    def to_Matrix(self, arg: LieGroupElement) -> ca.SX:
        assert self == arg.group
        return arg.param.reshape((3, 3))

    def from_Matrix(self, arg: ca.SX) -> LieGroupElement:
        assert arg.shape == (3, 3)
        return self.elem(arg.reshape((9, 1)))

    def from_Mrp(self, arg: LieGroupElement) -> LieGroupElement:
        assert arg.group == SO3Mrp
        return self.from_Quat(SO3Quat.from_Mrp(arg))

    def from_Quat(self, arg: LieGroupElement) -> LieGroupElement:
        assert arg.group == SO3Quat
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
        return self.from_Matrix(arg=R)

    def from_Mrp(self, arg: LieGroupElement) -> LieGroupElement:
        assert arg.group == SO3MrpLieGroup
        X = arg.to_Matrix()
        n_sq = ca.dot(arg.param, arg.param)
        X_sq = X @ X
        R = ca.SX.eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
        # return transpose, due to convention difference in book
        return self.from_Matrix(R.T)

    def from_EulerB321(self, arg: LieGroupElement) -> LieGroupElement:
        assert arg.group == SO3EulerB321
        return self.from_Quat(SO3Quat.from_EulerB321(arg))


@beartype
class SO3DcmLieGroupElement(LieGroupElement):
    """
    This is an SO3Dcm Lie group elem
    """

    def __init__(self, group: SO3DcmLieGroup, param: PARAM_TYPE):
        super().__init__(group, param)


SO3Dcm = SO3DcmLieGroup()


@beartype
class SO3EulerLieGroup(SO3LieGroup):
    def __init__(self, euler_type: EulerType, sequence: List[Axis]):
        super().__init__(algebra=so3, n_param=3, matrix_shape=(3, 3))
        self.euler_type = euler_type
        assert len(sequence) == 3
        self.sequence = sequence

    def elem(self, param: PARAM_TYPE) -> SO3EulerLieGroupElement:
        return SO3EulerLieGroupElement(group=self, param=param)

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
        return self.from_Dcm(SO3Dcm.exp(arg))

    def log(self, arg: LieGroupElement) -> LieAlgebraElement:
        assert self == arg.group
        return SO3Dcm.log(SO3Dcm.from_EulerB321(arg))

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

    def from_Matrix(self, arg: ca.SX) -> ca.SX:
        assert arg.shape == (3, 3)
        raise NotImplementedError("need to implement for each Euler sequence")

    def from_Dcm(self, arg: LieGroupElement) -> LieGroupElement:
        assert arg.group == SO3Dcm
        return self.from_Matrix(SO3Dcm.to_Matrix(arg))

    def from_Quat(self, arg: LieGroupElement) -> LieGroupElement:
        assert arg.group == SO3Quat
        return self.from_Matrix(SO3Quat.to_Matrix())


@beartype
class SO3EulerLieGroupElement(LieGroupElement):
    """
    This is an SO3Euler Lie group elem
    """

    def __init__(self, group: SO3EulerLieGroup, param: PARAM_TYPE):
        super().__init__(group, param)


@beartype
class SO3EulerB321LieGroup(SO3EulerLieGroup):
    def __init__(self):
        super().__init__(
            euler_type=EulerType.body_fixed, sequence=[Axis.z, Axis.y, Axis.z]
        )

    def elem(self, param: PARAM_TYPE) -> SO3EulerB321LieGroupElement:
        return SO3EulerB321LieGroupElement(group=self, param=param)

    def from_Matrix(self, arg: ca.SX) -> LieGroupElement:
        assert arg.shape == (3, 3)
        theta = ca.asin(-arg[2, 0])
        param = ca.if_else(
            ca.fabs(theta - ca.pi / 2) < 1e-3,
            ca.vertcat(0, theta, ca.atan2(arg[1, 2], arg[0, 2])),
            ca.if_else(
                ca.fabs(theta + ca.pi / 2) < 1e-3,
                ca.vertcat(0, theta, ca.atan2(-arg[1, 2], -arg[0, 2])),
                ca.vertcat(
                    ca.atan2(arg[2, 1], arg[2, 2]),
                    theta,
                    ca.atan2(arg[1, 0], arg[0, 0]),
                ),
            ),
        )
        return LieGroupElement(group=self, param=param)


@beartype
class SO3EulerB321LieGroupElement(LieGroupElement):
    """
    This is an SO3EulerB321 Lie group elem
    """

    def __init__(self, group: SO3EulerB321LieGroup, param: PARAM_TYPE):
        super().__init__(group, param)


SO3EulerB321 = SO3EulerB321LieGroup()


@beartype
class SO3QuatLieGroup(SO3LieGroup):
    def __init__(self):
        super().__init__(algebra=so3, n_param=4, matrix_shape=(3, 3))

    def elem(self, param: PARAM_TYPE) -> SO3QuatLieGroupElement:
        return SO3QuatLieGroupElement(group=self, param=param)

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

    def from_Matrix(self, arg: ca.SX) -> LieGroupElement:
        assert arg.shape == (3, 3)
        R = arg
        b1 = 0.5 * ca.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        b2 = 0.5 * ca.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        b3 = 0.5 * ca.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        b4 = 0.5 * ca.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])

        q1 = ca.SX(4, 1)
        q1[0] = b1
        q1[1] = (R[2, 1] - R[1, 2]) / (4 * b1)
        q1[2] = (R[0, 2] - R[2, 0]) / (4 * b1)
        q1[3] = (R[1, 0] - R[0, 1]) / (4 * b1)

        q2 = ca.SX(4, 1)
        q2[0] = (R[2, 1] - R[1, 2]) / (4 * b2)
        q2[1] = b2
        q2[2] = (R[0, 1] + R[1, 0]) / (4 * b2)
        q2[3] = (R[0, 2] + R[2, 0]) / (4 * b2)

        q3 = ca.SX(4, 1)
        q3[0] = (R[0, 2] - R[2, 0]) / (4 * b3)
        q3[1] = (R[0, 1] + R[1, 0]) / (4 * b3)
        q3[2] = b3
        q3[3] = (R[1, 2] + R[2, 1]) / (4 * b3)

        q4 = ca.SX(4, 1)
        q4[0] = (R[1, 0] - R[0, 1]) / (4 * b4)
        q4[1] = (R[0, 2] + R[2, 0]) / (4 * b4)
        q4[2] = (R[1, 2] + R[2, 1]) / (4 * b4)
        q4[3] = b4

        q = ca.if_else(
            ca.trace(R) > 0,
            q1,
            ca.if_else(
                ca.logic_and(R[0, 0] > R[1, 1], R[0, 0] > R[2, 2]),
                q2,
                ca.if_else(R[1, 1] > R[2, 2], q3, q4),
            ),
        )
        return SO3Quat.elem(q)

    def from_Mrp(self, arg: LieGroupElement) -> LieGroupElement:
        assert arg.group == SO3Mrp
        q = ca.SX(4, 1)
        n_sq = ca.dot(arg.param, arg.param)
        den = 1 + n_sq
        q[0] = (1 - n_sq) / den
        for i in range(3):
            q[i + 1] = 2 * arg.param[i] / den
        return SO3Quat.elem(q)

    def from_Dcm(self, arg: LieGroupElement) -> LieGroupElement:
        assert arg.group == SO3Dcm
        return self.from_Matrix(SO3Dcm.to_Matrix(arg))

    def from_EulerB321(self, arg: LieGroupElement) -> LieGroupElement:
        assert arg.group == SO3EulerB321
        q = ca.SX(4, 1)
        cosPhi_2 = ca.cos(arg.param[0] / 2)
        cosTheta_2 = ca.cos(arg.param[1] / 2)
        cosPsi_2 = ca.cos(arg.param[2] / 2)
        sinPhi_2 = ca.sin(arg.param[0] / 2)
        sinTheta_2 = ca.sin(arg.param[1] / 2)
        sinPsi_2 = ca.sin(arg.param[2] / 2)
        q[0] = cosPhi_2 * cosTheta_2 * cosPsi_2 + sinPhi_2 * sinTheta_2 * sinPsi_2
        q[1] = sinPhi_2 * cosTheta_2 * cosPsi_2 - cosPhi_2 * sinTheta_2 * sinPsi_2
        q[2] = cosPhi_2 * sinTheta_2 * cosPsi_2 + sinPhi_2 * cosTheta_2 * sinPsi_2
        q[3] = cosPhi_2 * cosTheta_2 * sinPsi_2 - sinPhi_2 * sinTheta_2 * cosPsi_2
        return SO3Quat.elem(q)


@beartype
class SO3QuatLieGroupElement(LieGroupElement):
    """
    This is an SO3Quat Lie group elem
    """

    def __init__(self, group: SO3QuatLieGroup, param: PARAM_TYPE):
        super().__init__(group, param)


SO3Quat = SO3QuatLieGroup()


@beartype
class SO3MrpLieGroup(SO3LieGroup):
    def __init__(self):
        super().__init__(algebra=so3, n_param=3, matrix_shape=(3, 3))

    def elem(self, param: PARAM_TYPE) -> SO3MrpLieGroupElement:
        return SO3MrpLieGroupElement(group=self, param=param)

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

    def from_Matrix(self, arg: ca.SX) -> LieGroupElement:
        assert arg.shape == (3, 3)
        return self.from_Dcm(SO3Dcm.from_Matrix(arg))

    def from_Dcm(self, arg: LieGroupElement) -> LieGroupElement:
        return self.from_Quat(SO3Quat.from_Dcm(arg))

    def from_Quat(self, arg: LieGroupElement) -> LieGroupElement:
        assert arg.group == SO3Quat
        x = ca.SX(3, 1)
        den = 1 + arg.param[0]
        x[0] = arg.param[1] / den
        x[1] = arg.param[2] / den
        x[2] = arg.param[3] / den
        X = self.elem(param=x)
        self.shadow_if_necessary(X)
        return X

    def from_EulerB321(self, arg: LieGroupElement) -> LieGroupElement:
        assert arg.shape == (3, 3)
        return self.from_Quat(SO3Quat.from_EulerB321(arg))


@beartype
class SO3MrpLieGroupElement(LieGroupElement):
    """
    This is an SO3Mrp Lie group elem
    """

    def __init__(self, group: SO3MrpLieGroup, param: PARAM_TYPE):
        super().__init__(group, param)


SO3Mrp = SO3MrpLieGroup()
