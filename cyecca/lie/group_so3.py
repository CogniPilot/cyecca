from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import casadi as ca

from beartype import beartype
from beartype.typing import List, Union

from cyecca.lie.base import *
from cyecca.lie.group_rn import R3LieGroupElement, R3LieAlgebraElement
from cyecca.symbolic import SERIES, SQUARED_SERIES

__all__ = [
    "so3",
    "Axis",
    "EulerType",
    "SO3Quat",
    "SO3Mrp",
    "SO3Dcm",
    "SO3EulerLieGroup",
    "SO3EulerB321",
    "SO3LieGroup",
]


@beartype
class SO3LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=3, matrix_shape=(3, 3))

    def elem(self, param: PARAM_TYPE) -> SO3LieAlgebraElement:
        return SO3LieAlgebraElement(algebra=self, param=param)

    def bracket(
        self, left: SO3LieAlgebraElement, right: SO3LieAlgebraElement
    ) -> SO3LieAlgebraElement:
        c = left.to_Matrix() @ right.to_Matrix() - right.to_Matrix() @ left.to_Matrix()
        return self.elem(param=ca.vertcat(c[2, 1], c[0, 2], c[1, 0]))

    def addition(
        self, left: SO3LieAlgebraElement, right: SO3LieAlgebraElement
    ) -> SO3LieAlgebraElement:
        return self.elem(param=left.param + right.param)

    def scalar_multiplication(
        self, left: SCALAR_TYPE, right: SO3LieAlgebraElement
    ) -> SO3LieAlgebraElement:
        return self.elem(param=left * right.param)

    def adjoint(self, left: SO3LieAlgebraElement) -> ca.SX:
        return left.to_Matrix()

    def wedge(self, left: PARAM_TYPE) -> SO3LieAlgebraElement:
        return self.elem(param=left)

    def vee(self, left: SO3LieAlgebraElement) -> ca.SX:
        return left.param

    def to_Matrix(self, left: SO3LieAlgebraElement) -> ca.SX:
        M = ca.SX(3, 3)
        M[0, 1] = -left.param[2, 0]
        M[1, 0] = left.param[2, 0]
        M[0, 2] = left.param[1, 0]
        M[2, 0] = -left.param[1, 0]
        M[1, 2] = -left.param[0, 0]
        M[2, 1] = left.param[0, 0]
        return ca.sparsify(M)

    def from_Matrix(self, arg: ca.SX) -> SO3LieAlgebraElement:
        assert arg.shape == (3, 3)
        return self.elem(ca.vertcat(arg[2, 1], arg[0, 2], arg[1, 0]))

    def left_jacobian(self, arg: SO3LieAlgebraElement) -> ca.SX:
        v = arg.param
        X = arg.to_Matrix()
        # must do series for theta_sq to avoid sqrt(x) nan for jacobian
        # near zero
        theta_sq = ca.dot(v, v)
        A = SQUARED_SERIES["(1 - cos(x))/x^2"](theta_sq)
        B = SQUARED_SERIES["(x - sin(x))/x^3"](theta_sq)
        return ca.SX.eye(3) + A * X + B * (X @ X)

    def left_jacobian_inv(self, arg: SO3LieAlgebraElement) -> ca.SX:
        v = arg.param
        X = arg.to_Matrix()
        # must do series for theta_sq to avoid sqrt(x) nan for jacobian
        # near zero
        theta_sq = ca.dot(v, v)
        A = SQUARED_SERIES["1/x^2 + sin(x)/(2 x (cos(x) - 1))"](theta_sq)
        return ca.SX.eye(3) - 0.5 * X + A * (X @ X)

    def right_jacobian(self, arg: SO3LieAlgebraElement) -> ca.SX:
        v = arg.param
        X = arg.to_Matrix()
        # must do series for theta_sq to avoid sqrt(x) nan for jacobian
        # near zero
        theta_sq = ca.dot(v, v)
        A = SQUARED_SERIES["(1 - cos(x))/x^2"](theta_sq)
        B = SQUARED_SERIES["(x - sin(x))/x^3"](theta_sq)
        return ca.SX.eye(3) - A * X + B * (X @ X)

    def right_jacobian_inv(self, arg: SO3LieAlgebraElement) -> ca.SX:
        v = arg.param
        X = arg.to_Matrix()
        # must do series for theta_sq to avoid sqrt(x) nan for jacobian
        # near zero
        theta_sq = ca.dot(v, v)
        A = SQUARED_SERIES["1/x^2 + sin(x)/(2 x (cos(x) - 1))"](theta_sq)
        return ca.SX.eye(3) + 0.5 * X + A * (X @ X)


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
    cos_angle = ca.cos(angle)
    sin_angle = ca.sin(angle)
    if axis == Axis.x:
        R = ca.SX.eye(3)
        R[1, 1] = cos_angle
        R[1, 2] = -sin_angle
        R[2, 1] = sin_angle
        R[2, 2] = cos_angle
        return R
    elif axis == Axis.y:
        R = ca.SX.eye(3)
        R[0, 0] = cos_angle
        R[2, 0] = -sin_angle
        R[0, 2] = sin_angle
        R[2, 2] = cos_angle
        return R
    elif axis == Axis.z:
        R = ca.SX.eye(3)
        R[0, 0] = cos_angle
        R[0, 1] = -sin_angle
        R[1, 0] = sin_angle
        R[1, 1] = cos_angle
        return R
    else:
        raise ValueError("unknown axis")


so3 = SO3LieAlgebra()


@beartype
class SO3LieGroup(LieGroup):
    """
    An abstract SO3 Lie Group
    """

    def product(
        self, left: SO3LieGroupElement, right: SO3LieGroupElement
    ) -> SO3LieGroupElement:
        """
        Default product uses matrix conversion
        """
        return self.from_Matrix(left.to_Matrix() @ right.to_Matrix())

    def product_r3(
        self, left: SO3LieGroupElement, right: R3LieAlgebraElement
    ) -> R3LieAlgebraElement:
        """
        Vector rotation for algebra r3, uses to_Matrix
        """
        v = left.to_Matrix() @ right.param
        return R3LieAlgebraElement(algebra=right.algebra, param=v)

    def product_vector(
        self, left: SO3LieGroupElement, right: Union[ca.SX, ca.DM]
    ) -> ca.SX:
        """
        Vector product, uses matrix conversion
        """
        return left.to_Matrix() @ right


@beartype
class SO3LieGroupElement(LieGroupElement):
    """
    An abstract SO3Dcm Lie group elem
    """

    def __init__(self, group: SO3LieGroup, param: PARAM_TYPE):
        super().__init__(group, param)

    def __matmul__(self, right):
        """
        override matrix mul operator to use as actions on 3 vectors
        """
        if isinstance(right, R3LieAlgebraElement):
            return self.group.product_r3(self, right)
        elif isinstance(right, ca.SX) and right.shape == (3, 1):
            return self.group.product_vector(self, right)
        elif isinstance(right, ca.DM) and right.shape == (3, 1):
            return self.group.product_vector(self, right)
        else:
            print(type(right))
            raise TypeError("unhandled type in product {:s}".format(str(type(right))))


@beartype
class SO3DcmLieGroup(SO3LieGroup):
    def __init__(self):
        super().__init__(algebra=so3, n_param=9, matrix_shape=(3, 3))

    def elem(self, param: PARAM_TYPE) -> SO3DcmLieGroupElement:
        return SO3DcmLieGroupElement(group=self, param=param)

    def inverse(self, arg: SO3DcmLieGroupElement) -> SO3DcmLieGroupElement:
        return self.from_Matrix(arg=arg.to_Matrix().T)

    def identity(self) -> SO3DcmLieGroupElement:
        return self.elem(param=ca.SX(self.n_param, 1))

    def adjoint(self, arg: SO3DcmLieGroupElement) -> ca.SX:
        return self.to_Matrix()

    def exp(self, arg: SO3LieAlgebraElement) -> SO3DcmLieGroupElement:
        v = arg.param
        X = arg.to_Matrix()
        # must do series for theta_sq to avoid sqrt(x) nan for jacobian
        # near zero
        theta_sq = ca.dot(v, v)
        C1 = SQUARED_SERIES["sin(x)/x"](theta_sq)
        C2 = SQUARED_SERIES["(1 - cos(x))/x^2"](theta_sq)
        return self.from_Matrix(ca.SX.eye(3) + C1 * X + C2 * X @ X)

    def log(self, arg: SO3DcmLieGroupElement) -> SO3LieAlgebraElement:
        R = self.to_Matrix(arg)
        e1 = (ca.trace(R) - 1) / 2
        theta = ca.if_else(e1 > 1, 0, ca.if_else(e1 < -1, ca.pi, ca.acos(e1)))
        C1 = SERIES["x/sin(x)"](theta) / 2
        return self.algebra.from_Matrix((R - R.T) * C1)

    def to_Matrix(self, arg: SO3DcmLieGroupElement) -> ca.SX:
        return arg.param.reshape((3, 3))

    def from_Matrix(self, arg: ca.SX) -> SO3DcmLieGroupElement:
        assert arg.shape == (3, 3)
        return self.elem(arg.reshape((9, 1)))

    def from_Quat(self, arg: SO3QuatLieGroupElement) -> SO3DcmLieGroupElement:
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

    def from_Mrp(self, arg: SO3MrpLieGroupElement) -> SO3DcmLieGroupElement:
        v = arg.param
        X = arg.to_Matrix()
        n_sq = ca.dot(v, v)
        X_sq = X @ X
        R = ca.SX.eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
        # return transpose, due to convention difference in book
        return self.from_Matrix(R.T)

    def from_Mrp_alternative(self, arg: SO3MrpLieGroupElement) -> SO3DcmLieGroupElement:
        return self.from_Quat(SO3Quat.from_Mrp(arg))

    def from_Euler(self, arg: SO3EulerLieGroupElement) -> SO3DcmLieGroupElement:
        return self.from_Quat(SO3Quat.from_Euler(arg))


@beartype
class SO3DcmLieGroupElement(SO3LieGroupElement):
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

    def inverse(self, arg: SO3EulerLieGroupElement) -> SO3EulerLieGroupElement:
        return self.from_Matrix(self.to_Matrix(arg).T)

    def identity(self) -> SO3EulerLieGroupElement:
        return self.elem(param=ca.SX(self.n_param, 1))

    def adjoint(self, arg: SO3EulerLieGroupElement):
        return arg.to_Matrix()

    def exp(self, arg: SO3LieAlgebraElement) -> SO3EulerLieGroupElement:
        return self.from_Dcm(SO3Dcm.exp(arg))

    def log(self, arg: SO3EulerLieGroupElement) -> SO3LieAlgebraElement:
        return SO3Dcm.log(SO3Dcm.from_Euler(arg))

    def to_Matrix(self, arg: SO3EulerLieGroupElement) -> ca.SX:
        m = ca.SX.eye(3)
        for axis, angle in zip(self.sequence, ca.vertsplit(arg.param)):
            if self.euler_type == EulerType.body_fixed:
                m = m @ rotation_matrix(axis=axis, angle=angle)
            elif self.euler_type == EulerType.space_fixed:
                m = rotation_matrix(axis=axis, angle=angle) @ m
            else:
                raise ValueError("euler_type must be body_fixed or space_fixed")
        return m

    def from_Matrix(self, arg: ca.SX) -> SO3EulerLieGroupElement:
        assert arg.shape == (3, 3)
        if self.euler_type == EulerType.body_fixed and self.sequence == [
            Axis.z,
            Axis.y,
            Axis.x,
        ]:
            theta = ca.asin(-arg[2, 0])

            cond1 = ca.fabs(theta - ca.pi / 2) < 1e-3
            phi1 = 0
            psi1 = ca.atan2(arg[1, 2], arg[0, 2])

            cond2 = ca.fabs(theta + ca.pi / 2) < 1e-3
            phi2 = 0
            psi2 = ca.atan2(-arg[1, 2], -arg[0, 2])

            phi3 = ca.atan2(arg[2, 1], arg[2, 2])
            psi3 = ca.atan2(arg[1, 0], arg[0, 0])

            param = ca.if_else(
                cond1,
                ca.vertcat(psi1, theta, phi1),
                ca.if_else(
                    cond2, ca.vertcat(psi2, theta, phi2), ca.vertcat(psi3, theta, phi3)
                ),
            )
        else:
            raise NotImplementedError(
                f"from_Matrix not implemented for {self.euler_type}, {self.sequence}"
            )
        return SO3EulerLieGroupElement(group=self, param=param)

    def from_Dcm(self, arg: SO3DcmLieGroupElement) -> SO3EulerLieGroupElement:
        return self.from_Matrix(SO3Dcm.to_Matrix(arg))

    def from_Quat(self, arg: SO3QuatLieGroupElement) -> SO3EulerLieGroupElement:
        return self.from_Matrix(arg.to_Matrix())

    def from_Mrp(self, arg: SO3MrpLieGroupElement) -> SO3EulerLieGroupElement:
        return self.from_Matrix(arg.to_Matrix())


@beartype
class SO3EulerLieGroupElement(SO3LieGroupElement):
    """
    This is an SO3Euler Lie group elem
    """

    def __init__(self, group: SO3EulerLieGroup, param: PARAM_TYPE):
        super().__init__(group, param)


@beartype
class SO3QuatLieGroup(SO3LieGroup):
    def __init__(self):
        super().__init__(algebra=so3, n_param=4, matrix_shape=(3, 3))

    def elem(self, param: PARAM_TYPE) -> SO3QuatLieGroupElement:
        return SO3QuatLieGroupElement(group=self, param=param)

    def product(
        self, left: SO3QuatLieGroupElement, right: SO3QuatLieGroupElement
    ) -> SO3QuatLieGroupElement:
        """
        provide a more efficient product
        """
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

    def inverse(self, arg: SO3QuatLieGroupElement) -> SO3QuatLieGroupElement:
        q = arg.param
        return self.elem(param=ca.vertcat(q[0], -q[1], -q[2], -q[3]))

    def identity(self) -> SO3QuatLieGroupElement:
        return self.elem(param=ca.SX([1, 0, 0, 0]))

    def adjoint(self, arg: SO3QuatLieGroupElement):
        return arg.to_Matrix()

    def exp(self, arg: SO3LieAlgebraElement) -> SO3QuatLieGroupElement:
        v = arg.param
        # must do series for theta_sq to avoid sqrt(x) nan for jacobian
        # near zero
        theta_sq = ca.dot(v, v)
        A = SQUARED_SERIES["sin(x)/x"](theta_sq / 4) / 2
        B = SQUARED_SERIES["cos(x)"](theta_sq / 4)
        return SO3Quat.elem(ca.vertcat(B, A * v[0], A * v[1], A * v[2]))

    def log(self, arg: SO3QuatLieGroupElement) -> SO3LieAlgebraElement:
        q = arg.param
        q = q / ca.norm_2(q)
        theta = 2 * ca.acos(q[0])
        A = SERIES["x/sin(x)"](theta / 2)
        omega = q[1:4] * A * 2
        return self.algebra.elem(omega)

    def left_jacobian(self, arg: SO3LieGroupElement) -> ca.SX:
        w = so3.elem(ca.SX.sym("w", 3))
        qw = SO3Quat.elem(ca.vertcat(0, w.param))
        q_dot_left = (qw * arg).param / 2
        return ca.jacobian(q_dot_left, w.param)

    def right_jacobian(self, arg: SO3LieGroupElement) -> ca.SX:
        w = so3.elem(ca.SX.sym("w", 3))
        qw = SO3Quat.elem(ca.vertcat(0, w.param))
        q_dot_right = (arg * qw).param / 2
        return ca.jacobian(q_dot_right, w.param)

    def to_Matrix(self, arg: SO3QuatLieGroupElement) -> ca.SX:
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

    def from_Matrix(self, arg: ca.SX) -> SO3QuatLieGroupElement:
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

    def from_Mrp(self, arg: SO3MrpLieGroupElement) -> SO3QuatLieGroupElement:
        q = ca.SX(4, 1)
        n_sq = ca.dot(arg.param, arg.param)
        den = 1 + n_sq
        q[0] = (1 - n_sq) / den
        for i in range(3):
            q[i + 1] = 2 * arg.param[i] / den
        return SO3Quat.elem(q)

    def from_Dcm(self, arg: SO3DcmLieGroupElement) -> SO3QuatLieGroupElement:
        return self.from_Matrix(arg.to_Matrix())

    def from_Euler(self, arg: SO3EulerLieGroupElement) -> SO3QuatLieGroupElement:
        return self.from_Matrix(arg.to_Matrix())


@beartype
class SO3QuatLieGroupElement(SO3LieGroupElement):
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

    def product(
        self, left: SO3MrpLieGroupElement, right: SO3MrpLieGroupElement
    ) -> SO3MrpLieGroupElement:
        """
        Provide a move efficient product
        """
        a = left.param[:3]
        b = right.param[:3]
        na_sq = ca.dot(a, a)
        nb_sq = ca.dot(b, b)
        den = 1 + na_sq * nb_sq - 2 * ca.dot(b, a)
        res = ((1 - na_sq) * b + (1 - nb_sq) * a - 2 * ca.cross(b, a)) / den
        return self.elem(param=res)

    def inverse(self, arg: SO3MrpLieGroupElement) -> SO3MrpLieGroupElement:
        return self.elem(param=-arg.param)

    def identity(self) -> SO3MrpLieGroupElement:
        return self.elem(param=ca.SX([0, 0, 0]))

    def shadow_if_necessary(self, arg: SO3MrpLieGroupElement) -> None:
        param = arg.param
        assert param.shape == (3, 1)
        n_sq = ca.dot(param, param)
        shadow_param = -param / n_sq
        param = ca.if_else(ca.dot(param, param) > 1, shadow_param, param)
        arg.param = param

    def adjoint(self, arg: SO3MrpLieGroupElement) -> ca.SX:
        return arg.to_Matrix()

    def exp(self, arg: SO3LieAlgebraElement) -> SO3MrpLieGroupElement:
        v = arg.param
        theta_sq = ca.dot(v, v)
        A = SQUARED_SERIES["tan(x/4)/x"](theta_sq)
        V = self.elem(param=A * v)
        self.shadow_if_necessary(arg=V)
        return V

    def log(self, arg: SO3MrpLieGroupElement) -> SO3LieAlgebraElement:
        r = arg.param
        theta_sq = ca.dot(r, r)
        A = SQUARED_SERIES["4 atan(x)/x"](theta_sq)
        return self.algebra.elem(param=A * r)

    def right_jacobian(self, arg: SO3LieGroupElement) -> ca.SX:
        r = arg.param
        n_sq = ca.dot(r, r)
        X = so3.elem(r).to_Matrix()
        B = 0.25 * ((1 - n_sq) * ca.SX.eye(3) + 2 * X + 2 * r @ r.T)
        return B

    def to_Matrix(self, arg: SO3MrpLieGroupElement) -> ca.SX:
        r = arg.param
        a = r[:3]
        X = so3.elem(param=a).to_Matrix()
        n_sq = ca.dot(a, a)
        X_sq = X @ X
        R = ca.SX.eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
        # return transpose, due to convention difference in book
        return R.T

    def from_Matrix(self, arg: ca.SX) -> SO3MrpLieGroupElement:
        assert arg.shape == (3, 3)
        return self.from_Dcm(SO3Dcm.from_Matrix(arg))

    def from_Dcm(self, arg: SO3DcmLieGroupElement) -> SO3MrpLieGroupElement:
        return self.from_Quat(SO3Quat.from_Dcm(arg))

    def from_Quat(self, arg: SO3QuatLieGroupElement) -> SO3MrpLieGroupElement:
        x = ca.SX(3, 1)
        den = 1 + arg.param[0]
        x[0] = arg.param[1] / den
        x[1] = arg.param[2] / den
        x[2] = arg.param[3] / den
        X = self.elem(param=x)
        self.shadow_if_necessary(X)
        return X

    def from_Euler(self, arg: SO3EulerLieGroupElement) -> SO3MrpLieGroupElement:
        return self.from_Matrix(arg.to_Matrix())


SO3EulerB321 = SO3EulerLieGroup(
    euler_type=EulerType.body_fixed, sequence=[Axis.z, Axis.y, Axis.x]
)


@beartype
class SO3MrpLieGroupElement(SO3LieGroupElement):
    """
    This is an SO3Mrp Lie group elem
    """

    def __init__(self, group: SO3MrpLieGroup, param: PARAM_TYPE):
        super().__init__(group, param)


SO3Mrp = SO3MrpLieGroup()
