from __future__ import annotations

import casadi as ca

from beartype import beartype
from beartype.typing import List, Union

from cyecca.lie.base import *

from cyecca.lie.group_rn import *
from cyecca.lie.group_rn import R3LieAlgebraElement

from cyecca.lie.group_so3 import *
from cyecca.lie.group_so3 import SO3LieGroupElement, SO3LieAlgebraElement

from cyecca.symbolic import SERIES, SQUARED_SERIES, taylor_series_near_zero

__all__ = ["se3", "SE3Quat", "SE3Mrp"]


@beartype
class SE3LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=6, matrix_shape=(4, 4))

    def elem(self, param: PARAM_TYPE) -> SE3LieAlgebraElement:
        return SE3LieAlgebraElement(algebra=self, param=param)

    def bracket(self, left: SE3LieAlgebraElement, right: SE3LieAlgebraElement):
        c = left.to_Matrix() @ right.to_Matrix() - right.to_Matrix() @ left.to_Matrix()
        return self.elem(
            param=ca.vertcat(c[0, 3], c[1, 3], c[2, 3], c[2, 1], c[0, 2], c[1, 0])
        )

    def addition(
        self, left: SE3LieAlgebraElement, right: SE3LieAlgebraElement
    ) -> SE3LieAlgebraElement:
        return self.elem(param=left.param + right.param)

    def scalar_multiplication(
        self, left: SCALAR_TYPE, right: SE3LieAlgebraElement
    ) -> SE3LieAlgebraElement:
        return self.elem(param=left * right.param)

    def adjoint(self, arg: SE3LieAlgebraElement):
        v = arg.param[:3]
        vx = so3.elem(arg.param[:3]).to_Matrix()
        w = so3.elem(arg.param[3:]).to_Matrix()
        horz1 = ca.horzcat(w, vx)
        horz2 = ca.horzcat(ca.SX(3, 3), w)
        return ca.vertcat(horz1, horz2)

    def to_Matrix(self, arg: SE3LieAlgebraElement) -> ca.SX:
        Omega = so3.elem(arg.param[3:]).to_Matrix()
        v = arg.param[:3]
        Z14 = ca.SX(1, 4)
        horz = ca.horzcat(Omega, v)
        return ca.vertcat(horz, Z14)

    def from_Matrix(self, arg: ca.SX) -> SE3LieAlgebraElement:
        assert arg.shape == self.matrix_shape
        return self.elem(
            ca.vertcat(arg[0, 3], arg[1, 3], arg[2, 3], arg[2, 1], arg[0, 2], arg[1, 0])
        )

    def wedge(self, arg: Union[ca.SX, ca.DM]) -> SE3LieAlgebraElement:
        return self.elem(param=arg)

    def vee(self, arg: SE3LieAlgebraElement) -> ca.SX:
        return arg.param

    def left_Q(self, arg: SE3LieAlgebraElement) -> ca.SX:
        V = so3.elem(arg.v_b.param).ad()
        O = arg.Omega.ad()
        O_sq = O @ O
        o = arg.Omega.param
        theta_sq = ca.dot(o, o)

        Coeff = ca.vertcat(
            SQUARED_SERIES["(1 - cos(x))/x^2"](theta_sq),  # C0
            SQUARED_SERIES["(x - sin(x))/x^3"](theta_sq),  # C1
            SQUARED_SERIES["(x^2 + 2 cos(x) - 2)/(2 x^4)"](theta_sq),  # C2
            SQUARED_SERIES["(x cos(x) + 2 x - 3 sin(x))/(2 x^5)"](theta_sq),  # C3
            SQUARED_SERIES["(x^2 + x sin(x) + 4 cos(x) - 4)/(2 x^6)"](theta_sq),  # C4
            SQUARED_SERIES["(2 - 2 cos(x) - x sin(x))/(2 x^4))"](theta_sq),  # C5
        )

        Ql = V / 2
        Ql += Coeff[1] * (O @ V + V @ O)
        Ql += Coeff[2] * (O_sq @ V + V @ O_sq)
        Ql += Coeff[3] * (O @ V @ O_sq + O_sq @ V @ O)
        Ql += Coeff[4] * (O_sq @ V @ O_sq)
        Ql += Coeff[5] * (O @ V @ O)

        return Ql

    def left_jacobian(self, arg: SE3LieAlgebraElement) -> ca.SX:
        Ql = arg.left_Q()
        R = arg.Omega.left_jacobian()
        Z = ca.SX.zeros(3, 3)
        return ca.sparsify(ca.vertcat(ca.horzcat(R, Ql), ca.horzcat(Z, R)))

    def left_jacobian_inv(self, arg: SE3LieAlgebraElement) -> ca.SX:
        Ql = arg.left_Q()
        R_inv = arg.Omega.left_jacobian_inv()
        Z = ca.SX.zeros(3, 3)
        return ca.sparsify(
            ca.vertcat(ca.horzcat(R_inv, -R_inv @ Ql @ R_inv), ca.horzcat(Z, R_inv))
        )

    def right_Q(self, arg: SE3LieAlgebraElement) -> ca.SX:
        return self.left_Q(-arg)

    def right_jacobian(self, arg: SE3LieAlgebraElement) -> ca.SX:
        Qr = arg.right_Q()
        R = arg.Omega.right_jacobian()
        Z = ca.SX.zeros(3, 3)
        Jr = ca.sparsify(ca.vertcat(ca.horzcat(R, Qr), ca.horzcat(Z, R)))
        return Jr

    def right_jacobian_inv(self, arg: SE3LieAlgebraElement) -> ca.SX:
        Qr = arg.right_Q()
        R_inv = arg.Omega.right_jacobian_inv()
        Z = ca.SX.zeros(3, 3)
        return ca.sparsify(
            ca.vertcat(ca.horzcat(R_inv, -R_inv @ Qr @ R_inv), ca.horzcat(Z, R_inv))
        )


@beartype
class SE3LieAlgebraElement(LieAlgebraElement):
    """
    This is an SE3 Lie algebra elem
    """

    def __init__(self, algebra: SE3LieAlgebra, param: PARAM_TYPE):
        super().__init__(algebra, param)

    @property
    def v_b(self) -> R3LieAlgebraElement:
        return r3.elem(self.param[:3])

    @property
    def Omega(self) -> SO3LieAlgebraElement:
        return so3.elem(self.param[3:6])

    def left_Q(self) -> ca.SX:
        return self.algebra.left_Q(self)

    def right_Q(self) -> ca.SX:
        return self.algebra.right_Q(self)


se3 = SE3LieAlgebra()


@beartype
class SE3LieGroup(LieGroup):
    def __init__(self, SO3: SO3LieGroup):
        super().__init__(algebra=se3, n_param=SO3.n_param + 3, matrix_shape=(4, 4))
        self.SO3 = SO3

    def elem(self, param: PARAM_TYPE) -> SE3LieGroupElement:
        return SE3LieGroupElement(group=self, param=param)

    def product(self, left: SE3LieGroupElement, right: SE3LieGroupElement):
        R = self.SO3.elem(left.param[3:]).to_Matrix()
        v = R @ right.param[:3] + left.param[:3]
        theta = (self.SO3.elem(left.param[3:]) * self.SO3.elem(right.param[3:])).param
        x = ca.vertcat(v, theta)
        return self.elem(param=x)

    def inverse(self, arg: SE3LieGroupElement):
        v = arg.param[:3]
        theta = arg.param[3:]
        theta_inv = self.SO3.elem(param=theta).inverse()
        R = self.SO3.elem(param=theta).to_Matrix()
        p = -R.T @ v
        return self.elem(param=ca.vertcat(p, theta_inv.param))

    def identity(self) -> SE3LieGroupElement:
        return self.elem(ca.vertcat(ca.SX(3, 1), self.SO3.identity().param))

    def adjoint(self, arg: SE3LieGroupElement):
        v = arg.param[:3]
        vx = so3.elem(param=v).to_Matrix()
        R = self.SO3.elem(param=arg.param[3:]).to_Matrix()
        horz1 = ca.horzcat(R, ca.times(vx, R))
        horz2 = ca.horzcat(ca.SX(3, 3), R)
        return ca.vertcat(horz1, horz2)

    def exp(self, arg: SE3LieAlgebraElement) -> SE3LieGroupElement:
        p = arg.Omega.left_jacobian() @ arg.v_b.param
        R = arg.Omega.exp(self.SO3).param
        return self.elem(ca.vertcat(p, R))

    def log(self, arg: SE3LieGroupElement) -> SE3LieAlgebraElement:
        Omega = arg.R.log()
        u = Omega.left_jacobian_inv() @ arg.p.param
        return self.algebra.elem(ca.vertcat(u, Omega.param))

    def to_Matrix(self, arg: SE3LieGroupElement) -> ca.SX:
        return ca.vertcat(
            ca.horzcat(arg.R.to_Matrix(), arg.p.param),
            ca.horzcat(ca.SX(1, 3), ca.SX.eye(1)),
        )

    def from_Matrix(self, arg: ca.SX) -> SE3LieGroupElement:
        assert arg.shape == self.matrix_shape
        raise NotImplementedError("")


@beartype
class SE3LieGroupElement(LieGroupElement):
    """
    This is an SE3 Lie group elem
    """

    def __init__(self, group: SE3LieGroup, param: PARAM_TYPE):
        super().__init__(group, param)

    @property
    def p(self) -> R3LieAlgebraElement:
        return r3.elem(self.param[:3])

    @property
    def R(self) -> SO3LieGroupElement:
        return self.group.SO3.elem(self.param[3:])


SE3Mrp = SE3LieGroup(SO3=SO3Mrp)
SE3Quat = SE3LieGroup(SO3=SO3Quat)
