from __future__ import annotations

import casadi as ca

from beartype import beartype
from beartype.typing import List

from cyecca.lie.base import *
from cyecca.lie.group_so3 import *
from cyecca.symbolic import SERIES, taylor_series_near_zero

__all__ = ["se3", "SE3EulerB321", "SE3Quat", "SE3Mrp"]


@beartype
class SE3LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=6, matrix_shape=(4, 4))

    def bracket(self, left: LieAlgebraElement, right: LieAlgebraElement):
        assert self == left.algebra
        assert self == right.algebra
        c = left.to_Matrix() @ right.to_Matrix() - right.to_Matrix() @ left.to_Matrix()
        return self.elem(
            param=ca.vertcat(c[0, 3], c[1, 3], c[2, 3], c[2, 1], c[0, 2], c[1, 0])
        )

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.elem(param=left.param + right.param)

    def scalar_multipication(
        self, left: (float, int), right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == right.algebra
        return self.elem(param=left * right.param)

    def adjoint(self, arg: LieAlgebraElement):
        assert self == arg.algebra
        v = arg.param[:3]
        vx = so3.elem(arg.param[:3]).to_Matrix()
        w = so3.elem(arg.param[3:]).to_Matrix()
        horz1 = ca.horzcat(w, vx)
        horz2 = ca.horzcat(ca.SX(3, 3), w)
        return ca.vertcat(horz1, horz2)

    def to_Matrix(self, arg: LieAlgebraElement) -> ca.SX:
        assert self == arg.algebra
        Omega = so3.elem(arg.param[3:]).to_Matrix()
        v = arg.param[:3]
        Z14 = ca.SX(1, 4)
        horz = ca.horzcat(Omega, v)
        return ca.vertcat(horz, Z14)

    def from_Matrix(self, arg: ca.SX) -> LieAlgebraElement:
        assert arg.shape == self.matrix_shape
        return self.elem(
            ca.vertcat(arg[0, 3], arg[1, 3], arg[2, 3], arg[2, 1], arg[0, 2], arg[1, 0])
        )

    def wedge(self, arg: (ca.SX, ca.DM)) -> LieAlgebraElement:
        return self.elem(param=arg)

    def vee(self, arg: LieAlgebraElement) -> ca.SX:
        assert self == arg.algebra
        return arg.param


se3 = SE3LieAlgebra()


@beartype
class SE3LieGroup(LieGroup):
    def __init__(self, SO3: SO3LieGroup):
        super().__init__(algebra=se3, n_param=SO3.n_param + 3, matrix_shape=(4, 4))
        self.SO3 = SO3

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        R = self.SO3.elem(left.param[3:]).to_Matrix()
        v = R @ right.param[:3] + left.param[:3]
        theta = (self.SO3.elem(left.param[3:]) * self.SO3.elem(right.param[3:])).param
        x = ca.vertcat(v, theta)
        return self.elem(param=x)

    def inverse(self, arg: LieGroupElement):
        assert self == arg.group
        v = arg.param[:3]
        theta = arg.param[3:]
        theta_inv = self.SO3.elem(param=theta).inverse()
        R = self.SO3.elem(param=theta).to_Matrix()
        p = -R.T @ v
        return self.elem(param=ca.vertcat(p, theta_inv.param))

    def identity(self) -> LieGroupElement:
        return self.elem(ca.vertcat(ca.SX(3, 1), self.SO3.identity().param))

    def adjoint(self, arg: LieGroupElement):
        assert self == arg.group
        v = arg.param[:3]
        vx = so3.elem(param=v).to_Matrix()
        R = self.SO3.elem(param=arg.param[3:]).to_Matrix()
        horz1 = ca.horzcat(R, ca.times(vx, R))
        horz2 = ca.horzcat(ca.SX(3, 3), R)
        return ca.vertcat(horz1, horz2)

    def exp(self, arg: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == arg.algebra
        v = arg.param
        omega_so3 = self.SO3.algebra.elem(
            v[3:]
        )  # grab only rotation terms for so3 uses ##corrected to v_so3 = v[3:6]
        Omega = omega_so3.to_Matrix()  # matrix for so3
        omega = ca.norm_2(
            v[3:]
        )  # theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)
        theta = omega_so3.exp(self.SO3).param

        # translational components u
        u = ca.vertcat(v[0], v[1], v[2])

        A = SERIES["(1 - cos(x))/x^2"](omega)
        B = SERIES["(1 - sin(x))/x^3"](omega)
        V = ca.SX.eye(3) + A * Omega + B * Omega @ Omega

        return self.elem(ca.vertcat(V @ u, theta))

    def log(self, arg: LieGroupElement) -> LieAlgebraElement:
        assert self == arg.group
        X = arg.to_Matrix()
        angle = arg.param[3:]
        R = X[0:3, 0:3]  # get the SO3 Lie group matrix
        omega = ca.acos((ca.trace(R) - 1) / 2)
        angle_so3 = self.SO3.elem(angle).log()
        Omega = angle_so3.to_Matrix()

        A = SERIES["sin(x)/x"](omega)
        B = SERIES["(1 - x*sin(x)/(2*(1 - cos(x))))/x^2"](omega)
        V_inv = ca.SX.eye(3) - A * Omega + B * Omega @ Omega
        t = X[0:3, 3]
        uInv = V_inv @ t
        return self.algebra.elem(ca.vertcat(uInv, angle_so3.param))

    def to_Matrix(self, arg: LieGroupElement) -> ca.SX:
        assert self == arg.group
        R = self.SO3.elem(arg.param[3:]).to_Matrix()
        t = arg.param[:3]
        Z13 = ca.SX(1, 3)
        I1 = ca.SX.eye(1)
        horz1 = ca.horzcat(R, t)
        horz2 = ca.horzcat(Z13, I1)
        return ca.vertcat(horz1, horz2)

    def from_Matrix(self, arg: ca.SX) -> LieGroupElement:
        assert arg.shape == self.matrix_shape
        raise NotImplementedError("")


SE3Mrp = SE3LieGroup(SO3=SO3Mrp)
SE3EulerB321 = SE3LieGroup(SO3=SO3EulerB321)
SE3Quat = SE3LieGroup(SO3=SO3Quat)
