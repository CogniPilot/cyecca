from __future__ import annotations

import casadi as ca

from beartype import beartype
from beartype.typing import List

from cyecca.lie.base import *
from cyecca.lie.group_so2 import *
from cyecca.symbolic import SERIES

__all__ = ["se2", "SE2"]


@beartype
class SE2LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=3, matrix_shape=(3, 3))

    def bracket(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        ## TODO: hard code the bracket here, avoid matrix math
        c = left.to_Matrix() @ right.to_Matrix() - right.to_Matrix() @ left.to_Matrix()
        return self.elem(param=ca.vertcat(c[0, 2], c[1, 2], c[1, 0]))

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

    def adjoint(self, arg: LieAlgebraElement) -> ca.SX:
        assert self == arg.algebra
        x = arg.param[0, 0]
        y = arg.param[1, 0]
        theta = arg.param[2, 0]
        ad = ca.SX(3, 3)
        ad[0, 1] = -theta
        ad[1, 0] = theta
        ad[0, 2] = y
        ad[1, 2] = -x
        return ad

    def to_Matrix(self, arg: LieAlgebraElement) -> ca.SX:
        assert self == arg.algebra
        Omega = so2.elem(arg.param[2:, 0]).to_Matrix()
        v = arg.param[:2, 0]
        Z13 = ca.SX(1, 3)
        horz = ca.horzcat(Omega, v)
        return ca.vertcat(horz, Z13)

    def from_Matrix(self, arg: ca.SX) -> LieAlgebraElement:
        raise NotImplementedError("")

    def wedge(self, arg: (ca.SX, ca.DM)) -> LieAlgebraElement:
        return self.elem(param=arg)

    def vee(self, arg: LieAlgebraElement) -> ca.SX:
        assert self == arg.algebra
        return arg.param


@beartype
class SE2LieGroup(LieGroup):
    def __init__(self):
        super().__init__(algebra=se2, n_param=3, matrix_shape=(3, 3))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        R = SO2.elem(left.param[2:, 0]).to_Matrix()
        v = R @ right.param[:2, 0] + left.param[:2, 0]
        x = ca.vertcat(v, left.param[2:, 0] + right.param[2:, 0])
        return self.elem(param=x)

    def inverse(self, arg: LieGroupElement) -> LieGroupElement:
        assert self == arg.group
        v = arg.param[:2, 0]
        theta = arg.param[2:, 0]
        R = SO2.elem(param=theta).to_Matrix()
        p = -R.T @ v
        return self.elem(param=ca.vertcat(p, -theta))

    def identity(self) -> LieGroupElement:
        return self.elem(param=ca.SX(self.n_param, 1))

    def adjoint(self, arg: LieGroupElement):
        assert self == arg.group
        v = ca.vertcat(arg.param[1], -arg.param[0])
        theta = SO2.elem(param=arg.param[2:])
        horz1 = ca.horzcat(theta.to_Matrix(), v)
        horz2 = ca.horzcat(ca.SX(1, 2), 1)
        return ca.vertcat(horz1, horz2)

    def exp(self, arg: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == arg.algebra
        theta = arg.param[2, 0]
        sin_th = ca.sin(theta)
        cos_th = ca.cos(theta)
        a = SERIES["sin(x)/x"](theta)
        b = SERIES["(1 - cos(x))/x"](theta)
        horz1 = ca.horzcat(a, -b)
        horz2 = ca.horzcat(b, a)
        V = ca.vertcat(horz1, horz2)
        v = V @ arg.param[:2]
        return self.elem(ca.vertcat(v, theta))

    def log(self, arg: LieGroupElement) -> LieAlgebraElement:
        assert self == arg.group
        v = arg.param[:2, 0]
        theta = arg.param[2, 0]
        x = ca.SX.sym("x")
        a = SERIES["sin(x)/x"](theta)
        b = SERIES["(1 - cos(x))/x"](theta)
        V_inv = ca.SX(2, 2)
        V_inv[0, 0] = a
        V_inv[0, 1] = b
        V_inv[1, 0] = -b
        V_inv[1, 1] = a
        V_inv = V_inv / (a**2 + b**2)
        p = V_inv @ v
        return self.algebra.elem(ca.vertcat(p, theta))

    def to_Matrix(self, arg: LieGroupElement) -> ca.SX:
        assert self == arg.group
        R = SO2.elem(arg.param[2:, 0]).to_Matrix()
        t = arg.param[:2, 0]
        Z12 = ca.SX(1, 2)
        I1 = ca.SX_eye(1)
        horz1 = ca.horzcat(R, t)
        horz2 = ca.horzcat(Z12, I1)
        return ca.vertcat(horz1, horz2)

    def from_Matrix(self, arg: ca.SX) -> LieAlgebraElement:
        assert arg.shape == self.matrix_shape
        return self.LieAlgebraElement(arg[0, 2], arg[1,], arg[1, 0])


se2 = SE2LieAlgebra()
SE2 = SE2LieGroup()
