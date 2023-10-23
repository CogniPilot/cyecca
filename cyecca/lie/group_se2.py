from __future__ import annotations

import casadi as ca

from beartype import beartype
from beartype.typing import List

from cyecca.lie.base import *
from cyecca.lie.group_so2 import *
from cyecca.lie.group_rn import *
from cyecca.lie.group_rn import R2LieAlgebraElement
from cyecca.lie.group_so2 import SO2LieGroupElement, SO2LieAlgebraElement
from cyecca.symbolic import SERIES

__all__ = ["se2", "SE2"]


@beartype
class SE2LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=3, matrix_shape=(3, 3))

    def elem(self, param: PARAM_TYPE) -> SE2LieAlgebraElement:
        return SE2LieAlgebraElement(algebra=self, param=param)

    def bracket(
        self, left: SE2LieAlgebraElement, right: SE2LieAlgebraElement
    ) -> SE2LieAlgebraElement:
        ## TODO: hard code the bracket here, avoid matrix math
        c = left.to_Matrix() @ right.to_Matrix() - right.to_Matrix() @ left.to_Matrix()
        return self.elem(param=ca.vertcat(c[0, 2], c[1, 2], c[1, 0]))

    def addition(
        self, left: SE2LieAlgebraElement, right: SE2LieAlgebraElement
    ) -> SE2LieAlgebraElement:
        return self.elem(param=left.param + right.param)

    def scalar_multiplication(
        self, left: SCALAR_TYPE, right: SE2LieAlgebraElement
    ) -> SE2LieAlgebraElement:
        return self.elem(param=left * right.param)

    def adjoint(self, arg: SE2LieAlgebraElement) -> ca.SX:
        x = arg.param[0, 0]
        y = arg.param[1, 0]
        theta = arg.param[2, 0]
        ad = ca.SX(3, 3)
        ad[0, 1] = -theta
        ad[1, 0] = theta
        ad[0, 2] = y
        ad[1, 2] = -x
        return ad

    def to_Matrix(self, arg: SE2LieAlgebraElement) -> ca.SX:
        return ca.vertcat(ca.horzcat(arg.Omega.to_Matrix(), arg.v_b.param), ca.SX(1, 3))

    def from_Matrix(self, arg: ca.SX) -> SE2LieAlgebraElement:
        raise NotImplementedError("")

    def wedge(self, arg: (ca.SX, ca.DM)) -> SE2LieAlgebraElement:
        return self.elem(param=arg)

    def vee(self, arg: SE2LieAlgebraElement) -> ca.SX:
        return arg.param


@beartype
class SE2LieAlgebraElement(LieAlgebraElement):
    """
    This is an SE2 Lie algebra elem
    """

    def __init__(self, algebra: SE2LieAlgebra, param: PARAM_TYPE):
        super().__init__(algebra, param)

    @property
    def v_b(self) -> R2LieAlgebraElement:
        return r2.elem(self.param[:2])

    @property
    def Omega(self) -> SO2LieAlgebraElement:
        return so2.elem(self.param[2])


@beartype
class SE2LieGroup(LieGroup):
    def __init__(self):
        super().__init__(algebra=se2, n_param=3, matrix_shape=(3, 3))

    def elem(self, param: PARAM_TYPE) -> SE2LieGroupElement:
        return SE2LieGroupElement(group=self, param=param)

    def product(self, left: SE2LieGroupElement, right: SE2LieGroupElement):
        p = left.R @ right.p + left.p
        R = left.R * right.R
        return self.elem(ca.vertcat(p.param, R.param))

    def inverse(self, arg: SE2LieGroupElement) -> SE2LieGroupElement:
        p = -(arg.R.inverse() @ arg.p)
        return self.elem(param=ca.vertcat(p.param, -arg.R.param))

    def identity(self) -> SE2LieGroupElement:
        return self.elem(param=ca.vertcat(R2.identity().param, SO2.identity().param))

    def adjoint(self, arg: SE2LieGroupElement):
        v = ca.vertcat(arg.param[1], -arg.param[0])
        theta = SO2.elem(param=arg.param[2:])
        horz1 = ca.horzcat(theta.to_Matrix(), v)
        horz2 = ca.horzcat(ca.SX(1, 2), 1)
        return ca.vertcat(horz1, horz2)

    def exp(self, arg: SE2LieAlgebraElement) -> SE2LieGroupElement:
        theta = arg.Omega.param
        sin_th = ca.sin(theta)
        cos_th = ca.cos(theta)
        a = SERIES["sin(x)/x"](theta)
        b = SERIES["(1 - cos(x))/x"](theta)
        V = ca.vertcat(ca.horzcat(a, -b), ca.horzcat(b, a))
        p = V @ arg.v_b.param
        return self.elem(ca.vertcat(p, theta))

    def log(self, arg: SE2LieGroupElement) -> SE2LieAlgebraElement:
        theta = arg.R.param
        x = ca.SX.sym("x")
        a = SERIES["sin(x)/x"](theta)
        b = SERIES["(1 - cos(x))/x"](theta)
        V_inv = ca.SX(2, 2)
        V_inv[0, 0] = a
        V_inv[0, 1] = b
        V_inv[1, 0] = -b
        V_inv[1, 1] = a
        V_inv = V_inv / (a**2 + b**2)  # TODO, check being 0 is impossible
        p = V_inv @ arg.p.param
        return self.algebra.elem(ca.vertcat(p, theta))

    def to_Matrix(self, arg: SE2LieGroupElement) -> ca.SX:
        return ca.vertcat(
            ca.horzcat(arg.R.to_Matrix(), arg.p.param),
            ca.horzcat(ca.SX(1, 2), ca.SX.eye(1)),
        )

    def from_Matrix(self, arg: ca.SX) -> SE2LieAlgebraElement:
        return self.LieAlgebraElement(
            arg[0, 2],
            arg[1,],
            arg[1, 0],
        )


@beartype
class SE2LieGroupElement(LieGroupElement):
    """
    This is an SE2 Lie group elem, not necessarily represented as a matrix
    """

    def __init__(self, group: SE2LieGroup, param: PARAM_TYPE):
        super().__init__(group, param)

    @property
    def p(self) -> R2LieAlgebraElement:
        return r2.elem(self.param[:2])

    @property
    def R(self) -> SO2LieGroupElement:
        return SO2.elem(self.param[2])


se2 = SE2LieAlgebra()
SE2 = SE2LieGroup()
