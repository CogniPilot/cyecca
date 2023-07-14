from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy import floating

import casadi as ca

from beartype import beartype
from beartype.typing import List

from ._base import LieAlgebra, LieAlgebraElement, LieGroup, LieGroupElement
from ._so2 import SO2, so2


@beartype
class SE2LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=3, matrix_shape=(3, 3))

    def bracket(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        c = left.to_matrix()@right.to_matrix() - right.to_matrix()@left.to_matrix()
        return self.element(param=ca.vertcat(c[0, 2], c[1, 2], c[1, 0]))

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(param=left.param + right.param)

    def scalar_multipication(self, left : (float, int), right: LieAlgebraElement) -> LieAlgebraElement:
        assert self == right.algebra
        return self.element(param=left * right.param)

    def adjoint(self, left: LieAlgebraElement) -> ca.SX:
        assert self == left.algebra
        x = left.param[0,0]
        y = left.param[1,0]
        theta = left.param[2,0]
        ad = ca.SX.zeros(3,3)
        ad[0, 1] = -theta
        ad[1, 0] = theta
        ad[0, 2] = y
        ad[1, 2] = -x
        return ad

    def to_matrix(self, left: LieAlgebraElement) -> ca.SX:
        assert self == left.algebra
        Omega = so2.element(left.param[2:,0]).to_matrix()
        v = left.param[:2,0]
        Z13 = ca.SX.zeros(1,3)
        horz = ca.horzcat(Omega, v)
        return ca.vertcat(horz, Z13)
    
    def wedge(self, left: (ca.SX, ca.DM)) -> LieAlgebraElement:
        self = SE2LieAlgebra()
        return self.element(param=left)
    
    def vee(self, left: LieAlgebraElement) -> ca.SX:
        assert self == left.algebra
        return left.param


@beartype
class SE2LieGroup(LieGroup):
    def __init__(self):
        super().__init__(algebra=se2, n_param=3, matrix_shape=(3, 3))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        R = SO2.element(left.param[2:,0]).to_matrix()
        v = (R@right.param[:2,0]+left.param[:2,0])
        x = ca.vertcat(v, left.param[2:,0]+right.param[2:,0])
        return self.element(param=x)

    def inverse(self, left: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        v = left.param[:2,0]
        theta = left.param[2:,0]
        R = SO2.element(param=theta).to_matrix()
        p = -R.T@v
        return self.element(param=ca.vertcat(p, -theta))

    def identity(self) -> LieGroupElement:
        return self.element(param=np.zeros(self.n_param))

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        v = np.array([left.param[1], -left.param[0]])
        theta = SO2.element(param=left.param[2:])
        horz1 = ca.horzcat(theta.to_matrix(), v.reshape(2,1))
        horz2 = ca.horzcat(ca.SX.zeros(1,2), 1)
        return ca.vertcat(horz1, horz2)
    
    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        theta = left.param[2,0]
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        a = sin_th / theta
        b = (1 - cos_th) / theta
        V = np.array([
            [a, -b],
            [b, a]])
        v = V @ left.param[:2,0]
        return self.element(ca.vertcat(v, theta))

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        v = left.param[:2,0]
        theta = left.param[2,0]
        x = ca.SX.sym('x')
        C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < 1e-3, 1 - x**2/6 + x**4/120, ca.sin(x)/x)])
        C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < 1e-3, x/2 - x**3/24 + x**5/720, (1 - ca.cos(x))/x)])
        a = C1(theta)
        b = C2(theta)
        V_inv = ca.SX(2,2)
        V_inv[0,0] = a
        V_inv[0,1] = b
        V_inv[1,0] = -b
        V_inv[1,1] = a
        V_inv = V_inv/(a**2 + b**2)
        p = V_inv@v
        return self.algebra.element(ca.vertcat(p, theta))

    def to_matrix(self, left: LieGroupElement) -> ca.SX:
        assert self == left.group
        R = SO2.element(left.param[2:,0]).to_matrix()
        t = left.param[:2,0]
        Z12 = ca.SX.zeros(1,2)
        I1 = ca.SX_eye(1)
        horz1 = ca.horzcat(R, t)
        horz2 = ca.horzcat(Z12, I1)
        return ca.vertcat(horz1, horz2)


se2 = SE2LieAlgebra()
SE2 = SE2LieGroup()
