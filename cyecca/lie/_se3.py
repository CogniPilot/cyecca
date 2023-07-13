from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy import floating

from beartype import beartype
from beartype.typing import List

from ._base import LieAlgebra, LieAlgebraElement, LieGroup, LieGroupElement
from ._so3 import so3, SO3Euler, SO3Quat, SO3MRP


@beartype
class SE3LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=6, matrix_shape=(4, 4))

    def bracket(self, left: LieAlgebraElement, right: LieAlgebraElement):
        assert self == left.algebra
        assert self == right.algebra
        return self.element(param=np.array([0]))

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(param=left.param + right.param)

    def scalar_multipication(
        self, left: Real, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == right.algebra
        return self.element(param=left * right.param)

    def adjoint(self, left: LieAlgebraElement):
        assert self == left.algebra
        v = left.param[:3]
        vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1],v[0],0]])
        w = so3(left.param[3:]).to_matrix()
        return np.block([[w, vx],[np.zeros((3,3)), w]])

    def to_matrix(self, left: LieAlgebraElement) -> npt.NDArray[np.floating]:
        assert self == left.algebra
        Omega = so3(left.param[2]).to_matrix()
        v = self.param[:3, 0]
        Z14 = np.zeros(4)
        return np.block([
            [Omega, v],
            [Z14]])
    
    def wedge(self, left: npt.NDArray[np.floating]) -> LieAlgebraElement:
        self = SE3LieAlgebra()
        return self.element(param=left)
    
    def vee(self, left: LieAlgebraElement) -> npt.NDArray[np.floating]:
        assert self == left.algebra
        return left.param


@beartype
class SE3LieGroup(LieGroup):
    def __init__(self, SO3=None):
        if SO3==None:
            self.SO3 = SO3Quat
        else:
            self.SO3 = SO3
        super().__init__(algebra=se3, n_param=7, matrix_shape=(4, 4))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        return self.element(left.param + right.param)

    def inverse(self, left):
        assert self == left.group
        v = left.param[:3]
        theta = left.param[3:]
        R = SO3.element(param=theta).to_matrix()
        p = -R.T@v
        return self.element(param=np.array([p[0], p[1], p[2], -theta]))

    def identity(self) -> LieGroupElement:
        return self.element(np.array.zeros(self.n_param, 1))

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        v = np.array([left.param[1], -left.param[0]])
        theta = SO2.element(param=left.param[2])
        return np.block([[theta.to_matrix(), v.reshape(2,1)],
                         [np.zeros((1,2)), 1]])

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        theta = left.param[2]
        sin_th = sin(theta)
        cos_th = cos(theta)
        a = sin_th / theta
        b = (1 - cos_th) / theta
        V = np.array([
            [a, -b],
            [b, a]])
        v = V @ left.param[:2, 0]
        return self.element(np.array([v[0], v[1], theta]))

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        v = left.param[:2]
        theta = left.param[2]
        with np.errstate(divide='ignore',invalid='ignore'):
            a = np.where(np.abs(theta) < 1e-3, 1 - theta**2/6 + theta**4/120, np.sin(theta)/theta)
            b = np.where(np.abs(theta) < 1e-3, theta/2 - theta**3/24 + theta**5/720, (1 - np.cos(theta))/theta)
        V_inv = np.array([
            [a, b],
            [-b, a]
        ])/(a**2 + b**2)
        p = V_inv@v
        return se2algebra.element(np.array([p[0], p[1], theta]))

    def to_matrix(selfm left: LieAlgebraElement) -> npt.NDArray[np.floating]:
        assert self == left.group
        R = SO2.to_matrix(left.param[3:])
        t = left.param[:3]
        Z13 = np.zeros(3)
        I1 = np.eye(1)
        return np.array(Blocknp.array([
            [R, t],
            [Z12, I1],
        ]))


se2 = SE2LieAlgebra()
SE2 = SE2LieGroup()
