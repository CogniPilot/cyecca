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
        c = left.to_matrix()@right.to_matrix() - right.to_matrix()@left.to_matrix()
        return self.element(param=np.array([c[0, 3], c[1, 3], c[2, 3], c[2, 1], c[0, 2], c[1, 0]]))

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
        w = so3.element(left.param[3:]).to_matrix()
        return np.block([
            [w, vx],
            [np.zeros((3,3)), w]
        ])

    def to_matrix(self, left: LieAlgebraElement) -> npt.NDArray[np.floating]:
        assert self == left.algebra
        Omega = so3.element(left.param[3:]).to_matrix()
        v = left.param[:3].reshape(3,1)
        Z14 = np.zeros(4)
        return np.block([
            [Omega, v],
            [Z14]
        ])
    
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
        R = self.SO3.element(left.param[3:]).to_matrix()
        v = (R@right.param[:3]+left.param[:3])
        theta = (self.SO3.element(left.param[3:])*self.SO3.element(right.param[3:])).param
        x = np.block([v, theta])
        return self.element(param=x)

    def inverse(self, left):
        assert self == left.group
        v = left.param[:3]
        theta = left.param[3:]
        theta_inv = self.SO3.element(param=theta).inverse()
        R = self.SO3.element(param=theta).to_matrix()
        p = -R.T@v
        return self.element(param=np.block([p, theta_inv.param]))

    def identity(self) -> LieGroupElement:
        return self.element(np.zeros((self.n_param, 1)))

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        v = left.param[:3]
        vx = so3.element(param=v).to_matrix()
        R = self.SO3.element(param=left.param[3:]).to_matrix()
        return np.block([
            [R, vx@R],
            [np.zeros((3,3)), R]
        ])

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        v = left.param
        omega_so3 = self.SO3.algebra.element(v[3:])  # grab only rotation terms for so3 uses ##corrected to v_so3 = v[3:6]
        omega_matrix = omega_so3.to_matrix()  # matrix for so3
        omega = np.linalg.norm(v[3:])  # theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)
        theta = omega_so3.exp(self.SO3).param
        
        # translational components u
        u = np.array([v[0],v[1],v[2]])

        C1 = np.where(np.abs(omega)<1e-7, 1 - omega ** 2 / 6 + omega ** 4 / 120, np.sin(omega)/omega)
        C2 = np.where(np.abs(omega)<1e-7, 0.5 - omega ** 2 / 24 + omega ** 4 / 720, (1 - np.cos(omega)) / omega ** 2)
        C = np.where(np.abs(omega)<1e-7, 1/6 - omega ** 2 /120 + omega ** 4 / 5040, (1 - C1) / omega ** 2)

        V = np.eye(3) + C2 * omega_matrix + C * omega_matrix @ omega_matrix

        return self.element(np.block([V@u, theta]))

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        X = left.to_matrix()
        angle = left.param[3:] 
        R = X[0:3, 0:3] # get the SO3 Lie group matrix
        theta = np.arccos((np.trace(R) - 1) / 2)
        angle_so3 = self.SO3.element(angle).log()
        wSkew = angle_so3.to_matrix()
        C1 = np.where(np.abs(theta)<1e-7, 1 - theta ** 2 / 6 + theta ** 4 / 120, np.sin(theta)/theta)
        C2 = np.where(np.abs(theta)<1e-7, 0.5 - theta ** 2 / 24 + theta ** 4 / 720, (1 - np.cos(theta)) / theta ** 2)
        V_inv = (
            np.eye(3)
            - wSkew / 2
            + (1 / theta**2) * (1 - C1 / (2 * C2)) * wSkew @ wSkew
        )

        t = X[0:3,3]
        uInv = V_inv @ t
        return self.algebra.element(np.block([uInv, angle_so3.param]))

    def to_matrix(self, left: LieGroupElement) -> npt.NDArray[np.floating]:
        assert self == left.group
        R = self.SO3.element(left.param[3:]).to_matrix()
        t = left.param[:3].reshape(3,1)
        Z13 = np.zeros(3)
        I1 = np.eye(1)
        return np.block([
            [R, t],
            [Z13, I1],
        ])


se3 = SE3LieAlgebra()
SE3 = SE3LieGroup
