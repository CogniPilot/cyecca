from __future__ import annotations

from beartype import beartype
from beartype.typing import List

import casadi as ca

from cyecca.lie.base import *
from cyecca.lie.group_rn import *
from cyecca.lie.group_so3 import *
from cyecca.lie.group_so3 import SO3LieGroupElement
from cyecca.symbolic import SERIES


__all__ = ["se23", "SE23Quat", "SE23Mrp"]


@beartype
class SE23LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=9, matrix_shape=(5, 5))

    def elem(self, param: PARAM_TYPE) -> SE23LieAlgebraElement:
        return SE23LieAlgebraElement(algebra=self, param=param)

    def bracket(self, left: SE23LieAlgebraElement, right: SE23LieAlgebraElement):
        c = left.to_Matrix() @ right.to_Matrix() - right.to_Matrix() @ left.to_Matrix()
        return self.elem(
            param=ca.vertcat(
                c[0, 4],
                c[1, 4],
                c[2, 4],
                c[0, 3],
                c[1, 3],
                c[2, 3],
                c[2, 1],
                c[0, 2],
                c[1, 0],
            )
        )

    def addition(
        self, left: SE23LieAlgebraElement, right: SE23LieAlgebraElement
    ) -> SE23LieAlgebraElement:
        return self.elem(param=left.param + right.param)

    def scalar_multiplication(
        self, left: SCALAR_TYPE, right: SE23LieAlgebraElement
    ) -> SE23LieAlgebraElement:
        return self.elem(param=left * right.param)

    def adjoint(self, arg: SE23LieAlgebraElement):
        a = arg.param[3:6]
        ax = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
        v = arg.param[0:3]
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        w = so3.elem(arg.param[6:]).to_Matrix()
        return np.block([[w, vx], [ca.SX(3, 3), w]])

    def to_Matrix(self, arg: SE23LieAlgebraElement) -> ca.SX:
        Omega = so3.elem(arg.param[6:]).to_Matrix()
        p = arg.param[:3].reshape(3, 1)
        v = arg.param[3:6].reshape(3, 1)
        Z15 = ca.SX(1, 5)
        return np.block([[Omega, v, p], [Z15]])

    def from_Matrix(self, arg: ca.SX) -> SE23LieAlgebraElement:
        raise NotImplementedError("")


@beartype
class SE23LieAlgebraElement(LieAlgebraElement):
    """
    This is an SE23 Lie algebra elem
    """

    def __init__(self, algebra: RnLieAlgebra, param: PARAM_TYPE):
        super().__init__(algebra, param)


@beartype
class SE23LieGroup(LieGroup):
    def __init__(self, SO3: SO3LieGroup):
        self.SO3 = SO3
        super().__init__(algebra=se23, n_param=SO3.n_param + 6, matrix_shape=(5, 5))

    def elem(self, param: PARAM_TYPE) -> SE23LieGroupElement:
        return SE23LieGroupElement(group=self, param=param)

    def product(self, left: SE23LieGroupElement, right: SE23LieGroupElement):
        p = left.p + left.R @ right.p
        v = left.v + left.R @ right.v
        R = left.R * right.R
        return self.elem(param=ca.vertcat(p, v, R.param))

    def inverse(self, arg):
        R_inv = arg.R.inverse()
        p_inv = -(R_inv @ arg.p)
        v_inv = -(R_inv @ arg.v)
        return self.elem(param=ca.vertcat(p_inv, v_inv, R_inv.param))

    def identity(self) -> SE23LieGroupElement:
        p = R3.identity()
        v = R3.identity()
        R = self.SO3.identity()
        return self.elem(param=ca.vertcat(p.param, v.param, R.param))

    def adjoint(self, arg: SE23LieGroupElement):
        v = arg.param[:3]
        vx = so3.elem(param=v).to_Matrix()
        R = self.SO3.elem(param=arg.param[3:]).to_Matrix()
        return np.block([[R, vx @ R], [ca.SX(3, 3), R]])

    def exp(self, arg: SE23LieAlgebraElement) -> SE23LieGroupElement:
        v = arg.param
        omega_so3 = self.SO3.algebra.elem(
            v[3:]
        )  # grab only rotation terms for so3 uses ##corrected to v_so3 = v[3:6]
        omega_matrix = omega_so3.to_Matrix()  # matrix for so3
        omega = np.linalg.norm(
            v[3:]
        )  # theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)
        theta = omega_so3.exp(self.SO3).param

        # translational components u
        u = np.array([v[0], v[1], v[2]])

        C1 = SERIES["sin(x)/x"]
        C2 = SERIES["(1 - cos(x))/x^2"]
        C = np.where(
            np.abs(omega) < 1e-7,
            1 / 6 - omega**2 / 120 + omega**4 / 5040,
            (1 - C1) / omega**2,
        )

        V = np.eye(3) + C2 * omega_matrix + C * omega_matrix @ omega_matrix

        return self.elem(np.block([V @ u, theta]))

    def log(self, arg: SE23LieGroupElement) -> SE23LieAlgebraElement:
        X = arg.to_Matrix()
        angle = arg.param[3:]
        R = X[0:3, 0:3]  # get the SO3 Lie groupbase matrix
        theta = np.arccos((np.trace(R) - 1) / 2)
        angle_so3 = self.SO3.elem(angle).log()
        wSkew = angle_so3.to_Matrix()
        C1 = SERIES["sin(x)/x"]
        C2 = SERIES["(1 - cos(x))/x^2"]
        V_inv = (
            np.eye(3)
            - wSkew / 2
            + (1 / theta**2) * (1 - C1 / (2 * C2)) * wSkew @ wSkew
        )

        t = X[0:3, 3]
        uInv = V_inv @ t
        return self.algebra.elem(np.block([uInv, angle_so3.param]))

    def to_Matrix(self, arg: SE23LieGroupElement) -> ca.SX:
        return ca.vertcat(
            ca.horzcat(arg.R.to_Matrix(), arg.v, arg.p),
            ca.horzcat(ca.SX(2, 3), ca.SX.eye(2)),
        )

    def from_Matrix(self, arg: ca.SX) -> SE23LieGroupElement:
        raise NotImplementedError("")


@beartype
class SE23LieGroupElement(LieGroupElement):
    """
    This is an SE23 Lie group elem, not necessarily represented as a matrix
    """

    def __init__(self, group: SE23LieGroup, param: PARAM_TYPE):
        super().__init__(group, param)

    @property
    def p(self) -> ca.SX:
        return self.param[:3]

    @property
    def v(self) -> ca.SX:
        return self.param[3:6]

    @property
    def R(self) -> SO3LieGroupElement:
        return self.group.SO3.elem(self.param[6:])


se23 = SE23LieAlgebra()
SE23Mrp = SE23LieGroup(SO3=SO3Mrp)
SE23Quat = SE23LieGroup(SO3=SO3Quat)
