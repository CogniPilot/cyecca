from __future__ import annotations

import casadi as ca

from beartype import beartype
from beartype.typing import List

from cyecca.lie.base import *

from cyecca.lie.group_rn import *
from cyecca.lie.group_rn import R3LieAlgebraElement

from cyecca.lie.group_so3 import *
from cyecca.lie.group_so3 import SO3LieGroupElement, SO3LieAlgebraElement

from cyecca.symbolic import SERIES, taylor_series_near_zero

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

    def wedge(self, arg: (ca.SX, ca.DM)) -> SE3LieAlgebraElement:
        return self.elem(param=arg)

    def vee(self, arg: SE3LieAlgebraElement) -> ca.SX:
        return arg.param

    def left_Q(self, vb: R3LieAlgebraElement, omega: SO3LieAlgebraElement) -> ca.SX:
        v = ca.SX.sym("v", 3)
        o = ca.SX.sym("o", 3)
        V = so3.elem(v).ad()
        O = so3.elem(o).ad()
        O_sq = O @ O
        theta = ca.norm_2(o)
        c_theta = ca.cos(theta)
        s_theta = ca.sin(theta)

        Coeff = ca.if_else(
            ca.fabs(theta) > 1e-3,
            ca.vertcat(
                (1 - c_theta) / (theta**2),  # C0
                (theta - s_theta) / (theta**3),  # C1
                (theta**2 + 2 * c_theta - 2) / (2 * theta**4),  # C2
                (theta * c_theta + 2 * theta - 3 * s_theta) / (2 * theta**5),  # C3
                (theta**2 + theta * s_theta + 4 * c_theta - 4)
                / (2 * theta**6),  # C4
                (2 - 2 * c_theta - theta * s_theta) / (2 * theta**4),  # C5
            ),
            ca.vertcat(
                1 / 2 - theta**2 / 24 + theta**4 / 720,
                1 / 6 - theta**2 / 120 + theta**4 / 5040,
                1 / 24 - theta**2 / 720 + theta**4 / 40320,
                1 / 120 - theta**2 / 2520 + theta**4 / 120960,
                1 / 720 - theta**2 / 20160 + theta**4 / 1209600,
                1 / 24 - theta**2 / 360 + theta**4 / 134400,
            ),
        )

        C = V / 2
        C += Coeff[1] * (O @ V + V @ O)
        C += Coeff[2] * (O_sq @ V + V @ O_sq)
        C += Coeff[3] * (O @ V @ O_sq + O_sq @ V @ O)
        C += Coeff[4] * (O_sq @ V @ O_sq)
        C += Coeff[5] * (O @ V @ O)

        f_Q = ca.Function("f_Q", [v, o], [C])

        Ql = f_Q(vb.param, omega.param)

        return Ql

    def left_jacobian(self, arg: SE3LieAlgebraElement) -> ca.SX:
        omega = arg.Omega
        vb = arg.v_b
        Ql = arg.left_Q(vb, omega)
        R = omega.left_jacobian()
        Z = ca.SX.zeros(3, 3)
        Jl = ca.sparsify(ca.vertcat(ca.horzcat(R, Ql), ca.horzcat(Z, R)))

        return Jl

    def left_jacobian_inv(self, arg: SE3LieAlgebraElement) -> ca.SX:
        omega = arg.Omega
        vb = arg.v_b
        Ql = arg.left_Q(vb, omega)
        R_inv = omega.left_jacobian_inv()
        Z = ca.SX.zeros(3, 3)
        Jl_inv = ca.sparsify(
            ca.vertcat(ca.horzcat(R_inv, -R_inv @ Ql @ R_inv), ca.horzcat(Z, R_inv))
        )

        return Jl_inv

    def right_Q(self, vb: R3LieAlgebraElement, omega: SO3LieAlgebraElement) -> ca.SX:
        Qr = self.left_Q(-vb, -omega)
        return Qr

    def right_jacobian(self, arg: SE3LieAlgebraElement) -> ca.SX:
        omega = arg.Omega
        vb = arg.v_b
        Qr = arg.right_Q(vb, omega)
        R = omega.right_jacobian()
        Z = ca.SX.zeros(3, 3)
        Jr = ca.sparsify(ca.vertcat(ca.horzcat(R, Qr), ca.horzcat(Z, R)))
        return Jr

    def right_jacobian_inv(self, arg: SE3LieAlgebraElement) -> ca.SX:
        omega = arg.Omega
        vb = arg.v_b
        Qr = arg.right_Q(vb, omega)
        R_inv = omega.right_jacobian_inv()
        Z = ca.SX.zeros(3, 3)
        Jr_inv = ca.sparsify(
            ca.vertcat(ca.horzcat(R_inv, -R_inv @ Qr @ R_inv), ca.horzcat(Z, R_inv))
        )
        return Jr_inv


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
        u = arg.param[:3]  # translation
        omega = self.SO3.algebra.elem(arg.param[3:])
        Omega = omega.to_Matrix()
        rotation = omega.exp(self.SO3).param
        theta = ca.norm_2(omega.param)

        A = SERIES["(1 - cos(x))/x^2"](theta)
        B = SERIES["(x - sin(x))/x^3"](theta)
        V = ca.SX.eye(3) + A * Omega + B * Omega @ Omega
        p = V @ u  # position
        return self.elem(ca.vertcat(p, rotation))

    def log(self, arg: SE3LieGroupElement) -> SE3LieAlgebraElement:
        Omega = arg.R.log()
        theta = ca.norm_2(Omega.param)
        A = SERIES["(1 - x*sin(x)/(2*(1 - cos(x))))/x^2"](theta)
        B = SERIES["1/x^2"](theta)
        Omega_mat = Omega.to_Matrix()
        V_inv = ca.SX.eye(3) - Omega_mat / 2 + A * (Omega_mat @ Omega_mat)
        u = V_inv @ arg.p.param
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
