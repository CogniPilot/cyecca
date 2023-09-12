from __future__ import annotations

import casadi as ca

from beartype import beartype
from beartype.typing import List

from cyecca.lie.base import *
from cyecca.lie.group_rn import *
from cyecca.lie.group_rn import R2LieAlgebraElement

__all__ = ["so2", "SO2"]


@beartype
class SO2LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=1, matrix_shape=(2, 2))

    def elem(self, param: PARAM_TYPE) -> SO2LieAlgebraElement:
        return SO2LieAlgebraElement(algebra=self, param=param)

    def bracket(
        self, left: SO2LieAlgebraElement, right: SO2LieAlgebraElement
    ) -> SO2LieAlgebraElement:
        return self.elem(param=ca.DM([0]))

    def addition(
        self, left: SO2LieAlgebraElement, right: SO2LieAlgebraElement
    ) -> SO2LieAlgebraElement:
        return self.elem(param=left.param + right.param)

    def scalar_multiplication(
        self, left: SCALAR_TYPE, right: SO2LieAlgebraElement
    ) -> SO2LieAlgebraElement:
        return self.elem(param=left * right.param)

    def adjoint(self, arg: SO2LieAlgebraElement) -> ca.SX:
        return ca.SX(1, 1)

    def to_Matrix(self, arg: SO2LieAlgebraElement) -> ca.SX:
        print(type(arg.param[0, 0]))
        M = ca.SX(2, 2)
        M[0, 1] = -arg.param[0, 0]
        M[1, 0] = arg.param[0, 0]
        return M

    def from_Matrix(self, arg: ca.SX) -> SO2LieAlgebraElement:
        return self.elem(M[1, 0])

    def wedge(self, arg: (ca.SX, ca.DM)) -> SO2LieAlgebraElement:
        return self.elem(param=arg)

    def vee(self, arg: SO2LieAlgebraElement) -> ca.SX:
        return arg.param


@beartype
class SO2LieAlgebraElement(LieAlgebraElement):
    """
    This is an SO2 Lie algebra elem
    """

    def __init__(self, algebra: SO2LieAlgebra, param: PARAM_TYPE):
        super().__init__(algebra, param)


@beartype
class SO2LieGroup(LieGroup):
    def __init__(self):
        super().__init__(algebra=so2, n_param=1, matrix_shape=(2, 2))

    def elem(self, param: PARAM_TYPE) -> SO2LieGroupElement:
        return SO2LieGroupElement(group=self, param=param)

    def product(
        self, left: SO2LieGroupElement, right: SO2LieGroupElement
    ) -> SO2LieGroupElement:
        """
        Default product uses matrix conversion
        """
        return self.elem(left.param + right.param)

    def product_r2(
        self, left: SO2LieGroupElement, right: R2LieAlgebraElement
    ) -> R2LieAlgebraElement:
        """
        Vector rotation for algebra r2, uses to_Matrix
        """
        v = left.to_Matrix() @ right.param
        return R2LieAlgebraElement(algebra=right.algebra, param=v)

    def product_vector(self, left: SO2LieGroupElement, right: ca.SX) -> ca.SX:
        """
        Vector product, uses matrix conversion
        """
        return left.to_Matrix() @ right

    def inverse(self, arg: SO2LieGroupElement) -> SO2LieGroupElement:
        return self.elem(param=-arg.param)

    def identity(self) -> SO2LieGroupElement:
        return self.elem(param=ca.SX(self.n_param, 1))

    def adjoint(self, arg: SO2LieGroupElement):
        return ca.SX_eye(1)

    def exp(self, arg: SO2LieAlgebraElement) -> SO2LieGroupElement:
        return self.elem(param=arg.param)

    def log(self, arg: SO2LieGroupElement) -> SO2LieAlgebraElement:
        return self.algebra.elem(param=arg.param)

    def to_Matrix(self, arg: SO2LieGroupElement) -> ca.SX:
        theta = arg.param[0, 0]
        c = ca.cos(theta)
        s = ca.sin(theta)
        M = ca.SX(2, 2)
        M[0, 0] = c
        M[0, 1] = -s
        M[1, 0] = s
        M[1, 1] = c
        return M

    def from_Matrix(self, arg: ca.SX) -> SO2LieGroupElement:
        return self.elem(ca.atan2(arg[0, 0], arg[1, 0]))


@beartype
class SO2LieGroupElement(LieGroupElement):
    """
    This is an SO2 Lie group elem
    """

    def __init__(self, group: SO2LieGroup, param: PARAM_TYPE):
        super().__init__(group, param)

    def __matmul__(self, right):
        """
        override matrix mul operator to use as actions on 3 vectors
        """
        if isinstance(right, R2LieAlgebraElement):
            return self.group.product_r2(self, right)
        if isinstance(right, ca.SX) and right.shape == (2, 1):
            return self.group.product_vector(self, right)
        else:
            raise TypeError("unhandled type in product {:s}".format(type(right)))


so2 = SO2LieAlgebra()
SO2 = SO2LieGroup()
