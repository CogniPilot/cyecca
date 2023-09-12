from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy import floating

import casadi as ca

from beartype import beartype
from beartype.typing import Union

from cyecca.lie.base import *

__all__ = ["r2", "R2", "r3", "R3"]


@beartype
class RnLieAlgebra(LieAlgebra):
    def __init__(self, n: int):
        super().__init__(n_param=n, matrix_shape=(n + 1, n + 1))

    def elem(self, param: PARAM_TYPE) -> RnLieAlgebraElement:
        return RnLieAlgebraElement(algebra=self, param=param)

    def bracket(self, left: RnLieAlgebraElement, right: RnLieAlgebraElement):
        return self.elem(param=ca.SX(self.n_param, 1))

    def addition(
        self, left: RnLieAlgebraElement, right: RnLieAlgebraElement
    ) -> RnLieAlgebraElement:
        return self.elem(param=left.param + right.param)

    def scalar_multiplication(
        self, left: SCALAR_TYPE, right: RnLieAlgebraElement
    ) -> RnLieAlgebraElement:
        return self.elem(param=left * right.param)

    def adjoint(self, arg: RnLieAlgebraElement) -> ca.SX:
        return ca.SX(*self.matrix_shape)

    def to_Matrix(self, arg: RnLieAlgebraElement) -> ca.SX:
        A = ca.SX(*self.matrix_shape)
        for i in range(self.n_param):
            A[i, self.n_param] = arg.param[i]
        return ca.sparsify(A)

    def from_Matrix(self, arg: ca.SX) -> RnLieAlgebraElement:
        raise NotImplementedError("")

    def __str__(self):
        return "{:s}({:d})".format(self.__class__.__name__, self.n_param)


@beartype
class RnLieAlgebraElement(LieAlgebraElement):
    """
    This is an Rn Lie algebra elem
    """

    def __init__(self, algebra: RnLieAlgebra, param: PARAM_TYPE):
        super().__init__(algebra, param)


@beartype
class RnLieGroup(LieGroup):
    def __init__(self, algebra: RnLieAlgebra):
        n = algebra.n_param
        super().__init__(algebra=algebra, n_param=n, matrix_shape=(n + 1, n + 1))

    def elem(self, param: PARAM_TYPE) -> RnLieGroupElement:
        return RnLieGroupElement(group=self, param=param)

    def product(
        self, left: RnLieGroupElement, right: RnLieGroupElement
    ) -> RnLieGroupElement:
        return self.elem(param=left.param + right.param)

    def inverse(self, arg: RnLieGroupElement) -> RnLieGroupElement:
        return self.elem(param=-arg.param)

    def identity(self) -> RnLieGroupElement:
        return self.elem(param=ca.SX(self.n_param, 1))

    def adjoint(self, arg: RnLieGroupElement) -> ca.SX:
        return ca.SX_eye(self.n_param + 1)

    def exp(self, arg: RnLieAlgebraElement) -> RnLieGroupElement:
        """It is the identity map"""
        return self.elem(param=arg.param)

    def log(self, arg: RnLieGroupElement) -> RnLieAlgebraElement:
        """It is the identity map"""
        return arg.group.algebra.elem(arg.param)

    def to_Matrix(self, arg: RnLieGroupElement) -> ca.SX:
        A = ca.SX_eye(self.n_param + 1)
        for i in range(self.n_param):
            A[i, self.n_param] = arg.param[i]
        return ca.sparsify(A)

    def from_Matrix(self, arg: ca.SX) -> RnLieGroupElement:
        raise NotImplementedError("")

    def __str__(self):
        return "{:s}({:d})".format(self.__class__.__name__, self.n_param)


@beartype
class RnLieGroupElement(LieGroupElement):
    """
    This is an Rn Lie group elem, not necessarily represented as a matrix
    """

    def __init__(self, group: RnLieGroup, param: PARAM_TYPE):
        super().__init__(group, param)


@beartype
class R3LieAlgebra(RnLieAlgebra):
    def __init__(self):
        super().__init__(3)

    def elem(self, param: PARAM_TYPE) -> R3LieAlgebraElement:
        return R3LieAlgebraElement(algebra=self, param=param)


@beartype
class R3LieAlgebraElement(RnLieAlgebraElement):
    def __init__(self, algebra: R3LieAlgebra, param: PARAM_TYPE):
        super().__init__(algebra, param)


@beartype
class R3LieGroup(RnLieGroup):
    def __init__(self, algebra: R3LieAlgebra):
        super().__init__(algebra)

    def elem(self, param: PARAM_TYPE) -> R3LieGroupElement:
        return R3LieGroupElement(group=self, param=param)


@beartype
class R3LieGroupElement(RnLieGroupElement):
    def __init__(self, group: R3LieGroup, param: PARAM_TYPE):
        super().__init__(group, param)


r3 = R3LieAlgebra()
R3 = R3LieGroup(algebra=r3)


@beartype
class R2LieAlgebra(RnLieAlgebra):
    def __init__(self):
        super().__init__(2)

    def elem(self, param: PARAM_TYPE) -> R2LieAlgebraElement:
        return R2LieAlgebraElement(algebra=self, param=param)


@beartype
class R2LieAlgebraElement(RnLieAlgebraElement):
    def __init__(self, algebra: R2LieAlgebra, param: PARAM_TYPE):
        super().__init__(algebra, param)


@beartype
class R2LieGroup(RnLieGroup):
    def __init__(self, algebra: R2LieAlgebra):
        super().__init__(algebra)

    def elem(self, param: PARAM_TYPE) -> R2LieGroupElement:
        return R2LieGroupElement(group=self, param=param)


@beartype
class R2LieGroupElement(RnLieGroupElement):
    def __init__(self, group: R2LieGroup, param: PARAM_TYPE):
        super().__init__(group, param)


r2 = R2LieAlgebra()
R2 = R2LieGroup(algebra=r2)
