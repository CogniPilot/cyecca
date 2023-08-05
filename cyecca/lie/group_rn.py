from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy import floating

import casadi as ca

from beartype import beartype

from cyecca.lie.base import *

__all__ = ["r2", "R2", "r3", "R3"]


@beartype
class RnLieAlgebra(LieAlgebra):
    def __init__(self, n: int):
        super().__init__(n_param=n, matrix_shape=(n + 1, n + 1))

    def bracket(self, left: LieAlgebraElement, right: LieAlgebraElement):
        assert self == left.algebra
        assert self == right.algebra
        return self.elem(param=ca.SX(self.n_param, 1))

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
        return ca.SX(self.matrix_shape)

    def to_Matrix(self, arg: LieAlgebraElement) -> ca.SX:
        assert self == arg.algebra
        A = ca.SX(*self.matrix_shape)
        print("A shape", A.shape)
        for i in range(self.n_param):
            A[i, self.n_param] = arg.param[i]
        return ca.sparsify(A)

    def from_Matrix(self, arg: ca.SX) -> LieAlgebraElement:
        raise NotImplementedError("")

    def __str__(self):
        return "{:s}({:d})".format(self.__class__.__name__, self.n_param)


@beartype
class RnLieGroup(LieGroup):
    def __init__(self, algebra: RnLieAlgebra):
        n = algebra.n_param
        super().__init__(algebra=algebra, n_param=n, matrix_shape=(n + 1, n + 1))

    def product(self, left: LieGroupElement, right: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        assert self == right.group
        return self.elem(param=left.param + right.param)

    def inverse(self, arg: LieGroupElement) -> LieGroupElement:
        assert self == arg.group
        return self.elem(param=-arg.param)

    def identity(self) -> LieGroupElement:
        return self.elem(param=ca.SX(self.n_param, 1))

    def adjoint(self, arg: LieGroupElement) -> ca.SX:
        assert self == arg.group
        return ca.SX_eye(self.n_param + 1)

    def exp(self, arg: LieAlgebraElement) -> LieGroupElement:
        """It is the identity map"""
        assert self.algebra == arg.algebra
        return self.elem(param=arg.param)

    def log(self, arg: LieGroupElement) -> LieAlgebraElement:
        """It is the identity map"""
        assert self == arg.group
        return arg.group.algebra.elem(arg.param)

    def to_Matrix(self, arg: LieGroupElement) -> ca.SX:
        assert self == arg.group
        A = ca.SX_eye(self.n_param + 1)
        for i in range(self.n_param):
            A[i, self.n_param] = arg.param[i]
        return ca.sparsify(A)

    def from_Matrix(self, arg: ca.SX) -> LieGroupElement:
        raise NotImplementedError("")

    def __str__(self):
        return "{:s}({:d})".format(self.__class__.__name__, self.n_param)


r2 = RnLieAlgebra(n=2)
R2 = RnLieGroup(algebra=r2)

r3 = RnLieAlgebra(n=3)
R3 = RnLieGroup(algebra=r3)
