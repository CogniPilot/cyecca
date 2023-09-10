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

    def elem(self, param: PARAM_TYPE) -> LieAlgebraElement:
        return RnLieAlgebraElement(algebra=self, param=param)

    def bracket(self, left: RnLieAlgebraElement, right: RnLieAlgebraElement):
        assert self == left.algebra
        assert self == right.algebra
        return self.elem(param=ca.SX(self.n_param, 1))

    def addition(
        self, left: RnLieAlgebraElement, right: RnLieAlgebraElement
    ) -> RnLieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.elem(param=left.param + right.param)

    def scalar_multiplication(
        self, left: SCALAR_TYPE, right: RnLieAlgebraElement
    ) -> RnLieAlgebraElement:
        assert self == right.algebra
        return self.elem(param=left * right.param)

    def adjoint(self, arg: RnLieAlgebraElement) -> ca.SX:
        assert self == arg.algebra
        return ca.SX(self.matrix_shape)

    def to_Matrix(self, arg: RnLieAlgebraElement) -> ca.SX:
        assert self == arg.algebra
        A = ca.SX(*self.matrix_shape)
        print("A shape", A.shape)
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
        self.algebra = algebra
        self.param = ca.SX(param)
        assert self.param.shape == (self.algebra.n_param, 1)

    def __eq__(self, other: RnLieAlgebraElement) -> bool:
        return bool(ca.logic_all(self.param == other.param))

    def __mul__(self, right: RnLieAlgebraElement) -> RnLieAlgebraElement:
        if isinstance(right, RnLieAlgebraElement):
            return self.algebra.bracket(left=self, right=right)
        elif isinstance(right, SCALAR_TYPE):
            return self.algebra.scalar_multiplication(left=right, right=self)

    def __rmul__(self, arg: SCALAR_TYPE) -> RnLieAlgebraElement:
        return self.algebra.scalar_multiplication(left=arg, right=self)

    def __add__(self, arg: RnLieAlgebraElement) -> RnLieAlgebraElement:
        return self.algebra.addition(left=self, right=arg)

    def exp(self, group: RnLieGroup) -> RnLieGroupElement:
        return group.exp(self)

    def __repr__(self):
        return "{:s}: {:s}".format(repr(self.algebra), repr(self.param))


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
        assert self == left.group
        assert self == right.group
        return self.elem(param=left.param + right.param)

    def inverse(self, arg: RnLieGroupElement) -> RnLieGroupElement:
        assert self == arg.group
        return self.elem(param=-arg.param)

    def identity(self) -> RnLieGroupElement:
        return self.elem(param=ca.SX(self.n_param, 1))

    def adjoint(self, arg: RnLieGroupElement) -> ca.SX:
        assert self == arg.group
        return ca.SX_eye(self.n_param + 1)

    def exp(self, arg: RnLieAlgebraElement) -> RnLieGroupElement:
        """It is the identity map"""
        assert self.algebra == arg.algebra
        return self.elem(param=arg.param)

    def log(self, arg: RnLieGroupElement) -> RnLieAlgebraElement:
        """It is the identity map"""
        assert self == arg.group
        return arg.group.algebra.elem(arg.param)

    def to_Matrix(self, arg: RnLieGroupElement) -> ca.SX:
        assert self == arg.group
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
        self.group = group
        self.param = ca.SX(param)
        assert self.param.shape == (self.group.n_param, 1)

    def inverse(self) -> RnLieGroupElement:
        return self.group.inverse(arg=self)

    def __add__(self, other: RnLieAlgebraElement) -> RnLieGroupElement:
        return self * other.exp(self.group)

    def __sub__(self, other: RnLieAlgebraElement) -> RnLieGroupElement:
        return self * (-other).exp(self.group)

    def __eq__(self, other: RnLieGroupElement) -> bool:
        return bool(ca.logic_all(self.param == other.param))

    def __mul__(self, right: RnLieGroupElement) -> RnLieGroupElement:
        return self.group.product(left=self, right=right)

    def log(self) -> RnLieAlgebraElement:
        return self.group.log(arg=self)


r2 = RnLieAlgebra(n=2)
R2 = RnLieGroup(algebra=r2)

r3 = RnLieAlgebra(n=3)
R3 = RnLieGroup(algebra=r3)
