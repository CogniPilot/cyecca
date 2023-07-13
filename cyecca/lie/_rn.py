from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy import floating

from beartype import beartype

from ._base import LieAlgebra, LieAlgebraElement, LieGroup, LieGroupElement


@beartype
class RnLieAlgebra(LieAlgebra):
    def __init__(self, n: int):
        super().__init__(n_param=n, matrix_shape=(n + 1, n + 1))

    def bracket(self, left: LieAlgebraElement, right: LieAlgebraElement):
        assert self == left.algebra
        assert self == right.algebra
        return self.element(param=np.zeros(self.n_param))

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(param=left.param + right.param)

    def scalar_multipication(self, left, right: LieAlgebraElement) -> LieAlgebraElement:
        assert self == right.algebra
        return self.element(param=left * right.param)

    def adjoint(self, left: LieAlgebraElement) -> npt.NDArray[np.floating]:
        assert self == left.algebra
        return np.zeros((self.n_param, self.n_param))

    def to_matrix(self, left: LieAlgebraElement) -> npt.NDArray[np.floating]:
        assert self == left.algebra
        A = np.zeros(self.matrix_shape)
        for i in range(self.n_param):
            A[i, self.n_param] = left.param[i]
        return A

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
        return self.element(param=left.param + right.param)

    def inverse(self, left: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        return self.element(param=-left.param)

    def identity(self) -> LieGroupElement:
        return self.element(param=np.zeros(self.n_param))

    def adjoint(self, left: LieGroupElement) -> npt.NDArray[np.floating]:
        assert self == left.group
        return np.eye(self.n_param + 1)

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        """It is the identity map"""
        assert self.algebra == left.algebra
        return self.element(param=left.param)

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        """It is the identity map"""
        assert self == left.group
        return left.group.algebra.element(left.param)

    def to_matrix(self, left: LieGroupElement) -> npt.NDArray[np.floating]:
        assert self == left.group
        A = np.eye(self.n_param + 1)
        for i in range(self.n_param):
            A[i, self.n_param] = left.param[i]
        return A

    def __str__(self):
        return "{:s}({:d})".format(self.__class__.__name__, self.n_param)


r2 = RnLieAlgebra(n=2)
R2 = RnLieGroup(algebra=r2)

r3 = RnLieAlgebra(n=3)
R3 = RnLieGroup(algebra=r3)
