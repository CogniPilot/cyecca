from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy import floating, cos, sin

from beartype import beartype

from ._base import LieAlgebra, LieAlgebraElement, LieGroup, LieGroupElement


@beartype
class SO2LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=1, matrix_shape=(2, 2))

    def bracket(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(param=Matrix([0]))

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(param=left.param + right.param)

    def scalar_multipication(self, left : Real, right: LieAlgebraElement) -> LieAlgebraElement:
        assert self == right.algebra
        return self.element(param=left * right.param)

    def adjoint(self, left: LieAlgebraElement) -> Matrix:
        assert self == left.algebra
        return Matrix.zeros(1, 1)

    def to_matrix(self, left: LieAlgebraElement) -> Matrix:
        assert self == left.algebra
        return Matrix([[0, -left.param[0]], [left.param[0], 0]])


@beartype
class SO2LieGroup(LieGroup):
    def __init__(self):
        super().__init__(algebra=so2, n_param=1, matrix_shape=(2, 2))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        return self.element(param=left.param + right.param)

    def inverse(self, left: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        return self.element(param=-left.param)

    def identity(self) -> LieGroupElement:
        return self.element(param=np.zeros(self.n_param))

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        return Matrix.eye(1)

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        return self.element(param=left.param)

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        return self.algebra.element(param=left.param)

    def to_matrix(self, left: LieGroupElement) -> Matrix:
        assert self == left.group
        theta = left.param[0]
        c = cos(theta)
        s = sin(theta)
        return Matrix([[c, -s], [s, c]])


so2 = SO2LieAlgebra()
SO2 = SO2LieGroup()
