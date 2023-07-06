from __future__ import annotations

import sympy
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
        return self.element(sympy.Matrix([0]))

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(left.param + right.param)

    def scalar_multipication(self, left, right: LieAlgebraElement) -> LieAlgebraElement:
        assert self == right.algebra
        return self.element(left * right.param)

    def adjoint(self, left: LieAlgebraElement) -> sympy.Matrix:
        assert self == left.algebra
        return sympy.Matrix.zeros(1, 1)

    def to_matrix(self, left: LieAlgebraElement) -> sympy.Matrix:
        assert self == left.algebra
        return sympy.Matrix([[0, -left.param[0]], [left.param[0], 0]])


@beartype
class SO2LieGroup(LieGroup):
    def __init__(self, algebra: SO2LieAlgebra):
        super().__init__(algebra=algebra, n_param=1, matrix_shape=(2, 2))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        return self.element(left.param + right.param)

    def inverse(self, left: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        return self.element(-left.param)

    def identity(self) -> LieGroupElement:
        return self.element(sympy.Matrix.zeros(self.n_param, 1))

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        return sympy.Matrix.eye(1)

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        return self.element(left.param)

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        return self.algebra.element(left.param)

    def to_matrix(self, left: LieGroupElement) -> sympy.Matrix:
        assert self == left.group
        theta = left.param[0]
        c = sympy.cos(theta)
        s = sympy.sin(theta)
        return sympy.Matrix([[c, -s], [s, c]])


so2 = SO2LieAlgebra()
SO2 = SO2LieGroup(algebra=so2)
