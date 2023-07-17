from __future__ import annotations

import casadi as ca

from beartype import beartype
from beartype.typing import List

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
        return self.element(param=ca.DM([0]))

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(param=left.param + right.param)

    def scalar_multipication(
        self, left: (float, int), right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == right.algebra
        return self.element(param=left * right.param)

    def adjoint(self, left: LieAlgebraElement) -> ca.SX:
        assert self == left.algebra
        return ca.SX.zeros(1, 1)

    def to_matrix(self, left: LieAlgebraElement) -> ca.SX:
        print(type(left.param[0, 0]))
        assert self == left.algebra
        M = ca.SX.zeros(2, 2)
        M[0, 1] = -left.param[0, 0]
        M[1, 0] = left.param[0, 0]
        return M

    def wedge(self, left: (ca.SX, ca.DM)) -> LieAlgebraElement:
        self = SO2LieAlgebra()
        return self.element(param=left)

    def vee(self, left: LieAlgebraElement) -> ca.SX:
        assert self == left.algebra
        return left.param


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
        return self.element(param=ca.SX.zeros(self.n_param))

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        return ca.SX_eye(1)

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        return self.element(param=left.param)

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        return self.algebra.element(param=left.param)

    def to_matrix(self, left: LieGroupElement) -> ca.SX:
        assert self == left.group
        theta = left.param[0, 0]
        c = ca.cos(theta)
        s = ca.sin(theta)
        M = ca.SX.zeros(2, 2)
        M[0, 0] = c
        M[0, 1] = -s
        M[1, 0] = s
        M[1, 1] = c
        return M


so2 = SO2LieAlgebra()
SO2 = SO2LieGroup()
