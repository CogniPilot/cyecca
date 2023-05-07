from __future__ import annotations

import sympy
from beartype import beartype

from ._base import LieAlgebra, LieAlgebraElement, LieGroup, LieGroupElement


@beartype
class SE2LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=1, matrix_shape=(2, 2))

    def bracket(self, left: SE2LieAlgebraElement, right: SE2LieAlgebraElement):
        assert self == left.algebra
        assert self == right.algebra
        return self.element(sympy.Matrix([0]))

    def addition(
        self, left: SE2LieAlgebraElement, right: SE2LieAlgebraElement
    ) -> SE2LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(left.param + right.param)

    def scalar_multipication(
        self, left, right: SE2LieAlgebraElement
    ) -> SE2LieAlgebraElement:
        assert self == right.algebra
        return self.element(left * right.param)

    def adjoint(self, left: SE2LieAlgebraElement):
        assert self == left.algebra
        raise NotImplementedError()

    def to_matrix(self) -> sympy.Matrix:
        Omega = SO2LieAlgebra.to_matrix(left.param[2])
        v = left.param[:2]
        Z13 = sympy.ZeroMatrix(1, 3)
        return sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [Omega, v],
                    [Z13],
                ]
            )
        )


@beartype
class SE2LieGroup(LieGroup):
    def __init__(self, algebra: SE2LieAlgebra):
        super().__init__(algebra=algebra, n_param=3, matrix_shape=(3, 3))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        return self.element(left.param + right.param)

    def inverse(self, left):
        assert self == left.group
        return self.element(-left.param)

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        raise NotImplementedError()

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        raise NotImplementedError()

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        raise NotImplementedError()

    def to_matrix(self) -> sympy.Matrix:
        R = SO2LieGroup.to_matrix(left.param[2])
        t = left.param[:2]
        Z12 = sympy.ZeroMatrix(1, 2)
        I1 = sympy.Identity(1)
        return sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [R, t],
                    [Z12, I1],
                ]
            )
        )


se2 = SE2LieAlgebra()
SE2 = SE2LieGroup(algebra=se2)
