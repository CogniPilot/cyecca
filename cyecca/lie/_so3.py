from __future__ import annotations

from sympy import cos, sin, Matrix, Identity
from enum import Enum
from beartype.typing import List
from numbers import Real

from beartype import beartype

from ._base import LieAlgebra, LieAlgebraElement, LieGroup, LieGroupElement


@beartype
class SO3LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=3, matrix_shape=(3, 3))

    def bracket(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        c = left.to_matrix()@right.to_matrix() - right.to_matrix()@left.to_matrix()
        return self.element(param=Matrix([c[2, 1], c[0, 2], c[1, 0]]))

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
        return Matrix([
            [0, -left.param[2], left.param[1]],
            [left.param[2], 0, -left.param[0]],
            [-left.param[1], left.param[0], 0]])


class Axis(Enum):
    x = 1
    y = 2
    z = 3

class EulerType(Enum):
    body = 1
    space = 2

def rotation_matrix(axis : Axis, angle : Real):
    if axis== Axis.x:
        return Matrix([
            [1, 0, 0],
            [0, cos(angle), -sin(angle)],
            [0, -sin(angle), cos(angle)]
        ])
    elif axis == Axis.y:
        return Matrix([
                [-cos(angle), 0, sin(angle)],
                [0, 1, 0],
                [sin(angle), 0, cos(angle)]
            ])
    elif axis == Axis.z:
        return Matrix([
            [cos(angle), -sin(angle), 0],
            [sin(angle), cos(angle), 0],
            [0, 0, 1]
        ])


@beartype
class SO3EulerLieGroup(LieGroup):
    def __init__(self, type : EulerType, sequence : List[Axis]):
        super().__init__(algebra=so3, n_param=3, matrix_shape=(3, 3))
        self.type = type
        assert len(sequence) == 3
        self.sequence = sequence

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        return self.element(left.param + right.param)

    def inverse(self, left: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        return self.element(param=-left.param)

    def identity(self) -> LieGroupElement:
        return self.element(param=Matrix.zeros(self.n_param, 1))

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
        m = Identity(3)
        for axis, angle in zip(self.sequence, left.param):
            m @= rotation_matrix(axis=axis, angle=angle)
        return m


so3 = SO3LieAlgebra()
SO3EulerB321 = SO3EulerLieGroup(type=EulerType.body, sequence=[Axis.z, Axis.y, Axis.x])
