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
        raise NotImplementedError("adjoint not implemented")

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
    body_fixed = 1
    space_fixed = 2

def rotation_matrix(axis : Axis, angle : Real):
    if axis== Axis.x:
        return Matrix([
            [1, 0, 0],
            [0, cos(angle), -sin(angle)],
            [0, sin(angle), cos(angle)]
        ])
    elif axis == Axis.y:
        return Matrix([
                [cos(angle), 0, sin(angle)],
                [0, 1, 0],
                [-sin(angle), 0, cos(angle)]
            ])
    elif axis == Axis.z:
        return Matrix([
            [cos(angle), -sin(angle), 0],
            [sin(angle), cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError('unknown axis')


so3 = SO3LieAlgebra()


@beartype
class SO3EulerLieGroup(LieGroup):
    def __init__(self, euler_type : EulerType, sequence : List[Axis]):
        super().__init__(algebra=so3, n_param=3, matrix_shape=(3, 3))
        self.euler_type = euler_type
        assert len(sequence) == 3
        self.sequence = sequence

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        return self.element(param=left.param + right.param)

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
        raise NotImplementedError("exp not implemented")

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        raise NotImplementedError("log not implemented")

    def to_matrix(self, left: LieGroupElement) -> Matrix:
        assert self == left.group
        m = Identity(3)
        for axis, angle in zip(self.sequence, left.param):
            if self.euler_type == EulerType.body_fixed:
                m = m @ rotation_matrix(axis=axis, angle=angle)
            elif self.euler_type == EulerType.space_fixed:
                m = rotation_matrix(axis=axis, angle=angle) @ m
            else:
                raise ValueError('euler_type must be body_fixed or space_fixed')
        return Matrix(m)


SO3Euler = SO3EulerLieGroup


@beartype
class SO3QuatLieGroup(LieGroup):
    def __init__(self):
        super().__init__(algebra=so3, n_param=4, matrix_shape=(3, 3))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        q = left.param
        p = right.param
        return self.element(param=Matrix([
            q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3],
            q[1] * p[0] + q[0] * p[1] - q[3] * p[2] + q[2] * p[3],
            q[2] * p[0] + q[3] * p[1] + q[0] * p[2] - q[1] * p[3],
            q[3] * p[0] - q[2] * p[1] + q[1] * p[2] + q[0] * p[3]
        ]))

    def inverse(self, left: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        q = left.param
        return self.element(param=Matrix([
            q[0], -q[1], -q[2], -q[3]
        ]))

    def identity(self) -> LieGroupElement:
        return self.element(param=Matrix([1, 0, 0, 0]))

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        raise NotImplementedError("adjoint not implemented")

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        raise NotImplementedError("exp not implemented")

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        raise NotImplementedError("exp not implemented")

    def to_matrix(self, left: LieGroupElement) -> Matrix:
        assert self == left.group
        a, b, c, d = left.param
        aa = a*a
        ab = a*b
        ac = a*c
        ad = a*d
        bb = b*b
        bc = b*c
        bd = b*d
        cc = c*c
        cd = c*d
        dd = d*d
        return Matrix([
            [aa + bb - cc - dd, 2 * (bc - ad), 2 * (ac + bd)],
            [2 * (bc + ad), aa - bb + cc - dd, 2 * (cd - ab), 2 * (bd - ac)],
            [2 * (bd - ac), 2 * (ab + cd), aa - bb - cc + dd]
        ])


SO3Quat = SO3QuatLieGroup()

@beartype

class SO3DCMLieGroup(LieGroup):
    def __init__(self):
        super().__init__(algebra=so3, n_param=9, matrix_shape=(3, 3))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        q = left.param
        p = right.param
        return self.element(param=Matrix([]))

    def inverse(self, left: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        q = left.param
        return self.element(param=Matrix([])) # transpose

    def identity(self) -> LieGroupElement:
        return self.element(param=Identity(3))

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        raise NotImplementedError("adjoint not implemented")

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        raise NotImplementedError("exp not implemented")

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        raise NotImplementedError("exp not implemented")

    def to_matrix(self, left: LieGroupElement) -> Matrix:
        assert self == left.group
        return self.element(param=left.param)

SO3DCM = SO3DCMLieGroup()

@beartype
class SO3MRPLieGroup(LieGroup):
    def __init__(self):
        super().__init__(algebra=so3, n_param=4, matrix_shape=(3, 3))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        q = left.param
        p = right.param
        return self.element(param=Matrix([

        ]))

    def inverse(self, left: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        r = left.param
        return self.element(param=Matrix([
            -r[0], -r[1], -r[2], r[3]
        ]))
    
    def identity(self) -> LieGroupElement:
        return self.element(param=Matrix([0, 0, 0, 0]))
    
    def shadow(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        raise NotImplementedError("shadow not implemented")

    def shadow_if_necessary(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        raise NotImplementedError("shadow not implemented")

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        raise NotImplementedError("adjoint not implemented")

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        raise NotImplementedError("exp not implemented")

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        raise NotImplementedError("exp not implemented")

    def to_matrix(self, left: LieGroupElement) -> Matrix:
        assert self == left.group
        raise NotImplementedError("to matrix not implemented")

SO3MRP = SO3MRPLieGroup()