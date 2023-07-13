from __future__ import annotations

from beartype import beartype

from ._base import LieAlgebra, LieAlgebraElement, LieGroup, LieGroupElement
from ._so2 import SO2


@beartype
class SE2LieAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__(n_param=3, matrix_shape=(3, 3))

    def bracket(self, left: LieAlgebraElement, right: LieAlgebraElement):
        assert self == left.algebra
        assert self == right.algebra
        return self.element(param=np.array([0]))

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(param=left.param + right.param)

    def scalar_multipication(
        self, left, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert self == right.algebra
        return self.element(param=left * right.param)

    def adjoint(self, left: LieAlgebraElement):
        assert self == left.algebra
        raise NotImplementedError()

    def to_matrix(self) -> np.array:
        Omega = LieAlgebraElement.to_matrix(self.param[2])
        v = self.param[:2, 0]
        Z13 = Zeronp.array(1, 3)
        return np.block([
            [Omega, v],
            [Z13]])


@beartype
class SE2LieGroup(LieGroup):
    def __init__(self):
        super().__init__(algebra=se2, n_param=3, matrix_shape=(3, 3))

    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        return self.element(left.param + right.param)

    def inverse(self, left):
        assert self == left.group
        return self.element(-left.param)

    def identity(self) -> LieGroupElement:
        return self.element(np.array.zeros(self.n_param, 1))

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        raise NotImplementedError()

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        theta = left.param[2]
        sin_th = sin(theta)
        cos_th = cos(theta)
        a = sin_th / theta
        b = (1 - cos_th) / theta
        V = np.array([
            [a, -b],
            [b, a]])
        v = V @ left.param[:2, 0]
        return self.element(np.array([v[0], v[1], theta]))

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        raise NotImplementedError()

    def to_matrix(self) -> np.array:
        R = SO2.to_matrix(self.param[2])
        t = self.param[:2]
        Z12 = Zeronp.array(1, 2)
        I1 = Identity(1)
        return np.array(Blocknp.array([
            [R, t],
            [Z12, I1],
        ]))


se2 = SE2LieAlgebra()
SE2 = SE2LieGroup()
