"""
This is the module for the n-dimensional translational group R^n
and the associated lie algebra r^n
"""

import casadi as ca

from .base import LieAlgebra, LieGroup


class LieAlgebraR(LieAlgebra):
    """
    Translation Lie Algebra
    """

    def __init__(self, n: int, param: ca.SX):
        super().__init__(param)
        self.n_dim = n
        assert self.param.shape == (n, 1)

    def wedge(self) -> ca.SX:
        algebra = ca.sparsify(ca.SX.zeros(self.n_dim + 1, self.n_dim + 1))
        algebra[: self.n_dim, self.n_dim] = self.param
        return algebra

    def vee(self) -> ca.SX:
        return self.param

    def neg(self) -> "LieAlgebraR":
        return LieAlgebraR(self.n_dim, -self.param)

    def add(self, other: "LieAlgebraR") -> "LieAlgebraR":
        return LieAlgebraR(self.n_dim, self.param + other.param)

    def rmul(self, other: float) -> "LieAlgebraR":
        other = ca.SX(other)
        assert ca.SX(other).shape == (1, 1)
        param = other * self.param
        return LieAlgebraR(self.n_dim, param)


class LieGroupR(LieGroup):
    """
    The Lie Group R^n
    """
    n_dim = 0
    def __init__(self, n_dim: int, param: ca.SX):
        super().__init__(param)
        self.n_dim = n_dim
        assert self.param.shape == (self.n_dim, 1)

    def inv(self) -> "LieGroupR":
        return LieGroupR(self.n_dim, -self.param)

    def log(self) -> "LieAlgebraR":
        return LieAlgebraR(self.n_dim, self.param)

    def product(self, other: "LieGroupR") -> "LieGroupR":
        param = self.param + other.param
        return LieGroupR(self.n_dim, param)

    @classmethod
    def identity(cls, n_dim: int) -> "LieGroupR":
        param = ca.sparsify(ca.SX.zeros(n_dim, 1))
        return LieGroupR(n_dim, param)

    def to_matrix(self):
        matrix = ca.sparsify(ca.SX.zeros(self.n_dim + 1, self.n_dim + 1))
        matrix[: self.n_dim, : self.n_dim] = ca.SX.eye(self.n_dim)
        matrix[: self.n_dim, self.n_dim] = self.param
        matrix[self.n_dim, self.n_dim] = 1
        return matrix

    @staticmethod
    def exp(g: LieAlgebraR):
        return LieGroupR(g.n_dim, g)


class LieGroupR2(LieGroupR):

    def __init__(self, param: ca.SX):
        super().__init__(2, param)


class LieAlgebraR2(LieAlgebraR):

    def __init__(self, param: ca.SX):
        super().__init__(2, param)


class LieGroupR3(LieGroupR):

    def __init__(self, param: ca.SX):
        super().__init__(3, param)


class LieAlgebraR3(LieAlgebraR):

    def __init__(param: ca.SX):
        super().__init__(3, param)
