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

    def wedge(self):
        algebra = ca.sparsify(ca.SX.zeros(self.n_dim + 1, self.n_dim + 1))
        algebra[: self.n_dim, self.n_dim] = self.param
        return algebra

    def vee(self):
        return self.param

    def neg(self):
        return LieAlgebraR(self.n_dim, -self.param)

    def add(self, other):
        return LieAlgebraR(self.n_dim, self.param + other.param)

    def rmul(self, other):
        other = ca.SX(other)
        assert ca.SX(other).shape == (1, 1)
        param = other * self.param
        return LieAlgebraR(self.n_dim, param)


class LieGroupR(LieGroup):
    """
    The Lie Group R^n
    """

    def __init__(self, n_dim: int, param: ca.SX):
        super().__init__(param)
        self.n_dim = n_dim
        assert self.param.shape == (self.n_dim, 1)

    def inv(self):
        return LieGroupR(self.n_dim, -self.param)

    def log(self):
        return LieAlgebraR(self.n_dim, self.param)

    def product(self, other: "LieGroupR"):
        param = self.param + other.param
        return LieGroupR(self.n_dim, param)

    def identity(self):
        param = ca.sparsify(ca.SX.zeros(self.n_dim, 1))
        return LieGroupR(self.n_dim, param)

    def to_matrix_lie_group(self):
        matrix = ca.sparsify(ca.SX.zeros(self.n_dim + 1, self.n_dim + 1))
        matrix[: self.n_dim, : self.n_dim] = ca.SX.eye(self.n_dim)
        matrix[: self.n_dim, self.n_dim] = self.param
        matrix[self.n_dim, self.n_dim] = 1
        return matrix

    @staticmethod
    def exp(g: LieAlgebraR):
        return LieGroupR(g.n_dim, g)
