"""
The SO(2), Special Orthogonal Lie Group of 2-Dimensions and the associated
so(2) Lie Algebra.
"""

import casadi as ca

from .base import LieAlgebra, LieGroup


class LieGroupSO2(LieGroup):
    """
    The SO(2) Lie Group
    """

    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (1, 1)

    def inv(self):
        return LieGroupSO2(-self.param)

    def log(self):
        return LieAlgebraSO2(self.param)

    def product(self, other):
        param = self.param + other.param
        return LieGroupSO2(param)

    def identity(self):
        return LieGroupSO2(0)

    def to_matrix_lie_group(self):
        matrix = ca.SX.zeros(2, 2)
        matrix[0, 0] = ca.cos(self.param)
        matrix[0, 1] = -ca.sin(self.param)
        matrix[1, 0] = ca.sin(self.param)
        matrix[1, 1] = ca.cos(self.param)
        return matrix


class LieAlgebraSO2(LieAlgebra):
    """
    The so(2) Lie Algebra
    """

    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (1, 1)

    def wedge(self):
        algebra = ca.sparsify(ca.SX.zeros(2, 2))
        algebra[0, 1] = -self.param
        algebra[1, 0] = self.param
        return algebra

    def vee(self):
        return self.param

    def exp(self):
        return NotImplementedError("")

    def neg(self):
        return LieAlgebraSO2(-self.param)

    def add(self, other):
        return LieAlgebraSO2(self.param + other.param)

    def rmul(self, other):
        other = ca.SX(other)
        assert ca.SX(other).shape == (1, 1)
        return LieAlgebraSO2(other * self.param)
