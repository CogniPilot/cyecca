import casadi as ca

from .base import LieAlgebra, LieGroup


class LieGroup_SO2(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (1, 1)

    def inv(self):
        return LieGroup_SO2(self.n, -self.param)

    def log(self):
        return LieAlgebra_so2(self.n, self.param)

    def product(self, other):
        v = self.param + other.param
        return LieGroup_SO2(self.n, v)

    def identity(self):
        return LieGroup_SO2(0)

    def to_matrix_lie_group(self):
        G = ca.SX.zeros(2, 2)
        G[0, 0] = ca.cos(self.param)
        G[0, 1] = -ca.sin(self.param)
        G[1, 0] = ca.sin(self.param)
        G[1, 1] = ca.cos(self.param)
        return G


class LieAlgebra_so2(LieAlgebra):
    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (1, 1)

    def wedge(self):
        X = ca.sparsify(ca.SX.zeros(self.n + 1, self.n + 1))
        X[: self.n, self.n] = self.param
        return X

    def vee(self):
        return self.param

    def exp(self):
        return LieGroup_SO2(self.n, self.param)

    def neg(self):
        return LieAlgebra_so2(self.n, -self.param)

    def add(self, other):
        return LieAlgebra_so2(self.n, self.param + other.param)

    def rmul(self, other):
        other = ca.SX(other)
        assert ca.SX(other).shape == (1, 1)
        return LieAlgebra_so2(self.n, other * self.param)
