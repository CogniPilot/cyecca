import casadi as ca

from .base import LieAlgebra, LieGroup


class LieGroup_R(LieGroup):
    def __init__(self, n, param):
        super().__init__(param)
        self.n = n
        assert self.param.shape == (n, 1)

    def inv(self):
        return LieGroup_R(self.n, -self.param)

    def log(self):
        return LieAlgebra_r(self.n, self.param)

    def product(self, other):
        v = self.param + other.param
        return LieGroup_R(self.n, v)

    def identity(self):
        v = ca.sparsify(ca.SX.zeros(self.n, 1))
        return LieGroup_R(self.n, v)

    def to_matrix_lie_group(self):
        G = ca.sparsify(ca.SX.zeros(self.n + 1, self.n + 1))
        G[: self.n, : self.n] = ca.SX.eye(self.n)
        G[: self.n, self.n] = self.param
        G[self.n, self.n] = 1
        return G


class LieAlgebra_r(LieAlgebra):
    def __init__(self, n, param):
        super().__init__(param)
        self.n = n
        assert self.param.shape == (n, 1)

    def wedge(self):
        X = ca.sparsify(ca.SX.zeros(self.n + 1, self.n + 1))
        X[: self.n, self.n] = self.param
        return X

    def vee(self):
        return self.param

    def exp(self):
        return LieGroup_R(self.n, self.param)

    def neg(self):
        return LieAlgebra_r(self.n, -self.param)

    def add(self, other):
        return LieAlgebra_r(self.n, self.param + other.param)

    def rmul(self, other):
        other = ca.SX(other)
        assert ca.SX(other).shape == (1, 1)
        v = other * self.param
        return LieAlgebra_r(self.n, v)
