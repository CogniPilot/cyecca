from __future__ import annotations

import casadi as ca
from .base import *


class LieAlgebraDirectProduct(LieAlgebra):
    def __init__(self, algebras: List[LieAlgebra]):
        self.algebras = algebras
        self.n_param_list = [algebra.n_param for algebra in self.algebras]
        n_param = sum(self.n_param_list)

        # param start indices for subgroups
        count = 0
        self.subparam_start = [0]
        matrix_shape = [0, 0]
        for i in range(len(self.algebras)):
            count += self.n_param_list[i]
            self.subparam_start.append(count)
            matrix_shape[0] += self.algebras[i].matrix_shape[0]
            matrix_shape[1] += self.algebras[i].matrix_shape[1]
        super().__init__(n_param=n_param, matrix_shape=tuple(matrix_shape))

    def __mul__(self, other: LieAlgebra):
        """
        Implements Direct Product of Lie Algebras
        """
        return LieAlgebraDirectProduct(algebras=self.algebras + [other])

    def sub_param(self, i: int, param: PARAM_TYPE):
        start = self.subparam_start[i]
        stop = start + self.algebras[i].n_param
        return param[start:stop]

    def sub_elements(self, arg: LieAlgebraElement):
        return [
            LieAlgebraElement(self.algebras[i], self.sub_param(i=i, param=arg.param))
            for i in range(len(self.algebras))
        ]

    def bracket(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        raise NotImplementedError("")

    def scalar_multipication(
        self, left: SCALAR_TYPE, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        assert right.algebra == self
        return LieAlgebraElement(algebra=self, param=left * right.param)

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        return LieAlgebraElement(group=self, param=left.param + right.param)

    def adjoint(self, arg: LieAlgebraElement) -> ca.SX:
        assert arg.group == self
        return ca.diagcat(*[x.ad() for x in self.sub_elements(arg)])

    def to_Matrix(self, arg: LieAlgebraElement) -> ca.SX:
        assert arg.algebra == self
        matrix_list = []
        for i in range(len(self.groups)):
            X = self.sub_element(i=i, arg=left).to_Matrix()
            matrix_list.append(X)
        return ca.diagcat(*matrix_list)

    def __repr__(self):
        return " x ".join([algebra.__class__.__name__ for algebra in self.algebras])


class LieGroupDirectProduct(LieGroup):
    def __init__(self, groups: List[LieGroup]):
        self.groups = groups
        self.n_param_list = [group.n_param for group in self.groups]
        n_param = sum(self.n_param_list)

        # param start indices for subgroups
        count = 0
        self.subparam_start = [0]
        algebra = None
        matrix_shape = [0, 0]
        for i in range(len(self.groups)):
            group = self.groups[i]
            if algebra is None:
                algebra = group.algebra
            else:
                algebra = algebra * group.algebra
            count += self.n_param_list[i]
            self.subparam_start.append(count)
            matrix_shape[0] += group.matrix_shape[0]
            matrix_shape[1] += group.matrix_shape[1]

        super().__init__(
            algebra=algebra, n_param=n_param, matrix_shape=tuple(matrix_shape)
        )

    def __mul__(self, other: LieGroup):
        """
        Implements Direct Product of Lie Groups
        """
        return LieGroupDirectProduct(groups=self.groups + [other])

    def sub_elements(self, arg: LieGroupElement):
        return [
            LieGroupElement(self.groups[i], self.sub_param(i=i, param=arg.param))
            for i in range(len(self.groups))
        ]

    def sub_param(self, i: int, param: PARAM_TYPE):
        start = self.subparam_start[i]
        stop = start + self.groups[i].n_param
        return param[start:stop]

    def product(self, left: LieGroupElement, right: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        assert self == right.group
        return LieGroupElement(
            group=self,
            param=ca.vertcat(
                *[
                    (X1 * X2).param
                    for X1, X2 in zip(
                        self.sub_elements(arg=left), self.sub_elements(arg=right)
                    )
                ]
            ),
        )

    def inverse(self, arg: LieGroupElement) -> LieGroupElement:
        assert self == arg.group
        return LieGroupElement(
            group=self,
            param=ca.vertcat(*[X.inverse().param for X in self.sub_elements(arg)]),
        )

    def identity(self) -> LieGroupElement:
        return LieGroupElement(
            group=self,
            param=ca.vertcat(*[group.identity().param for group in self.groups]),
        )

    def adjoint(self, arg: LieGroupElement) -> ca.SX:
        raise NotImplementedError("")

    def exp(self, arg: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == arg.algebra
        algebra = arg.algebra  # type: LieAlgebraDirectProduct
        return LieGroupElement(
            group=self,
            param=ca.vertcat(
                *[
                    x1.exp(group=group).param
                    for group, x1 in zip(self.groups, algebra.sub_elements(arg))
                ]
            ),
        )

    def log(self, arg: LieGroupElement) -> LieAlgebraElement:
        return LieAlgebraElement(
            algebra=self.algebra,
            param=ca.vertcat(*[X1.log().param for X1 in self.sub_elements(arg)]),
        )

    def to_Matrix(self, arg: LieGroupElement) -> ca.SX:
        assert arg.group == self
        return ca.diagcat(*[X.to_Matrix() for X in self.sub_elements(arg)])

    def __repr__(self):
        return " x ".join([group.__class__.__name__ for group in self.groups])
