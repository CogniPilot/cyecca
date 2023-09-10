from __future__ import annotations

import casadi as ca
from cyecca.lie.base import *
from beartype import beartype
from beartype.typing import List, Union


@beartype
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

    def elem(self, param: PARAM_TYPE) -> LieAlgebraDirectProductElement:
        return LieAlgebraDirectProductElement(algebra=self, param=param)

    def __mul__(self, other: LieAlgebra):
        """
        Implements Direct Product of Lie Algebras
        """
        return LieAlgebraDirectProduct(algebras=self.algebras + [other])

    def sub_param(self, i: int, param: PARAM_TYPE) -> ca.SX:
        start = self.subparam_start[i]
        stop = start + self.algebras[i].n_param
        return param[start:stop]

    def sub_elems(self, arg: LieAlgebraDirectProductElement) -> List[LieAlgebraElement]:
        return [
            self.algebras[i].elem(self.sub_param(i=i, param=arg.param))
            for i in range(len(self.algebras))
        ]

    def bracket(
        self,
        left: LieAlgebraDirectProductElement,
        right: LieAlgebraDirectProductElement,
    ) -> LieAlgebraDirectProductElement:
        raise NotImplementedError("")

    def scalar_multiplication(
        self, left: SCALAR_TYPE, right: LieAlgebraDirectProductElement
    ) -> LieAlgebraDirectProductElement:
        assert right.algebra == self
        return LieAlgebraDirectProductElement(algebra=self, param=left * right.param)

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        return LieAlgebraDirectProductElement(
            algebra=self, param=left.param + right.param
        )

    def adjoint(self, arg: LieAlgebraElement) -> ca.SX:
        assert arg.algebra == self
        return ca.diagcat(*[x.ad() for x in self.sub_elems(arg)])

    def to_Matrix(self, arg: LieAlgebraElement) -> ca.SX:
        assert arg.algebra == self
        return ca.diagcat(*[X.to_Matrix() for X in self.sub_elems(arg)])

    def from_Matrix(self, arg: ca.SX) -> LieAlgebraDirectProduct:
        assert arg.shape == self.matrix_shape
        raise NotImplementedError("")

    def __repr__(self):
        return " x ".join([algebra.__class__.__name__ for algebra in self.algebras])


@beartype
class LieAlgebraDirectProductElement(LieAlgebraElement):
    """
    This is an Direct Product Lie algebra elem
    """

    def __init__(self, algebra: LieAlgebraDirectProduct, param: PARAM_TYPE):
        self.algebra = algebra
        self.param = ca.SX(param)
        assert self.param.shape == (self.algebra.n_param, 1)

    def __eq__(self, other: LieAlgebraDirectProductElement) -> bool:
        return bool(ca.logic_all(self.param == other.param))

    def __mul__(
        self, right: Union[LieAlgebraDirectProductElement, SCALAR_TYPE]
    ) -> LieAlgebraDirectProductElement:
        if isinstance(right, LieAlgebraDirectProductElement):
            return self.algebra.bracket(left=self, right=right)
        elif isinstance(right, SCALAR_TYPE):
            return self.algebra.scalar_multiplication(left=right, right=self)

    def __neg__(self) -> LieAlgebraDirectProductElement:
        return -1 * self

    def __rmul__(self, arg: SCALAR_TYPE) -> LieAlgebraDirectProductElement:
        return self.algebra.scalar_multiplication(left=arg, right=self)

    def __add__(
        self, arg: LieAlgebraDirectProductElement
    ) -> LieAlgebraDirectProductElement:
        return self.algebra.addition(left=self, right=arg)

    def __sub__(
        self, arg: LieAlgebraDirectProductElement
    ) -> LieAlgebraDirectProductElement:
        return self.algebra.addition(left=self, right=-arg)

    def exp(self, group: LieGroupDirectProduct) -> LieGroupDirectProductElement:
        return group.exp(self)

    def __repr__(self):
        return "{:s}: {:s}".format(repr(self.algebra), repr(self.param))


@beartype
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

    def elem(self, param: PARAM_TYPE) -> LieGroupDirectProductElement:
        return LieGroupDirectProductElement(group=self, param=param)

    def __add__(self, other: LieAlgebraElement) -> LieGroupElement:
        return self * other.exp(self.group)

    def __mul__(self, other: LieGroup) -> LieGroupDirectProduct:
        """
        Implements Direct Product of Lie Groups
        """
        return LieGroupDirectProduct(groups=self.groups + [other])

    def sub_elems(self, arg: LieGroupDirectProductElement) -> List[LieGroupElement]:
        return [
            self.groups[i].elem(self.sub_param(i=i, param=arg.param))
            for i in range(len(self.groups))
        ]

    def sub_param(self, i: int, param: PARAM_TYPE) -> ca.SX:
        start = self.subparam_start[i]
        stop = start + self.groups[i].n_param
        return param[start:stop]

    def product(
        self, left: LieGroupDirectProductElement, right: LieGroupDirectProductElement
    ) -> LieGroupDirectProductElement:
        assert self == left.group
        assert self == right.group
        return LieGroupDirectProductElement(
            group=self,
            param=ca.vertcat(
                *[
                    (X1 * X2).param
                    for X1, X2 in zip(
                        self.sub_elems(arg=left), self.sub_elems(arg=right)
                    )
                ]
            ),
        )

    def inverse(
        self, arg: LieGroupDirectProductElement
    ) -> LieGroupDirectProductElement:
        assert self == arg.group
        return LieGroupDirectProductElement(
            group=self,
            param=ca.vertcat(*[X.inverse().param for X in self.sub_elems(arg)]),
        )

    def identity(self) -> LieGroupDirectProductElement:
        return LieGroupDirectProductElement(
            group=self,
            param=ca.vertcat(*[group.identity().param for group in self.groups]),
        )

    def adjoint(self, arg: LieGroupDirectProductElement) -> ca.SX:
        raise NotImplementedError("")

    def exp(self, arg: LieAlgebraDirectProductElement) -> LieGroupDirectProductElement:
        assert self.algebra == arg.algebra
        algebra = arg.algebra  # type: LieAlgebraDirectProduct
        return LieGroupDirectProductElement(
            group=self,
            param=ca.vertcat(
                *[
                    x1.exp(group=group).param
                    for group, x1 in zip(self.groups, algebra.sub_elems(arg))
                ]
            ),
        )

    def log(self, arg: LieGroupDirectProductElement) -> LieAlgebraDirectProductElement:
        return LieAlgebraDirectProductElement(
            algebra=self.algebra,
            param=ca.vertcat(*[X1.log().param for X1 in self.sub_elems(arg)]),
        )

    def to_Matrix(self, arg: LieGroupDirectProductElement) -> ca.SX:
        assert arg.group == self
        return ca.diagcat(*[X.to_Matrix() for X in self.sub_elems(arg)])

    def from_Matrix(self, arg: ca.SX) -> LieGroupDirectProductElement:
        assert arg.shape == self.matrix_shape
        raise NotImplementedError("")

    def __repr__(self) -> str:
        return " x ".join([group.__class__.__name__ for group in self.groups])


@beartype
class LieGroupDirectProductElement(LieGroupElement):
    """
    This is a Lie group directo product elem
    """

    def __init__(self, group: LieGroupDirectProduct, param: PARAM_TYPE):
        self.group = group
        self.param = ca.SX(param)
        assert self.param.shape == (self.group.n_param, 1)

    def inverse(self) -> LieGroupDirectProductElement:
        return self.group.inverse(arg=self)

    def __add__(
        self, other: LieAlgebraDirectProductElement
    ) -> LieGroupDirectProductElement:
        return self * other.exp(self.group)

    def __sub__(
        self, other: LieAlgebraDirectProductElement
    ) -> LieGroupDirectProductElement:
        return self * (-other).exp(self.group)

    def __eq__(self, other: LieGroupDirectProductElement) -> bool:
        return bool(ca.logic_all(self.param == other.param))

    def __mul__(
        self, right: LieGroupDirectProductElement
    ) -> LieGroupDirectProductElement:
        return self.group.product(left=self, right=right)

    def log(self) -> LieAlgebraDirectProductElement:
        return self.group.log(arg=self)
