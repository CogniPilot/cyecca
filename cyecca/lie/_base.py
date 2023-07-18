from __future__ import annotations

import casadi as ca

from abc import ABC, abstractmethod
from beartype import beartype
from beartype.typing import List


SCALAR_TYPE = (ca.SX, ca.DM, float, int)
PARAM_TYPE = (ca.SX, ca.DM)


@beartype
class LieAlgebraElement:
    """
    This is a generic Lie algebra element, not necessarily represented as a matrix
    """

    def __init__(self, algebra: LieAlgebra, param: PARAM_TYPE):
        self.algebra = algebra
        assert param.shape == (self.algebra.n_param, 1)
        self.param = ca.SX(param)

    def ad(self) -> ca.SX:
        """returns the adjoint as a linear operator on the parameter vector"""
        return self.algebra.adjoint(self)

    def vee(self) -> ca.SX:
        """maps from Lie algebra to its parameters as a vector"""
        return self.algebra.vee(self)

    def __neg__(self):
        return -1 * self

    def __eq__(self, other: LieAlgebraElement):
        return ca.logic_all(self.param == other.param)

    def __mul__(self, right) -> LieAlgebraElement:
        if isinstance(right, LieAlgebraElement):
            return self.algebra.bracket(left=self, right=right)
        elif isinstance(right, SCALAR_TYPE):
            return self.algebra.scalar_multipication(left=right, right=self)

    def __rmul__(self, left: SCALAR_TYPE) -> LieAlgebraElement:
        return self.algebra.scalar_multipication(left=left, right=self)

    def __add__(self, right: "LieAlgebraElement") -> "LieAlgebraElement":
        return self.algebra.addition(left=self, right=right)

    def to_matrix(self) -> ca.SX:
        return self.algebra.to_matrix(self)

    def exp(self, group: "LieGroup") -> "LieGroupElement":
        return group.exp(self)

    def __repr__(self):
        return "{:s}: {:s}".format(repr(self.algebra), repr(self.param))


@beartype
class LieAlgebra(ABC):
    """
    This is a generic Lie algebra, not necessarily represented as a matrix
    """

    def __init__(self, n_param: int, matrix_shape: tuple[int, int]):
        self.n_param = n_param
        self.matrix_shape = matrix_shape

    def __mul__(self, other: LieAlgebra):
        """
        Implements Direct Product of Lie Algebras
        """
        return LieAlgebraDirectProduct(algebras=[self, other])

    def element(self, param: PARAM_TYPE) -> LieAlgebraElement:
        return LieAlgebraElement(algebra=self, param=param)

    def wedge(self, left: PARAM_TYPE) -> LieAlgebraElement:
        """given a parameter vector, creates a LieAlgebraElement"""
        return self.element(param=left)

    def vee(self, left: LieAlgebraElement) -> ca.SX:
        """given a LieAlgebraElement, returns a parameter vector"""
        return left.param

    @abstractmethod
    def bracket(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        pass

    @abstractmethod
    def scalar_multipication(
        self, left: SCALAR_TYPE, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        pass

    @abstractmethod
    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        pass

    @abstractmethod
    def adjoint(self, left: LieAlgebraElement) -> ca.SX:
        pass

    @abstractmethod
    def to_matrix(self, left: LieAlgebraElement) -> ca.SX:
        pass

    def __repr__(self):
        return self.__class__.__name__


@beartype
class LieGroupElement:
    """
    This is a generic Lie group element, not necessarily represented as a matrix
    """

    def __init__(self, group: LieGroup, param: PARAM_TYPE):
        self.group = group
        assert param.shape == (self.group.n_param, 1)
        self.param = ca.SX(param)

    def inverse(self) -> LieGroupElement:
        return self.group.inverse(left=self)

    def __add__(self, other: LieAlgebraElement):
        return self * other.exp(self.group)

    def __sub__(self, other: LieAlgebraElement):
        return self * (-other).exp(self.group)

    def __eq__(self, other):
        return ca.logic_all(self.param == other.param)

    def __mul__(self, right: LieGroupElement) -> LieGroupElement:
        return self.group.product(left=self, right=right)

    def Ad(self) -> ca.SX:
        return self.group.adjoint(left=self)

    def to_matrix(self):
        return self.group.to_matrix(left=self)

    def log(self) -> LieAlgebraElement:
        return self.group.log(left=self)

    def __repr__(self):
        return "{:s}: {:s}".format(repr(self.group), repr(self.param))


@beartype
class LieGroup(ABC):
    """
    This is a generic Lie group, not necessarily represented as a matrix
    """

    def __init__(
        self, algebra: LieAlgebra, n_param: int, matrix_shape: tuple[int, int]
    ):
        self.algebra = algebra
        self.n_param = n_param
        self.matrix_shape = matrix_shape

    def element(self, param: PARAM_TYPE) -> LieGroupElement:
        return LieGroupElement(group=self, param=param)

    def __mul__(self, other: LieGroup):
        """
        Implements Direct Product of Groups
        """
        return LieGroupDirectProduct(groups=[self, other])

    @abstractmethod
    def product(self, left: LieGroupElement, right: LieGroupElement) -> LieGroupElement:
        pass

    @abstractmethod
    def inverse(self, left: LieGroupElement) -> LieGroupElement:
        pass

    @abstractmethod
    def identity(self) -> LieGroupElement:
        pass

    @abstractmethod
    def adjoint(self, left: LieGroupElement) -> ca.SX:
        pass

    @abstractmethod
    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        pass

    @abstractmethod
    def log(self, algebra: LieAlgebra, left: LieGroupElement) -> LieAlgebraElement:
        pass

    @abstractmethod
    def to_matrix(self, left: LieGroupElement) -> ca.SX:
        pass

    def __repr__(self):
        return self.__class__.__name__


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

    def subalgebra_param(self, i: int, param: PARAM_TYPE):
        start = self.subparam_start[i]
        stop = start + self.groups[i].n_param
        return param[start:stop]

    def bracket(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        raise NotImplementedError("")

    def scalar_multipication(
        self, left: SCALAR_TYPE, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        raise NotImplementedError("")

    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        raise NotImplementedError("")

    def adjoint(self, left: LieAlgebraElement) -> ca.SX:
        raise NotImplementedError("")

    def to_matrix(self, left: LieAlgebraElement) -> ca.SX:
        raise NotImplementedError("")

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

    def subgroup_param(self, i: int, param: PARAM_TYPE):
        start = self.subparam_start[i]
        stop = start + self.groups[i].n_param
        return param[start:stop]

    def product(self, left: LieGroupElement, right: LieGroupElement) -> LieGroupElement:
        assert left.group == self
        assert right.group == self
        param_list = []
        for i in range(len(self.groups)):
            group = self.groups[i]
            X1 = LieGroupElement(group, self.subgroup_param(i=i, param=left.param))
            X2 = LieGroupElement(group, self.subgroup_param(i=i, param=right.param))
            X3 = X1 * X2
            param_list.append(X3.param)
        param = ca.vertcat(*param_list)
        return LieGroupElement(group=self, param=param)

    def inverse(self, left: LieGroupElement) -> LieGroupElement:
        assert left.group == self
        param_list = []
        for i in range(len(self.groups)):
            group = self.groups[i]
            X = LieGroupElement(group, self.subgroup_param(i=i, param=left.param))
            X_inv = X.inverse()
            param_list.append(X_inv.param)
        param = ca.vertcat(*param_list)
        return LieGroupElement(group=self, param=param)

    def identity(self) -> LieGroupElement:
        param_list = []
        for i in range(len(self.groups)):
            group = self.groups[i]
            param_list.append(group.identity().param)
        param = ca.vertcat(*param_list)
        return LieGroupElement(group=self, param=param)

    def adjoint(self, left: LieGroupElement) -> ca.SX:
        raise NotImplementedError("")

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        raise NotImplementedError("")

    def log(self, algebra: LieAlgebra, left: LieGroupElement) -> LieAlgebraElement:
        raise NotImplementedError("")

    def to_matrix(self, left: LieGroupElement) -> ca.SX:
        raise NotImplementedError("")

    def __repr__(self):
        return " x ".join([group.__class__.__name__ for group in self.groups])
