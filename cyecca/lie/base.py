from __future__ import annotations

import casadi as ca

from abc import ABC, abstractmethod
from beartype import beartype
from beartype.typing import List, Union

from cyecca.symbolic import casadi_to_sympy
import sympy

__all__ = [
    "LieAlgebraElement",
    "LieAlgebra",
    "LieGroupElement",
    "LieGroup",
    "SCALAR_TYPE",
    "PARAM_TYPE",
]

SCALAR_TYPE = Union[ca.SX, ca.DM, float, int]
PARAM_TYPE = Union[ca.SX, ca.DM]


@beartype
class LieAlgebraElement:
    """
    This is a generic Lie algebra elem, not necessarily represented as a matrix
    """

    def __init__(self, algebra: LieAlgebra, param: PARAM_TYPE):
        self.algebra = algebra
        self.param = ca.SX(param)
        assert self.param.shape == (self.algebra.n_param, 1)

    def ad(self) -> ca.SX:
        """returns the adjoint as a linear operator on the parameter vector"""
        return self.algebra.adjoint(self)

    def vee(self) -> ca.SX:
        """maps from Lie algebra to its parameters as a vector"""
        return self.algebra.vee(self)

    def left_jacobian(self) -> ca.SX:
        return self.algebra.left_jacobian(self)

    def left_jacobian_inv(self) -> ca.SX:
        return self.algebra.left_jacobian_inv(self)

    def left_Q(self, vb: LieAlgebraElement, omega: LieAlgebraElement) -> ca.SX:
        return self.algebra.left_Q(vb, omega)

    def right_jacobian(self) -> ca.SX:
        return self.algebra.right_jacobian(self)

    def right_jacobian_inv(self) -> ca.SX:
        return self.algebra.right_jacobian_inv(self)

    def right_Q(self, vb: LieAlgebraElement, omega: LieAlgebraElement) -> ca.SX:
        return self.algebra.right_Q(vb, omega)

    def __neg__(self) -> LieAlgebraElement:
        return -1 * self

    def __eq__(self, other: LieAlgebraElement) -> ca.SX:
        return ca.logic_all(ca.eq(self.param, other.param))

    def __mul__(
        self, right: Union[LieAlgebraElement, SCALAR_TYPE]
    ) -> LieAlgebraElement:
        if isinstance(right, LieAlgebraElement):
            return self.algebra.bracket(left=self, right=right)
        elif isinstance(right, SCALAR_TYPE):
            return self.algebra.scalar_multiplication(left=right, right=self)

    def __rmul__(self, arg: SCALAR_TYPE) -> LieAlgebraElement:
        return self.algebra.scalar_multiplication(left=arg, right=self)

    def __add__(self, arg: LieAlgebraElement) -> LieAlgebraElement:
        return self.algebra.addition(left=self, right=arg)

    def __sub__(self, arg: LieAlgebraElement) -> LieAlgebraElement:
        return self.algebra.addition(left=self, right=-arg)

    def to_Matrix(self) -> ca.SX:
        return self.algebra.to_Matrix(self)

    def from_Matrix(self) -> LieGroupElement:
        return self.algebra.from_Matrix(self)

    def exp(self, group: LieGroup) -> LieGroupElement:
        return group.exp(self)

    def __repr__(self) -> str:
        return "{:s}: {:s}".format(repr(self.algebra), repr(self.param))


@beartype
class LieAlgebra(ABC):
    """
    This is a generic Lie algebra, not necessarily represented as a matrix
    """

    def __init__(self, n_param: int, matrix_shape: tuple[int, int]):
        self.n_param = n_param
        self.matrix_shape = matrix_shape

    def __mul__(self, other: LieAlgebra) -> LieAlgebraDirectProduct:
        """
        Implements Direct Product of Lie Algebras
        """
        return LieAlgebraDirectProduct(algebras=[self, other])

    @abstractmethod
    def elem(self, param: PARAM_TYPE) -> LieAlgebraElement:
        ...

    def wedge(self, arg: PARAM_TYPE) -> LieAlgebraElement:
        """given a parameter vector, creates a LieAlgebraElement"""
        return self.elem(param=arg)

    def vee(self, arg: LieAlgebraElement) -> ca.SX:
        """given a LieAlgebraElement, returns a parameter vector"""
        return arg.param

    def left_jacobian(self, arg: LieAlgebraElement) -> ca.SX:
        ...

    def left_jacobian_inv(self, arg: LieAlgebraElement) -> ca.SX:
        ...

    def left_Q(self, vb: LieAlgebraElement, omega: LieAlgebraElement) -> ca.SX:
        ...

    def right_jacobian(self, arg: LieAlgebraElement) -> ca.SX:
        ...

    def right_jacobian_inv(self, arg: LieAlgebraElement) -> ca.SX:
        ...

    def right_Q(self, vb: LieAlgebraElement, omega: LieAlgebraElement) -> ca.SX:
        ...

    def scalar_multiplication(
        self, left: SCALAR_TYPE, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        return LieGroupElement(self.groups[i], self.sub_param(i=i, param=arg.param))

    @abstractmethod
    def bracket(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        ...

    @abstractmethod
    def scalar_multiplication(
        self, left: SCALAR_TYPE, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        ...

    @abstractmethod
    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        ...

    @abstractmethod
    def adjoint(self, arg: LieAlgebraElement) -> ca.SX:
        ...

    @abstractmethod
    def to_Matrix(self, arg: LieAlgebraElement) -> ca.SX:
        ...

    @abstractmethod
    def from_Matrix(self, arg: LieAlgebraElement) -> ca.SX:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


@beartype
class LieGroupElement:
    """
    This is a generic Lie group elem, not necessarily represented as a matrix
    """

    def __init__(self, group: LieGroup, param: PARAM_TYPE):
        self.group = group
        self.param = ca.SX(param)
        assert self.param.shape == (self.group.n_param, 1)

    def inverse(self) -> LieGroupElement:
        return self.group.inverse(arg=self)

    def __add__(
        self, other: Union[LieGroupElement, LieAlgebraElement]
    ) -> LieGroupElement:
        if isinstance(other, LieGroupElement):
            if hasattr(self.group, "addition"):
                return self.group.addition(self, other)
            else:
                raise TypeError(
                    "{:s} does not support addition".format(
                        self.group.__class__.__name__
                    )
                )
        elif isinstance(other, LieAlgebraElement):
            return self * other.exp(self.group)

    def __sub__(
        self, other: Union[LieGroupElement, LieAlgebraElement]
    ) -> LieGroupElement:
        if isinstance(other, LieGroupElement):
            if hasattr(self.group, "subtraction"):
                return self.group.subtraction(self, other)
            else:
                raise TypeError(
                    "{:s} does not support subtraction".format(
                        self.group.__class__.__name__
                    )
                )
        elif isinstance(other, LieAlgebraElement):
            return self * (-other).exp(self.group)

    def __eq__(self, other: LieGroupElement) -> ca.SX:
        return ca.logic_all(ca.eq(self.param, other.param))

    def __mul__(self, right: LieGroupElement) -> LieGroupElement:
        return self.group.product(left=self, right=right)

    def Ad(self) -> ca.SX:
        return self.group.adjoint(arg=self)

    def to_Matrix(self) -> ca.SX:
        return self.group.to_Matrix(arg=self)

    def from_Matrix(self) -> LieGroupElement:
        return self.group.from_Matrix(self)

    def log(self) -> LieAlgebraElement:
        return self.group.log(arg=self)

    def __repr__(self) -> str:
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

    def elem(self, param: PARAM_TYPE) -> LieGroupElement:
        return LieGroupElement(group=self, param=param)

    def __mul__(self, other: LieGroup) -> LieGroupDirectProduct:
        """
        Implements Direct Product of Groups
        """
        return LieGroupDirectProduct(groups=[self, other])

    @abstractmethod
    def product(self, left: LieGroupElement, right: LieGroupElement) -> LieGroupElement:
        ...

    @abstractmethod
    def inverse(self, arg: LieGroupElement) -> LieGroupElement:
        ...

    @abstractmethod
    def identity(self) -> LieGroupElement:
        ...

    @abstractmethod
    def adjoint(self, arg: LieGroupElement) -> ca.SX:
        ...

    @abstractmethod
    def exp(self, arg: LieAlgebraElement) -> LieGroupElement:
        ...

    @abstractmethod
    def log(self, arg: LieGroupElement) -> LieAlgebraElement:
        ...

    @abstractmethod
    def to_Matrix(self, arg: LieGroupElement) -> ca.SX:
        ...

    @abstractmethod
    def from_Matrix(self, arg: LieAlgebraElement) -> ca.SX:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


from .direct_product import LieGroupDirectProduct, LieAlgebraDirectProduct
