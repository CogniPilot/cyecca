from __future__ import annotations

from abc import ABC, abstractmethod
import sympy
from beartype import beartype


@beartype
class LieAlgebraElement:
    """
    This is a generic Lie algebra element, not necessarily represented as a matrix
    """

    def __init__(self, algebra: "LieAlgebra", param: sympy.Matrix):
        self.algebra = algebra
        assert param.shape == (self.algebra.n_param, 1)
        self.param = param

    def ad(self) -> sympy.Matrix:
        """returns the adjoint as a linear operator on the parameter vector"""
        return self.algebra.adjoint(self)

    def vee(self) -> sympy.Matrix:
        """maps from Lie algebra to its parameters as a vector"""
        return self.algebra.vee(self)

    def __mul__(self, right: "LieAlgebraElement") -> "LieAlgebraElement":
        return self.algebra.bracket(self, right)

    def __rmul__(self, left) -> "LieAlgebraElement":
        return self.algebra.scalar_multipication(left, self)

    def __add__(self, right: "LieAlgebraElement") -> "LieAlgebraElement":
        return self.algebra.addition(self, right)

    def to_matrix(self) -> sympy.Matrix:
        return self.algebra.to_matrix(self)

    def exp(self, group: "LieGroup") -> "LieGroupElement":
        return group.exp(self)

    def __repr__(self):
        return "{:s}({:s}, {:s})".format(
            self.__class__.__name__, self.algebra.__class__.__name__, repr(self.param)
        )


@beartype
class LieAlgebra(ABC):
    """
    This is a generic Lie algebra, not necessarily represented as a matrix
    """

    def __init__(self, n_param: int, matrix_shape: tuple[int, int]):
        self.n_param = n_param
        self.matrix_shape = matrix_shape

    def element(self, param: sympy.Matrix) -> LieAlgebraElement:
        return LieAlgebraElement(self, param)

    def wedge(self, left: sympy.Matrix) -> LieAlgebraElement:
        """given a parameter vector, creates a LieAlgebraElement"""
        return LieAlgebraElement(self, left)

    def vee(self, left: LieAlgebraElement) -> sympy.Matrix:
        """given a LieAlgebraElement, returns a parameter vector"""
        return left.param

    @abstractmethod
    def bracket(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        ...

    @abstractmethod
    def scalar_multipication(
        self, left: float, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        ...

    @abstractmethod
    def addition(
        self, left: LieAlgebraElement, right: LieAlgebraElement
    ) -> LieAlgebraElement:
        ...

    @abstractmethod
    def adjoint(self, left: LieAlgebraElement) -> sympy.Matrix:
        ...

    @abstractmethod
    def to_matrix(self, left: LieAlgebraElement) -> sympy.Matrix:
        ...

    def __repr__(self):
        return self.__class__.__name__


@beartype
class LieGroupElement:
    """
    This is a generic Lie group element, not necessarily represented as a matrix
    """

    def __init__(self, group: "LieGroup", param: sympy.Matrix):
        self.group = group
        assert param.shape == (self.group.n_param, 1)
        self.param = param

    def inverse(self) -> LieGroupElement:
        return self.group.inverse(self)

    def __mul__(self, right: LieGroupElement) -> LieGroupElement:
        return self.group.product(self, right)

    def Ad(self) -> sympy.Matrix:
        return self.group.adjoint(self)

    def to_matrix(self):
        return self.group.to_matrix(self)

    def log(self, algebra: LieAlgebra) -> LieGroupElement:
        return self.group.log(algebra, self)

    def __repr__(self):
        return "{:s}({:s}, {:s})".format(
            self.__class__.__name__, self.group.__class__.__name__, repr(self.param)
        )


@beartype
class LieGroup:
    """
    This is a generic Lie group, not necessarily represented as a matrix
    """

    def __init__(
        self, algebra: LieAlgebra, n_param: int, matrix_shape: tuple[int, int]
    ):
        self.algebra = algebra
        self.n_param = n_param
        self.matrix_shape = matrix_shape

    def element(self, param: sympy.Matrix) -> LieGroupElement:
        return LieGroupElement(self, param)

    @abstractmethod
    def product(self, left: LieGroupElement, right: LieGroupElement) -> LieGroupElement:
        ...

    @abstractmethod
    def inverse(self, left: LieGroupElement) -> LieGroupElement:
        ...

    @abstractmethod
    def identity(self) -> LieGroupElement:
        ...

    @abstractmethod
    def adjoint(self, left: LieGroupElement) -> sympy.Matrix:
        ...

    @abstractmethod
    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        ...

    @abstractmethod
    def log(self, algebra: LieAlgebra, left: LieGroupElement) -> LieAlgebraElement:
        ...

    @abstractmethod
    def to_matrix(self, left: LieGroupElement) -> sympy.Matrix:
        ...

    def __repr__(self):
        return self.__class__.__name__
