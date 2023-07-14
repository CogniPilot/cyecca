from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy import floating

import casadi as ca

from abc import ABC, abstractmethod
from beartype import beartype


@beartype
class LieAlgebraElement:
    """
    This is a generic Lie algebra element, not necessarily represented as a matrix
    """

    def __init__(self, algebra: LieAlgebra, param: ca.SX):
        self.algebra = algebra
        assert param.shape == (self.algebra.n_param,1)
        self.param = param

    def ad(self) -> ca.SX:
        """returns the adjoint as a linear operator on the parameter vector"""
        return self.algebra.adjoint(self)

    def vee(self) -> ca.SX:
        """maps from Lie algebra to its parameters as a vector"""
        return self.algebra.vee(self)

    def __eq__(self, other):
        return self.param == other.param

    def __mul__(self, right: LieAlgebraElement) -> LieAlgebraElement:
        return self.algebra.bracket(left=self, right=right)

    def __rmul__(self, left) -> LieAlgebraElement:
        return self.algebra.scalar_multipication(left=left, right=self)

    def __add__(self, right: "LieAlgebraElement") -> "LieAlgebraElement":
        return self.algebra.addition(left=self, right=right)

    def to_matrix(self) -> ca.SX:
        return self.algebra.to_matrix(self)

    def exp(self, group: "LieGroup") -> "LieGroupElement":
        return group.exp(self)

    def __str__(self):
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

    def element(self, param: ca.SX) -> LieAlgebraElement:
        return LieAlgebraElement(algebra=self, param=param)

    def wedge(self, left: ca.SX) -> LieAlgebraElement:
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
        self, left: Real, right: LieAlgebraElement
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

    def __init__(self, group: LieGroup, param: ca.SX):
        self.group = group
        assert param.shape == (self.group.n_param,1)
        self.param = param

    def inverse(self) -> LieGroupElement:
        return self.group.inverse(left=self)

    def __eq__(self, other):
        return self.param == other.param

    def __mul__(self, right: LieGroupElement) -> LieGroupElement:
        return self.group.product(left=self, right=right)

    def Ad(self) -> ca.SX:
        return self.group.adjoint(left=self)

    def to_matrix(self):
        return self.group.to_matrix(left=self)

    def log(self) -> LieAlgebraElement:
        return self.group.log(left=self)

    def __str__(self):
        return "{:s}({:s}, {:s})".format(
            self.__class__.__name__, self.group.__class__.__name__, repr(self.param)
        )
    

@beartype
class LieGroup:
    """
    This is a generic Lie group, not necessarily represented as a matrix
    """

    def __init__(self, algebra: LieAlgebra, n_param: int, matrix_shape: tuple[int, int]):
        self.algebra = algebra
        self.n_param = n_param
        self.matrix_shape = matrix_shape

    def element(self, param: ca.SX) -> LieGroupElement:
        return LieGroupElement(group=self, param=param)

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
