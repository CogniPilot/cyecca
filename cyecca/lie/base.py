"""
Abstract base classes for Lie Groups and Lie Algebras

Note, these are not restricted to Matrix Lie Groups.
"""

import abc

import casadi as ca


EPS = 1e-7


class LieAlgebra(abc.ABC):
    """
    Abstract Lie Algebra base class.
    """

    def __init__(self, param):
        self.param = ca.SX(param)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.add(other.neg())

    def __rmul__(self, other):
        return self.rmul(other)

    def __neg__(self):
        return self.neg()

    def __eq__(self, other) -> bool:
        return ca.norm_2(self.param - other.param) < EPS

    @abc.abstractmethod
    def neg(self):
        """
        Negative of Lie algebra
        """

    @abc.abstractmethod
    def add(self, other):
        """
        Add to elements of the Lie algebra
        """

    @abc.abstractmethod
    def rmul(self, other):
        """
        Add to elements of the Lie algebra
        """

    @abc.abstractmethod
    def wedge(self):
        """
        Map a vector to a Lie algebra matrix.
        """

    @abc.abstractmethod
    def vee(self):
        """
        Map a Lie algebra matrix to a avector
        """

    def __repr__(self):
        return repr(self.param)

    def __str__(self):
        return str(self.param)


class LieGroup(abc.ABC):
    """
    A Lie Group with group operator (*) is:

    (C)losed under operator (*)
    (A)ssociative with operator (*), (G1*G2)*G3 = G1*(G2*G3)
    (I)nverse: has an inverse such that G*G^-1 = e
    (N)uetral: has a neutral element: G*e = G
    """

    def __init__(self, param: ca.SX):
        self.param = ca.SX(param)

    def __mul__(self, other):
        """
        The * operator will be used as the Group multiplication operator
        (see product)
        """
        if not isinstance(other, type(self)):
            return TypeError("Lie Group types must match for product")
        assert isinstance(other, LieGroup)
        return self.product(other)

    def __eq__(self, other) -> bool:
        return ca.logic_all(self.param == other.param)
    
    @staticmethod
    @abc.abstractmethod
    def identity():
        """
        The identity element of the gorup, e
        """

    @abc.abstractmethod
    def product(self, other):
        """
        The group operator (*), returns an element of the group: G1*G2 = G3
        """

    @abc.abstractmethod
    def inv(self):
        """
        The inverse operator G1*G1.inv() = e
        """

    @abc.abstractmethod
    def log(self):
        """
        Returns the Lie logarithm of a group element, an element of the
        Lie algebra
        """

    @abc.abstractmethod
    def to_matrix_lie_group(self):
        """
        Returns the matrix lie group representation
        """

    @staticmethod
    @abc.abstractmethod
    def exp(g: LieAlgebra):
        """
        Compute the Lie group exponential of a Lie algebra element
        """

    def __repr__(self):
        return repr(self.param)

    def __str__(self):
        return str(self.param)

    def __eq__(self, other) -> bool:
        return ca.norm_2(self.param - other.param) < EPS



