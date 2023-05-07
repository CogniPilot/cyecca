from __future__ import annotations

from abc import ABC, abstractmethod
import sympy
from beartype import beartype


@beartype
class LieAlgebraElement:
    """
    This is a generic Lie algebra element, not necessarily represented as a matrix
    """

    def __init__(self, algebra: 'LieAlgebra', param: sympy.Matrix):
        self.algebra = algebra
        assert param.shape == (self.algebra.n_param, 1)
        self.param = param
    
    def ad(self) -> sympy.Matrix:
        """returns the adjoint as a linear operator on the parameter vector"""
        return self.algebra.adjoint(self)

    def vee(self) -> sympy.Matrix:
        """maps from Lie algebra to its parameters as a vector"""
        return self.algebra.vee(self)

    def __mul__(self, right: 'LieAlgebraElement') -> 'LieAlgebraElement':
        return self.algebra.bracket(self, right)

    def __rmul__(self, left: float) -> 'LieAlgebraElement':
        return self.algebra.scalar_multipication(left, self)

    def __add__(self, right: 'LieAlgebraElement') -> 'LieAlgebraElement':
        return self.algebra.addition(self, right)

    def to_matrix(self) -> sympy.Matrix:
        return self.algebra.to_matrix(self)
    
    def exp(self, group:'LieGroup') -> 'LieGroupElement':
        return group.exp(self)

    def __repr__(self):
        return '{:s}({:s}, {:s})'.format(
            self.__class__.__name__,
            self.algebra.__class__.__name__,
            repr(self.param))


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
    def bracket(self, left: LieAlgebraElement, right: LieAlgebraElement) -> LieAlgebraElement:
        ...

    @abstractmethod
    def scalar_multipication(self, left: float, right : LieAlgebraElement) -> LieAlgebraElement:
        ...

    @abstractmethod
    def addition(self, left: LieAlgebraElement, right : LieAlgebraElement) -> LieAlgebraElement:
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

    def __init__(self, group : 'LieGroup', param : sympy.Matrix):
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
        return '{:s}({:s}, {:s})'.format(
            self.__class__.__name__,
            self.group.__class__.__name__,
            repr(self.param))


@beartype
class LieGroup:
    """
    This is a generic Lie group, not necessarily represented as a matrix
    """

    def __init__(self, algebra: LieAlgebra, n_param: int, matrix_shape: tuple[int, int]):
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


@beartype
class RnLieAlgebra(LieAlgebra):
    
    def __init__(self, n: int):
        super().__init__(n_param=n, matrix_shape=(n+1, n+1))
    
    def bracket(self, left: LieAlgebraElement, right: LieAlgebraElement):
        assert self == left.algebra
        assert self == right.algebra
        return self.element(sympy.Matrix([0]))
    
    def addition(self, left : LieAlgebraElement, right : LieAlgebraElement) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(a.param + b.param)
    
    def scalar_multipication(self, left: float, right: LieAlgebraElement) -> LieAlgebraElement:
        assert self == right.algebra
        return self.element(left*right.param)

    def adjoint(self, left: LieAlgebraElement) -> sympy.Matrix:
        assert self == left.algebra
        return sympy.Matrix.zeros(self.n_param, self.n_param)

    def to_matrix(self, left: LieAlgebraElement) -> sympy.Matrix:
        assert self == left.algebra
        A = sympy.Matrix(self.matrix_shape)
        for i in range(self.n_param):
            A[i, self.n_param] = left.param[i]
        return A

    def __repr__(self):
        return '{:s}({:d})'.format(self.__class__.__name__, self.n_param)


@beartype
class RnLieGroup(LieGroup):
    
    def __init__(self, algebra: RnLieAlgebra):
        n = algebra.n_param
        super().__init__(algebra=algebra, n_param=n, matrix_shape=(n+1, n+1))
    
    def product(self, left: LieGroupElement, right: LieGroupElement) -> LieGroupElement:
        assert self == left.group
        assert self == right.group
        return self.element(left.param + right.param)

    def inverse(self, left: LieAlgebraElement) -> LieAlgebraElement:
        assert self == left.group
        return self.element(-left.param)

    def adjoint(self, left: LieGroupElement) -> sympy.Matrix:
        assert self == left.group
        return sympy.Matrix.eye(self.n_param)

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        """It is the identity map"""
        assert self.algebra == left.algebra
        return self.element(left.param)

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        """It is the identity map"""
        assert self == left.group
        return left.group.algebra.element(left.param)

    def to_matrix(self, left: LieGroupElement) -> sympy.Matrix:
        assert self == left.group
        A = sympy.Matrix.eye(self.n_param + 1)
        for i in range(self.n_param):
            A[i, self.n_param] = left.param[i]
        return A

    def __repr__(self):
        return '{:s}({:d})'.format(self.__class__.__name__, self.n_param)


@beartype
class SO2LieAlgebra(LieAlgebra):
    
    def __init__(self):
        super().__init__(n_param=1, matrix_shape=(2, 2))
    
    def bracket(self, left : LieAlgebraElement, right : LieAlgebraElement) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(sympy.Matrix([0]))
    
    def addition(self, left: LieAlgebraElement, right: LieAlgebraElement) -> LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(left.param + right.param)
    
    def scalar_multipication(self, left: float, right: LieAlgebraElement) -> LieAlgebraElement:
        assert self == right.algebra
        return self.element(left*right.param)

    def adjoint(self, left: LieAlgebraElement) -> sympy.Matrix:
        assert self == left.algebra
        return sympy.Matrix.zeros(1, 1)

    def to_matrix(self, left: LieAlgebraElement) -> sympy.Matrix:
        assert self == left.algebra
        return sympy.Matrix([
            [0, -left.param[0]],
            [left.param[0], 0]])


@beartype
class SO2LieGroup(LieGroup):
    
    def __init__(self, algebra: SO2LieAlgebra):
        super().__init__(algebra=algebra, n_param=1, matrix_shape=(2, 2))
    
    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        return self.element(left.param + right.param)

    def inverse(self, left):
        assert self == left.group
        return self.element(-left.param)

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        return sympy.Matrix.eye(1)
    
    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        return self.element(left.param)

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        return algebra.element(left.param)

    def to_matrix(self, left: LieGroupElement) -> sympy.Matrix:
        assert self == left.group
        theta = left.param[0]
        c = sympy.cos(theta)
        s = sympy.sin(theta)
        return sympy.Matrix([
            [c, -s],
            [s,  c]])


@beartype
class SE2LieAlgebra(LieAlgebra):
    
    def __init__(self):
        super().__init__(n_param=1, matrix_shape=(2, 2))
    
    def bracket(self, left : SE2LieAlgebraElement, right : SE2LieAlgebraElement):
        assert self == left.algebra
        assert self == right.algebra
        return self.element(sympy.Matrix([0]))
    
    def addition(self, left : SE2LieAlgebraElement, right : SE2LieAlgebraElement) -> SE2LieAlgebraElement:
        assert self == left.algebra
        assert self == right.algebra
        return self.element(left.param + right.param)
    
    def scalar_multipication(self, left : float, right : SE2LieAlgebraElement) -> SE2LieAlgebraElement:
        assert self == right.algebra
        return self.element(left*right.param)

    def adjoint(self, left: SE2LieAlgebraElement):
        assert self == left.algebra
        raise NotImplementedError()

    def to_matrix(self) -> sympy.Matrix:
        Omega = SO2LieAlgebra.to_matrix(left.param[2])
        v = left.param[:2]
        Z13 = sympy.ZeroMatrix(1, 3)
        return sympy.Matrix(sympy.BlockMatrix([
            [Omega, v],
            [Z13],
        ]))


@beartype
class SE2LieGroup(LieGroup):
    
    def __init__(self, algebra: SE2LieAlgebra):
        super().__init__(algebra=algebra, n_param=3, matrix_shape=(3, 3))
    
    def product(self, left: LieGroupElement, right: LieGroupElement):
        assert self == left.group
        assert self == right.group
        return self.element(left.param + right.param)

    def inverse(self, left):
        assert self == left.group
        return self.element(-left.param)

    def adjoint(self, left: LieGroupElement):
        assert self == left.group
        raise NotImplementedError()
    
    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        assert self.algebra == left.algebra
        raise NotImplementedError()

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        assert self == left.group
        raise NotImplementedError()

    def to_matrix(self) -> sympy.Matrix:
        R = SO2LieGroup.to_matrix(left.param[2])
        t = left.param[:2]
        Z12 = sympy.ZeroMatrix(1, 2)
        I1 = sympy.Identity(1)
        return sympy.Matrix(sympy.BlockMatrix([
            [R, t],
            [Z12, I1],
        ]))
