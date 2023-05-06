from abc import ABC, abstractmethod
import sympy
from typing import Tuple


class LieAlgebraElement:
    """
    This is a generic Lie algebra element, not necessarily represented as a matrix
    """

    def __init__(self, algebra: 'LieAlgebra', param: sympy.Matrix):
        self.algebra = algebra
        assert isinstance(algebra, LieAlgebra)
        assert param.shape == (self.algebra.n_param, 1)
        self.param = param
    
    def ad(self):
        """returns the adjoint as a linear operator on the parameter vector"""
        return self.algebra.adjoint(self)

    def vee(self):
        """maps from Lie algebra to its parameters as a vector"""
        return self.algebra.vee(self)

    def __mul__(self, right: 'LieAlgebraElement') -> 'LieAlgebraElement':
        assert isinstance(right, LieAlgebraElement)
        return self.algebra.bracket(self, right)

    def __rmul__(self, left: float) -> 'LieAlgebraElement':
        return self.algebra.scalar_multipication(left, self)

    def __add__(self, right: 'LieAlgebraElement') -> 'LieAlgebraElement':
        return self.algebra.addition(self, right)

    def to_matrix(self):
        return self.algebra.to_matrix(self)
    
    def exp(self, group:'LieGroup') -> 'LieGroupElement':
        return group.exp(self)

    def __repr__(self):
        return '{:s}({:s}, {:s})'.format(
            self.__class__.__name__,
            self.algebra.__class__.__name__,
            repr(self.param))


class LieAlgebra(ABC):
    """
    This is a generic Lie algebra, not necessarily represented as a matrix
    """

    def __init__(self, n_param: int, matrix_shape: Tuple):
        self.n_param = n_param
        self.matrix_shape = matrix_shape
    
    def element(self, param: sympy.Matrix) -> LieAlgebraElement:
        return LieAlgebraElement(self, param)

    def wedge(self, left: sympy.Matrix) -> LieAlgebraElement:
        """given a parameter vector, creates a LieAlgebraElement"""
        return LieAlgebraElement(self, left)

    def vee(self, left: LieAlgebraElement) -> LieAlgebraElement:
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


class LieGroupElement:
    """
    This is a generic Lie group element, not necessarily represented as a matrix
    """

    def __init__(self, group : 'LieGroup', param : sympy.Matrix):
        self.group = group
        assert isinstance(group, LieGroup)
        assert param.shape == (self.group.n_param, 1)
        self.param = param
        
    def inverse(self) -> 'LieGroupElement':
        return self.group.inverse(self)

    def __mul__(self, right: 'LieGroupElement') -> 'LieGroupElement':
        return self.group.product(self, right)

    def Ad(self) -> sympy.Matrix:
        return self.group.adjoint(self)

    def to_matrix(self):
        return self.group.to_matrix(self)
    
    def log(self, algebra: LieAlgebra) -> LieAlgebraElement:
        return self.group.log(algebra, self)

    def __repr__(self):
        return '{:s}({:s}, {:s})'.format(
            self.__class__.__name__,
            self.group.__class__.__name__,
            repr(self.param))


class LieGroup:
    """
    This is a generic Lie group, not necessarily represented as a matrix
    """

    def __init__(self, n_param : int, matrix_shape : Tuple):
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


class RnLieAlgebra(LieAlgebra):
    
    def __init__(self, n: int):
        super().__init__(n_param=n, matrix_shape=(n+1, n+1))
    
    def bracket(self, left: LieAlgebraElement, right: LieAlgebraElement):
        assert isinstance(left.algebra, RnLieAlgebra)
        assert isinstance(right.algebra, RnLieAlgebra)
        return self.element(sympy.Matrix([0]))
    
    def addition(self, left : LieAlgebraElement, right : LieAlgebraElement) -> LieAlgebraElement:
        assert isinstance(left.algebra, RnLieAlgebra)
        assert isinstance(right.algebra, RnLieAlgebra)
        return self.element(a.param + b.param)
    
    def scalar_multipication(self, left: float, right: LieAlgebraElement) -> LieAlgebraElement:
        return self.element(left*right.param)

    def adjoint(self, left: LieAlgebraElement) -> sympy.Matrix:
        assert isinstance(left.algebra, RnLieAlgebra)
        return sympy.Matrix.zeros(self.n_param, self.n_param)

    def to_matrix(self, left: LieAlgebraElement) -> sympy.Matrix:
        A = sympy.Matrix(self.matrix_shape)
        for i in range(self.n_param):
            A[i, self.n_param] = left.param[i]
        return A

    def __repr__(self):
        return '{:s}({:d})'.format(self.__class__.__name__, self.n_param)


class RnLieGroup(LieGroup):
    
    def __init__(self, n: int):
        super().__init__(n_param=n, matrix_shape=(n+1, n+1))
    
    def product(self, left: LieAlgebraElement, right: LieAlgebraElement) -> LieAlgebraElement:
        assert isinstance(left.group, RnLieGroup)
        assert isinstance(right.group, RnLieGroup)
        return self.element(left.param + right.param)

    def inverse(self, left: LieAlgebraElement) -> LieAlgebraElement:
        return self.element(-left.param)

    def adjoint(self, left: LieGroupElement) -> sympy.Matrix:
        return sympy.Matrix.eye(self.n_param)

    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        """It is the identity map"""
        return self.element(left.param)

    def log(self, algebra: LieAlgebra, left: LieGroupElement) -> LieAlgebraElement:
        """It is the identity map"""
        return algebra.element(left.param)

    def to_matrix(self, left: LieGroupElement) -> sympy.Matrix:
        A = sympy.Matrix.eye(self.n_param + 1)
        for i in range(self.n_param):
            A[i, self.n_param] = left.param[i]
        return A

    def __repr__(self):
        return '{:s}({:d})'.format(self.__class__.__name__, self.n_param)


class SO2LieAlgebra(LieAlgebra):
    
    def __init__(self):
        super().__init__(n_param=1, matrix_shape=(2, 2))
    
    def bracket(self, left : LieAlgebraElement, right : LieAlgebraElement) -> LieAlgebraElement:
        assert left.algebra == right.algebra
        return self.element(sympy.Matrix([0]))
    
    def addition(self, left: LieAlgebraElement, right: LieAlgebraElement) -> LieAlgebraElement:
        assert left.algebra == right.algebra
        return self.element(left.param + right.param)
    
    def scalar_multipication(self, left: float, right: LieAlgebraElement) -> LieAlgebraElement:
        return self.element(left*right.param)

    def adjoint(self, left: LieAlgebraElement) -> sympy.Matrix:
        return sympy.Matrix.zeros(1, 1)

    def to_matrix(self, left: LieAlgebraElement) -> sympy.Matrix:
        return sympy.Matrix([
            [0, -left.param[0]],
            [left.param[0], 0]])


class SO2LieGroup(LieGroup):
    
    def __init__(self):
        super().__init__(n_param=1, matrix_shape=(2, 2))
    
    def product(self, left: LieAlgebraElement, right: LieAlgebraElement):
        assert isinstance(left.group, SO2LieGroup)
        assert isinstance(right.group, SO2LieGroup)
        return self.element(left.param + right.param)

    def inverse(self, left):
        return self.element(-left.param)

    def adjoint(self, left: LieGroupElement):
        return sympy.Matrix.eye(1)
    
    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        return self.element(left.param)

    def log(self, algebra: LieAlgebra, left: LieGroupElement) -> LieAlgebraElement:
        return algebra.element(left.param)

    def to_matrix(self, left: LieGroupElement) -> sympy.Matrix:
        theta = left.param[0]
        cos = sympy.cos
        sin = sympy.sin
        return sympy.Matrix([
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)]])


class SE2LieAlgebra(LieAlgebra):
    
    def __init__(self):
        super().__init__(n_param=1, matrix_shape=(2, 2))
    
    def bracket(self, left : LieAlgebraElement, right : LieAlgebraElement):
        assert left.algebra == right.algebra
        return self.element(sympy.Matrix([0]))
    
    def addition(self, left : LieAlgebraElement, right : LieAlgebraElement) -> LieAlgebraElement:
        assert left.algebra == right.algebra
        return self.element(left.param + right.param)
    
    def scalar_multipication(self, left : float, right : LieAlgebraElement) -> LieAlgebraElement:
        return self.element(left*right.param)

    def adjoint(self, lef: LieAlgebraElement):
        raise NotImplementedError()

    def to_matrix(self) -> sympy.Matrix:
        raise NotImplementedError()


class SE2LieGroup(LieGroup):
    
    def __init__(self):
        super().__init__(n_param=1, matrix_shape=(2, 2))
    
    def product(self, left, right):
        assert isinstance(left.group, SO2LieGroup)
        assert isinstance(right.group, SO2LieGroup)
        return self.element(left.param + right.param)

    def inverse(self, left):
        return self.element(-left.param)

    def adjoint(self, left: LieGroupElement):
        raise NotImplementedError()
    
    def exp(self, left: LieAlgebraElement) -> LieGroupElement:
        raise NotImplementedError()

    def log(self, left: LieGroupElement) -> LieAlgebraElement:
        raise NotImplementedError()

    def to_matrix(self) -> sympy.Matrix:
        raise NotImplementedError()



r2 = RnLieAlgebra(2)
R2 = RnLieGroup(2)
r3 = RnLieAlgebra(3)
R3 = RnLieGroup(3)
so2 = SO2LieAlgebra()
SO2 = SO2LieGroup()
