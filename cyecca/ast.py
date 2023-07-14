from __future__ import annotations

import abc
from beartype import beartype
from beartype.typing import List, Tuple


@beartype
class Scalar(abc.ABC):

    def __init__(self):
        pass

    def __neg__(self):
        return Negative(self)

    def __add__(self, right: (Scalar, float, int)):
        if isinstance(right, float):
            return Add(self, Float(right))
        elif isinstance(right, int):
            return Add(self, Integer(right))
        elif isinstance(right, Scalar):
            return Add(self, right)
        else:
            raise ValueError(f'unhandled type {type(right):s}')

    def __radd__(self, left: (float, int)):
        if isinstance(left, float):
            return Add(Float(left), self)
        elif isinstance(left, int):
            return Add(Integer(left), self)
        else:
            raise ValueError(f'unhandled type {type(left):s}')

    def __mul__(self, right: (Scalar, float, int)):
        if isinstance(right, float):
            return Multiply(self, Float(right))
        elif isinstance(right, int):
            return Multiply(self, Integer(right))
        elif isinstance(right, Scalar):
            return Multiply(self, right)
        else:
            raise ValueError(f'unhandled type {type(right):s}')

    def __rmul__(self, left: (float, int)):
        if isinstance(left, float):
            return Multiply(Float(left), self)
        elif isinstance(left, int):
            return Multiply(Integer(left), self)

    def __sub__(self, right: (Scalar, float, int)):
        if isinstance(right, float):
            return Subtract(self, Float(right))
        elif isinstance(right, int):
            return Subtract(self, Integer(right))
        elif isinstance(right, Scalar):
            return Subtract(self, right)
        else:
            raise ValueError(f'unhandled type {type(right):s}')

    def __rsub__(self, left: (float, int)):
        if isinstance(left, float):
            return Subtract(Float(left), self)
        elif isinstance(left, int):
            return Subtract(Integer(left), self)
        else:
            raise ValueError(f'unhandled type {type(left):s}')

    @abc.abstractmethod
    def _repr_latex_(self, depth: int=0):
        pass


@beartype
class Symbol(Scalar):

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        pass

    def _repr_latex_(self, depth: int=0):
        latex = self.name
        if depth == 0:
            latex = f'${latex:s}$'
        return latex


@beartype
class Float(Scalar):

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def _repr_latex_(self, depth: int=0):
        latex = f'{self.value:g}'
        if depth == 0:
            latex = f'${latex:s}$'
        return latex


@beartype
class Integer(Scalar):

    def __init__(self, value: int):
        super().__init__()
        self.value = value

    def _repr_latex_(self, depth: int=0):
        latex = f'{self.value:d}'
        if depth == 0:
            latex = f'${latex:d}$'
        return latex


@beartype
class BinaryOperator(Scalar):

    def __init__(self, op: str, left: Scalar, right: Scalar):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def _repr_latex_(self, depth: int=0):
        latex = f'{self.left._repr_latex_(depth+1):s} {self.op:s} {self.right._repr_latex_(depth+1):s}'
        if depth == 0:
            latex = f'${latex:s}$'
        else:
            latex = f'({latex:s})'
        return latex


@beartype
class Add(BinaryOperator):

    def __init__(self, left: Scalar, right: Scalar):
        super().__init__(op='+', left=left, right=right)


@beartype
class Subtract(BinaryOperator):

    def __init__(self, left: Scalar, right: Scalar):
        super().__init__(op='-', left=left, right=right)


@beartype
class Multiply(BinaryOperator):

    def __init__(self, left: Scalar, right: Scalar):
        super().__init__(op='\cdot', left=left, right=right)

@beartype
class Divide(BinaryOperator):

    def __init__(self, left: Scalar, right: Scalar):
        super().__init__(op='/', left=left, right=right)


@beartype
class UnaryOperator(Scalar):

    def __init__(self, op: str, right: Scalar):
        self.op = op
        self.right = right

    def _repr_latex_(self, depth: int=0):
        latex = f'{self.op:s}({self.right._repr_latex_(depth+1):s})'
        if depth == 0:
            latex = f'${latex:s}$'
        else:
            latex = f'{latex:s}'
        return latex


@beartype
class Sin(UnaryOperator):

    def __init__(self, right: Scalar):
        super().__init__(op='\sin', right=right)

@beartype
class Cos(UnaryOperator):

    def __init__(self, right: Scalar):
        super().__init__(op='\cos', right=right)

@beartype
class Negative(UnaryOperator):

    def __init__(self, right: Scalar):
        super().__init__(op='-', right=right)


@beartype
def sin(right: Scalar) -> Sin:
    return Sin(right)

@beartype
def cos(right: Scalar) -> Cos:
    return Cos(right)
