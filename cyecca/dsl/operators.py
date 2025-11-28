"""
Math operators for the Cyecca DSL.

Provides math functions that work with symbolic expressions.

================================================================================
PROTOTYPE MODE - API IS IN FLUX
================================================================================

This DSL is in active prototype development. The API may change significantly
between versions. Do NOT maintain backward compatibility - iterate rapidly.

================================================================================
DESIGN PRINCIPLES - DO NOT REMOVE OR IGNORE
================================================================================

1. MODELICA CONFORMANCE: This DSL conforms as closely as possible to the
   Modelica Language Specification v3.7-dev.
   Reference: https://specification.modelica.org/master/
   
   - Math functions follow Modelica built-in function semantics
   - See Modelica Spec Section 3.7 for built-in math functions

2. TYPE SAFETY: All functions MUST use beartype for runtime type checking.
   - All public functions decorated with @beartype
   - Do NOT remove beartype decorators
   - WHEN ADDING NEW FUNCTIONS: Always add @beartype decorator

3. SELF-CONTAINED: This module uses NO external compute libraries.
   - Returns Expr nodes for symbolic expressions
   - Returns Python floats for numeric inputs
   - Backends (CasADi, JAX, etc.) compile Expr trees to executable code
   
   DO NOT import CasADi, JAX, or other compute backends in this module.

================================================================================
"""

from __future__ import annotations

from typing import Any, Union

from beartype import beartype

from cyecca.dsl.model import DerivativeExpr, Expr, ExprKind, SymbolicVar, TimeVar, _to_expr


def _to_symbolic(x: Any) -> Union[Expr, float]:
    """Convert to Expr or return float for numeric values."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (SymbolicVar, DerivativeExpr, TimeVar, Expr)):
        return _to_expr(x)
    raise TypeError(f"Cannot convert {type(x)} to symbolic expression")


# Math functions that work with both Python floats and symbolic expressions


@beartype
def sin(x: Any) -> Union[Expr, float]:
    """Sine function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.sin(val)
    return Expr(ExprKind.SIN, (val,))


@beartype
def cos(x: Any) -> Union[Expr, float]:
    """Cosine function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.cos(val)
    return Expr(ExprKind.COS, (val,))


@beartype
def tan(x: Any) -> Union[Expr, float]:
    """Tangent function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.tan(val)
    return Expr(ExprKind.TAN, (val,))


@beartype
def asin(x: Any) -> Union[Expr, float]:
    """Arcsine function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.asin(val)
    return Expr(ExprKind.ASIN, (val,))


@beartype
def acos(x: Any) -> Union[Expr, float]:
    """Arccosine function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.acos(val)
    return Expr(ExprKind.ACOS, (val,))


@beartype
def atan(x: Any) -> Union[Expr, float]:
    """Arctangent function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.atan(val)
    return Expr(ExprKind.ATAN, (val,))


@beartype
def atan2(y: Any, x: Any) -> Union[Expr, float]:
    """Two-argument arctangent function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Parameters
    ----------
    y : numeric or Expr
        Numerator (opposite side)
    x : numeric or Expr
        Denominator (adjacent side)

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val_y = _to_symbolic(y)
    val_x = _to_symbolic(x)
    if isinstance(val_y, float) and isinstance(val_x, float):
        import math

        return math.atan2(val_y, val_x)
    # At least one is symbolic - convert both to Expr
    if isinstance(val_y, float):
        val_y = Expr(ExprKind.CONSTANT, value=val_y)
    if isinstance(val_x, float):
        val_x = Expr(ExprKind.CONSTANT, value=val_x)
    return Expr(ExprKind.ATAN2, (val_y, val_x))


@beartype
def sqrt(x: Any) -> Union[Expr, float]:
    """Square root function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.sqrt(val)
    return Expr(ExprKind.SQRT, (val,))


@beartype
def exp(x: Any) -> Union[Expr, float]:
    """Exponential function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.exp(val)
    return Expr(ExprKind.EXP, (val,))


@beartype
def log(x: Any) -> Union[Expr, float]:
    """Natural logarithm function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.log(val)
    return Expr(ExprKind.LOG, (val,))


@beartype
def abs(x: Any) -> Union[Expr, float]:
    """Absolute value function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        return __builtins__["abs"](val) if isinstance(__builtins__, dict) else val if val >= 0 else -val
    return Expr(ExprKind.ABS, (val,))


@beartype
def sign(x: Any) -> Union[Expr, float]:
    """Sign function.

    Returns -1 if x < 0, 0 if x == 0, +1 if x > 0.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        if val < 0:
            return -1.0
        elif val > 0:
            return 1.0
        else:
            return 0.0
    return Expr(ExprKind.SIGN, (val,))


@beartype
def floor(x: Any) -> Union[Expr, float]:
    """Floor function - largest integer not greater than x.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return float(math.floor(val))
    return Expr(ExprKind.FLOOR, (val,))


@beartype
def ceil(x: Any) -> Union[Expr, float]:
    """Ceiling function - smallest integer not less than x.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return float(math.ceil(val))
    return Expr(ExprKind.CEIL, (val,))


@beartype
def sinh(x: Any) -> Union[Expr, float]:
    """Hyperbolic sine function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.sinh(val)
    return Expr(ExprKind.SINH, (val,))


@beartype
def cosh(x: Any) -> Union[Expr, float]:
    """Hyperbolic cosine function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.cosh(val)
    return Expr(ExprKind.COSH, (val,))


@beartype
def tanh(x: Any) -> Union[Expr, float]:
    """Hyperbolic tangent function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.tanh(val)
    return Expr(ExprKind.TANH, (val,))


@beartype
def log10(x: Any) -> Union[Expr, float]:
    """Base-10 logarithm function.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val = _to_symbolic(x)
    if isinstance(val, float):
        import math

        return math.log10(val)
    return Expr(ExprKind.LOG10, (val,))


@beartype
def min(x: Any, y: Any) -> Union[Expr, float]:
    """Minimum of two values.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Parameters
    ----------
    x : numeric or Expr
        First value
    y : numeric or Expr
        Second value

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val_x = _to_symbolic(x)
    val_y = _to_symbolic(y)
    if isinstance(val_x, float) and isinstance(val_y, float):
        return val_x if val_x <= val_y else val_y
    # At least one is symbolic - convert both to Expr
    if isinstance(val_x, float):
        val_x = Expr(ExprKind.CONSTANT, value=val_x)
    if isinstance(val_y, float):
        val_y = Expr(ExprKind.CONSTANT, value=val_y)
    return Expr(ExprKind.MIN, (val_x, val_y))


@beartype
def max(x: Any, y: Any) -> Union[Expr, float]:
    """Maximum of two values.

    Modelica Spec: Section 3.7.3 - Built-in Mathematical Functions.

    Parameters
    ----------
    x : numeric or Expr
        First value
    y : numeric or Expr
        Second value

    Returns
    -------
    Expr or float
        Expr node for symbolic inputs, Python float for numeric inputs.
    """
    val_x = _to_symbolic(x)
    val_y = _to_symbolic(y)
    if isinstance(val_x, float) and isinstance(val_y, float):
        return val_x if val_x >= val_y else val_y
    # At least one is symbolic - convert both to Expr
    if isinstance(val_x, float):
        val_x = Expr(ExprKind.CONSTANT, value=val_x)
    if isinstance(val_y, float):
        val_y = Expr(ExprKind.CONSTANT, value=val_y)
    return Expr(ExprKind.MAX, (val_x, val_y))
