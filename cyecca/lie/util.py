import casadi as ca
import sympy

from ..symbolic import taylor_series_near_zero

# see https://ethaneade.com/lie.pdf

x = sympy.symbols("x")
series_dict = {}
series_dict["sin(x)/x"] = taylor_series_near_zero(x, sympy.sin(x) / x)
series_dict["(1 - cos(x))/x"] = taylor_series_near_zero(x, (1 - sympy.cos(x)) / x)
series_dict["(1 - cos(x))/x^2"] = taylor_series_near_zero(
    x, (1 - sympy.cos(x)) / x**2
)
series_dict["(1 - sin(x))/x^3"] = taylor_series_near_zero(
    x, (1 - sympy.sin(x)) / x**3
)
