from cyecca.symbolic import sympy_to_casadi, casadi_to_sympy

import sympy
from .common import *


class Test_Symbolic(ProfiledTestCase):
    def setUp(self):
        super().setUp()

    def test_sympy_to_casadi(self):
        x = sympy.symbols("x")
        y = sympy.sin(x) + 2
        sympy_to_casadi(y)

    def test_casadi_to_sympy(self):
        x = ca.SX.sym("x")
        y = ca.tan(x) + ca.cos(x) * ca.sin(x) + 2
        casadi_to_sympy(y)
