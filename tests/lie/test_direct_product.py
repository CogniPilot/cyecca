from ..common import *

from cyecca.lie.direct_product import *
from cyecca.lie.group_se2 import *
from cyecca.lie.group_rn import *


class Test_LieGroupDirectProduct(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.G = SE2 * R3 * R3

    def test_product(self):
        G1 = self.G.elem(ca.SX([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        G2 = self.G.elem(ca.SX([0, 0, 0, 0, 0, 0, 0, 0, 0]))
        G3 = G1 * G2

    def test_inverse(self):
        G1 = self.G.elem(ca.SX([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        self.assertTrue(SX_close((G1 * G1.inverse()).param, self.G.identity().param))

    def test_log(self):
        G1 = self.G.elem(ca.SX([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        G1.log()

    def test_repr(self):
        repr(self.G)


class Test_LieAlgebraDirectProduct(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.G = SE2 * R3 * R3
        self.g = self.G.algebra

    def test_ctor(self):
        g1 = self.g.elem(ca.SX([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def test_repr(self):
        repr(self.g)

    def test_exp(self):
        g1 = self.g.elem(ca.SX([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        g1.exp(self.G)
