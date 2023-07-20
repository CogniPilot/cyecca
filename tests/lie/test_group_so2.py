from ..common import *

from cyecca.lie.group_so2 import *


class Test_LieAlgebraSO2(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0])
        self.v2 = ca.DM([4.0])

    def test_ctor(self):
        so2.elem(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = so2.elem(self.v1)
        g2 = so2.elem(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_Matrix(self):
        g1 = so2.elem(self.v1)
        X = g1.to_Matrix()

    def test_ad(self):
        g1 = so2.elem(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = so2.elem(self.v1)
        g1.vee()

    def test_wedge(self):
        so2.wedge(self.v1)

    def test_mul(self):
        g1 = so2.elem(self.v1)
        g2 = so2.elem(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = so2.elem(self.v1)
        3 * g1

    def test_exp(self):
        g1 = so2.elem(self.v1)
        g1.exp(SO2)

    def test_str(self):
        g1 = so2.elem(self.v1)
        print(g1)

    def test_eq(self):
        g1 = so2.elem(self.v1)
        g2 = so2.elem(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(so2)

    def test_repr(self):
        repr(so2)


class Test_LieGroupSO2(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0])
        self.v2 = ca.DM([2.0])

    def test_ctor(self):
        SO2.elem(self.v1)

    def test_bad_operations(self):
        G1 = SO2.elem(self.v1)
        G2 = SO2.elem(self.v2)
        s = 1
        with self.assertRaises(BeartypeCallHintParamViolation):
            G1 + G2
        with self.assertRaises(BeartypeCallHintParamViolation):
            G1 - G2
        with self.assertRaises(TypeError):
            G1 @ G2
        with self.assertRaises(TypeError):
            s * G2

    def test_product(self):
        v3 = self.v1 + self.v2
        G1 = SO2.elem(self.v1)
        G2 = SO2.elem(self.v2)
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, v3))

    def test_identity(self):
        G1 = SO2.elem(self.v1)
        G2 = G1 * SO2.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = SO2.elem(self.v1)
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = SO2.elem(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SO2.identity().param))

    def test_log(self):
        G1 = SO2.elem(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SO2.elem(self.v1)
        G2 = G1.log().exp(SO2)
        print(G1, G2)
        self.assertTrue(G1 == G2)

    def test_print_group(self):
        print(SO2)

    def test_print_group_elem(self):
        G1 = SO2.elem(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SO2)

    def test_repr_group_elem(self):
        G1 = SO2.elem(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SO2.elem(self.v1)
        G2 = SO2.elem(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SO2.elem(self.v1)
        G1.Ad()
