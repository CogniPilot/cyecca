from ..common import *

from cyecca.lie.group_se2 import *


class Test_LieAlgebraSE2(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0])

    def test_ctor(self):
        se2.elem(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = se2.elem(self.v1)
        g2 = se2.elem(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_Matrix(self):
        g1 = se2.elem(self.v1)
        X = g1.to_Matrix()

    def test_ad(self):
        g1 = se2.elem(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = se2.elem(self.v1)
        g1.vee()

    def test_wedge(self):
        se2.wedge(self.v1)

    def test_mul(self):
        g1 = se2.elem(self.v1)
        g2 = se2.elem(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = se2.elem(self.v1)
        3 * g1

    def test_exp(self):
        g1 = se2.elem(self.v1)
        g1.exp(SE2)

    def test_str(self):
        g1 = se2.elem(self.v1)
        print(g1)

    def test_eq(self):
        g1 = se2.elem(self.v1)
        g2 = se2.elem(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(se2)

    def test_repr(self):
        repr(se2)


class Test_LieGroupSE2(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([3.0, 2.0, 1.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0])

    def test_ctor(self):
        SE2.elem(self.v1)

    def test_bad_operations(self):
        G1 = SE2.elem(self.v1)
        G2 = SE2.elem(self.v2)
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
        G1 = SE2.elem(self.v1)
        G2 = SE2.elem(ca.DM([0, 0, 0]))
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, G1.param))

    def test_identity(self):
        G1 = SE2.elem(self.v1)
        G2 = G1 * SE2.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = SE2.elem(self.v1)
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = SE2.elem(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SE2.identity().param))

    def test_log(self):
        G1 = SE2.elem(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SE2.elem(self.v1)
        G2 = G1.log().exp(SE2)
        print(G1, G2)
        self.assertTrue(G1 == G2)

    def test_print_group(self):
        print(SE2)

    def test_print_group_elem(self):
        G1 = SE2.elem(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SE2)

    def test_repr_group_elem(self):
        G1 = SE2.elem(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SE2.elem(self.v1)
        G2 = SE2.elem(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SE2.elem(self.v1)
        G1_Ad = G1.Ad()
