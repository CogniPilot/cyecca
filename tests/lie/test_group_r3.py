from ..common import *

from cyecca.lie.group_rn import *


class Test_LieGroupR3(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0])

    def test_print(self):
        print(R3)

    def test_ctor(self):
        v = ca.DM([1.0, 2.0, 3.0])
        G1 = R3.element(v)
        self.assertTrue(SX_close(G1.param, v))
        self.assertEqual(G1.group.n_param, 3)

    def test_bad_operations(self):
        G1 = R3.element(self.v1)
        G2 = R3.element(self.v2)
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
        G1 = R3.element(self.v1)
        G2 = R3.element(self.v2)
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, v3))

    def test_identity(self):
        G1 = R3.element(self.v1)
        G2 = G1 * R3.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = R3.element(ca.DM([1.0, 2.0, 3.0]))
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = R3.element(ca.DM([1.0, 2.0, 3.0]))
        self.assertTrue(SX_close((G1 * G1.inverse()).param, R3.identity().param))

    def test_print_group(self):
        print(R3)

    def test_print_group_element(self):
        G1 = R3.element(ca.DM([1.0, 2.0, 3.0]))
        print(G1)

    def test_repr_group(self):
        repr(R3)

    def test_repr_group_element(self):
        G1 = R3.element(ca.DM([1.0, 2.0, 3.0]))
        repr(G1)

    def test_eq(self):
        G1 = R3.element(self.v1)
        G2 = R3.element(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = R3.element(self.v1)
        G1_Ad = G1.Ad()

    def test_log(self):
        G1 = R3.element(self.v1)
        g1 = G1.log()
        self.assertTrue(SX_close(g1.param, self.v1))


class Test_LieAlgebraR3(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0])

    def test_ctor(self):
        g1 = r3.element(self.v1)
        self.assertTrue(SX_close(g1.param, self.v1))
        self.assertEqual(g1.algebra.n_param, 3)

    def test_bad_operations(self):
        pass

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = r3.element(self.v1)
        g2 = r3.element(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_Matrix(self):
        g1 = r3.element(self.v1)
        X = g1.to_Matrix()

    def test_ad(self):
        g1 = r3.element(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = r3.element(self.v1)
        g1.vee()

    def test_wedge(self):
        r3.wedge(self.v1)

    def test_mul(self):
        g1 = r3.element(self.v1)
        g2 = r3.element(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = r3.element(self.v1)
        3 * g1

    def test_exp(self):
        g1 = r3.element(self.v1)
        g1.exp(R3)

    def test_str(self):
        g1 = r3.element(self.v1)
        print(g1)

    def test_eq(self):
        g1 = r3.element(self.v1)
        g2 = r3.element(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(r3)

    def test_repr(self):
        repr(r3)
