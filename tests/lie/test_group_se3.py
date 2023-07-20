from ..common import *

from cyecca.lie.group_se3 import *


class Test_LieAlgebraSE3(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    def test_ctor(self):
        se3.elem(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = se3.elem(self.v1)
        g2 = se3.elem(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_Matrix(self):
        g1 = se3.elem(self.v1)
        X = g1.to_Matrix()

    def test_ad(self):
        g1 = se3.elem(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = se3.elem(self.v1)
        g1.vee()

    def test_wedge(self):
        se3.wedge(self.v1)

    def test_mul(self):
        g1 = se3.elem(self.v1)
        g2 = se3.elem(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = se3.elem(self.v1)
        3 * g1

    def test_exp_mrp(self):
        g1 = se3.elem(self.v1)
        g1.exp(SE3Mrp)

    def test_exp_quat(self):
        g1 = se3.elem(self.v1)
        g1.exp(SE3Quat)

    def test_str(self):
        g1 = se3.elem(self.v1)
        print(g1)

    def test_eq(self):
        g1 = se3.elem(self.v1)
        g2 = se3.elem(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(se3)

    def test_repr(self):
        repr(se3)


class Test_LieGroupSE3Mrp(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([3.0, 2.0, 1.0, 1.0, 0.0, 0.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0, 0.0, 1.0, 0.0])

    def test_ctor(self):
        SE3Mrp.elem(self.v1)

    def test_bad_operations(self):
        G1 = SE3Mrp.elem(self.v1)
        G2 = SE3Mrp.elem(self.v2)
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
        G1 = SE3Mrp.elem(self.v1)
        G2 = SE3Mrp.elem(ca.DM([0, 0, 0, 0, 0, 0]))
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, G1.param))

    def test_identity(self):
        G1 = SE3Mrp.elem(self.v1)
        G2 = G1 * SE3Mrp.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = SE3Mrp.elem(self.v1)
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = SE3Mrp.elem(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SE3Mrp.identity().param))

    def test_log(self):
        G1 = SE3Mrp.elem(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SE3Mrp.elem(self.v1)
        G2 = G1.log().exp(SE3Mrp)
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_print_group(self):
        print(SE3Mrp)

    def test_print_group_elem(self):
        G1 = SE3Mrp.elem(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SE3Mrp)

    def test_repr_group_elem(self):
        G1 = SE3Mrp.elem(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SE3Mrp.elem(self.v1)
        G2 = SE3Mrp.elem(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SE3Mrp.elem(self.v1)
        G1.Ad()


class Test_LieGroupSE3Quat(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([3.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0])

    def test_ctor(self):
        SE3Quat.elem(self.v1)

    def test_bad_operations(self):
        G1 = SE3Quat.elem(self.v1)
        G2 = SE3Quat.elem(self.v2)
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
        G1 = SE3Quat.elem(ca.DM([1, 2, 3, 0, 1, 0, 0]))
        G2 = SE3Quat.elem(ca.DM([0, 0, 0, 1, 0, 0, 0]))
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, G1.param))

    def test_identity(self):
        G1 = SE3Quat.elem(self.v1)
        G2 = G1 * SE3Quat.identity()
        print(SE3Quat.identity().to_Matrix())
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = SE3Quat.elem(self.v1)
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = SE3Quat.elem(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SE3Quat.identity().param))

    def test_log(self):
        G1 = SE3Quat.elem(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SE3Quat.elem(self.v1)
        G2 = G1.log().exp(SE3Quat)
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_print_group(self):
        print(SE3Quat)

    def test_print_group_elem(self):
        G1 = SE3Quat.elem(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SE3Quat)

    def test_repr_group_elem(self):
        G1 = SE3Quat.elem(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SE3Quat.elem(self.v1)
        G2 = SE3Quat.elem(self.v1)
        self.assertTrue(G1 == G2)
