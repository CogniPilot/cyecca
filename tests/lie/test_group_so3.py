from ..common import *

from cyecca.lie.group_so3 import *


class Test_LieAlgebraSO3(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0])

    def test_ctor(self):
        so3.element(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = so3.element(self.v1)
        g2 = so3.element(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_Matrix(self):
        g1 = so3.element(self.v1)
        X = g1.to_Matrix()

    def test_ad(self):
        g1 = so3.element(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = so3.element(self.v1)
        g1.vee()

    def test_wedge(self):
        so3.wedge(self.v1)

    def test_mul(self):
        g1 = so3.element(self.v1)
        g2 = so3.element(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = so3.element(self.v1)
        3 * g1

    def test_exp_mrp(self):
        g1 = so3.element(self.v1)
        g1.exp(SO3Mrp)

    def test_exp_quat(self):
        g1 = so3.element(self.v1)
        g1.exp(SO3Quat)

    def test_exp_Euler(self):
        SO3EulerB321.element(self.v1)
        g1 = so3.element(self.v1)
        g1.exp(SO3EulerB321)

    def test_str(self):
        g1 = so3.element(self.v1)
        print(g1)

    def test_eq(self):
        g1 = so3.element(self.v1)
        g2 = so3.element(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(so3)

    def test_repr(self):
        repr(so3)


class Test_LieGroupSO3Euler(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([0.1, 0.2, 0.3])
        self.v2 = ca.DM([0.4, 0.5, 0.6])

    def test_ctor(self):
        SO3EulerB321.element(self.v1)

    def test_ineverse(self):
        G1 = SO3EulerB321.element(self.v1)
        self.assertTrue(G1 * G1.inverse() == SO3EulerB321.identity())

    def test_bad_operations(self):
        G1 = SO3EulerB321.element(self.v1)
        G1 = SO3EulerB321.element(self.v1)
        G2 = SO3EulerB321.element(self.v2)
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
        G1 = SO3EulerB321.element(self.v1)
        G2 = SO3EulerB321.element(self.v2)
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, v3))

    def test_identity(self):
        G1 = SO3EulerB321.element(self.v1)
        G2 = G1 * SO3EulerB321.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = SO3EulerB321.element(self.v1)
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = SO3EulerB321.element(self.v1)
        self.assertTrue(
            SX_close((G1 * G1.inverse()).param, SO3EulerB321.identity().param)
        )

    def test_log(self):
        G1 = SO3EulerB321.element(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SO3EulerB321.element(self.v1)
        G2 = G1.log().exp(SO3EulerB321)
        print(G1, G2)
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_print_group(self):
        print(SO3EulerB321)

    def test_print_group_element(self):
        G1 = SO3EulerB321.element(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SO3EulerB321)

    def test_repr_group_element(self):
        G1 = SO3EulerB321.element(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SO3EulerB321.element(self.v1)
        G2 = SO3EulerB321.element(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SO3EulerB321.element(self.v1)
        G1.Ad()


class Test_LieGroupSO3Quat(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 0.0, 0.0, 0.0])
        self.v2 = ca.DM([0.0, 1.0, 0.0, 0.0])

    def test_ctor(self):
        SO3Quat.element(self.v1)

    def test_bad_operations(self):
        G1 = SO3Quat.element(self.v1)
        G2 = SO3Quat.element(self.v2)
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
        q0 = SO3Quat.element(self.v1)
        q1 = SO3Quat.element(self.v2)
        q2 = q0 * q1
        assert q2 == q1

    def test_identity(self):
        G1 = SO3Quat.element(self.v1)
        G2 = G1 * SO3Quat.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = SO3Quat.element(self.v1)
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = SO3Quat.element(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SO3Quat.identity().param))

    def test_log(self):
        G1 = SO3Quat.element(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SO3Quat.element(self.v1)
        G2 = G1.log().exp(SO3Quat)
        print(G1, G2)
        self.assertTrue(G1 == G2)

    def test_print_group(self):
        print(SO3Quat)

    def test_print_group_element(self):
        G1 = SO3Quat.element(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SO3Quat)

    def test_repr_group_element(self):
        G1 = SO3Quat.element(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SO3Quat.element(self.v1)
        G2 = SO3Quat.element(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SO3Quat.element(self.v1)
        G1.Ad()


class Test_LieGroupSO3Mrp(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 0.0, 0.0])
        self.v2 = ca.DM([0.0, 1.0, 0.0])

    def test_ctor(self):
        SO3Mrp.element(self.v1)

    def test_bad_operations(self):
        G1 = SO3Mrp.element(self.v1)
        G2 = SO3Mrp.element(self.v2)
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
        q0 = SO3Mrp.element(self.v1)
        q1 = SO3Mrp.element(ca.DM([0, 0, 0]))
        q2 = q0 * q1
        self.assertTrue(SX_close(q0.param, q2.param))

    def test_identity(self):
        G1 = SO3Mrp.element(self.v1)
        G2 = G1 * SO3Mrp.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = SO3Mrp.element(self.v1)
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = SO3Mrp.element(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SO3Mrp.identity().param))

    def test_log(self):
        G1 = SO3Mrp.element(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SO3Mrp.element(self.v1)
        G2 = G1.log().exp(SO3Mrp)
        print(G1, G2)
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_print_group(self):
        print(SO3Mrp)

    def test_print_group_element(self):
        G1 = SO3Mrp.element(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SO3Mrp)

    def test_repr_group_element(self):
        G1 = SO3Mrp.element(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SO3Mrp.element(self.v1)
        G2 = SO3Mrp.element(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SO3Mrp.element(self.v1)
        G1.Ad()
