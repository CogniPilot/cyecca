import unittest

from pathlib import Path
import cProfile
from pstats import Stats

import casadi as ca

from cyecca.lie import r3, R3
from cyecca.lie import so2, SO2
from cyecca.lie import so3, SO3Mrp, SO3Quat, SO3EulerB321
from cyecca.lie import se2, SE2
from cyecca.lie import se3, SE3Mrp, SE3Quat, SE3EulerB321
from cyecca.lie import se23, SE23Mrp, SE23Quat, SE23EulerB321


EPS = 1e-9


def SX_close(e1: (ca.SX, ca.DM), e2: (ca.SX, ca.DM)):
    return ca.norm_2(e1 - e2) < EPS


class ProfiledTestCase(unittest.TestCase):
    def setUp(self):
        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self) -> None:
        p = Stats(self.pr)
        p.strip_dirs()
        p.sort_stats("cumtime")
        profile_dir = Path(".profile")
        profile_dir.mkdir(exist_ok=True)
        p.dump_stats(profile_dir / self.id())


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
        with self.assertRaises(TypeError):
            G1 + G2
        with self.assertRaises(TypeError):
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

    def test_to_matrix(self):
        G1 = R3.element(ca.DM([1.0, 2.0, 3.0]))
        X = G1.to_matrix()

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


class Test_LieAlgebraR(ProfiledTestCase):
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

    def test_to_matrix(self):
        g1 = r3.element(self.v1)
        X = g1.to_matrix()

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


class Test_LieAlgebraSE2(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0])

    def test_ctor(self):
        se2.element(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = se2.element(self.v1)
        g2 = se2.element(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_matrix(self):
        g1 = se2.element(self.v1)
        X = g1.to_matrix()

    def test_ad(self):
        g1 = se2.element(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = se2.element(self.v1)
        g1.vee()

    def test_wedge(self):
        se2.wedge(self.v1)

    def test_mul(self):
        g1 = se2.element(self.v1)
        g2 = se2.element(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = se2.element(self.v1)
        3 * g1

    def test_exp(self):
        g1 = se2.element(self.v1)
        g1.exp(SE2)

    def test_str(self):
        g1 = se2.element(self.v1)
        print(g1)

    def test_eq(self):
        g1 = se2.element(self.v1)
        g2 = se2.element(self.v1)
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
        SE2.element(self.v1)

    def test_bad_operations(self):
        G1 = SE2.element(self.v1)
        G2 = SE2.element(self.v2)
        s = 1
        with self.assertRaises(TypeError):
            G1 + G2
        with self.assertRaises(TypeError):
            G1 - G2
        with self.assertRaises(TypeError):
            G1 @ G2
        with self.assertRaises(TypeError):
            s * G2

    def test_product(self):
        G1 = SE2.element(self.v1)
        G2 = SE2.element(ca.DM([0, 0, 0]))
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, G1.param))

    def test_identity(self):
        G1 = SE2.element(self.v1)
        G2 = G1 * SE2.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        G1 = SE2.element(self.v1)
        X = G1.to_matrix()

    def test_inverse(self):
        G1 = SE2.element(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SE2.identity().param))

    def test_log(self):
        G1 = SE2.element(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SE2.element(self.v1)
        G2 = G1.log().exp(SE2)
        print(G1, G2)
        self.assertTrue(G1 == G2)

    def test_print_group(self):
        print(SE2)

    def test_print_group_element(self):
        G1 = SE2.element(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SE2)

    def test_repr_group_element(self):
        G1 = SE2.element(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SE2.element(self.v1)
        G2 = SE2.element(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SE2.element(self.v1)
        G1_Ad = G1.Ad()


class Test_LieAlgebraSO2(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0])
        self.v2 = ca.DM([4.0])

    def test_ctor(self):
        so2.element(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = so2.element(self.v1)
        g2 = so2.element(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_matrix(self):
        g1 = so2.element(self.v1)
        X = g1.to_matrix()

    def test_ad(self):
        g1 = so2.element(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = so2.element(self.v1)
        g1.vee()

    def test_wedge(self):
        so2.wedge(self.v1)

    def test_mul(self):
        g1 = so2.element(self.v1)
        g2 = so2.element(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = so2.element(self.v1)
        3 * g1

    def test_exp(self):
        g1 = so2.element(self.v1)
        g1.exp(SO2)

    def test_str(self):
        g1 = so2.element(self.v1)
        print(g1)

    def test_eq(self):
        g1 = so2.element(self.v1)
        g2 = so2.element(self.v1)
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
        SO2.element(self.v1)

    def test_bad_operations(self):
        G1 = SO2.element(self.v1)
        G2 = SO2.element(self.v2)
        s = 1
        with self.assertRaises(TypeError):
            G1 + G2
        with self.assertRaises(TypeError):
            G1 - G2
        with self.assertRaises(TypeError):
            G1 @ G2
        with self.assertRaises(TypeError):
            s * G2

    def test_product(self):
        v3 = self.v1 + self.v2
        G1 = SO2.element(self.v1)
        G2 = SO2.element(self.v2)
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, v3))

    def test_identity(self):
        G1 = SO2.element(self.v1)
        G2 = G1 * SO2.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        G1 = SO2.element(self.v1)
        X = G1.to_matrix()

    def test_inverse(self):
        G1 = SO2.element(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SO2.identity().param))

    def test_log(self):
        G1 = SO2.element(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SO2.element(self.v1)
        G2 = G1.log().exp(SO2)
        print(G1, G2)
        self.assertTrue(G1 == G2)

    def test_print_group(self):
        print(SO2)

    def test_print_group_element(self):
        G1 = SO2.element(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SO2)

    def test_repr_group_element(self):
        G1 = SO2.element(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SO2.element(self.v1)
        G2 = SO2.element(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SO2.element(self.v1)
        G1.Ad()


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

    def test_to_matrix(self):
        g1 = so3.element(self.v1)
        X = g1.to_matrix()

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
        with self.assertRaises(TypeError):
            G1 + G2
        with self.assertRaises(TypeError):
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

    def test_to_matrix(self):
        G1 = SO3EulerB321.element(self.v1)
        X = G1.to_matrix()

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
        with self.assertRaises(TypeError):
            G1 + G2
        with self.assertRaises(TypeError):
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

    def test_to_matrix(self):
        G1 = SO3Quat.element(self.v1)
        X = G1.to_matrix()

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
        self.v1 = ca.DM([1.0, 0.0, 0.0, 0.0])
        self.v2 = ca.DM([0.0, 1.0, 0.0, 0.0])

    def test_ctor(self):
        SO3Mrp.element(self.v1)

    def test_bad_operations(self):
        G1 = SO3Mrp.element(self.v1)
        G2 = SO3Mrp.element(self.v2)
        s = 1
        with self.assertRaises(TypeError):
            G1 + G2
        with self.assertRaises(TypeError):
            G1 - G2
        with self.assertRaises(TypeError):
            G1 @ G2
        with self.assertRaises(TypeError):
            s * G2

    def test_product(self):
        q0 = SO3Mrp.element(self.v1)
        q1 = SO3Mrp.element(ca.DM([0, 0, 0, 0]))
        q2 = q0 * q1
        self.assertTrue(SX_close(q0.param, q2.param))

    def test_identity(self):
        G1 = SO3Mrp.element(self.v1)
        G2 = G1 * SO3Mrp.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        G1 = SO3Mrp.element(self.v1)
        X = G1.to_matrix()

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


class Test_LieAlgebraSE3(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    def test_ctor(self):
        se3.element(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = se3.element(self.v1)
        g2 = se3.element(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_matrix(self):
        g1 = se3.element(self.v1)
        X = g1.to_matrix()

    def test_ad(self):
        g1 = se3.element(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = se3.element(self.v1)
        g1.vee()

    def test_wedge(self):
        se3.wedge(self.v1)

    def test_mul(self):
        g1 = se3.element(self.v1)
        g2 = se3.element(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = se3.element(self.v1)
        3 * g1

    def test_exp_mrp(self):
        g1 = se3.element(self.v1)
        g1.exp(SE3Mrp)

    def test_exp_quat(self):
        g1 = se3.element(self.v1)
        g1.exp(SE3Quat)

    def test_str(self):
        g1 = se3.element(self.v1)
        print(g1)

    def test_eq(self):
        g1 = se3.element(self.v1)
        g2 = se3.element(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(se3)

    def test_repr(self):
        repr(se3)


class Test_LieGroupSE3Mrp(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([3.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0])

    def test_ctor(self):
        SE3Mrp.element(self.v1)

    def test_bad_operations(self):
        G1 = SE3Mrp.element(self.v1)
        G2 = SE3Mrp.element(self.v2)
        s = 1
        with self.assertRaises(TypeError):
            G1 + G2
        with self.assertRaises(TypeError):
            G1 - G2
        with self.assertRaises(TypeError):
            G1 @ G2
        with self.assertRaises(TypeError):
            s * G2

    def test_product(self):
        G1 = SE3Mrp.element(self.v1)
        G2 = SE3Mrp.element(ca.DM([0, 0, 0, 0, 0, 0, 0]))
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, G1.param))

    def test_identity(self):
        G1 = SE3Mrp.element(self.v1)
        G2 = G1 * SE3Mrp.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        G1 = SE3Mrp.element(self.v1)
        X = G1.to_matrix()

    def test_inverse(self):
        G1 = SE3Mrp.element(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SE3Mrp.identity().param))

    def test_log(self):
        G1 = SE3Mrp.element(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SE3Mrp.element(self.v1)
        G2 = G1.log().exp(SE3Mrp)
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_print_group(self):
        print(SE3Mrp)

    def test_print_group_element(self):
        G1 = SE3Mrp.element(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SE3Mrp)

    def test_repr_group_element(self):
        G1 = SE3Mrp.element(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SE3Mrp.element(self.v1)
        G2 = SE3Mrp.element(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SE3Mrp.element(self.v1)
        G1.Ad()


class Test_LieGroupSE3Quat(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([3.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0])

    def test_ctor(self):
        SE3Quat.element(self.v1)

    def test_bad_operations(self):
        G1 = SE3Quat.element(self.v1)
        G2 = SE3Quat.element(self.v2)
        s = 1
        with self.assertRaises(TypeError):
            G1 + G2
        with self.assertRaises(TypeError):
            G1 - G2
        with self.assertRaises(TypeError):
            G1 @ G2
        with self.assertRaises(TypeError):
            s * G2

    def test_product(self):
        G1 = SE3Quat.element(ca.DM([1, 2, 3, 0, 1, 0, 0]))
        G2 = SE3Quat.element(ca.DM([0, 0, 0, 1, 0, 0, 0]))
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, G1.param))

    def test_identity(self):
        G1 = SE3Quat.element(self.v1)
        G2 = G1 * SE3Quat.identity()
        print(SE3Quat.identity().to_matrix())
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        G1 = SE3Quat.element(self.v1)
        X = G1.to_matrix()

    def test_inverse(self):
        G1 = SE3Quat.element(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SE3Quat.identity().param))

    def test_log(self):
        G1 = SE3Quat.element(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SE3Quat.element(self.v1)
        G2 = G1.log().exp(SE3Quat)
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_print_group(self):
        print(SE3Quat)

    def test_print_group_element(self):
        G1 = SE3Quat.element(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SE3Quat)

    def test_repr_group_element(self):
        G1 = SE3Quat.element(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SE3Quat.element(self.v1)
        G2 = SE3Quat.element(self.v1)
        self.assertTrue(G1 == G2)


class Test_LieGroupSE23Mrp(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])

    def test_ctor(self):
        SE23Mrp.element(param=self.v1)


class Test_LieGroupDirectProduct(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.G = SE2 * R3 * R3

    def test_group_product(self):
        G1 = self.G.element(ca.SX([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        G2 = self.G.element(ca.SX([0, 0, 0, 0, 0, 0, 0, 0, 0]))
        G3 = G1 * G2

    def test_group_inverse(self):
        G1 = self.G.element(ca.SX([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        self.assertTrue(SX_close((G1 * G1.inverse()).param, self.G.identity().param))

    def test_repr(self):
        repr(self.G)


class Test_LieAlgebraDirectProduct(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.g = se2 * r3 * r3

    def test_repr(self):
        repr(self.g)
