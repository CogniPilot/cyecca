import unittest

from pathlib import Path
import cProfile
from pstats import Stats

import casadi as ca

from cyecca import lie


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
        print(lie.R3)

    def test_ctor(self):
        v = ca.DM([1.0, 2.0, 3.0])
        G1 = lie.R3.element(v)
        self.assertTrue(SX_close(G1.param, v))
        self.assertEqual(G1.group.n_param, 3)

    def test_bad_operations(self):
        G1 = lie.R3.element(self.v1)
        G2 = lie.R3.element(self.v2)
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
        G1 = lie.R3.element(self.v1)
        G2 = lie.R3.element(self.v2)
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, v3))

    def test_identity(self):
        G1 = lie.R3.element(self.v1)
        G2 = G1 * lie.R3.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        G1 = lie.R3.element(ca.DM([1.0, 2.0, 3.0]))
        X = G1.to_matrix()

    def test_inverse(self):
        G1 = lie.R3.element(ca.DM([1.0, 2.0, 3.0]))
        self.assertTrue(SX_close((G1*G1.inverse()).param, lie.R3.identity().param))

    def test_print_group(self):
        print(lie.R3)

    def test_print_group_element(self):
        G1 = lie.R3.element(ca.DM([1.0, 2.0, 3.0]))
        print(G1)

    def test_repr_group(self):
        repr(lie.R3)

    def test_repr_group_element(self):
        G1 = lie.R3.element(ca.DM([1.0, 2.0, 3.0]))
        repr(G1)

    def test_eq(self):
        G1 = lie.r3.element(self.v1)
        G2 = lie.r3.element(self.v1)
        self.assertTrue(G1 == G2)


class Test_LieAlgebraR(ProfiledTestCase):

    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0])

    def test_ctor(self):
        g1 = lie.r3.element(self.v1)
        self.assertTrue(SX_close(g1.param, self.v1))
        self.assertEqual(g1.algebra.n_param, 3)

    def test_bad_operations(self):
        pass

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = lie.r3.element(self.v1)
        g2 = lie.r3.element(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_matrix(self):
        g1 = lie.r3.element(self.v1)
        X = g1.to_matrix()

    def test_ad(self):
        g1 = lie.r3.element(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = lie.r3.element(self.v1)
        g1.vee()

    def test_wedge(self):
        lie.r3.wedge(self.v1)

    def test_mul(self):
        g1 = lie.r3.element(self.v1)
        g2 = lie.r3.element(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = lie.r3.element(self.v1)
        3 * g1

    def test_exp(self):
        g1 = lie.r3.element(self.v1)
        g1.exp(lie.R3)

    def test_str(self):
        g1 = lie.r3.element(self.v1)
        print(g1)

    def test_eq(self):
        g1 = lie.r3.element(self.v1)
        g2 = lie.r3.element(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(lie.r3)

    def test_repr(self):
        repr(lie.r3)


class Test_LieAlgebraSE2(ProfiledTestCase):

    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0])

    def test_ctor(self):
        lie.se2.element(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = lie.se2.element(self.v1)
        g2 = lie.se2.element(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_matrix(self):
        g1 = lie.se2.element(self.v1)
        X = g1.to_matrix()

    def test_ad(self):
        g1 = lie.se2.element(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = lie.se2.element(self.v1)
        g1.vee()

    def test_wedge(self):
        lie.se2.wedge(self.v1)

    def test_mul(self):
        g1 = lie.se2.element(self.v1)
        g2 = lie.se2.element(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = lie.se2.element(self.v1)
        3 * g1

    def test_exp(self):
        g1 = lie.se2.element(self.v1)
        g1.exp(lie.SE2)

    def test_str(self):
        g1 = lie.se2.element(self.v1)
        print(g1)

    def test_eq(self):
        g1 = lie.se2.element(self.v1)
        g2 = lie.se2.element(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(lie.se2)

    def test_repr(self):
        repr(lie.se2)



class Test_LieGroupSE2(ProfiledTestCase):
    def test_ctor(self):
        v = ca.DM([1.0, 2.0, 3.0])
        lie.SE2.element(v)


class Test_LieAlgebraSO2(ProfiledTestCase):
    def test_ctor(self):
        v = ca.DM([1.0])
        lie.so2.element(v)


class Test_LieGroupSO2(ProfiledTestCase):
    def test_ctor(self):
        v = ca.DM([1.0])
        lie.SO2.element(v)

    def test_identity(self):
        e = lie.SO2.identity()
        G = lie.SO2.element(ca.DM([2.0]))
        self.assertTrue(ca.norm_2(G.param - (e * G).param) <  EPS)


# class Test_LieAlgebraSO3(ProfiledTestCase):

    # def setUp(self):
        # super().setUp()
        # self.v1 = ca.DM([1.0, 2.0, 3.0])
        # self.v2 = ca.DM([4.0, 5.0, 6.0])

    # def test_ctor(self):
        # lie.so3.element(self.v1)


# class Test_LieGroupSO3Euler(ProfiledTestCase):

    # def setUp(self):
        # super().setUp()
        # self.v1 = ca.DM([1.0, 2.0, 3.0])
        # self.v2 = ca.DM([4.0, 5.0, 6.0])


    # def test_ctor(self):
        # SO3EulerB321 = lie.SO3Euler(
            # euler_type=lie.EulerType.body_fixed,
            # sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        # SO3EulerB321.element(self.v1)

    # def test_ineverse(self):
        # SO3EulerB321 = lie.SO3Euler(
            # euler_type=lie.EulerType.body_fixed,
            # sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        # G1 = SO3EulerB321.element(self.v1)
        # self.assertTrue(G1*G1.inverse() == SO3EulerB321.identity())


# class Test_LieGroupSO3Quat(ProfiledTestCase):

    # def setUp(self):
        # super().setUp()


    # def test_ctor(self):
        # lie.SO3Quat.element(ca.DM([1, 0, 0, 0]))

    # def test_product(self):
        # q0 = lie.SO3Quat.element(ca.DM([1, 0, 0, 0]))
        # q1 = lie.SO3Quat.element(ca.DM([0, 1, 0, 0]))
        # q2 = q0 * q1
        # assert q2 == q1

    # def test_inverse(self):
        # q1 = lie.SO3Quat.element(ca.DM([0, 1, 0, 0]))
        # assert q1 * q1.inverse() == lie.SO3Quat.identity()


# class Test_LieGroupSO3MRP(ProfiledTestCase):

    # def setUp(self):
        # super().setUp()


    # def test_ctor(self):
        # m0 = lie.SO3MRP.element(ca.DM([0, 0, 0, 0]))

    # def test_identity(self):
        # m0 = lie.SO3MRP.element(ca.DM([1, 0, 0, 0]))
        # e = lie.SO3MRP.identity()
        # assert m0 == m0*e


if __name__ == "__main__":
    unittest.main()
