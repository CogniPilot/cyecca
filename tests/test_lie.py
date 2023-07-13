import unittest

from pathlib import Path
import cProfile
from pstats import Stats
import numpy as np
import numpy.testing as nptest

from cyecca import lie


EPS = 1e-9


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
    def test_print(self):
        print(lie.R3)

    def test_ctor(self):
        v = np.array([1.0, 2.0, 3.0])
        G1 = lie.R3.element(v)
        nptest.assert_array_almost_equal(G1.param, v, EPS)
        self.assertEqual(G1.group.n_param, 3)

    def test_bad_operations(self):
        G1 = lie.R3.element(np.array([1.0, 2.0, 3.0]))
        G2 = lie.R3.element(np.array([4.0, 5.0, 6.0]))
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
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        v3 = v1 + v2
        G1 = lie.R3.element(np.array([1.0, 2.0, 3.0]))
        G2 = lie.R3.element(np.array([4.0, 5.0, 6.0]))
        G3 = G1 * G2
        nptest.assert_array_almost_equal(G3.param, v3, EPS)

    def test_identity(self):
        G1 = lie.R3.element(np.array([1.0, 2.0, 3.0]))
        G2 = G1 * lie.R3.identity()
        nptest.assert_array_almost_equal(G1.param, G2.param, EPS)

    def test_to_matrix(self):
        G1 = lie.R3.element(np.array([1.0, 2.0, 3.0]))
        X = G1.to_matrix()

    def test_inverse(self):
        G1 = lie.R3.element(np.array([1.0, 2.0, 3.0]))
        nptest.assert_array_almost_equal((G1*G1.inverse()).param, lie.R3.identity().param)

    def test_print_group(self):
        print(lie.R3)

    def test_print_group_element(self):
        G1 = lie.R3.element(np.array([1.0, 2.0, 3.0]))
        print(G1)

    def test_repr_group(self):
        repr(lie.R3)

    def test_repr_group_element(self):
        G1 = lie.R3.element(np.array([1.0, 2.0, 3.0]))
        repr(G1)

    def test_eq(self):
        G1 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        G2 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.all(G1 == G2))


class Test_LieAlgebraR(ProfiledTestCase):
    def test_ctor(self):
        v = np.array([1.0, 2.0, 3.0])
        g1 = lie.r3.element(v)
        self.assertTrue(np.linalg.norm(g1.param - v) < EPS)
        self.assertEqual(g1.algebra.n_param, 3)

    def test_bad_operations(self):
        pass

    def test_add(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        v3 = v1 + v2
        g1 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        g2 = lie.r3.element(np.array([4.0, 5.0, 6.0]))
        g3 = g1 + g2
        self.assertTrue(np.linalg.norm(g3.param - v3) < EPS)

    def test_to_matrix(self):
        g1 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        X = g1.to_matrix()

    def test_ad(self):
        g1 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        g1.ad()

    def test_vee(self):
        g1 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        g1.vee()

    def test_wedge(self):
        lie.r3.wedge(np.array([1.0, 2.0, 3.0]))

    def test_mul(self):
        g1 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        g2 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        g3 = g1 * g2

    def test_rmul(self):
        g1 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        3 * g1

    def test_exp(self):
        g1 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        g1.exp(lie.R3)

    def test_str(self):
        g1 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        print(g1)

    def test_eq(self):
        g1 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        g2 = lie.r3.element(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.all(g1 == g2))

    def test_print(self):
        print(lie.r3)

    def test_repr(self):
        repr(lie.r3)


class Test_LieAlgebraSO2(ProfiledTestCase):
    def test_ctor(self):
        v = np.array([1.0])
        lie.so2.element(v)


class Test_nieGroupSO2(ProfiledTestCase):
    def test_ctor(self):
        v = np.array([1.0])
        lie.SO2.element(v)

    def test_identity(self):
        e = lie.SO2.identity()
        G = lie.SO2.element(np.array([2.0]))
        self.assertTrue(np.linalg.norm(G.param - (e * G).param) < EPS)


class Test_LieAlgebraSO3(ProfiledTestCase):
    def test_ctor(self):
        v = np.array([1.0, 2.0, 3.0])
        lie.so3.element(v)


class Test_LieGroupSO3Euler(ProfiledTestCase):
    def test_ctor(self):
        v = np.array([1.0, 2.0, 3.0])
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        SO3EulerB321.element(v)


if __name__ == "__main__":
    unittest.main()
