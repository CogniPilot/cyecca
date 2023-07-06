import unittest

from pathlib import Path
import cProfile
from pstats import Stats

import sympy

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


class Test_LieGroupRn(ProfiledTestCase):
    def test_ctor(self):
        v = sympy.Matrix([1, 2, 3])
        G1 = lie.R3.element(v)
        self.assertTrue((G1.param - v).norm() < EPS)
        self.assertEqual(G1.group.n_param, 3)

    def test_bad_operations(self):
        G1 = lie.R3.element(sympy.Matrix([1, 2, 3]))
        G2 = lie.R3.element(sympy.Matrix([4, 5, 6]))
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
        v1 = sympy.Matrix([1, 2, 3])
        v2 = sympy.Matrix([4, 5, 6])
        v3 = v1 + v2
        G1 = lie.R3.element(sympy.Matrix([1, 2, 3]))
        G2 = lie.R3.element(sympy.Matrix([4, 5, 6]))
        G3 = G1 * G2
        self.assertTrue((G3.param - v3).norm() < EPS)

    def test_identity(self):
        G1 = lie.R3.element(sympy.Matrix([1, 2, 3]))
        G2 = G1 * lie.R3.identity()
        self.assertTrue((G1.param - G2.param).norm() < EPS)


class Test_LieAlgebraR(ProfiledTestCase):
    def test_ctor(self):
        v = sympy.Matrix([1, 2, 3])
        g1 = lie.r3.element(v)
        self.assertTrue((g1.param - v).norm() < EPS)
        self.assertEqual(g1.algebra.n_param, 3)

    def test_bad_operations(self):
        pass

    def test_add(self):
        v1 = sympy.Matrix([1, 2, 3])
        v2 = sympy.Matrix([4, 5, 6])
        v3 = v1 + v2
        g1 = lie.r3.element(sympy.Matrix([1, 2, 3]))
        g2 = lie.r3.element(sympy.Matrix([4, 5, 6]))
        g3 = g1 + g2
        self.assertTrue((g3.param - v3).norm() < EPS)


class Test_LieAlgebraSO2(ProfiledTestCase):
    def test_ctor(self):
        v = sympy.Matrix([1])
        lie.SO2.element(v)

    def test_identity(self):
        e = lie.SO2.identity()
        G = lie.SO2.element(sympy.Matrix([2]))
        self.assertTrue((G.param - (e * G).param).norm() < EPS)


class Test_LieGroupSO2(ProfiledTestCase):
    def test_ctor(self):
        v = sympy.Matrix([1])
        lie.SO2.element(v)


# class Test_LieGroupSO3(ProfiledTestCase):
#     def test_ctor(self):
#         v = sympy.Matrix([1, 2, 3])
#         G1 = LieGroupSO3Quat([1, 0, 0, 0])

#     def test_identity(self):
#         e = LieGroupSO3Quat.identity()
#         G2 = LieGroupSO3Quat([0, 1, 0, 0])
#         self.assertEqual(e * G2, G2)
#         self.assertEqual(G2 * e, G2)
#         self.assertEqual(G2, G2)

#     def test_addition(self):
#         g = LieAlgebraSO3([1, 2, 3])
#         self.assertEqual(g + g, LieAlgebraSO3(2 * g.param))
#         self.assertEqual(g - g, LieAlgebraSO3([0, 0, 0]))
#         self.assertEqual(-g, LieAlgebraSO3([-1, -2, -3]))

#     def test_exp_log(self):
#         g = lie.SO3([1, 2, 3])
#         self.assertEqual(g, LieGroupSO3Quat.exp(g).log())


if __name__ == "__main__":
    unittest.main()
