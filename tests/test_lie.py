import unittest

import casadi as ca

from cyecca.lie.r import LieGroup_R


class Test_LieGroup_R(unittest.TestCase):
    def test_ctor(self):
        v = ca.DM([1, 2, 3])
        G1 = LieGroup_R(3, v)
        for i in range(3):
            self.assertEqual(G1.param[i], v[i])
        self.assertEqual(G1.n, 3)

    def test_bad_operations(self):
        G1 = LieGroup_R(3, ca.DM([1, 2, 3]))
        G2 = LieGroup_R(3, ca.DM([4, 5, 6]))
        s = ca.SX.sym("s")
        with self.assertRaises(TypeError):
            G1 + G2
        with self.assertRaises(TypeError):
            G1 - G2
        with self.assertRaises(TypeError):
            G1 @ G2
        with self.assertRaises(TypeError):
            s * G2

    def test_product(self):
        v1 = ca.DM([1, 2, 3])
        v2 = ca.DM([4, 5, 6])
        v3 = v1 + v2
        G1 = LieGroup_R(3, ca.DM([1, 2, 3]))
        G2 = LieGroup_R(3, ca.DM([4, 5, 6]))
        G3 = G1 * G2
        for i in range(3):
            self.assertEqual(G3.param[i], v3[i])


class Test_LieGroup_SO2(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
