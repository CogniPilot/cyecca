import casadi as ca
from ..common import SX_close, ProfiledTestCase, is_finite

from cyecca.lie.group_se23 import SE23Mrp, se23
from beartype import beartype
import scipy.linalg
import numpy as np


@beartype
class Test_LieGroupSE23Mrp(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([0.1, 0.2, 0.3, 0.4, 5.0, 6.0, 0.15, 0.25, 0.35])
        self.v2 = ca.DM([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.15, 0.25, 0.35])

    def test_ctor(self):
        SE23Mrp.elem(param=self.v1)

    def test_bad_operations(self):
        G1 = SE23Mrp.elem(self.v1)
        G2 = SE23Mrp.elem(self.v2)
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
        G1 = SE23Mrp.elem(self.v1)
        G2 = SE23Mrp.elem(self.v2)
        G3 = G1 * G2

    def test_identity(self):
        G1 = SE23Mrp.elem(self.v1)
        G2 = G1 * SE23Mrp.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = SE23Mrp.elem(self.v1)
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = SE23Mrp.elem(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SE23Mrp.identity().param))

    def test_exp(self):
        g1 = SE23Mrp.algebra.elem(self.v1)
        g1.exp(SE23Mrp)

    def test_log(self):
        G1 = SE23Mrp.elem(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SE23Mrp.elem(self.v1)
        G2 = G1.log().exp(SE23Mrp)
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_ad_Ad_exp(self):
        x = se23.elem(self.v1)
        exp_ad_x = scipy.linalg.expm(ca.DM(x.ad()))
        Ad_exp_x = np.array(ca.DM(x.exp(SE23Mrp).Ad()))
        self.assertTrue(np.linalg.norm(exp_ad_x - Ad_exp_x) < 1e-12)

    def test_print_group(self):
        print(SE23Mrp)

    def test_left_jacobian(self):
        x = ca.SX.sym("x", 9)
        omega = se23.elem(x)
        Jl = omega.left_jacobian()
        self.assertTrue(is_finite(ca.substitute(ca.jacobian(Jl, x), x, ca.DM.zeros(9))))

        Jl_inv = omega.left_jacobian_inv()
        self.assertTrue(
            is_finite(ca.substitute(ca.jacobian(Jl_inv, x), x, ca.DM.zeros(9)))
        )

        I_check = ca.substitute(Jl @ Jl_inv, x, ca.DM.zeros(9))
        self.assertTrue(SX_close(I_check, ca.DM.eye(9)))

    def test_right_jacobian(self):
        x = ca.SX.sym("x", 9)
        omega = se23.elem(x)
        Jr = omega.right_jacobian()
        self.assertTrue(is_finite(ca.substitute(ca.jacobian(Jr, x), x, ca.DM.zeros(9))))

        Jr_inv = omega.right_jacobian_inv()
        self.assertTrue(
            is_finite(ca.substitute(ca.jacobian(Jr_inv, x), x, ca.DM.zeros(9)))
        )

        I_check = ca.substitute(Jr @ Jr_inv, x, ca.DM.zeros(9))
        self.assertTrue(SX_close(I_check, ca.DM.eye(9)))

    def test_ad_exp_jacobian(self):
        x1 = ca.DM([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        omega = se23.elem(x1)
        Jl = omega.left_jacobian()
        Jr = omega.right_jacobian()
        Jl_inv = omega.left_jacobian_inv()
        Jr_inv = omega.right_jacobian_inv()

        self.assertTrue(SX_close(Jl, scipy.linalg.expm(ca.DM(omega.ad())) @ Jr))

        self.assertTrue(SX_close(Jr_inv, scipy.linalg.expm(ca.DM(omega.ad())) @ Jl_inv))
