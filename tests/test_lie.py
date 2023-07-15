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
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([3.0, 2.0, 1.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0])

    def test_ctor(self):
        lie.SE2.element(self.v1)

    def test_bad_operations(self):
        G1 = lie.SE2.element(self.v1)
        G2 = lie.SE2.element(self.v2)
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
        G1 = lie.SE2.element(self.v1)
        G2 = lie.SE2.element(ca.DM([0,0,0]))
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, G1.param))

    def test_identity(self):
        G1 = lie.SE2.element(self.v1)
        G2 = G1 * lie.SE2.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        G1 = lie.SE2.element(self.v1)
        X = G1.to_matrix()

    def test_inverse(self):
        G1 = lie.SE2.element(self.v1)
        self.assertTrue(SX_close((G1*G1.inverse()).param, lie.SE2.identity().param))
    
    def test_log(self):
        G1 = lie.SE2.element(self.v1)
        G1.log()
    
    def test_exp_log(self):
        G1 = lie.SE2.element(self.v1)
        G2 = G1.log().exp(lie.SE2)
        print(G1, G2)
        self.assertTrue(G1 == G2)
        
    def test_print_group(self):
        print(lie.SE2)

    def test_print_group_element(self):
        G1 = lie.SE2.element(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(lie.SE2)

    def test_repr_group_element(self):
        G1 = lie.SE2.element(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = lie.SE2.element(self.v1)
        G2 = lie.SE2.element(self.v1)
        self.assertTrue(G1 == G2)


class Test_LieAlgebraSO2(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0])
        self.v2 = ca.DM([4.0])

    def test_ctor(self):
        lie.so2.element(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = lie.so2.element(self.v1)
        g2 = lie.so2.element(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_matrix(self):
        g1 = lie.so2.element(self.v1)
        X = g1.to_matrix()

    def test_ad(self):
        g1 = lie.so2.element(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = lie.so2.element(self.v1)
        g1.vee()

    def test_wedge(self):
        lie.so2.wedge(self.v1)

    def test_mul(self):
        g1 = lie.so2.element(self.v1)
        g2 = lie.so2.element(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = lie.so2.element(self.v1)
        3 * g1

    def test_exp(self):
        g1 = lie.so2.element(self.v1)
        g1.exp(lie.SO2)

    def test_str(self):
        g1 = lie.so2.element(self.v1)
        print(g1)

    def test_eq(self):
        g1 = lie.so2.element(self.v1)
        g2 = lie.so2.element(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(lie.so2)

    def test_repr(self):
        repr(lie.so2)

class Test_LieGroupSO2(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0])
        self.v2 = ca.DM([2.0])

    def test_ctor(self):
        lie.SO2.element(self.v1)

    def test_bad_operations(self):
        G1 = lie.SO2.element(self.v1)
        G2 = lie.SO2.element(self.v2)
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
        G1 = lie.SO2.element(self.v1)
        G2 = lie.SO2.element(self.v2)
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, v3))

    def test_identity(self):
        G1 = lie.SO2.element(self.v1)
        G2 = G1 * lie.SO2.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        G1 = lie.SO2.element(self.v1)
        X = G1.to_matrix()

    def test_inverse(self):
        G1 = lie.SO2.element(self.v1)
        self.assertTrue(SX_close((G1*G1.inverse()).param, lie.SO2.identity().param))
    
    def test_log(self):
        G1 = lie.SO2.element(self.v1)
        G1.log()
    
    def test_exp_log(self):
        G1 = lie.SO2.element(self.v1)
        G2 = G1.log().exp(lie.SO2)
        print(G1, G2)
        self.assertTrue(G1 == G2)
        
    def test_print_group(self):
        print(lie.SO2)

    def test_print_group_element(self):
        G1 = lie.SO2.element(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(lie.SO2)

    def test_repr_group_element(self):
        G1 = lie.SO2.element(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = lie.SO2.element(self.v1)
        G2 = lie.SO2.element(self.v1)
        self.assertTrue(G1 == G2)


class Test_LieAlgebraSO3(ProfiledTestCase):

    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0])

    def test_ctor(self):
        lie.so3.element(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = lie.so3.element(self.v1)
        g2 = lie.so3.element(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_matrix(self):
        g1 = lie.so3.element(self.v1)
        X = g1.to_matrix()

    def test_ad(self):
        g1 = lie.so3.element(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = lie.so3.element(self.v1)
        g1.vee()

    def test_wedge(self):
        lie.so3.wedge(self.v1)

    def test_mul(self):
        g1 = lie.so3.element(self.v1)
        g2 = lie.so3.element(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = lie.so3.element(self.v1)
        3 * g1

    def test_exp_mrp(self):
        g1 = lie.so3.element(self.v1)
        g1.exp(lie.SO3MRP)
        
    def test_exp_quat(self):
        g1 = lie.so3.element(self.v1)
        g1.exp(lie.SO3Quat)
        
    def test_exp_Euler(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        SO3EulerB321.element(self.v1)
        g1 = lie.so3.element(self.v1)
        g1.exp(SO3EulerB321)

    def test_str(self):
        g1 = lie.so3.element(self.v1)
        print(g1)

    def test_eq(self):
        g1 = lie.so3.element(self.v1)
        g2 = lie.so3.element(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(lie.so3)

    def test_repr(self):
        repr(lie.so3)


class Test_LieGroupSO3Euler(ProfiledTestCase):

    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([0.1, 0.2, 0.3])
        self.v2 = ca.DM([0.4, 0.5, 0.6])


    def test_ctor(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        SO3EulerB321.element(self.v1)

    def test_ineverse(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        G1 = SO3EulerB321.element(self.v1)
        self.assertTrue(G1*G1.inverse() == SO3EulerB321.identity())

    def test_bad_operations(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
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
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        v3 = self.v1 + self.v2
        G1 = SO3EulerB321.element(self.v1)
        G2 = SO3EulerB321.element(self.v2)
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, v3))

    def test_identity(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        G1 = SO3EulerB321.element(self.v1)
        G2 = G1 * SO3EulerB321.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        G1 = SO3EulerB321.element(self.v1)
        X = G1.to_matrix()

    def test_inverse(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        G1 = SO3EulerB321.element(self.v1)
        self.assertTrue(SX_close((G1*G1.inverse()).param, SO3EulerB321.identity().param))
    
    def test_log(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        G1 = SO3EulerB321.element(self.v1)
        G1.log()
    
    def test_exp_log(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        G1 = SO3EulerB321.element(self.v1)
        G2 = G1.log().exp(SO3EulerB321)
        print(G1, G2)
        self.assertTrue(SX_close(G1.param, G2.param))
        
    def test_print_group(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        print(SO3EulerB321)

    def test_print_group_element(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        G1 = SO3EulerB321.element(self.v1)
        print(G1)

    def test_repr_group(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        repr(SO3EulerB321)

    def test_repr_group_element(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        G1 = SO3EulerB321.element(self.v1)
        repr(G1)

    def test_eq(self):
        SO3EulerB321 = lie.SO3Euler(
            euler_type=lie.EulerType.body_fixed,
            sequence=[lie.Axis.z, lie.Axis.y, lie.Axis.x])
        G1 = SO3EulerB321.element(self.v1)
        G2 = SO3EulerB321.element(self.v1)
        self.assertTrue(G1 == G2)


class Test_LieGroupSO3Quat(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 0.0, 0.0, 0.0])
        self.v2 = ca.DM([0.0, 1.0, 0.0, 0.0])

    def test_ctor(self):
        lie.SO3Quat.element(self.v1)

    def test_bad_operations(self):
        G1 = lie.SO3Quat.element(self.v1)
        G2 = lie.SO3Quat.element(self.v2)
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
        q0 = lie.SO3Quat.element(self.v1)
        q1 = lie.SO3Quat.element(self.v2)
        q2 = q0 * q1
        assert q2 == q1

    def test_identity(self):
        G1 = lie.SO3Quat.element(self.v1)
        G2 = G1 * lie.SO3Quat.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        G1 = lie.SO3Quat.element(self.v1)
        X = G1.to_matrix()

    def test_inverse(self):
        G1 = lie.SO3Quat.element(self.v1)
        self.assertTrue(SX_close((G1*G1.inverse()).param, lie.SO3Quat.identity().param))
    
    def test_log(self):
        G1 = lie.SO3Quat.element(self.v1)
        G1.log()
    
    def test_exp_log(self):
        G1 = lie.SO3Quat.element(self.v1)
        G2 = G1.log().exp(lie.SO3Quat)
        print(G1, G2)
        self.assertTrue(G1 == G2)
        
    def test_print_group(self):
        print(lie.SO3Quat)

    def test_print_group_element(self):
        G1 = lie.SO3Quat.element(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(lie.SO3Quat)

    def test_repr_group_element(self):
        G1 = lie.SO3Quat.element(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = lie.SO3Quat.element(self.v1)
        G2 = lie.SO3Quat.element(self.v1)
        self.assertTrue(G1 == G2)


class Test_LieGroupSO3MRP(ProfiledTestCase):    
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 0.0, 0.0, 0.0])
        self.v2 = ca.DM([0.0, 1.0, 0.0, 0.0])

    def test_ctor(self):
        lie.SO3MRP.element(self.v1)

    def test_bad_operations(self):
        G1 = lie.SO3MRP.element(self.v1)
        G2 = lie.SO3MRP.element(self.v2)
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
        q0 = lie.SO3MRP.element(self.v1)
        q1 = lie.SO3MRP.element(ca.DM([0,0,0,0]))
        q2 = q0 * q1
        self.assertTrue(SX_close(q0.param, q2.param))

    def test_identity(self):
        G1 = lie.SO3MRP.element(self.v1)
        G2 = G1 * lie.SO3MRP.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        G1 = lie.SO3MRP.element(self.v1)
        X = G1.to_matrix()

    def test_inverse(self):
        G1 = lie.SO3MRP.element(self.v1)
        self.assertTrue(SX_close((G1*G1.inverse()).param, lie.SO3MRP.identity().param))
    
    def test_log(self):
        G1 = lie.SO3MRP.element(self.v1)
        G1.log()
    
    def test_exp_log(self):
        G1 = lie.SO3MRP.element(self.v1)
        G2 = G1.log().exp(lie.SO3MRP)
        print(G1, G2)
        self.assertTrue(SX_close(G1.param, G2.param))
        
    def test_print_group(self):
        print(lie.SO3MRP)

    def test_print_group_element(self):
        G1 = lie.SO3MRP.element(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(lie.SO3MRP)

    def test_repr_group_element(self):
        G1 = lie.SO3MRP.element(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = lie.SO3MRP.element(self.v1)
        G2 = lie.SO3MRP.element(self.v1)
        self.assertTrue(G1 == G2)

class Test_LieAlgebraSE3(ProfiledTestCase):

    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    def test_ctor(self):
        lie.se3.element(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = lie.se3.element(self.v1)
        g2 = lie.se3.element(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_matrix(self):
        g1 = lie.se3.element(self.v1)
        X = g1.to_matrix()

    def test_ad(self):
        g1 = lie.se3.element(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = lie.se3.element(self.v1)
        g1.vee()

    def test_wedge(self):
        lie.se3.wedge(self.v1)

    def test_mul(self):
        g1 = lie.se3.element(self.v1)
        g2 = lie.se3.element(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = lie.se3.element(self.v1)
        3 * g1

    def test_exp_mrp(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        g1 = lie.se3.element(self.v1)
        g1.exp(SE3MRP)
        
    def test_exp_quat(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        g1 = lie.se3.element(self.v1)
        g1.exp(SE3Quat)

    def test_str(self):
        g1 = lie.se3.element(self.v1)
        print(g1)

    def test_eq(self):
        g1 = lie.se3.element(self.v1)
        g2 = lie.se3.element(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(lie.se3)

    def test_repr(self):
        repr(lie.se3)

class Test_LieGroupSE3MRP(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([3.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0])

    def test_ctor(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        SE3MRP.element(self.v1)

    def test_bad_operations(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        G1 = SE3MRP.element(self.v1)
        G2 = SE3MRP.element(self.v2)
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
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        G1 = SE3MRP.element(self.v1)
        G2 = SE3MRP.element(ca.DM([0,0,0,0,0,0,0]))
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, G1.param))

    def test_identity(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        G1 = SE3MRP.element(self.v1)
        G2 = G1 * SE3MRP.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        G1 = SE3MRP.element(self.v1)
        X = G1.to_matrix()

    def test_inverse(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        G1 = SE3MRP.element(self.v1)
        self.assertTrue(SX_close((G1*G1.inverse()).param, SE3MRP.identity().param))
    
    def test_log(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        G1 = SE3MRP.element(self.v1)
        G1.log()
    
    def test_exp_log(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        G1 = SE3MRP.element(self.v1)
        G2 = G1.log().exp(SE3MRP)
        self.assertTrue(SX_close(G1.param, G2.param))
        
    def test_print_group(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        print(SE3MRP)

    def test_print_group_element(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        G1 = SE3MRP.element(self.v1)
        print(G1)

    def test_repr_group(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        repr(SE3MRP)

    def test_repr_group_element(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        G1 = SE3MRP.element(self.v1)
        repr(G1)

    def test_eq(self):
        SE3MRP = lie.SE3(SO3=lie.SO3MRP)
        G1 = SE3MRP.element(self.v1)
        G2 = SE3MRP.element(self.v1)
        self.assertTrue(G1 == G2)

class Test_LieGroupSE3Quat(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([3.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0])

    def test_ctor(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        SE3Quat.element(self.v1)

    def test_bad_operations(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
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
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        G1 = SE3Quat.element(self.v1)
        G2 = SE3Quat.element(ca.DM([0,0,0,1,0,0,0]))
        G3 = G1 * G2
        self.assertTrue(SX_close(G3.param, G1.param))

    def test_identity(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        G1 = SE3Quat.element(self.v1)
        G2 = G1 * SE3Quat.identity()
        print(SE3Quat.identity().to_matrix())
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_matrix(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        G1 = SE3Quat.element(self.v1)
        X = G1.to_matrix()

    def test_inverse(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        G1 = SE3Quat.element(self.v1)
        self.assertTrue(SX_close((G1*G1.inverse()).param, SE3Quat.identity().param))
    
    def test_log(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        G1 = SE3Quat.element(self.v1)
        G1.log()
    
    def test_exp_log(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        G1 = SE3Quat.element(self.v1)
        G2 = G1.log().exp(SE3Quat)
        self.assertTrue(SX_close(G1.param, G2.param))
        
    def test_print_group(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        print(SE3Quat)

    def test_print_group_element(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        G1 = SE3Quat.element(self.v1)
        print(G1)

    def test_repr_group(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        repr(SE3Quat)

    def test_repr_group_element(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        G1 = SE3Quat.element(self.v1)
        repr(G1)

    def test_eq(self):
        SE3Quat = lie.SE3(SO3=lie.SO3Quat)
        G1 = SE3Quat.element(self.v1)
        G2 = SE3Quat.element(self.v1)
        self.assertTrue(G1 == G2)

if __name__ == "__main__":
    unittest.main()
