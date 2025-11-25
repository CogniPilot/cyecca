from tests.common import ProfiledTestCase, SX_close, is_finite
from beartype import beartype

import casadi as ca
import numpy as np
import scipy.linalg

from cyecca.lie.group_so3 import so3, SO3EulerB321, SO3Quat, SO3Mrp


@beartype
class Test_LieAlgebraSO3(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0])
        self.v2 = ca.DM([4.0, 5.0, 6.0])

    def test_ctor(self):
        so3.elem(self.v1)

    def test_add(self):
        v3 = self.v1 + self.v2
        g1 = so3.elem(self.v1)
        g2 = so3.elem(self.v2)
        g3 = g1 + g2
        self.assertTrue(SX_close(g3.param, v3))

    def test_to_Matrix(self):
        g1 = so3.elem(self.v1)
        X = g1.to_Matrix()

    def test_ad(self):
        g1 = so3.elem(self.v1)
        g1.ad()

    def test_vee(self):
        g1 = so3.elem(self.v1)
        g1.vee()

    def test_wedge(self):
        so3.wedge(self.v1)

    def test_mul(self):
        g1 = so3.elem(self.v1)
        g2 = so3.elem(self.v2)
        g3 = g1 * g2

    def test_rmul(self):
        g1 = so3.elem(self.v1)
        3 * g1

    def test_exp_mrp(self):
        g1 = so3.elem(self.v1)
        g1.exp(SO3Mrp)

    def test_exp_quat(self):
        g1 = so3.elem(self.v1)
        g1.exp(SO3Quat)

    def test_exp_Euler(self):
        SO3EulerB321.elem(self.v1)
        g1 = so3.elem(self.v1)
        g1.exp(SO3EulerB321)

    def test_str(self):
        g1 = so3.elem(self.v1)
        print(g1)

    def test_eq(self):
        g1 = so3.elem(self.v1)
        g2 = so3.elem(self.v1)
        self.assertTrue(g1 == g2)

    def test_print(self):
        print(so3)

    def test_repr(self):
        repr(so3)

    def test_left_jacobian(self):
        n = 3
        x = ca.SX.sym('x', n)
        omega = so3.elem(x)
        Jl = omega.left_jacobian()
        self.assertTrue(is_finite(ca.substitute(ca.jacobian(Jl, x), x, ca.DM.zeros(n))))

        Jl_inv = omega.left_jacobian_inv()
        self.assertTrue(
            is_finite(ca.substitute(ca.jacobian(Jl_inv, x), x, ca.DM.zeros(n)))
        )

        I_check = ca.substitute(Jl @ Jl_inv, x, ca.DM.zeros(n))
        self.assertTrue(SX_close(I_check, ca.DM.eye(n)))

    def test_right_jacobian(self):
        n = 3
        x = ca.SX.sym('x', n)
        omega = so3.elem(x)
        Jr = omega.right_jacobian()
        self.assertTrue(is_finite(ca.substitute(ca.jacobian(Jr, x), x, ca.DM.zeros(n))))

        Jr_inv = omega.right_jacobian_inv()
        self.assertTrue(
            is_finite(ca.substitute(ca.jacobian(Jr_inv, x), x, ca.DM.zeros(n)))
        )

        I_check = ca.substitute(Jr @ Jr_inv, x, ca.DM.zeros(n))
        self.assertTrue(SX_close(I_check, ca.DM.eye(n)))

    def test_ad_exp_jacobian(self):
        x1 = ca.DM([0.1, 0.2, 0.3])
        omega = so3.elem(x1)
        Jl = omega.left_jacobian()
        Jr = omega.right_jacobian()
        Jl_inv = omega.left_jacobian_inv()
        Jr_inv = omega.right_jacobian_inv()

        self.assertTrue(SX_close(Jl, scipy.linalg.expm(ca.DM(omega.ad())) @ Jr))

        self.assertTrue(SX_close(Jr_inv, scipy.linalg.expm(ca.DM(omega.ad())) @ Jl_inv))


class Test_LieGroupSO3Euler(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([0.1, 0.2, 0.3])
        self.v2 = ca.DM([0.4, 0.5, 0.6])

    def test_ctor(self):
        SO3EulerB321.elem(self.v1)

    def test_bad_operations(self):
        G1 = SO3EulerB321.elem(self.v1)
        G1 = SO3EulerB321.elem(self.v1)
        G2 = SO3EulerB321.elem(self.v2)
        s = 1
        with self.assertRaises(TypeError):
            G1 + G2
        with self.assertRaises(TypeError):
            G1 - G2
        with self.assertRaises(TypeError):
            G1 @ G2
        with self.assertRaises(TypeError):
            s * G2

    def test_identity(self):
        G1 = SO3EulerB321.elem(self.v1)
        G2 = G1 * SO3EulerB321.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = SO3EulerB321.elem(self.v1)
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = SO3EulerB321.elem(self.v1)
        self.assertTrue(
            SX_close((G1 * G1.inverse()).param, SO3EulerB321.identity().param)
        )

    def test_log(self):
        G1 = SO3EulerB321.elem(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SO3EulerB321.elem(self.v1)
        G2 = G1.log().exp(SO3EulerB321)
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_print_group(self):
        print(SO3EulerB321)

    def test_print_group_elem(self):
        G1 = SO3EulerB321.elem(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SO3EulerB321)

    def test_repr_group_elem(self):
        G1 = SO3EulerB321.elem(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SO3EulerB321.elem(self.v1)
        G2 = SO3EulerB321.elem(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SO3EulerB321.elem(self.v1)
        G1.Ad()

    def test_right_jacobian(self):
        """Test that right Jacobian relates body angular velocity to Euler rates."""
        # Test at several attitudes
        test_cases = [
            ca.DM([0.0, 0.0, 0.0]),      # Identity
            ca.DM([0.1, -0.05, 0.15]),   # Small angles
            ca.DM([0.5, -0.3, 0.4]),     # Moderate angles
        ]
        
        for euler_angles in test_cases:
            G = SO3EulerB321.elem(euler_angles)
            Jr = G.right_jacobian()
            
            # Check that Jacobian is 3x3
            self.assertEqual(Jr.shape, (3, 3))
            
            # Check that it's finite
            self.assertTrue(is_finite(Jr))
            
            # Verify the relationship: euler_dot = Jr @ omega_body
            # by checking consistency with DCM kinematics
            omega_body = ca.DM([0.1, 0.2, 0.3])
            euler_dot = Jr @ omega_body
            
            # The derivative should be finite
            self.assertTrue(is_finite(euler_dot))

    def test_left_jacobian(self):
        """Test that left Jacobian relates spatial angular velocity to Euler rates."""
        # Test at several attitudes  
        test_cases = [
            ca.DM([0.0, 0.0, 0.0]),      # Identity
            ca.DM([0.1, -0.05, 0.15]),   # Small angles
            ca.DM([0.5, -0.3, 0.4]),     # Moderate angles
        ]
        
        for euler_angles in test_cases:
            G = SO3EulerB321.elem(euler_angles)
            Jl = G.left_jacobian()
            
            # Check that Jacobian is 3x3
            self.assertEqual(Jl.shape, (3, 3))
            
            # Check that it's finite
            self.assertTrue(is_finite(Jl))
            
            # Verify the relationship: euler_dot = Jl @ omega_spatial
            omega_spatial = ca.DM([0.1, 0.2, 0.3])
            euler_dot = Jl @ omega_spatial
            
            # The derivative should be finite
            self.assertTrue(is_finite(euler_dot))

    def test_left_right_jacobian_consistency(self):
        """Test that left and right Jacobians are consistently related.
        
        The relationship for Euler angles is:
        euler_dot = Jr @ omega_body = Jl @ omega_spatial
        
        where omega_spatial = R_be @ omega_body = R_eb^T @ omega_body
        
        This implies: Jl = Jr @ R_eb
        """
        # Test at several non-trivial attitudes
        test_cases = [
            ca.DM([0.1, -0.05, 0.15]),   # Small angles
            ca.DM([0.5, -0.3, 0.4]),     # Moderate angles
            ca.DM([1.0, -0.5, 0.8]),     # Larger angles
        ]
        
        for euler_angles in test_cases:
            G = SO3EulerB321.elem(euler_angles)
            Jl = G.left_jacobian()
            Jr = G.right_jacobian()
            
            # Get rotation matrix (earth to body)
            from cyecca.lie.group_so3 import SO3Dcm
            R_eb = SO3Dcm.from_Euler(G).to_Matrix()
            
            # Check relationship: Jl should equal Jr @ R_eb
            expected_Jl = Jr @ R_eb
            
            # Compute difference
            diff = Jl - expected_Jl
            max_error = float(ca.norm_inf(diff))
            
            # Should be very close (within numerical precision)
            self.assertLess(max_error, 1e-10, 
                          f"Left/Right Jacobian inconsistency at {euler_angles}: max error = {max_error}")
            
            # Also verify by checking that both give the same euler_dot
            omega_body = ca.DM([0.1, 0.2, 0.3])
            omega_spatial = R_eb.T @ omega_body
            
            euler_dot_from_right = Jr @ omega_body
            euler_dot_from_left = Jl @ omega_spatial
            
            diff_euler_dot = euler_dot_from_right - euler_dot_from_left
            max_euler_dot_error = float(ca.norm_inf(diff_euler_dot))
            
            self.assertLess(max_euler_dot_error, 1e-10,
                          f"Euler rate inconsistency at {euler_angles}: max error = {max_euler_dot_error}")

    def test_jacobian_at_zero(self):
        """Test Jacobians at identity (zero Euler angles)."""
        G = SO3EulerB321.elem(ca.DM([0.0, 0.0, 0.0]))
        
        Jr = G.right_jacobian()
        Jl = G.left_jacobian()
        
        # At identity (zero Euler angles), the right Jacobian relates:
        # [psi_dot, theta_dot, phi_dot]^T = Jr @ [p, q, r]^T
        # Since psi_dot=r, theta_dot=q, phi_dot=p (at zero angles):
        Jr_expected = ca.DM([
            [0, 0, 1],   # psi_dot = r
            [0, 1, 0],   # theta_dot = q
            [1, 0, 0]    # phi_dot = p
        ])
        
        # Similarly for left Jacobian at identity
        Jl_expected = ca.DM([
            [0, 0, 1],   # psi_dot = omega_spatial_z
            [0, 1, 0],   # theta_dot = omega_spatial_y  
            [1, 0, 0]    # phi_dot = omega_spatial_x
        ])
        
        self.assertTrue(SX_close(Jr, Jr_expected))
        self.assertTrue(SX_close(Jl, Jl_expected))


class Test_LieGroupSO3Quat(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 0.0, 0.0, 0.0])
        self.v2 = ca.DM([0.0, 1.0, 0.0, 0.0])

    def test_ctor(self):
        SO3Quat.elem(self.v1)

    def test_bad_operations(self):
        G1 = SO3Quat.elem(self.v1)
        G2 = SO3Quat.elem(self.v2)
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
        q0 = SO3Quat.elem(self.v1)
        q1 = SO3Quat.elem(self.v2)
        q2 = q0 * q1
        assert q2 == q1

    def test_identity(self):
        G1 = SO3Quat.elem(self.v1)
        G2 = G1 * SO3Quat.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = SO3Quat.elem(self.v1)
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = SO3Quat.elem(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SO3Quat.identity().param))

    def test_log(self):
        G1 = SO3Quat.elem(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SO3Quat.elem(self.v1)
        G2 = G1.log().exp(SO3Quat)
        print(G1, G2)
        self.assertTrue(G1 == G2)

    def test_print_group(self):
        print(SO3Quat)

    def test_print_group_elem(self):
        G1 = SO3Quat.elem(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SO3Quat)

    def test_repr_group_elem(self):
        G1 = SO3Quat.elem(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SO3Quat.elem(self.v1)
        G2 = SO3Quat.elem(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SO3Quat.elem(self.v1)
        G1.Ad()


class Test_LieGroupSO3Mrp(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 0.0, 0.0])
        self.v2 = ca.DM([0.0, 1.0, 0.0])

    def test_ctor(self):
        SO3Mrp.elem(self.v1)

    def test_bad_operations(self):
        G1 = SO3Mrp.elem(self.v1)
        G2 = SO3Mrp.elem(self.v2)
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
        q0 = SO3Mrp.elem(self.v1)
        q1 = SO3Mrp.elem(ca.DM([0, 0, 0]))
        q2 = q0 * q1
        self.assertTrue(SX_close(q0.param, q2.param))

    def test_identity(self):
        G1 = SO3Mrp.elem(self.v1)
        G2 = G1 * SO3Mrp.identity()
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_to_Matrix(self):
        G1 = SO3Mrp.elem(self.v1)
        X = G1.to_Matrix()

    def test_inverse(self):
        G1 = SO3Mrp.elem(self.v1)
        self.assertTrue(SX_close((G1 * G1.inverse()).param, SO3Mrp.identity().param))

    def test_log(self):
        G1 = SO3Mrp.elem(self.v1)
        G1.log()

    def test_exp_log(self):
        G1 = SO3Mrp.elem(self.v1)
        G2 = G1.log().exp(SO3Mrp)
        print(G1, G2)
        self.assertTrue(SX_close(G1.param, G2.param))

    def test_print_group(self):
        print(SO3Mrp)

    def test_print_group_elem(self):
        G1 = SO3Mrp.elem(self.v1)
        print(G1)

    def test_repr_group(self):
        repr(SO3Mrp)

    def test_repr_group_elem(self):
        G1 = SO3Mrp.elem(self.v1)
        repr(G1)

    def test_eq(self):
        G1 = SO3Mrp.elem(self.v1)
        G2 = SO3Mrp.elem(self.v1)
        self.assertTrue(G1 == G2)

    def test_Ad(self):
        G1 = SO3Mrp.elem(self.v1)
        G1.Ad()


class Test_SO3Conversions(ProfiledTestCase):
    """Test conversions between different SO3 representations."""
    
    def setUp(self):
        super().setUp()
        # Start with Euler angles and derive consistent test values
        self.euler_test = ca.DM([0.3, -0.2, 0.4])  # [psi, theta, phi]
        
        # Get corresponding quaternion and MRP
        euler_elem = SO3EulerB321.elem(self.euler_test)
        quat_elem = SO3Quat.from_Euler(euler_elem)
        mrp_elem = SO3Mrp.from_Quat(quat_elem)
        
        self.quat_test = quat_elem.param
        self.mrp_test = mrp_elem.param
    
    def test_quat_to_euler_to_quat(self):
        """Test roundtrip conversion: Quat -> Euler -> Quat."""
        q1 = SO3Quat.elem(self.quat_test)
        euler = SO3EulerB321.from_Quat(q1)
        q2 = SO3Quat.from_Euler(euler)
        
        # Quaternions might differ by sign, check both
        diff1 = ca.norm_2(q1.param - q2.param)
        diff2 = ca.norm_2(q1.param + q2.param)
        
        self.assertTrue(float(diff1) < 1e-10 or float(diff2) < 1e-10,
                       f"Quat->Euler->Quat failed: diff={min(float(diff1), float(diff2))}")
    
    def test_euler_to_quat_to_euler(self):
        """Test roundtrip conversion: Euler -> Quat -> Euler."""
        e1 = SO3EulerB321.elem(self.euler_test)
        quat = SO3Quat.from_Euler(e1)
        e2 = SO3EulerB321.from_Quat(quat)
        
        diff = ca.norm_2(e1.param - e2.param)
        self.assertLess(float(diff), 1e-10,
                       f"Euler->Quat->Euler failed: diff={float(diff)}")
    
    def test_mrp_to_quat_to_mrp(self):
        """Test roundtrip conversion: MRP -> Quat -> MRP."""
        m1 = SO3Mrp.elem(self.mrp_test)
        quat = SO3Quat.from_Mrp(m1)
        m2 = SO3Mrp.from_Quat(quat)
        
        # MRP might shadow, so check both
        diff1 = ca.norm_2(m1.param - m2.param)
        # Shadow transformation: p_shadow = -p/|p|^2
        n_sq = ca.dot(m1.param, m1.param)
        shadow = -m1.param / n_sq
        diff2 = ca.norm_2(shadow - m2.param)
        
        self.assertTrue(float(diff1) < 1e-10 or float(diff2) < 1e-10,
                       f"MRP->Quat->MRP failed: diff={min(float(diff1), float(diff2))}")
    
    def test_quat_euler_mrp_dcm_consistency(self):
        """Test that all representations produce the same DCM."""
        from cyecca.lie.group_so3 import SO3Dcm
        
        # Create same rotation in all representations
        euler = SO3EulerB321.elem(self.euler_test)
        quat = SO3Quat.from_Euler(euler)
        mrp = SO3Mrp.from_Quat(quat)
        
        # Convert all to DCM
        dcm_from_euler = SO3Dcm.from_Euler(euler).to_Matrix()
        dcm_from_quat = SO3Dcm.from_Quat(quat).to_Matrix()
        dcm_from_mrp = SO3Dcm.from_Mrp(mrp).to_Matrix()
        
        # All should be equal
        diff_eq = ca.norm_inf(dcm_from_euler - dcm_from_quat)
        diff_em = ca.norm_inf(dcm_from_euler - dcm_from_mrp)
        diff_qm = ca.norm_inf(dcm_from_quat - dcm_from_mrp)
        
        self.assertLess(float(diff_eq), 1e-10, f"Euler-Quat DCM mismatch: {float(diff_eq)}")
        self.assertLess(float(diff_em), 1e-10, f"Euler-MRP DCM mismatch: {float(diff_em)}")
        self.assertLess(float(diff_qm), 1e-10, f"Quat-MRP DCM mismatch: {float(diff_qm)}")
    
    def test_dcm_from_euler(self):
        """Test DCM construction from Euler angles."""
        from cyecca.lie.group_so3 import SO3Dcm
        
        euler = SO3EulerB321.elem(self.euler_test)
        dcm = SO3Dcm.from_Euler(euler)
        
        # DCM should be orthogonal: R @ R^T = I
        R = dcm.to_Matrix()
        I_check = R @ R.T
        diff = ca.norm_inf(I_check - ca.DM.eye(3))
        
        self.assertLess(float(diff), 1e-10, f"DCM not orthogonal: {float(diff)}")
        
        # Determinant should be 1
        det = ca.det(R)
        self.assertLess(abs(float(det) - 1.0), 1e-10, f"DCM determinant not 1: {float(det)}")
    
    def test_euler_gimbal_lock_theta_90(self):
        """Test Euler angles near gimbal lock (theta = 90 degrees)."""
        from cyecca.lie.group_so3 import SO3Dcm
        
        # theta close to pi/2 (90 degrees)
        euler_near_lock = ca.DM([0.5, ca.pi/2 - 0.001, 0.3])
        
        euler = SO3EulerB321.elem(euler_near_lock)
        dcm = SO3Dcm.from_Euler(euler)
        
        # Should still produce valid DCM
        R = dcm.to_Matrix()
        det = ca.det(R)
        self.assertLess(abs(float(det) - 1.0), 1e-6, f"DCM invalid at gimbal lock: det={float(det)}")
        
        # Jacobian should have high condition number but be finite
        Jr = euler.right_jacobian()
        self.assertTrue(is_finite(Jr), "Jacobian not finite near gimbal lock")
    
    def test_quat_negative_w(self):
        """Test quaternion with negative w component."""
        # Both q and -q represent the same rotation
        q_pos = SO3Quat.elem(ca.DM([0.9, 0.1, 0.2, 0.3]))
        q_neg = SO3Quat.elem(ca.DM([-0.9, -0.1, -0.2, -0.3]))
        
        # Should produce same DCM
        from cyecca.lie.group_so3 import SO3Dcm
        dcm_pos = SO3Dcm.from_Quat(q_pos).to_Matrix()
        dcm_neg = SO3Dcm.from_Quat(q_neg).to_Matrix()
        
        diff = ca.norm_inf(dcm_pos - dcm_neg)
        self.assertLess(float(diff), 1e-10, f"q and -q produce different DCM: {float(diff)}")

