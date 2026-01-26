import unittest

import numpy as np

from skrobot.pycompat import HAS_JAX


def requires_jax(test_func):
    """Decorator to skip tests if JAX is not available."""
    return unittest.skipUnless(HAS_JAX, "JAX not available")(test_func)


class TestDifferentiableKinematics(unittest.TestCase):
    """Test differentiable kinematics with JAX backend."""

    @classmethod
    def setUpClass(cls):
        if not HAS_JAX:
            return

        from skrobot.models import Panda
        from skrobot.models import R8_6

        cls.panda = Panda()
        cls.r8_6 = R8_6()

    @requires_jax
    def test_jax_backend_available(self):
        """Test that JAX backend can be loaded."""
        from skrobot.backend import get_backend

        backend = get_backend('jax')
        self.assertEqual(backend.name, 'jax')
        self.assertTrue(backend.supports_autodiff)
        self.assertTrue(backend.supports_jit)

    @requires_jax
    def test_forward_kinematics_panda(self):
        """Test forward kinematics with JAX backend on Panda robot."""
        from skrobot.backend import get_backend
        from skrobot.kinematics.differentiable import extract_fk_parameters
        from skrobot.kinematics.differentiable import forward_kinematics_ee

        panda = self.panda
        panda.reset_manip_pose()

        backend = get_backend('jax')
        link_list = panda.rarm.link_list
        move_target = panda.rarm.end_coords

        # Extract FK parameters
        fk_params = extract_fk_parameters(panda, link_list, move_target)

        # Get current joint angles
        joint_angles = np.array([link.joint.joint_angle() for link in link_list])

        # Compute FK with JAX backend
        pos_jax, rot_jax = forward_kinematics_ee(
            backend, backend.array(joint_angles), fk_params
        )

        # Get expected position from robot model
        expected_pos = move_target.worldpos()
        expected_rot = move_target.worldrot()

        # Compare positions (should be very close)
        pos_error = np.linalg.norm(backend.to_numpy(pos_jax) - expected_pos)
        self.assertLess(pos_error, 1e-6, f"Position error too large: {pos_error}")

        # Compare rotations
        rot_error = np.linalg.norm(backend.to_numpy(rot_jax) - expected_rot)
        self.assertLess(rot_error, 1e-6, f"Rotation error too large: {rot_error}")

    @requires_jax
    def test_forward_kinematics_at_different_poses(self):
        """Test FK at multiple joint configurations."""
        from skrobot.backend import get_backend
        from skrobot.kinematics.differentiable import extract_fk_parameters
        from skrobot.kinematics.differentiable import forward_kinematics_ee

        panda = self.panda
        backend = get_backend('jax')
        link_list = panda.rarm.link_list
        move_target = panda.rarm.end_coords

        # Test at different poses
        test_poses = [
            np.zeros(7),  # Zero pose
            np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]),  # Random pose
        ]

        for angles in test_poses:
            # Set robot to this pose
            for i, link in enumerate(link_list):
                # Clip to joint limits
                min_angle = link.joint.min_angle
                max_angle = link.joint.max_angle
                clipped_angle = np.clip(angles[i], min_angle, max_angle)
                link.joint.joint_angle(clipped_angle)

            # Extract FK params (need to re-extract after changing pose)
            fk_params = extract_fk_parameters(panda, link_list, move_target)

            # Get actual joint angles after clipping
            actual_angles = np.array([link.joint.joint_angle() for link in link_list])

            # Compute FK
            pos_jax, rot_jax = forward_kinematics_ee(
                backend, backend.array(actual_angles), fk_params
            )

            # Compare with robot model
            expected_pos = move_target.worldpos()
            pos_error = np.linalg.norm(backend.to_numpy(pos_jax) - expected_pos)
            self.assertLess(pos_error, 1e-6)

    @requires_jax
    def test_jacobian_computation(self):
        """Test Jacobian computation with JAX autodiff."""
        from skrobot.backend import get_backend
        from skrobot.kinematics.differentiable import compute_jacobian
        from skrobot.kinematics.differentiable import extract_fk_parameters

        panda = self.panda
        panda.reset_manip_pose()

        backend = get_backend('jax')
        link_list = panda.rarm.link_list
        move_target = panda.rarm.end_coords

        fk_params = extract_fk_parameters(panda, link_list, move_target)
        joint_angles = np.array([link.joint.joint_angle() for link in link_list])

        # Compute Jacobian
        jacobian = compute_jacobian(backend, backend.array(joint_angles), fk_params)
        jacobian_np = backend.to_numpy(jacobian)

        # Check shape
        self.assertEqual(jacobian_np.shape, (3, 7))

        # Jacobian should not be all zeros
        self.assertGreater(np.linalg.norm(jacobian_np), 0.1)

    @requires_jax
    def test_batch_ik_solver_creation(self):
        """Test that batch IK solver can be created."""
        from skrobot.kinematics.differentiable import create_batch_ik_solver

        panda = self.panda
        panda.reset_manip_pose()

        link_list = panda.rarm.link_list
        move_target = panda.rarm.end_coords

        # Create solver
        solver = create_batch_ik_solver(panda, link_list, move_target, backend_name='jax')

        # Check solver attributes
        self.assertEqual(solver.n_joints, 7)
        self.assertIsNotNone(solver.fk_params)

    @requires_jax
    def test_batch_ik_simple_target(self):
        """Test batch IK with a simple target near current position."""
        from skrobot.kinematics.differentiable import create_batch_ik_solver

        panda = self.panda
        panda.reset_manip_pose()

        link_list = panda.rarm.link_list
        move_target = panda.rarm.end_coords

        solver = create_batch_ik_solver(panda, link_list, move_target, backend_name='jax')

        # Create target at current position (should converge easily)
        current_pos = move_target.worldpos()
        current_rot = move_target.worldrot()

        target_positions = np.array([current_pos])
        target_rotations = np.array([current_rot])

        # Get current joint angles as initial guess
        initial_angles = np.array([[link.joint.joint_angle() for link in link_list]])

        # Solve
        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=50
        )

        # Should converge since we're already at the target
        self.assertTrue(success_flags[0], f"IK failed with error {errors[0]}")
        self.assertLess(errors[0], 0.01)


class TestDifferentiableKinematicsR8_6(unittest.TestCase):
    """Test differentiable kinematics with R8_6 robot (has mimic joints)."""

    @classmethod
    def setUpClass(cls):
        if not HAS_JAX:
            return

        from skrobot.models import R8_6
        cls.r8_6 = R8_6()

    @requires_jax
    def test_forward_kinematics_r8_6(self):
        """Test forward kinematics with R8_6 robot (mimic joints).

        R8_6 has mimic joints in its elbow mechanism. This test verifies
        that the differentiable FK correctly handles mimic joints.
        """
        from skrobot.backend import get_backend
        from skrobot.kinematics.differentiable import extract_fk_parameters
        from skrobot.kinematics.differentiable import forward_kinematics_ee

        r8_6 = self.r8_6
        r8_6.reset_pose()

        backend = get_backend('jax')
        link_list = r8_6.rarm.link_list
        move_target = r8_6.rarm.end_coords

        # Extract FK parameters
        fk_params = extract_fk_parameters(r8_6, link_list, move_target)

        # Get current joint angles
        joint_angles = np.array([link.joint.joint_angle() for link in link_list])

        # Compute FK with JAX backend
        pos_jax, rot_jax = forward_kinematics_ee(
            backend, backend.array(joint_angles), fk_params
        )

        # Get expected position from robot model
        expected_pos = move_target.worldpos()

        # Compare positions
        pos_error = np.linalg.norm(backend.to_numpy(pos_jax) - expected_pos)
        self.assertLess(
            pos_error, 0.01,
            f"Position error too large for R8_6: {pos_error}m. "
            f"JAX pos: {backend.to_numpy(pos_jax)}, Expected: {expected_pos}"
        )

    @requires_jax
    def test_batch_ik_r8_6(self):
        """Test batch IK with R8_6 robot."""
        from skrobot.kinematics.differentiable import create_batch_ik_solver

        r8_6 = self.r8_6
        r8_6.reset_pose()

        link_list = r8_6.rarm.link_list
        move_target = r8_6.rarm.end_coords

        solver = create_batch_ik_solver(r8_6, link_list, move_target, backend_name='jax')

        # Create target at current position
        current_pos = move_target.worldpos()
        current_rot = move_target.worldrot()

        target_positions = np.array([current_pos])
        target_rotations = np.array([current_rot])

        # Get current joint angles as initial guess
        initial_angles = np.array([[link.joint.joint_angle() for link in link_list]])

        # Solve
        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=100
        )

        # Check if converged (may fail if mimic joints not properly handled)
        self.assertLess(
            errors[0], 0.05,
            f"R8_6 IK error too large: {errors[0]}m"
        )


if __name__ == '__main__':
    unittest.main()
