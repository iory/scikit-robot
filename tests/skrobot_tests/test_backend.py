"""Tests for skrobot.backend module."""

import unittest

import numpy as np

from skrobot.coordinates import Coordinates


class TestBackendKinematics(unittest.TestCase):
    """Tests for backend kinematics functions."""

    def test_extract_fk_parameters(self):
        """Test FK parameter extraction."""
        from skrobot.kinematics.differentiable import extract_fk_parameters
        from skrobot.models.panda import Panda

        robot = Panda()
        link_list = robot.rarm.link_list
        move_target = robot.rarm_end_coords

        fk_params = extract_fk_parameters(robot, link_list, move_target)

        self.assertEqual(fk_params['n_joints'], 7)
        self.assertEqual(fk_params['link_translations'].shape, (7, 3))
        self.assertEqual(fk_params['link_rotations'].shape, (7, 3, 3))
        self.assertEqual(fk_params['joint_axes'].shape, (7, 3))
        self.assertEqual(len(fk_params['joint_limits_lower']), 7)
        self.assertEqual(len(fk_params['joint_limits_upper']), 7)

    def test_scipy_ik_solver(self):
        """Test scipy-based IK solver."""
        from skrobot.kinematics.differentiable import solve_ik_scipy
        from skrobot.models.panda import Panda

        robot = Panda()
        link_list = robot.rarm.link_list
        move_target = robot.rarm_end_coords

        # Get current position and add small offset
        current_pos = move_target.worldpos()
        target = Coordinates(pos=current_pos + np.array([0.02, 0.02, -0.02]))
        target.rotation = move_target.worldrot().copy()

        result = solve_ik_scipy(
            robot, target, link_list, move_target,
            max_iterations=100, pos_threshold=0.01
        )

        self.assertIsNotNone(result)
        self.assertFalse(result is False)
        self.assertEqual(len(result), 7)


class TestBackendDynamics(unittest.TestCase):
    """Tests for backend dynamics functions."""

    def test_extract_inverse_dynamics_parameters(self):
        """Test inverse dynamics parameter extraction."""
        from skrobot.dynamics import extract_inverse_dynamics_parameters
        from skrobot.models import Kuka

        robot = Kuka()

        # Get arm links (7 DOF)
        link_list = [
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_1'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_2'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_3'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_4'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_5'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_6'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_7'],
        ]

        params = extract_inverse_dynamics_parameters(robot, link_list=link_list)

        self.assertEqual(params['n_joints'], 7)
        self.assertEqual(params['link_translations'].shape, (7, 3))
        self.assertEqual(params['link_rotations'].shape, (7, 3, 3))
        self.assertEqual(params['joint_axes'].shape, (7, 3))
        self.assertEqual(params['link_masses'].shape, (7,))
        self.assertEqual(params['link_coms'].shape, (7, 3))
        self.assertEqual(params['link_inertias'].shape, (7, 3, 3))

    def test_build_inverse_dynamics_fn_numpy(self):
        """Test inverse dynamics function with NumPy backend."""
        from skrobot.backend import get_backend
        from skrobot.dynamics import build_inverse_dynamics_fn
        from skrobot.models import Kuka

        robot = Kuka()
        link_list = [
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_1'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_2'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_3'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_4'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_5'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_6'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_7'],
        ]

        backend = get_backend('numpy')
        id_fn = build_inverse_dynamics_fn(robot, link_list=link_list, backend=backend)

        # Test at zero position
        q = np.zeros(7)
        tau = id_fn(q)
        self.assertEqual(tau.shape, (7,))

        # Test at non-zero position
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        tau = id_fn(q)
        self.assertEqual(tau.shape, (7,))

        # Test with different gravity
        tau_pos_gravity = id_fn(q, gravity=backend.array([0.0, 0.0, 9.81]))
        tau_neg_gravity = id_fn(q, gravity=backend.array([0.0, 0.0, -9.81]))
        # Gravity in opposite direction should give opposite torques
        np.testing.assert_allclose(tau_pos_gravity, -tau_neg_gravity, rtol=1e-5)

    def test_inverse_dynamics_matches_existing(self):
        """Test that build_inverse_dynamics_fn matches robot.torque_vector().

        Both should use the same underlying differentiable implementation.
        """
        from skrobot.backend import get_backend
        from skrobot.dynamics import build_inverse_dynamics_fn
        from skrobot.models import Kuka

        robot = Kuka()
        # Use all joints from the robot for fair comparison
        link_list = [j.child_link for j in robot.joint_list if j is not None]
        n_joints = len(link_list)

        backend = get_backend('numpy')
        id_fn = build_inverse_dynamics_fn(
            robot, link_list=link_list, backend=backend,
            include_all_mass_links=True
        )

        # Use explicit gravity to ensure consistency
        gravity = np.array([0, 0, -9.80665])

        # Test with multiple configurations
        test_configs = [
            np.zeros(n_joints),
            np.random.RandomState(42).uniform(-0.5, 0.5, n_joints),
            np.random.RandomState(123).uniform(-1.0, 1.0, n_joints),
        ]

        for q in test_configs:
            # Compute torques with build_inverse_dynamics_fn (explicit gravity)
            tau_new = id_fn(q, gravity=gravity)

            # Compute torques with robot.torque_vector (uses same gravity default)
            tau_existing = robot.torque_vector(av=q, gravity=gravity)

            # Verify they match exactly (same implementation)
            np.testing.assert_allclose(
                tau_new, tau_existing, rtol=1e-10, atol=1e-12,
                err_msg=f"Torques don't match for q={q}"
            )

    def test_build_torque_vector_fn(self):
        """Test torque vector function."""
        from skrobot.backend import get_backend
        from skrobot.dynamics import build_torque_vector_fn
        from skrobot.models import Kuka

        robot = Kuka()
        link_list = [
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_1'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_2'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_3'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_4'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_5'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_6'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_7'],
        ]

        backend = get_backend('numpy')
        torque_fn = build_torque_vector_fn(robot, link_list=link_list, backend=backend)

        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        tau = torque_fn(q)
        self.assertEqual(tau.shape, (7,))


class TestBackendDynamicsJax(unittest.TestCase):
    """Tests for JAX backend dynamics functions."""

    @classmethod
    def setUpClass(cls):
        """Check if JAX is available and compatible with current NumPy."""
        try:
            import jax
            import jax.numpy as jnp

            # Test JIT computation to catch runtime incompatibilities
            # JAX 0.9+ requires NumPy 2.0+ and fails inside JIT traced context
            # with "asarray() got an unexpected keyword argument 'copy'"
            @jax.jit
            def _test_fn(x):
                return jnp.eye(3) @ x

            _ = _test_fn(jnp.array([1.0, 2.0, 3.0]))
            cls.jax_available = True
        except Exception:
            # JAX may fail to import or run due to NumPy version incompatibility
            cls.jax_available = False

    def test_build_inverse_dynamics_fn_jax(self):
        """Test inverse dynamics function with JAX backend."""
        if not self.jax_available:
            self.skipTest("JAX not available")

        import jax.numpy as jnp

        from skrobot.backend import get_backend
        from skrobot.dynamics import build_inverse_dynamics_fn
        from skrobot.models import Kuka

        robot = Kuka()
        link_list = [
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_1'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_2'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_3'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_4'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_5'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_6'],
            robot.__dict__['lbr_iiwa_with_wsg50__lbr_iiwa_link_7'],
        ]

        backend = get_backend('jax')
        id_fn = build_inverse_dynamics_fn(robot, link_list=link_list, backend=backend)

        # Test basic computation
        q = jnp.zeros(7)
        tau = id_fn(q)
        self.assertEqual(tau.shape, (7,))

        # Test autodiff
        import jax

        def torque_sum_sq(q):
            tau = id_fn(q)
            return jnp.sum(tau ** 2)

        grad_fn = jax.grad(torque_sum_sq)
        q_test = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        grad = grad_fn(q_test)
        self.assertEqual(grad.shape, (7,))


if __name__ == '__main__':
    unittest.main()
