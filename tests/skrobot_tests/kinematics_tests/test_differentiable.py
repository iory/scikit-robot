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

        fk_params = extract_fk_parameters(panda, link_list, move_target)
        joint_angles = np.array([link.joint.joint_angle() for link in link_list])

        pos_jax, rot_jax = forward_kinematics_ee(
            backend, backend.array(joint_angles), fk_params
        )

        expected_pos = move_target.worldpos()
        expected_rot = move_target.worldrot()

        pos_error = np.linalg.norm(backend.to_numpy(pos_jax) - expected_pos)
        self.assertLess(pos_error, 1e-6, f"Position error too large: {pos_error}")

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

        test_poses = [
            np.zeros(7),
            np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]),
        ]

        for angles in test_poses:
            for i, link in enumerate(link_list):
                min_angle = link.joint.min_angle
                max_angle = link.joint.max_angle
                clipped_angle = np.clip(angles[i], min_angle, max_angle)
                link.joint.joint_angle(clipped_angle)

            fk_params = extract_fk_parameters(panda, link_list, move_target)
            actual_angles = np.array([link.joint.joint_angle() for link in link_list])

            pos_jax, rot_jax = forward_kinematics_ee(
                backend, backend.array(actual_angles), fk_params
            )

            expected_pos = move_target.worldpos()
            pos_error = np.linalg.norm(backend.to_numpy(pos_jax) - expected_pos)
            self.assertLess(pos_error, 1e-6)

    @requires_jax
    def test_forward_kinematics_r8_6(self):
        """Test forward kinematics with R8_6 robot (mimic joints)."""
        from skrobot.backend import get_backend
        from skrobot.kinematics.differentiable import extract_fk_parameters
        from skrobot.kinematics.differentiable import forward_kinematics_ee

        r8_6 = self.r8_6
        r8_6.reset_pose()

        backend = get_backend('jax')
        link_list = r8_6.rarm.link_list
        move_target = r8_6.rarm.end_coords

        fk_params = extract_fk_parameters(r8_6, link_list, move_target)
        joint_angles = np.array([link.joint.joint_angle() for link in link_list])

        pos_jax, rot_jax = forward_kinematics_ee(
            backend, backend.array(joint_angles), fk_params
        )

        expected_pos = move_target.worldpos()
        pos_error = np.linalg.norm(backend.to_numpy(pos_jax) - expected_pos)
        self.assertLess(
            pos_error, 0.01,
            f"Position error too large for R8_6: {pos_error}m. "
            f"JAX pos: {backend.to_numpy(pos_jax)}, Expected: {expected_pos}"
        )

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

        jacobian = compute_jacobian(backend, backend.array(joint_angles), fk_params)
        jacobian_np = backend.to_numpy(jacobian)

        self.assertEqual(jacobian_np.shape, (3, 7))
        self.assertGreater(np.linalg.norm(jacobian_np), 0.1)

    @requires_jax
    def test_batch_ik_solver_creation(self):
        """Test that batch IK solver can be created."""
        from skrobot.kinematics.differentiable import create_batch_ik_solver

        panda = self.panda
        panda.reset_manip_pose()

        link_list = panda.rarm.link_list
        move_target = panda.rarm.end_coords

        solver = create_batch_ik_solver(panda, link_list, move_target, backend_name='jax')

        self.assertEqual(solver.n_joints, 7)
        self.assertIsNotNone(solver.fk_params)


class TestDifferentiableBatchIK(unittest.TestCase):
    """Test batch IK solving functionality with various configurations."""

    @classmethod
    def setUpClass(cls):
        if not HAS_JAX:
            return

        from skrobot.models import Panda
        from skrobot.models import R8_6

        cls.panda = Panda()
        cls.r8_6 = R8_6()

    def _get_solver_and_target(self, robot, offset=None):
        """Helper to create solver and target at current or offset position."""
        from skrobot.kinematics.differentiable import create_batch_ik_solver

        link_list = robot.rarm.link_list
        move_target = robot.rarm.end_coords

        solver = create_batch_ik_solver(robot, link_list, move_target, backend_name='jax')

        current_pos = move_target.worldpos()
        current_rot = move_target.worldrot()

        if offset is not None:
            target_pos = current_pos + np.array(offset)
        else:
            target_pos = current_pos

        target_positions = np.array([target_pos])
        target_rotations = np.array([current_rot])

        initial_angles = np.array([[link.joint.joint_angle() for link in link_list]])

        return solver, target_positions, target_rotations, initial_angles

    @requires_jax
    def test_batch_ik_simple_target(self):
        """Test batch IK with a simple target near current position."""
        panda = self.panda
        panda.reset_manip_pose()

        solver, target_positions, target_rotations, initial_angles = \
            self._get_solver_and_target(panda)

        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=50
        )

        self.assertTrue(success_flags[0], f"IK failed with error {errors[0]}")
        self.assertLess(errors[0], 0.01)

    @requires_jax
    def test_batch_ik_r8_6(self):
        """Test batch IK with R8_6 robot (mimic joints)."""
        r8_6 = self.r8_6
        r8_6.reset_pose()

        solver, target_positions, target_rotations, initial_angles = \
            self._get_solver_and_target(r8_6)

        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=100
        )

        self.assertLess(errors[0], 0.05, f"R8_6 IK error too large: {errors[0]}m")

    @requires_jax
    def test_position_mask_and_rotation_mask(self):
        """Test batch IK with various position_mask and rotation_mask combinations."""
        from skrobot.coordinates.math import rotation_matrix_from_rpy

        panda = self.panda
        panda.reset_manip_pose()

        link_list = panda.rarm.link_list
        move_target = panda.rarm.end_coords

        solver, _, _, initial_angles = self._get_solver_and_target(panda)

        position_masks = [True, False, 'x', 'xy', [1, 0, 1]]
        rotation_masks = [True, False, 'x', 'yz']

        current_pos = move_target.worldpos()
        current_rot = move_target.worldrot()
        target_pos = current_pos + np.array([0.02, 0.01, 0.01])
        target_rot = rotation_matrix_from_rpy([0.1, 0.05, 0.1]) @ current_rot

        target_positions = np.array([target_pos])
        target_rotations = np.array([target_rot])

        def mask_to_array(mask):
            if mask is None or mask is False:
                return np.array([0, 0, 0])
            if mask is True:
                return np.array([1, 1, 1])
            if isinstance(mask, str):
                arr = np.array([0, 0, 0])
                if 'x' in mask:
                    arr[0] = 1
                if 'y' in mask:
                    arr[1] = 1
                if 'z' in mask:
                    arr[2] = 1
                return arr
            return np.array(mask)

        for pos_mask in position_masks:
            for rot_mask in rotation_masks:
                panda.reset_manip_pose()

                pos_arr = mask_to_array(pos_mask)
                rot_arr = mask_to_array(rot_mask)
                if np.sum(pos_arr) == 0 and np.sum(rot_arr) == 0:
                    continue

                try:
                    solutions, success_flags, errors = solver(
                        target_positions,
                        target_rotations,
                        initial_angles=initial_angles,
                        max_iterations=150,
                        learning_rate=0.15,
                        rot_weight=0.5,
                        position_mask=pos_mask,
                        rotation_mask=rot_mask,
                    )

                    for i, link in enumerate(link_list):
                        link.joint.joint_angle(solutions[0, i])

                    achieved_pos = move_target.worldpos()
                    achieved_rot = move_target.worldrot()

                    pos_threshold = 0.05 if np.sum(rot_arr) > 0 else 0.02
                    if np.sum(pos_arr) > 0:
                        for axis_idx in range(3):
                            if pos_arr[axis_idx] == 1:
                                axis_error = abs(achieved_pos[axis_idx] - target_pos[axis_idx])
                                self.assertLess(
                                    axis_error, pos_threshold,
                                    f"pos_mask={pos_mask}, rot_mask={rot_mask}, "
                                    f"axis {axis_idx} error: {axis_error}"
                                )

                    if np.sum(rot_arr) > 0 and np.sum(rot_arr) <= 2:
                        for axis_idx in range(3):
                            if rot_arr[axis_idx] == 1:
                                target_axis = target_rot[:, axis_idx]
                                achieved_axis = achieved_rot[:, axis_idx]
                                dot_product = abs(np.dot(target_axis, achieved_axis))
                                axis_error = 1 - dot_product
                                self.assertLess(
                                    axis_error, 0.15,
                                    f"pos_mask={pos_mask}, rot_mask={rot_mask}, "
                                    f"rotation axis {axis_idx} direction error: {axis_error}"
                                )

                except Exception as e:
                    self.fail(f"Failed for pos_mask={pos_mask}, rot_mask={rot_mask}: {e}")

    @requires_jax
    def test_convergence_with_pos_and_rot_threshold(self):
        """Test that success flag considers both position and rotation errors."""
        panda = self.panda
        panda.reset_manip_pose()

        solver, target_positions, target_rotations, initial_angles = \
            self._get_solver_and_target(panda)

        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=50,
            pos_threshold=0.001,
            rot_threshold=0.01,
            position_mask=True,
            rotation_mask=True,
        )

        self.assertTrue(success_flags[0], f"Should succeed at current pose. Error: {errors[0]}")

    @requires_jax
    def test_convergence_position_only(self):
        """Test convergence with position constraint only."""
        panda = self.panda
        panda.reset_manip_pose()

        solver, target_positions, target_rotations, initial_angles = \
            self._get_solver_and_target(panda)

        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=50,
            pos_threshold=0.001,
            rot_threshold=0.01,
            position_mask=True,
            rotation_mask=False,
        )

        self.assertTrue(success_flags[0], f"Should succeed with position only. Error: {errors[0]}")

    @requires_jax
    def test_convergence_rotation_only(self):
        """Test convergence with rotation constraint only."""
        panda = self.panda
        panda.reset_manip_pose()

        solver, target_positions, target_rotations, initial_angles = \
            self._get_solver_and_target(panda)

        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=50,
            pos_threshold=0.001,
            rot_threshold=0.1,
            position_mask=False,
            rotation_mask=True,
        )

        self.assertTrue(success_flags[0], f"Should succeed with rotation only. Error: {errors[0]}")

    @requires_jax
    def test_convergence_single_axis_rotation(self):
        """Test convergence with single-axis rotation constraint."""
        panda = self.panda
        panda.reset_manip_pose()

        solver, target_positions, target_rotations, initial_angles = \
            self._get_solver_and_target(panda)

        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=50,
            pos_threshold=0.001,
            rot_threshold=0.1,
            position_mask=True,
            rotation_mask='z',
        )

        self.assertTrue(
            success_flags[0],
            f"Should succeed with single-axis rotation. Error: {errors[0]}"
        )

    @requires_jax
    def test_rotation_mirror_basic(self):
        """Test that rotation_mirror parameter works."""
        panda = self.panda
        panda.reset_manip_pose()

        solver, target_positions, target_rotations, initial_angles = \
            self._get_solver_and_target(panda)

        for mirror_axis in ['x', 'y', 'z']:
            solutions, success_flags, errors = solver(
                target_positions,
                target_rotations,
                initial_angles=initial_angles,
                max_iterations=50,
                rotation_mirror=mirror_axis,
            )

            self.assertTrue(success_flags[0], f"Should succeed with rotation_mirror='{mirror_axis}'")

    @requires_jax
    def test_rotation_mirror_flipped_target(self):
        """Test that rotation_mirror allows flipped orientation."""
        panda = self.panda
        panda.reset_manip_pose()

        move_target = panda.rarm.end_coords
        solver, _, _, initial_angles = self._get_solver_and_target(panda)

        current_pos = move_target.worldpos()
        current_rot = move_target.worldrot()

        Rx_180 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        flipped_rot = current_rot @ Rx_180

        target_positions = np.array([current_pos])
        target_rotations = np.array([flipped_rot])

        solutions_no_mirror, _, errors_no_mirror = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=100,
            rotation_mirror=None,
        )

        solutions_with_mirror, success_with_mirror, errors_with_mirror = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=100,
            rotation_mirror='x',
        )

        err_mirror = float(errors_with_mirror[0])
        err_no_mirror = float(errors_no_mirror[0])
        # When both errors are near machine epsilon, the comparison is
        # meaningless — both solutions have converged perfectly.
        if err_no_mirror > 1e-6:
            self.assertLess(
                err_mirror,
                err_no_mirror,
                "Error with mirror ({}) should be less than "
                "without mirror ({})".format(err_mirror, err_no_mirror)
            )

    @requires_jax
    def test_rotation_mirror_convergence(self):
        """Test that rotation_mirror helps convergence to mirrored pose."""
        panda = self.panda
        panda.reset_manip_pose()

        move_target = panda.rarm.end_coords
        solver, _, _, initial_angles = self._get_solver_and_target(panda)

        current_pos = move_target.worldpos()
        current_rot = move_target.worldrot()

        Rz_180 = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        flipped_rot = current_rot @ Rz_180

        target_positions = np.array([current_pos])
        target_rotations = np.array([flipped_rot])

        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=50,
            rotation_mirror='z',
            pos_threshold=0.01,
            rot_threshold=0.1,
        )

        self.assertTrue(success_flags[0], f"Should succeed with z-axis mirror. Error: {errors[0]}")

    @requires_jax
    def test_early_stopping_at_solution(self):
        """Test that solver stops early when starting at solution."""
        import time

        panda = self.panda
        panda.reset_manip_pose()

        solver, target_positions, target_rotations, initial_angles = \
            self._get_solver_and_target(panda)

        # Warm up JIT
        _ = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=10,
            pos_threshold=0.001,
            rot_threshold=0.1,
        )

        start = time.time()
        solutions, success, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=1000,
            pos_threshold=0.001,
            rot_threshold=0.1,
        )
        elapsed = time.time() - start

        self.assertTrue(success[0])
        self.assertLess(errors[0], 0.01)
        self.assertLess(elapsed, 5.0, "Should be fast due to early stopping")

    @requires_jax
    def test_attempts_per_pose_basic(self):
        """Test that attempts_per_pose parameter works correctly."""
        panda = self.panda
        panda.reset_manip_pose()

        solver, target_positions, target_rotations, initial_angles = \
            self._get_solver_and_target(panda)

        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=50,
            attempts_per_pose=5,
            use_current_angles=True,
        )

        self.assertEqual(solutions.shape[0], 1)
        self.assertEqual(len(success_flags), 1)
        self.assertEqual(len(errors), 1)
        self.assertLess(errors[0], 0.01)

    @requires_jax
    def test_attempts_per_pose_improves_success(self):
        """Test that multiple attempts improve success rate for difficult targets."""
        panda = self.panda
        panda.reset_manip_pose()

        solver, _, _, _ = self._get_solver_and_target(panda, offset=[0.1, 0.0, 0.0])

        move_target = panda.rarm.end_coords
        current_pos = move_target.worldpos()
        current_rot = move_target.worldrot()

        target_pos = current_pos + np.array([0.1, 0.0, 0.0])
        target_positions = np.array([target_pos])
        target_rotations = np.array([current_rot])

        np.random.seed(42)
        solutions_single, _, errors_single = solver(
            target_positions,
            target_rotations,
            initial_angles=None,
            max_iterations=100,
            attempts_per_pose=1,
        )

        np.random.seed(42)
        solutions_multi, _, errors_multi = solver(
            target_positions,
            target_rotations,
            initial_angles=None,
            max_iterations=100,
            attempts_per_pose=10,
            use_current_angles=False,
        )

        self.assertLessEqual(
            float(errors_multi[0]),
            float(errors_single[0]) + 0.001,
            f"Multi-attempt error {errors_multi[0]} should be <= single-attempt {errors_single[0]}"
        )

    @requires_jax
    def test_attempts_per_pose_with_batch(self):
        """Test attempts_per_pose with multiple targets."""
        panda = self.panda
        panda.reset_manip_pose()

        move_target = panda.rarm.end_coords
        solver, _, _, initial_angles = self._get_solver_and_target(panda)

        current_pos = move_target.worldpos()
        current_rot = move_target.worldrot()

        target_positions = np.array([
            current_pos + np.array([0.02, 0.0, 0.0]),
            current_pos + np.array([0.0, 0.02, 0.0]),
            current_pos + np.array([0.0, 0.0, 0.02]),
        ])
        target_rotations = np.array([current_rot, current_rot, current_rot])

        initial_angles = np.tile(initial_angles, (3, 1))

        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=100,
            attempts_per_pose=3,
            use_current_angles=True,
        )

        self.assertEqual(solutions.shape[0], 3)
        self.assertEqual(len(success_flags), 3)
        self.assertEqual(len(errors), 3)

        for i, err in enumerate(errors):
            self.assertLess(err, 0.05, f"Target {i} error {err} too large")

    @requires_jax
    def test_use_current_angles_flag(self):
        """Test that use_current_angles=True uses provided initial angles first."""
        panda = self.panda
        panda.reset_manip_pose()

        solver, target_positions, target_rotations, initial_angles = \
            self._get_solver_and_target(panda)

        solutions, _, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=10,
            attempts_per_pose=5,
            use_current_angles=True,
        )

        self.assertLess(
            errors[0], 0.001,
            f"Error {errors[0]} too large when starting from solution"
        )


class TestDifferentiableDynamicLimits(unittest.TestCase):
    """Test dynamic joint limit constraints in batch IK."""

    @classmethod
    def setUpClass(cls):
        from skrobot.models import DifferentialWristSample

        cls.robot = DifferentialWristSample(use_joint_limit_table=True)

    def _check_dynamic_limit_violations(self, solutions, fk_params, tolerance=1e-6):
        """Check if any solutions violate dynamic joint limits.

        Parameters
        ----------
        solutions : numpy.ndarray
            Joint angle solutions of shape (n_samples, n_joints).
        fk_params : dict
            FK parameters containing dynamic_limit_tables.
        tolerance : float
            Numerical tolerance for limit checking.

        Returns
        -------
        violations : list
            List of dicts with violation details.
        """
        dynamic_tables = fk_params.get('dynamic_limit_tables', [])
        violations = []

        for table in dynamic_tables:
            dep_idx = table['dependent_joint_index']
            tgt_idx = table['target_joint_index']
            sample_angles = table['sample_angles']
            min_angles = table['min_angles']
            max_angles = table['max_angles']

            target_angles = solutions[:, tgt_idx]
            dependent_angles = solutions[:, dep_idx]

            dynamic_min = np.interp(target_angles, sample_angles, min_angles)
            dynamic_max = np.interp(target_angles, sample_angles, max_angles)

            below_min = np.where(dependent_angles < dynamic_min - tolerance)[0]
            above_max = np.where(dependent_angles > dynamic_max + tolerance)[0]

            for i in below_min:
                violations.append({
                    'sample_idx': i,
                    'violation_amount': dynamic_min[i] - dependent_angles[i],
                })
            for i in above_max:
                violations.append({
                    'sample_idx': i,
                    'violation_amount': dependent_angles[i] - dynamic_max[i],
                })

        return violations

    def _run_batch_ik_dynamic_limits_test(self, backend_name):
        """Run batch IK test for dynamic limits with specified backend."""
        from skrobot.kinematics.differentiable import create_batch_ik_solver
        from skrobot.kinematics.differentiable import create_dynamic_limit_mask
        from skrobot.kinematics.differentiable import extract_fk_parameters

        robot = self.robot
        robot.reset_manip_pose()

        link_list = [
            robot.ARM_LINK0, robot.ARM_LINK1, robot.ARM_LINK2,
            robot.ARM_LINK3, robot.ARM_LINK4,
            robot.WRIST_GEAR, robot.WRIST_END
        ]
        move_target = robot.end_coords

        fk_params = extract_fk_parameters(robot, link_list, move_target)

        self.assertGreater(
            len(fk_params['dynamic_limit_tables']), 0,
            "Robot should have dynamic limit tables"
        )

        solver = create_batch_ik_solver(
            robot, link_list, move_target, backend_name=backend_name)

        n_samples = 200
        np.random.seed(42)

        robot.reset_manip_pose()
        base_pos = robot.end_coords.worldpos()

        target_positions = base_pos + np.random.uniform(-0.1, 0.1, size=(n_samples, 3))
        target_rotations = np.array([
            robot.end_coords.worldrot() for _ in range(n_samples)
        ])

        initial_angles = np.tile(
            robot.angle_vector()[None, :],
            (n_samples, 1)
        )

        solutions, success_flags, errors = solver(
            target_positions,
            target_rotations,
            initial_angles=initial_angles,
            max_iterations=50,
        )

        if hasattr(solutions, 'block_until_ready'):
            solutions = np.asarray(solutions)
            success_flags = np.asarray(success_flags)

        successful_solutions = solutions[success_flags]

        if len(successful_solutions) == 0:
            self.skipTest("No successful IK solutions to test")

        violations = self._check_dynamic_limit_violations(
            successful_solutions, fk_params)

        self.assertEqual(
            len(violations), 0,
            f"Found {len(violations)} dynamic limit violations "
            f"in {len(successful_solutions)} successful solutions"
        )

        mask = create_dynamic_limit_mask(fk_params, successful_solutions)
        self.assertTrue(
            np.all(mask),
            f"create_dynamic_limit_mask found {np.sum(~mask)} invalid solutions"
        )

    def test_batch_ik_respects_dynamic_limits_numpy(self):
        """Test that NumPy batch IK solutions respect dynamic joint limits."""
        self._run_batch_ik_dynamic_limits_test('numpy')

    @requires_jax
    def test_batch_ik_respects_dynamic_limits_jax(self):
        """Test that JAX batch IK solutions respect dynamic joint limits."""
        self._run_batch_ik_dynamic_limits_test('jax')


if __name__ == '__main__':
    unittest.main()
