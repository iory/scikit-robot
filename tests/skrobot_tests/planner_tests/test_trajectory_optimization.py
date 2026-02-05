import os
import unittest

import numpy as np
from numpy import testing


os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import skrobot  # noqa: E402
from skrobot.planner.trajectory_optimization.collision import create_self_collision_pairs  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import compute_collision_residuals  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import compute_sphere_obstacle_distances  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import prepare_fk_data  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import rotation_error_vector  # noqa: E402
from skrobot.planner.trajectory_optimization.problem import TrajectoryProblem  # noqa: E402
from skrobot.planner.trajectory_optimization.trajectory import interpolate_trajectory  # noqa: E402
from skrobot.pycompat import HAS_JAX  # noqa: E402


HAS_JAXLS = False
if HAS_JAX:
    try:
        import jaxls  # noqa: F401
        HAS_JAXLS = True
    except ImportError:
        pass

requires_jax = unittest.skipUnless(HAS_JAX, 'JAX is required')
requires_jaxls = unittest.skipUnless(
    HAS_JAX and HAS_JAXLS, 'JAX and jaxls are required')


def _make_kuka():
    """Create a Kuka robot and return robot, link_list, n_joints."""
    robot = skrobot.models.Kuka()
    robot.reset_manip_pose()
    link_list = robot.rarm.link_list
    n_joints = len(link_list)
    return robot, link_list, n_joints


class TestInterpolateTrajectory(unittest.TestCase):

    def test_endpoints(self):
        start = np.array([0.0, 1.0, 2.0])
        end = np.array([1.0, 2.0, 3.0])
        traj = interpolate_trajectory(start, end, 5)
        testing.assert_almost_equal(traj[0], start)
        testing.assert_almost_equal(traj[-1], end)

    def test_shape(self):
        start = np.zeros(4)
        end = np.ones(4)
        traj = interpolate_trajectory(start, end, 10)
        self.assertEqual(traj.shape, (10, 4))

    def test_midpoint(self):
        start = np.array([0.0, 0.0])
        end = np.array([2.0, 4.0])
        traj = interpolate_trajectory(start, end, 3)
        expected_mid = (start + end) / 2.0
        testing.assert_almost_equal(traj[1], expected_mid)

    def test_two_waypoints(self):
        start = np.array([1.0, 2.0, 3.0])
        end = np.array([4.0, 5.0, 6.0])
        traj = interpolate_trajectory(start, end, 2)
        self.assertEqual(traj.shape, (2, 3))
        testing.assert_almost_equal(traj[0], start)
        testing.assert_almost_equal(traj[1], end)


class TestRotationErrorVector(unittest.TestCase):

    def test_identity_gives_zero(self):
        R = np.eye(3)
        err = rotation_error_vector(R, R, np)
        testing.assert_almost_equal(err, np.zeros(3))

    def test_known_rotation(self):
        theta = 0.1
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        err = rotation_error_vector(Rz, np.eye(3), np)
        # R_err = Rz @ I^T = Rz
        # err[0] = R[1,0] - R[0,1] = sin(theta) - (-sin(theta)) = 2*sin(theta)
        # err[1] = R[2,0] - R[0,2] = 0
        # err[2] = R[2,1] - R[1,2] = 0
        expected = np.array([2 * np.sin(theta), 0.0, 0.0])
        testing.assert_almost_equal(err, expected, decimal=10)

    def test_antisymmetry(self):
        rng = np.random.RandomState(42)
        A, _ = np.linalg.qr(rng.randn(3, 3))
        B, _ = np.linalg.qr(rng.randn(3, 3))
        # Ensure proper rotation matrices (det = +1)
        if np.linalg.det(A) < 0:
            A[:, 0] *= -1
        if np.linalg.det(B) < 0:
            B[:, 0] *= -1
        err_a_b = rotation_error_vector(A, B, np)
        err_b_a = rotation_error_vector(B, A, np)
        testing.assert_almost_equal(err_a_b, -err_b_a, decimal=10)


class TestCollisionUtils(unittest.TestCase):

    def test_self_collision_pairs(self):
        # 4 dummy links: indices 0,1,2,3
        # Pairs skipping adjacent: (0,2), (0,3), (1,3)
        dummy_links = [None, None, None, None]
        pairs = create_self_collision_pairs(dummy_links, ignore_adjacent=True)
        expected = [(0, 2), (0, 3), (1, 3)]
        self.assertEqual(pairs, expected)

    def test_sphere_obstacle_distances(self):
        sphere_positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        sphere_radii = np.array([0.1, 0.1])
        obstacle_centers = np.array([[0.5, 0.0, 0.0]])
        obstacle_radii = np.array([0.1])

        dists = compute_sphere_obstacle_distances(
            sphere_positions, sphere_radii,
            obstacle_centers, obstacle_radii, np)

        # sphere0 -> obs: |0.5| - 0.1 - 0.1 = 0.3
        # sphere1 -> obs: |0.5| - 0.1 - 0.1 = 0.3
        self.assertEqual(dists.shape, (2, 1))
        # Allow small epsilon from sqrt(... + 1e-10)
        testing.assert_almost_equal(dists[0, 0], 0.3, decimal=4)
        testing.assert_almost_equal(dists[1, 0], 0.3, decimal=4)

    def test_collision_residuals(self):
        signed_dists = np.array([0.1, 0.03, -0.05])
        activation = 0.05
        residuals = compute_collision_residuals(signed_dists, activation, np)
        # max(0, 0.05 - 0.1) = 0
        # max(0, 0.05 - 0.03) = 0.02
        # max(0, 0.05 - (-0.05)) = 0.1
        expected = np.array([0.0, 0.02, 0.1])
        testing.assert_almost_equal(residuals, expected)


class TestTrajectoryProblem(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.robot, cls.link_list, cls.n_joints = _make_kuka()

    def test_initialization(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        self.assertEqual(problem.n_joints, self.n_joints)
        self.assertEqual(problem.n_waypoints, 3)

    def test_add_smoothness_cost(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_smoothness_cost(weight=2.0)
        self.assertEqual(len(problem.residuals), 1)
        self.assertEqual(problem.residuals[0].name, 'smoothness')
        self.assertEqual(problem.residuals[0].weight, 2.0)

    def test_add_joint_limit_constraint(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_joint_limit_constraint()
        self.assertEqual(len(problem.residuals), 1)
        self.assertEqual(problem.residuals[0].kind, 'geq')

    def test_set_fixed_endpoints(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.set_fixed_endpoints(start=False, end=True)
        self.assertFalse(problem.fixed_start)
        self.assertTrue(problem.fixed_end)

    def test_add_waypoint_constraint(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=5)
        angles = np.zeros(self.n_joints)
        problem.add_waypoint_constraint(2, angles)
        self.assertEqual(len(problem.waypoint_constraints), 1)
        self.assertEqual(problem.waypoint_constraints[0][0], 2)
        testing.assert_almost_equal(
            problem.waypoint_constraints[0][1], angles)

    def test_add_posture_cost(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        nominal = np.zeros(self.n_joints)
        problem.add_posture_cost(nominal, weight=0.5)
        self.assertEqual(len(problem.residuals), 1)
        self.assertEqual(problem.residuals[0].name, 'posture')
        self.assertEqual(problem.residuals[0].weight, 0.5)
        testing.assert_almost_equal(
            problem.residuals[0].params['nominal_angles'], nominal)

    def test_add_ee_waypoint_cost(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=5,
            move_target=self.robot.rarm_end_coords)
        target_pos = np.array([0.5, 0.0, 0.5])
        target_rot = np.eye(3)
        problem.add_ee_waypoint_cost(
            2, target_pos, target_rot,
            position_weight=100.0, rotation_weight=10.0)
        self.assertEqual(len(problem.ee_waypoint_costs), 1)
        self.assertEqual(problem.ee_waypoint_costs[0]['waypoint_index'], 2)
        testing.assert_almost_equal(
            problem.ee_waypoint_costs[0]['target_position'], target_pos)

    def test_add_cartesian_path_cost(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3,
            move_target=self.robot.rarm_end_coords)
        target_pos = np.zeros((3, 3))
        problem.add_cartesian_path_cost(
            target_pos, rotation_weight=0.5, weight=5.0)
        spec = problem.residuals[0]
        self.assertEqual(spec.name, 'cartesian_path')
        self.assertEqual(spec.params['rotation_weight'], 0.5)
        self.assertEqual(spec.weight, 5.0)

    def test_fk_params(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3,
            move_target=self.robot.rarm_end_coords)
        fk = problem.fk_params
        required_keys = [
            'link_translations', 'link_rotations', 'joint_axes',
            'base_position', 'base_rotation', 'n_joints',
            'ee_offset_position', 'ee_offset_rotation', 'ref_angles',
        ]
        for key in required_keys:
            self.assertIn(key, fk, msg=f"Missing key: {key}")


class TestFKUtils(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.robot, cls.link_list, cls.n_joints = _make_kuka()

    def test_link_transforms_match_robot(self):
        rng = np.random.RandomState(123)
        angles = rng.uniform(-0.5, 0.5, self.n_joints)

        # Set robot to test angles
        for link, angle in zip(self.link_list, angles):
            link.joint.joint_angle(angle)

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3,
            move_target=self.robot.rarm_end_coords)
        fk_data = prepare_fk_data(problem, np)
        get_link_transforms, _, _, _ = build_fk_functions(fk_data, np)

        positions, rotations = get_link_transforms(angles)

        for i, link in enumerate(self.link_list):
            expected_pos = link.worldpos()
            testing.assert_almost_equal(
                positions[i], expected_pos, decimal=4,
                err_msg=f"Position mismatch at link {i}")

    def test_ee_pose_matches_robot(self):
        rng = np.random.RandomState(456)
        angles = rng.uniform(-0.5, 0.5, self.n_joints)

        for link, angle in zip(self.link_list, angles):
            link.joint.joint_angle(angle)

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3,
            move_target=self.robot.rarm_end_coords)
        fk_data = prepare_fk_data(problem, np)
        _, _, _, get_ee_pose = build_fk_functions(fk_data, np)

        ee_pos, ee_rot = get_ee_pose(angles)

        expected_pos = self.robot.rarm_end_coords.worldpos()
        expected_rot = self.robot.rarm_end_coords.worldrot()

        testing.assert_almost_equal(ee_pos, expected_pos, decimal=4)
        testing.assert_almost_equal(ee_rot, expected_rot, decimal=4)

    def test_prepare_fk_data_keys(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3,
            move_target=self.robot.rarm_end_coords)
        fk_data = prepare_fk_data(problem, np)
        required_keys = [
            'link_translations', 'link_rotations', 'joint_axes',
            'base_position', 'base_rotation', 'n_joints',
            'ee_offset_position', 'ee_offset_rotation', 'ref_angles',
        ]
        for key in required_keys:
            self.assertIn(key, fk_data, msg=f"Missing key: {key}")


class TestScipySolver(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.robot, cls.link_list, cls.n_joints = _make_kuka()

    def _make_simple_problem(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_smoothness_cost(weight=1.0)
        problem.add_joint_limit_constraint()
        return problem

    def test_smoothness_only(self):
        from skrobot.planner.trajectory_optimization.solvers.scipy_solver import ScipySolver

        problem = self._make_simple_problem()
        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.3
        initial_traj = interpolate_trajectory(start, end, 3)

        solver = ScipySolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        self.assertTrue(result.success)
        self.assertEqual(result.trajectory.shape, (3, self.n_joints))

    def test_preserves_endpoints(self):
        from skrobot.planner.trajectory_optimization.solvers.scipy_solver import ScipySolver

        problem = self._make_simple_problem()
        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.3
        initial_traj = interpolate_trajectory(start, end, 3)

        solver = ScipySolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        testing.assert_almost_equal(result.trajectory[0], start, decimal=4)
        testing.assert_almost_equal(result.trajectory[-1], end, decimal=4)

    def test_joint_limits(self):
        from skrobot.planner.trajectory_optimization.solvers.scipy_solver import ScipySolver

        problem = self._make_simple_problem()
        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.3
        initial_traj = interpolate_trajectory(start, end, 3)

        solver = ScipySolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        lower = problem.joint_limits_lower
        upper = problem.joint_limits_upper
        for t in range(result.trajectory.shape[0]):
            self.assertTrue(
                np.all(result.trajectory[t] >= lower - 1e-6),
                msg=f"Joint limits lower violated at waypoint {t}")
            self.assertTrue(
                np.all(result.trajectory[t] <= upper + 1e-6),
                msg=f"Joint limits upper violated at waypoint {t}")


@requires_jax
class TestGradientDescentSolver(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.robot, cls.link_list, cls.n_joints = _make_kuka()

    def test_smoothness_convergence(self):
        from skrobot.planner.trajectory_optimization.solvers.gradient_descent import GradientDescentSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_smoothness_cost(weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.5
        # Create a non-smooth initial trajectory
        initial_traj = interpolate_trajectory(start, end, 3)
        initial_traj[1] += 0.3  # perturb middle

        solver = GradientDescentSolver(max_iterations=200, learning_rate=0.01)
        result = solver.solve(problem, initial_traj)

        # After optimization, trajectory should be smoother
        diff_before = np.sum(
            (initial_traj[1:] - initial_traj[:-1]) ** 2)
        diff_after = np.sum(
            (result.trajectory[1:] - result.trajectory[:-1]) ** 2)
        self.assertLess(diff_after, diff_before)

    def test_endpoints_preserved(self):
        from skrobot.planner.trajectory_optimization.solvers.gradient_descent import GradientDescentSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_smoothness_cost(weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.3
        initial_traj = interpolate_trajectory(start, end, 3)

        solver = GradientDescentSolver(max_iterations=100)
        result = solver.solve(problem, initial_traj)

        testing.assert_almost_equal(result.trajectory[0], start, decimal=4)
        testing.assert_almost_equal(result.trajectory[-1], end, decimal=4)

    def test_cartesian_path(self):
        from skrobot.planner.trajectory_optimization.solvers.gradient_descent import GradientDescentSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3,
            move_target=self.robot.rarm_end_coords)
        problem.add_smoothness_cost(weight=0.1)

        # Get current EE position and create target path
        fk_data = prepare_fk_data(problem, np)
        _, _, _, get_ee_pose = build_fk_functions(fk_data, np)

        start_angles = np.zeros(self.n_joints)
        end_angles = np.ones(self.n_joints) * 0.2
        ee_start, _ = get_ee_pose(start_angles)
        ee_end, _ = get_ee_pose(end_angles)

        target_positions = np.stack([
            ee_start,
            (ee_start + ee_end) / 2.0,
            ee_end,
        ])

        problem.add_cartesian_path_cost(
            target_positions, weight=10.0)

        initial_traj = interpolate_trajectory(
            start_angles, end_angles, 3)
        solver = GradientDescentSolver(max_iterations=200, learning_rate=0.001)
        result = solver.solve(problem, initial_traj)

        # Compute EE position errors after optimization
        total_err = 0.0
        for t in range(3):
            ee_pos, _ = get_ee_pose(result.trajectory[t])
            total_err += np.sum((np.array(ee_pos) - target_positions[t]) ** 2)

        # Compute EE position errors before optimization
        total_err_before = 0.0
        for t in range(3):
            ee_pos, _ = get_ee_pose(initial_traj[t])
            total_err_before += np.sum(
                (np.array(ee_pos) - target_positions[t]) ** 2)

        self.assertLess(total_err, total_err_before)


@requires_jaxls
class TestJaxlsSolver(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.robot, cls.link_list, cls.n_joints = _make_kuka()

    def test_smoothness_convergence(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_smoothness_cost(weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.5
        initial_traj = interpolate_trajectory(start, end, 3)
        initial_traj[1] += 0.3

        solver = JaxlsSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        diff_before = np.sum(
            (initial_traj[1:] - initial_traj[:-1]) ** 2)
        diff_after = np.sum(
            (result.trajectory[1:] - result.trajectory[:-1]) ** 2)
        self.assertLess(diff_after, diff_before)

    def test_endpoints_preserved(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_smoothness_cost(weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.3
        initial_traj = interpolate_trajectory(start, end, 3)

        solver = JaxlsSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        testing.assert_almost_equal(result.trajectory[0], start, decimal=4)
        testing.assert_almost_equal(result.trajectory[-1], end, decimal=4)

    def test_waypoint_constraint(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_smoothness_cost(weight=1.0)

        mid_angles = np.ones(self.n_joints) * 0.2
        problem.add_waypoint_constraint(1, mid_angles)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.5
        initial_traj = interpolate_trajectory(start, end, 3)

        solver = JaxlsSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        testing.assert_almost_equal(
            result.trajectory[1], mid_angles, decimal=3)

    def test_posture_cost(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=5)
        problem.add_smoothness_cost(weight=0.1)

        # Set nominal angles to zeros
        nominal = np.zeros(self.n_joints)
        problem.add_posture_cost(nominal, weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.3
        initial_traj = interpolate_trajectory(start, end, 5)
        # Perturb middle waypoints away from nominal
        initial_traj[1] += 0.5
        initial_traj[2] += 0.5
        initial_traj[3] += 0.5

        solver = JaxlsSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        # After optimization, middle waypoints should be closer to nominal
        # than the perturbed initial trajectory
        deviation_before = np.sum(
            (initial_traj[1:-1] - nominal) ** 2)
        deviation_after = np.sum(
            (result.trajectory[1:-1] - nominal) ** 2)
        self.assertLess(deviation_after, deviation_before)

    def test_ee_waypoint_cost(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        move_target = self.robot.rarm_end_coords
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=5,
            move_target=move_target)
        problem.add_smoothness_cost(weight=0.1)

        # Compute EE pose at a target configuration
        target_angles = np.ones(self.n_joints) * 0.3
        for link, angle in zip(self.link_list, target_angles):
            link.joint.joint_angle(angle)
        target_pos = move_target.worldpos().copy()
        target_rot = move_target.worldrot().copy()

        # Add EE waypoint cost at middle waypoint
        problem.add_ee_waypoint_cost(
            2, target_pos, target_rot,
            position_weight=100.0, rotation_weight=10.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.5
        initial_traj = interpolate_trajectory(start, end, 5)

        solver = JaxlsSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        # Verify EE pose at waypoint 2 is close to target
        for link, angle in zip(
            self.link_list, result.trajectory[2]
        ):
            link.joint.joint_angle(angle)
        result_pos = move_target.worldpos()

        pos_err = np.linalg.norm(result_pos - target_pos)
        self.assertLess(pos_err, 0.05)

    def test_caching(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_smoothness_cost(weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.3
        initial_traj = interpolate_trajectory(start, end, 3)

        solver = JaxlsSolver(max_iterations=20)

        # First solve builds cache
        solver.solve(problem, initial_traj)
        cache_key_after_first = solver._cache_key

        # Second solve should use cache
        end2 = np.ones(self.n_joints) * 0.4
        initial_traj2 = interpolate_trajectory(start, end2, 3)
        solver.solve(problem, initial_traj2)

        self.assertEqual(solver._cache_key, cache_key_after_first)
        self.assertIsNotNone(solver._cached_problem)


@requires_jax
class TestAugmentedLagrangianSolver(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.robot, cls.link_list, cls.n_joints = _make_kuka()

    def test_smoothness_convergence(self):
        from skrobot.planner.trajectory_optimization.solvers.augmented_lagrangian import AugmentedLagrangianSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_smoothness_cost(weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.5
        initial_traj = interpolate_trajectory(start, end, 3)
        initial_traj[1] += 0.3

        solver = AugmentedLagrangianSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        diff_before = np.sum(
            (initial_traj[1:] - initial_traj[:-1]) ** 2)
        diff_after = np.sum(
            (result.trajectory[1:] - result.trajectory[:-1]) ** 2)
        self.assertLess(diff_after, diff_before)

    def test_endpoints_preserved(self):
        from skrobot.planner.trajectory_optimization.solvers.augmented_lagrangian import AugmentedLagrangianSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_smoothness_cost(weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.3
        initial_traj = interpolate_trajectory(start, end, 3)

        solver = AugmentedLagrangianSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        testing.assert_almost_equal(result.trajectory[0], start, decimal=4)
        testing.assert_almost_equal(result.trajectory[-1], end, decimal=4)

    def test_joint_limit_constraint(self):
        from skrobot.planner.trajectory_optimization.solvers.augmented_lagrangian import AugmentedLagrangianSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=5)
        problem.add_smoothness_cost(weight=1.0)
        problem.add_joint_limit_constraint()

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.3
        initial_traj = interpolate_trajectory(start, end, 5)

        solver = AugmentedLagrangianSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        lower = problem.joint_limits_lower
        upper = problem.joint_limits_upper
        for t in range(result.trajectory.shape[0]):
            self.assertTrue(
                np.all(result.trajectory[t] >= lower - 1e-4),
                msg=f"Joint limits lower violated at waypoint {t}")
            self.assertTrue(
                np.all(result.trajectory[t] <= upper + 1e-4),
                msg=f"Joint limits upper violated at waypoint {t}")

    def test_caching(self):
        from skrobot.planner.trajectory_optimization.solvers.augmented_lagrangian import AugmentedLagrangianSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=3)
        problem.add_smoothness_cost(weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.3
        initial_traj = interpolate_trajectory(start, end, 3)

        solver = AugmentedLagrangianSolver(max_iterations=20)

        # First solve builds cache
        solver.solve(problem, initial_traj)
        cache_keys_after_first = list(solver._jit_cache.keys())

        # Second solve with same structure should reuse cache
        end2 = np.ones(self.n_joints) * 0.4
        initial_traj2 = interpolate_trajectory(start, end2, 3)
        solver.solve(problem, initial_traj2)

        cache_keys_after_second = list(solver._jit_cache.keys())
        self.assertEqual(cache_keys_after_first, cache_keys_after_second)


if __name__ == '__main__':
    unittest.main()
