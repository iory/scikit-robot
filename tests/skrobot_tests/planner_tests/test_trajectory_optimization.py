import os
import unittest

import numpy as np
from numpy import testing


os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import skrobot  # noqa: E402
from skrobot.planner.trajectory_optimization.collision import create_self_collision_pairs  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import compute_collision_residuals  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import compute_self_collision_distances  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import compute_sphere_obstacle_distances  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import prepare_fk_data  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import rotation_error_vector  # noqa: E402
from skrobot.planner.trajectory_optimization.gridsdf_collision import build_gridsdf_self_data  # noqa: E402
from skrobot.planner.trajectory_optimization.gridsdf_collision import gridsdf_self_distances  # noqa: E402
from skrobot.planner.trajectory_optimization.problem import TrajectoryProblem  # noqa: E402
from skrobot.planner.trajectory_optimization.trajectory import interpolate_trajectory  # noqa: E402
from skrobot.pycompat import HAS_JAX  # noqa: E402


HAS_JAXLS = False
if HAS_JAX:
    try:
        import jaxls  # noqa: F401
        if hasattr(jaxls, 'LeastSquaresProblem'):
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


def _make_panda():
    """Create a Panda and return robot, joint links and collision links.

    The fingers are left out of the collision link list: they are physically
    adjacent to the hand but far apart in the list, and
    ``create_self_collision_pairs`` only knows about list adjacency, so they
    would show up as a permanent (bogus) contact.
    """
    robot = skrobot.models.Panda()
    robot.reset_manip_pose()
    link_list = [link for link in robot.link_list
                 if link.joint is not None
                 and link.joint.__class__.__name__ != 'FixedJoint']
    collision_link_list = [link for link in robot.link_list
                           if link.collision_mesh is not None
                           and 'finger' not in link.name]
    return robot, link_list, collision_link_list


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

    def test_five_point_velocity_cost(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=10, dt=0.1)
        problem.add_smoothness_cost(weight=0.1)
        problem.add_five_point_velocity_cost(weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.5
        initial_traj = interpolate_trajectory(start, end, 10)
        # Add perturbation that creates high velocity
        initial_traj[3] += 0.8
        initial_traj[4] -= 0.5

        solver = JaxlsSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        # After optimization, velocities should be smoother
        # Compute 5-point velocity at waypoint 5
        dt = 0.1
        vel_before = (
            -initial_traj[7] + 8 * initial_traj[6]
            - 8 * initial_traj[4] + initial_traj[3]
        ) / (12 * dt)
        vel_after = (
            -result.trajectory[7] + 8 * result.trajectory[6]
            - 8 * result.trajectory[4] + result.trajectory[3]
        ) / (12 * dt)
        self.assertLess(
            np.max(np.abs(vel_after)),
            np.max(np.abs(vel_before)))

    def test_five_point_acceleration_cost(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=10, dt=0.1)
        problem.add_smoothness_cost(weight=0.1)
        problem.add_five_point_acceleration_cost(weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.5
        initial_traj = interpolate_trajectory(start, end, 10)
        # Add perturbation that creates high acceleration
        initial_traj[4] += 0.5
        initial_traj[5] -= 0.3

        solver = JaxlsSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        # Compute 5-point acceleration at waypoint 5
        dt = 0.1

        def compute_5pt_acc(traj, t):
            return (
                -traj[t + 2] + 16 * traj[t + 1] - 30 * traj[t]
                + 16 * traj[t - 1] - traj[t - 2]
            ) / (12 * dt ** 2)

        acc_before = compute_5pt_acc(initial_traj, 5)
        acc_after = compute_5pt_acc(result.trajectory, 5)
        self.assertLess(
            np.max(np.abs(acc_after)),
            np.max(np.abs(acc_before)))

    def test_five_point_jerk_cost(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=12, dt=0.1)
        problem.add_smoothness_cost(weight=0.1)
        problem.add_five_point_jerk_cost(weight=1.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.5
        initial_traj = interpolate_trajectory(start, end, 12)
        # Add perturbation that creates high jerk
        initial_traj[5] += 0.5
        initial_traj[6] -= 0.4

        solver = JaxlsSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        # Compute 7-point jerk at waypoint 6
        dt = 0.1

        def compute_7pt_jerk(traj, t):
            return (
                -traj[t + 3] + 8 * traj[t + 2] - 13 * traj[t + 1]
                + 13 * traj[t - 1] - 8 * traj[t - 2] + traj[t - 3]
            ) / (8 * dt ** 3)

        jerk_before = compute_7pt_jerk(initial_traj, 6)
        jerk_after = compute_7pt_jerk(result.trajectory, 6)
        self.assertLess(
            np.max(np.abs(jerk_after)),
            np.max(np.abs(jerk_before)))

    def test_acceleration_limit(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=10, dt=0.1)
        problem.add_smoothness_cost(weight=0.1)
        problem.add_acceleration_limit(acceleration_limit=5.0, weight=10.0)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.5
        initial_traj = interpolate_trajectory(start, end, 10)
        # Add perturbation that creates high acceleration
        initial_traj[4] += 1.0
        initial_traj[5] -= 0.8

        solver = JaxlsSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        # Compute 5-point acceleration
        dt = 0.1
        for t in range(2, 8):
            acc = (
                -result.trajectory[t + 2] + 16 * result.trajectory[t + 1]
                - 30 * result.trajectory[t] + 16 * result.trajectory[t - 1]
                - result.trajectory[t - 2]
            ) / (12 * dt ** 2)
            # Allow some slack for optimization tolerance
            self.assertTrue(
                np.all(np.abs(acc) < 6.0),
                msg=f"Acceleration limit violated at waypoint {t}")

    def test_five_point_waypoint_requirement(self):
        # Test that 5-point methods raise error for insufficient waypoints
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=4)
        with self.assertRaises(ValueError):
            problem.add_five_point_velocity_cost()
        with self.assertRaises(ValueError):
            problem.add_five_point_acceleration_cost()

    def test_seven_point_waypoint_requirement(self):
        # Test that 7-point jerk method raises error for insufficient waypoints
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=6)
        with self.assertRaises(ValueError):
            problem.add_five_point_jerk_cost()

    def test_smooth_trajectory_costs_high_precision(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        # Test with enough waypoints for full high precision (>=7)
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=12, dt=0.1)
        problem.add_smooth_trajectory_costs(weight=1.0, use_high_precision=True)

        # Should have added 3 residuals: velocity, acceleration, jerk
        residual_names = [r.name for r in problem.residuals]
        self.assertIn('five_point_velocity', residual_names)
        self.assertIn('five_point_acceleration', residual_names)
        self.assertIn('five_point_jerk', residual_names)

        start = np.zeros(self.n_joints)
        end = np.ones(self.n_joints) * 0.5
        initial_traj = interpolate_trajectory(start, end, 12)
        initial_traj[5] += 0.5

        solver = JaxlsSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)

        # Trajectory should be smoother after optimization
        diff_before = np.sum((initial_traj[1:] - initial_traj[:-1]) ** 2)
        diff_after = np.sum(
            (result.trajectory[1:] - result.trajectory[:-1]) ** 2)
        self.assertLess(diff_after, diff_before)

    def test_smooth_trajectory_costs_medium_waypoints(self):
        # Test with 5-6 waypoints (no jerk, but velocity and acceleration)
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=6, dt=0.1)
        problem.add_smooth_trajectory_costs(weight=1.0, use_high_precision=True)

        residual_names = [r.name for r in problem.residuals]
        self.assertIn('five_point_velocity', residual_names)
        self.assertIn('five_point_acceleration', residual_names)
        self.assertNotIn('five_point_jerk', residual_names)

    def test_smooth_trajectory_costs_low_waypoints(self):
        # Test with few waypoints (fallback to simple differences)
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=4, dt=0.1)
        problem.add_smooth_trajectory_costs(weight=1.0, use_high_precision=True)

        residual_names = [r.name for r in problem.residuals]
        self.assertIn('smoothness', residual_names)
        self.assertIn('acceleration', residual_names)
        self.assertNotIn('five_point_velocity', residual_names)

    def test_smooth_trajectory_costs_disabled_high_precision(self):
        # Test with high precision disabled
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=12, dt=0.1)
        problem.add_smooth_trajectory_costs(
            weight=1.0, use_high_precision=False)

        residual_names = [r.name for r in problem.residuals]
        self.assertIn('smoothness', residual_names)
        self.assertIn('acceleration', residual_names)
        self.assertNotIn('five_point_velocity', residual_names)


@requires_jaxls
class TestFloatingBaseAndMultiChain(unittest.TestCase):
    """Multi-chain FK, floating base, CoG cost, and per-waypoint
    multi-EE pose cost."""

    def test_multi_chain_problem_shape(self):
        robot = skrobot.models.PR2()
        robot.reset_manip_pose()
        rarm_links = list(robot.rarm.link_list)
        larm_links = list(robot.larm.link_list)

        problem = TrajectoryProblem(
            robot,
            link_list=[rarm_links, larm_links],
            n_waypoints=4,
            move_target=[robot.rarm_end_coords, robot.larm_end_coords],
        )

        self.assertTrue(problem.is_multi_chain)
        self.assertEqual(len(problem.link_lists), 2)
        self.assertEqual(problem.n_joints,
                         len(rarm_links) + len(larm_links))
        self.assertEqual(problem.n_total_dof, problem.n_joints)
        self.assertEqual(len(problem.fk_params_per_chain), 2)
        self.assertEqual(problem.fk_params_per_chain[0]['n_joints'],
                         len(rarm_links))
        self.assertEqual(problem.fk_params_per_chain[1]['n_joints'],
                         len(larm_links))
        self.assertEqual(problem.centroid_data['n_chains'], 2)

    def test_n_base_dof_extends_total_dof(self):
        robot, link_list, n_joints = _make_kuka()

        for n_base in (3, 6):
            problem = TrajectoryProblem(
                robot, link_list, n_waypoints=3, n_base_dof=n_base)
            self.assertEqual(problem.n_base_dof, n_base)
            self.assertEqual(problem.n_total_dof, n_joints + n_base)

        with self.assertRaises(ValueError):
            TrajectoryProblem(
                robot, link_list, n_waypoints=3, n_base_dof=5)

    def test_add_com_cost_drives_centroid(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        robot = skrobot.models.Fetch()
        robot.reset_pose()
        link_list = list(robot.rarm.link_list)
        n_waypoints = 3

        nominal = np.array([link.joint.joint_angle()
                            for link in link_list])
        cog_initial = robot.update_mass_properties()['total_centroid'].copy()

        # Pick a CoG target that the rarm can actually realise: forward
        # kinematics from a perturbed config, then ask the optimiser to
        # drive the CoG there from nominal.
        for link, q in zip(link_list, nominal + 0.3):
            link.joint.joint_angle(float(q))
        cog_reachable = robot.update_mass_properties()[
            'total_centroid'].copy()
        for link, q in zip(link_list, nominal):
            link.joint.joint_angle(float(q))

        problem = TrajectoryProblem(
            robot, link_list, n_waypoints=n_waypoints,
            move_target=robot.rarm_end_coords)
        problem.set_fixed_endpoints(start=False, end=False)
        problem.add_smoothness_cost(weight=0.01)
        problem.add_com_cost(
            target_positions=np.tile(cog_reachable, (n_waypoints, 1)),
            translation_axis='xy',
            weight=200.0,
        )

        initial_traj = np.tile(nominal, (n_waypoints, 1))

        solver = JaxlsSolver(max_iterations=100)
        result = solver.solve(problem, initial_traj)
        self.assertTrue(result.success)

        # Cost was registered.
        residual_names = [r.name for r in problem.residuals]
        self.assertIn('com', residual_names)

        for link, q in zip(link_list, result.trajectory[-1]):
            link.joint.joint_angle(float(q))
        cog_final = robot.update_mass_properties()['total_centroid']

        err_initial = np.linalg.norm((cog_initial - cog_reachable)[:2])
        err_final = np.linalg.norm((cog_final - cog_reachable)[:2])
        self.assertLess(err_final, err_initial)

    def test_add_multi_ee_waypoint_cost(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        robot = skrobot.models.PR2()
        robot.reset_manip_pose()
        rarm_links = list(robot.rarm.link_list)
        larm_links = list(robot.larm.link_list)
        n_waypoints = 3

        nominal = np.concatenate([
            np.array([link.joint.joint_angle() for link in rarm_links]),
            np.array([link.joint.joint_angle() for link in larm_links]),
        ])

        rarm_initial = robot.rarm_end_coords.copy_worldcoords()
        larm_initial = robot.larm_end_coords.copy_worldcoords()
        rarm_target_pos = rarm_initial.worldpos() + np.array(
            [0.02, 0.0, 0.02])
        larm_target_pos = larm_initial.worldpos() + np.array(
            [0.02, 0.0, 0.02])
        rarm_target_rot = rarm_initial.worldrot()
        larm_target_rot = larm_initial.worldrot()

        rarm_targets_pos = np.tile(rarm_target_pos, (n_waypoints, 1))
        larm_targets_pos = np.tile(larm_target_pos, (n_waypoints, 1))
        rarm_targets_rot = np.tile(rarm_target_rot, (n_waypoints, 1, 1))
        larm_targets_rot = np.tile(larm_target_rot, (n_waypoints, 1, 1))

        problem = TrajectoryProblem(
            robot,
            link_list=[rarm_links, larm_links],
            n_waypoints=n_waypoints,
            move_target=[robot.rarm_end_coords, robot.larm_end_coords],
        )
        problem.set_fixed_endpoints(start=False, end=False)
        problem.add_smoothness_cost(weight=0.01)
        problem.add_multi_ee_waypoint_cost(
            target_positions_per_chain=[rarm_targets_pos, larm_targets_pos],
            target_rotations_per_chain=[rarm_targets_rot, larm_targets_rot],
            position_weight=500.0,
            rotation_weight=50.0,
        )

        initial_traj = np.tile(nominal, (n_waypoints, 1))

        solver = JaxlsSolver(max_iterations=80)
        result = solver.solve(problem, initial_traj)
        self.assertTrue(result.success)

        n_rarm = len(rarm_links)
        for link, q in zip(rarm_links, result.trajectory[-1, :n_rarm]):
            link.joint.joint_angle(float(q))
        for link, q in zip(larm_links, result.trajectory[-1, n_rarm:]):
            link.joint.joint_angle(float(q))

        rarm_err = np.linalg.norm(
            robot.rarm_end_coords.worldpos() - rarm_target_pos)
        larm_err = np.linalg.norm(
            robot.larm_end_coords.worldpos() - larm_target_pos)
        self.assertLess(rarm_err, 0.02)
        self.assertLess(larm_err, 0.02)

    def test_add_base_pose_cost_registers(self):
        """``add_base_pose_cost`` adds a base_pose residual and solves."""
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        robot, link_list, n_joints = _make_kuka()
        n_waypoints = 3
        n_base = 6
        problem = TrajectoryProblem(
            robot, link_list, n_waypoints=n_waypoints, n_base_dof=n_base)
        problem.set_fixed_endpoints(start=False, end=False)
        problem.add_smoothness_cost(weight=0.01)

        target_pos = np.tile(np.array([0.2, -0.1, 0.05]), (n_waypoints, 1))
        target_rot = np.tile(np.eye(3), (n_waypoints, 1, 1))
        problem.add_base_pose_cost(
            target_positions=target_pos,
            target_rotations=target_rot,
            position_weight=300.0,
            rotation_weight=300.0,
        )

        residual_names = [r.name for r in problem.residuals]
        self.assertIn('base_pose', residual_names)

        initial_traj = np.zeros((n_waypoints, n_joints + n_base))
        solver = JaxlsSolver(max_iterations=40)
        result = solver.solve(problem, initial_traj)
        self.assertTrue(result.success)
        self.assertTrue(np.all(np.isfinite(result.trajectory)))

    def test_update_base_pose_targets_reuses_graph(self):
        """``update_base_pose_targets`` swaps targets without recompiling."""
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        robot, link_list, n_joints = _make_kuka()
        n_waypoints = 3
        n_base = 6
        problem = TrajectoryProblem(
            robot, link_list, n_waypoints=n_waypoints, n_base_dof=n_base)
        problem.set_fixed_endpoints(start=False, end=False)
        problem.add_smoothness_cost(weight=0.01)

        problem.add_base_pose_cost(
            target_positions=np.zeros((n_waypoints, 3)),
            target_rotations=np.tile(np.eye(3), (n_waypoints, 1, 1)),
            position_weight=200.0,
            rotation_weight=200.0,
        )

        # First solve traces the cost graph and populates the cached
        # param vars.
        initial_traj = np.zeros((n_waypoints, n_joints + n_base))
        solver = JaxlsSolver(max_iterations=20)
        result = solver.solve(problem, initial_traj)
        self.assertTrue(result.success)

        # Swap targets — should not raise and the next solve still runs.
        problem.update_base_pose_targets(
            target_positions=np.tile([0.1, 0.0, 0.0], (n_waypoints, 1)),
            target_rotations=np.tile(np.eye(3), (n_waypoints, 1, 1)),
        )
        result2 = solver.solve(problem, initial_traj)
        self.assertTrue(result2.success)
        self.assertTrue(np.all(np.isfinite(result2.trajectory)))

    def test_multi_ee_per_chain_weights_zero_disables(self):
        """``rotation_weights_per_chain`` = 0 disables rotation tracking
        for that chain (position still tracked)."""
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        robot = skrobot.models.PR2()
        robot.reset_manip_pose()
        rarm_links = list(robot.rarm.link_list)
        larm_links = list(robot.larm.link_list)
        n_waypoints = 3

        nominal = np.concatenate([
            np.array([link.joint.joint_angle() for link in rarm_links]),
            np.array([link.joint.joint_angle() for link in larm_links]),
        ])

        rarm_initial = robot.rarm_end_coords.copy_worldcoords()
        larm_initial = robot.larm_end_coords.copy_worldcoords()
        # Position-only target for rarm; full pose for larm.
        rarm_target_pos = rarm_initial.worldpos() + np.array(
            [0.02, 0.0, 0.0])
        larm_target_pos = larm_initial.worldpos() + np.array(
            [0.02, 0.0, 0.02])

        problem = TrajectoryProblem(
            robot,
            link_list=[rarm_links, larm_links],
            n_waypoints=n_waypoints,
            move_target=[robot.rarm_end_coords, robot.larm_end_coords],
        )
        problem.set_fixed_endpoints(start=False, end=False)
        problem.add_smoothness_cost(weight=0.01)
        problem.add_multi_ee_waypoint_cost(
            target_positions_per_chain=[
                np.tile(rarm_target_pos, (n_waypoints, 1)),
                np.tile(larm_target_pos, (n_waypoints, 1))],
            target_rotations_per_chain=[
                np.tile(rarm_initial.worldrot(), (n_waypoints, 1, 1)),
                np.tile(larm_initial.worldrot(), (n_waypoints, 1, 1))],
            position_weight=500.0,
            rotation_weight=50.0,
            rotation_weights_per_chain=[0.0, 50.0],
        )

        initial_traj = np.tile(nominal, (n_waypoints, 1))
        solver = JaxlsSolver(max_iterations=80)
        result = solver.solve(problem, initial_traj)
        self.assertTrue(result.success)

        n_rarm = len(rarm_links)
        for link, q in zip(rarm_links, result.trajectory[-1, :n_rarm]):
            link.joint.joint_angle(float(q))
        for link, q in zip(larm_links, result.trajectory[-1, n_rarm:]):
            link.joint.joint_angle(float(q))

        rarm_err = np.linalg.norm(
            robot.rarm_end_coords.worldpos() - rarm_target_pos)
        larm_err = np.linalg.norm(
            robot.larm_end_coords.worldpos() - larm_target_pos)
        self.assertLess(rarm_err, 0.02)
        self.assertLess(larm_err, 0.02)


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


class TestGridSDFSelfCollision(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.robot, cls.link_list, cls.collision_link_list = _make_panda()

    def _link_transforms(self):
        positions = np.stack(
            [link.worldpos() for link in self.collision_link_list])
        rotations = np.stack(
            [link.worldrot() for link in self.collision_link_list])
        return positions, rotations

    def test_reset_pose_is_reported_collision_free(self):
        """GridSDF must not report the rest pose as self-colliding.

        The sphere approximation does, which is the reason this mode exists.
        """
        self.robot.reset_manip_pose()
        data = build_gridsdf_self_data(
            self.robot, self.collision_link_list, dim_grid=40, n_surface=32)
        positions, rotations = self._link_transforms()
        distances = gridsdf_self_distances(positions, rotations, data, np)

        n_pairs = len(data['pairs_a'])
        self.assertEqual(distances.shape, (n_pairs, 32))
        self.assertGreater(distances.min(), 0.0)

        # Same pose through the sphere model: heavily over-conservative.
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=2)
        problem.add_collision_cost(
            self.collision_link_list, world_obstacles=[])
        problem.add_self_collision_cost(mode='sphere')
        spheres = problem.collision_spheres
        pairs_i, pairs_j = problem.residuals[-1].params['pair_indices']
        sphere_positions = np.stack([
            self.collision_link_list[link_idx].worldcoords()
            .transform_vector(center)
            for link_idx, center in zip(
                spheres['link_indices'], spheres['sphere_centers_local'])])
        sphere_distances = compute_self_collision_distances(
            sphere_positions, spheres['sphere_radii'], pairs_i, pairs_j, np)
        self.assertLess(sphere_distances.min(), 0.0)

    def test_matches_gridsdf_reference_interpolation(self):
        """The batched lookup must agree with ``GridSDF`` itself."""
        from skrobot.sdf.signed_distance_function import trimesh2sdf

        self.robot.reset_manip_pose()
        data = build_gridsdf_self_data(
            self.robot, self.collision_link_list, dim_grid=16, n_surface=8)
        positions, rotations = self._link_transforms()
        actual = gridsdf_self_distances(positions, rotations, data, np)

        expected = np.empty_like(actual)
        for row, (a, b) in enumerate(zip(data['pairs_a'], data['pairs_b'])):
            mesh = self.collision_link_list[b].collision_mesh.copy()
            mesh.metadata = {key: value
                             for key, value in mesh.metadata.items()
                             if key != 'shape'}
            sdf = trimesh2sdf(mesh, dim_grid=16)
            world = positions[a] + data['surface_points'][a].dot(rotations[a].T)
            local_b = (world - positions[b]).dot(rotations[b])
            expected[row] = sdf._signed_distance(local_b)

        # GridSDF fills points outside its grid with inf; those are exactly
        # the ones this module replaces with a finite, differentiable value.
        inside = np.isfinite(expected)
        self.assertTrue(inside.any())
        testing.assert_allclose(actual[inside], expected[inside], atol=1e-12)

    def test_trilinear_handles_grids_of_different_shapes(self):
        """Padding cells must never leak into the interpolated value."""
        from skrobot.planner.trajectory_optimization.gridsdf_collision import _trilinear

        small = np.arange(2 * 3 * 2, dtype=np.float64).reshape(2, 3, 2)
        large = np.arange(4 * 3 * 5, dtype=np.float64).reshape(4, 3, 5)
        padded = np.stack([
            np.pad(small, [(0, 2), (0, 0), (0, 3)], mode='edge'), large])
        dims = np.array([[2.0, 3.0, 2.0], [4.0, 3.0, 5.0]])
        resolutions = np.array([0.1, 0.1])

        # Query the far corner of each grid, which is padding for the small one.
        coords = np.array([[[1.0, 2.0, 1.0]], [[3.0, 2.0, 4.0]]])
        values = _trilinear(padded, coords, dims, resolutions, np)
        testing.assert_allclose(
            values[:, 0], [small[1, 2, 1], large[3, 2, 4]])

        # Outside the small grid: edge value plus the distance to the boundary.
        outside = np.array([[[3.0, 2.0, 1.0]], [[3.0, 2.0, 4.0]]])
        values = _trilinear(padded, outside, dims, resolutions, np)
        testing.assert_allclose(
            values[0, 0], small[1, 2, 1] + 2.0 * 0.1, atol=1e-9)

    def test_pairs_are_evaluated_in_both_directions(self):
        data = build_gridsdf_self_data(
            self.robot, self.collision_link_list, dim_grid=16, n_surface=8)
        pairs = set(zip(data['pairs_a'].tolist(), data['pairs_b'].tolist()))
        expected = set(
            create_self_collision_pairs(self.collision_link_list))
        self.assertEqual(pairs, expected | {(b, a) for a, b in expected})

    @requires_jax
    def test_numpy_and_jax_agree(self):
        import jax.numpy as jnp

        self.robot.reset_manip_pose()
        data = build_gridsdf_self_data(
            self.robot, self.collision_link_list, dim_grid=16, n_surface=8)
        positions, rotations = self._link_transforms()

        expected = gridsdf_self_distances(positions, rotations, data, np)
        actual = gridsdf_self_distances(
            jnp.asarray(positions), jnp.asarray(rotations),
            {key: jnp.asarray(value) for key, value in data.items()}, jnp)
        testing.assert_allclose(np.asarray(actual), expected, atol=1e-5)

    def test_add_self_collision_cost_mode(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=2)
        problem.add_collision_cost(
            self.collision_link_list, world_obstacles=[])
        problem.add_self_collision_cost(
            mode='gridsdf', dim_grid=16, n_surface=8)

        spec = problem.residuals[-1]
        self.assertEqual(spec.name, 'self_collision')
        self.assertEqual(spec.params['mode'], 'gridsdf')
        self.assertIn('gridsdf_data', spec.params)

    def test_unknown_mode_raises(self):
        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=2)
        problem.add_collision_cost(
            self.collision_link_list, world_obstacles=[])
        with self.assertRaises(ValueError):
            problem.add_self_collision_cost(mode='capsule')

    @requires_jaxls
    def test_jaxls_resolves_self_collision(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver

        # A configuration the GridSDF model reports as penetrating.
        colliding = np.array(
            [1.116, -1.309, 0.575, -3.0, 0.074, 0.008, -0.842, 0.015, 0.018])
        n_waypoints = 3

        problem = TrajectoryProblem(
            self.robot, self.link_list, n_waypoints=n_waypoints)
        problem.set_fixed_endpoints(start=False, end=False)
        problem.add_collision_cost(
            self.collision_link_list, world_obstacles=[])
        problem.add_self_collision_cost(
            mode='gridsdf', dim_grid=16, n_surface=8,
            weight=1000.0, activation_distance=0.02, as_constraint=False)
        problem.add_smoothness_cost(weight=0.01)

        data = problem.residuals[-2].params['gridsdf_data']

        def min_distance(angles):
            for link, angle in zip(self.link_list, angles):
                link.joint.joint_angle(float(angle))
            positions, rotations = self._link_transforms()
            return gridsdf_self_distances(
                positions, rotations, data, np).min()

        initial_traj = np.tile(colliding, (n_waypoints, 1))
        before = min(min_distance(q) for q in initial_traj)
        self.assertLess(before, 0.0)

        solver = JaxlsSolver(max_iterations=50)
        result = solver.solve(problem, initial_traj)
        self.assertTrue(result.success)

        after = min(min_distance(q) for q in result.trajectory)
        self.assertGreater(after, 0.0)


if __name__ == '__main__':
    unittest.main()
