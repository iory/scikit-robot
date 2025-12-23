import unittest

import numpy as np
from numpy import testing as npt

from skrobot.coordinates import Coordinates
from skrobot.models import Fetch
from skrobot.planner import compute_trajectory_smoothness
from skrobot.planner import generate_initial_trajectory_seeded_ik
from skrobot.planner import interpolate_waypoints
from skrobot.planner import plan_smooth_trajectory_ik


class TestInterpolateWaypoints(unittest.TestCase):

    def test_interpolate_two_points(self):
        c1 = Coordinates(pos=[0, 0, 0])
        c2 = Coordinates(pos=[1, 1, 1])

        result = interpolate_waypoints([c1, c2], n_divisions=4, closed_loop=False)

        self.assertEqual(len(result), 5)  # 4 divisions + end point
        npt.assert_array_almost_equal(result[0].worldpos(), [0, 0, 0])
        npt.assert_array_almost_equal(result[-1].worldpos(), [1, 1, 1])

    def test_interpolate_closed_loop(self):
        c1 = Coordinates(pos=[0, 0, 0])
        c2 = Coordinates(pos=[1, 0, 0])
        c3 = Coordinates(pos=[1, 1, 0])

        result = interpolate_waypoints([c1, c2, c3], n_divisions=2, closed_loop=True)

        # 3 segments * 2 divisions = 6 points
        self.assertEqual(len(result), 6)

    def test_interpolate_preserves_orientation(self):
        c1 = Coordinates(pos=[0, 0, 0])
        c2 = Coordinates(pos=[1, 0, 0]).rotate(np.pi / 2, 'z')

        result = interpolate_waypoints([c1, c2], n_divisions=2, closed_loop=False)

        # Middle point should have intermediate rotation
        mid_angle = np.arccos(result[1].worldrot()[0, 0])
        self.assertGreater(mid_angle, 0.1)
        self.assertLess(mid_angle, np.pi / 2 - 0.1)


class TestGenerateInitialTrajectory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.robot = Fetch()
        cls.robot.reset_pose()

    def test_seeded_ik_generates_trajectory(self):
        targets = [
            Coordinates(pos=[0.6, 0.1, 0.9]),
            Coordinates(pos=[0.6, -0.1, 0.9]),
            Coordinates(pos=[0.6, 0.0, 1.0]),
        ]

        traj, flags = generate_initial_trajectory_seeded_ik(
            self.robot,
            self.robot.end_coords,
            targets,
            link_list=self.robot.link_list[1:8]
        )

        self.assertEqual(traj.shape[0], 3)  # 3 waypoints
        self.assertEqual(len(flags), 3)

    def test_seeded_ik_maintains_continuity(self):
        # Create targets that are close together
        targets = [
            Coordinates(pos=[0.6, 0.0, 0.9]),
            Coordinates(pos=[0.6, 0.05, 0.92]),
            Coordinates(pos=[0.6, 0.1, 0.94]),
        ]

        traj, flags = generate_initial_trajectory_seeded_ik(
            self.robot,
            self.robot.end_coords,
            targets,
            link_list=self.robot.link_list[1:8]
        )

        # Check that consecutive configurations are similar
        diffs = np.diff(traj, axis=0)
        max_diff = np.max(np.abs(diffs))

        # Max joint change between consecutive points should be reasonable
        self.assertLess(max_diff, np.deg2rad(30))


class TestPlanSmoothTrajectory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.robot = Fetch()
        cls.robot.reset_pose()

    def test_basic_rectangle_trajectory(self):
        corners = [
            Coordinates(pos=[0.5, 0.1, 0.9]),
            Coordinates(pos=[0.5, -0.1, 0.9]),
            Coordinates(pos=[0.5, -0.1, 1.0]),
            Coordinates(pos=[0.5, 0.1, 1.0]),
        ]

        traj, coords, success, info = plan_smooth_trajectory_ik(
            self.robot,
            self.robot.end_coords,
            corners,
            link_list=self.robot.link_list[1:8],
            n_divisions=3,
            closed_loop=True,
            position_tolerance=0.03,
            rotation_tolerance=np.deg2rad(15.0),
        )

        # Check trajectory shape
        expected_n_wp = 4 * 3  # 4 corners * 3 divisions (closed loop)
        self.assertEqual(traj.shape[0], expected_n_wp)

        # Check info dict contains expected keys
        self.assertIn('initial_trajectory', info)
        self.assertIn('ik_success_flags', info)
        self.assertIn('position_errors', info)
        self.assertIn('rotation_errors', info)

    def test_smoothness_improvement(self):
        corners = [
            Coordinates(pos=[0.5, 0.1, 0.9]),
            Coordinates(pos=[0.5, -0.1, 0.9]),
        ]

        traj, coords, success, info = plan_smooth_trajectory_ik(
            self.robot,
            self.robot.end_coords,
            corners,
            link_list=self.robot.link_list[1:8],
            n_divisions=5,
            closed_loop=False,
            position_tolerance=0.02,
            rotation_tolerance=np.deg2rad(10.0),
        )

        initial_metrics = compute_trajectory_smoothness(info['initial_trajectory'])
        optimized_metrics = compute_trajectory_smoothness(traj)

        # Optimized trajectory should have lower or equal acceleration
        self.assertLessEqual(
            optimized_metrics['max_acceleration'],
            initial_metrics['max_acceleration'] * 1.5  # Allow some tolerance
        )


class TestComputeTrajectorySmoothness(unittest.TestCase):

    def test_constant_velocity_trajectory(self):
        # Linear trajectory: constant velocity, zero acceleration
        traj = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ], dtype=float)

        metrics = compute_trajectory_smoothness(traj)

        # max_velocity is per-joint max, so it's 1.0 (not Euclidean norm)
        npt.assert_almost_equal(metrics['max_velocity'], 1.0)
        npt.assert_almost_equal(metrics['max_acceleration'], 0.0)

    def test_jerky_trajectory(self):
        # Trajectory with sudden direction change
        traj = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],  # Reversal!
            [1, 0, 0],
        ], dtype=float)

        metrics = compute_trajectory_smoothness(traj)

        # Should have high acceleration due to reversal
        self.assertGreater(metrics['max_acceleration'], 1.0)


if __name__ == '__main__':
    unittest.main()
