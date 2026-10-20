import unittest

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.kinematics import find_base_placement
from skrobot.kinematics import set_base_pose
from skrobot.kinematics import vectorized_target
from skrobot.kinematics.base_placement import _chain_fk_params
from skrobot.kinematics.base_placement import _resolve_link_list
from skrobot.kinematics.base_placement import _resolve_masks
from skrobot.kinematics.base_placement import _SharedBaseProblem
from skrobot.models import Fetch


def _targets_from_base(robot, base_pose, n_targets, seed):
    """Targets the arm provably reaches from ``base_pose``."""
    rng = np.random.RandomState(seed)
    move_target = robot.rarm.end_coords
    joints = robot.rarm.joint_list
    set_base_pose(robot, base_pose)
    targets = []
    for _ in range(n_targets):
        robot.reset_pose()
        for j in joints:
            lo = max(j.min_angle, -2.0)
            hi = min(j.max_angle, 2.0)
            j.joint_angle(j.joint_angle() + rng.uniform(-0.3, 0.3)
                          * (hi - lo) / 2.0)
        targets.append(move_target.copy_worldcoords())
    robot.reset_pose()
    set_base_pose(robot, Coordinates())
    return targets


class TestFindBasePlacement(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.robot = Fetch()

    def setUp(self):
        self.robot.reset_pose()
        set_base_pose(self.robot, Coordinates())

    def test_reaches_targets_from_a_known_base(self):
        robot = self.robot
        true_base = Coordinates(pos=[2.0, 1.0, 0.0]).rotate(0.7, 'z')
        targets = _targets_from_base(robot, true_base, 4, seed=0)
        starts = [Coordinates(pos=[x, y, 0.0]).rotate(yaw, 'z')
                  for x in (1.0, 2.5) for y in (0.0, 1.5)
                  for yaw in (0.0, 1.5)]
        result = find_base_placement(
            robot, targets, move_target=robot.rarm.end_coords,
            joint_list=robot.rarm.joint_list, initial_base_poses=starts)
        self.assertTrue(result.all_reached, result)
        self.assertEqual(len(result.candidates), len(starts))

        # Check against the model's own FK, not the solver's.
        set_base_pose(robot, result.base_pose)
        for av, target in zip(result.angle_vectors, targets):
            robot.angle_vector(av)
            ee = robot.rarm.end_coords.worldcoords()
            np.testing.assert_allclose(
                ee.worldpos(), target.worldpos(), atol=2e-3)
            np.testing.assert_allclose(
                ee.worldrot(), target.worldrot(), atol=2e-2)
        np.testing.assert_allclose(
            result.base_pose.worldpos()[2], 0.0, atol=1e-12)

    def test_base_jacobian_matches_finite_difference(self):
        robot = self.robot
        move_target = robot.rarm.end_coords
        link_list = _resolve_link_list(
            robot, move_target, None, robot.rarm.joint_list)
        fk_params = _chain_fk_params(robot, link_list, move_target)
        base = np.array([[0.3, -0.2, 0.4, 0.0]])  # x, y, yaw, height
        q = np.array([[[j.joint_angle() + 0.1 for j in
                        robot.rarm.joint_list]]])
        # The log-map error has the geometric Jacobian as its derivative
        # only at zero error, so the target is the pose itself.
        problem = _SharedBaseProblem(
            fk_params, np.zeros((1, 3)), np.eye(3)[None],
            np.ones(3), np.ones(3))
        pos, rot, _, _, _ = problem._ee_world(base, q, False)
        problem.target_pos = pos[0]
        problem.target_rot = rot[0]
        err, base_jac, arm_jac = problem.linearize(base, q)
        np.testing.assert_allclose(err, 0.0, atol=1e-12)
        eps = 1e-6
        for i in range(3):
            d = np.zeros((1, 4))
            d[0, i] = eps
            num = -(problem.residual(base + d, q)
                    - problem.residual(base - d, q)) / (2 * eps)
            np.testing.assert_allclose(
                num[0, 0], base_jac[0, 0, :, i], atol=1e-5)
        for i in range(q.shape[-1]):
            d = np.zeros_like(q)
            d[..., i] = eps
            num = -(problem.residual(base, q + d)
                    - problem.residual(base, q - d)) / (2 * eps)
            np.testing.assert_allclose(
                num[0, 0], arm_jac[0, 0, :, i], atol=1e-5)

    def test_reports_unreachable_target(self):
        robot = self.robot
        targets = [Coordinates(pos=[1.0, 0.0, 0.8]),
                   Coordinates(pos=[1.1, 0.1, 0.9]),
                   Coordinates(pos=[1.0, 0.0, 5.0])]
        result = find_base_placement(
            robot, targets, move_target=robot.rarm.end_coords,
            rotation_mask=False)
        np.testing.assert_array_equal(result.success, [True, True, False])
        self.assertFalse(result.all_reached)
        self.assertEqual(result.n_reached, 2)

    def test_base_bounds(self):
        robot = self.robot
        targets = [Coordinates(pos=[2.0, 0.0, 0.8])]
        bounds = ((-0.5, 0.5), (-0.5, 0.5))
        result = find_base_placement(
            robot, targets, move_target=robot.rarm.end_coords,
            rotation_mask=False, base_bounds=bounds)
        x, y = result.base_pose.worldpos()[:2]
        self.assertTrue(-0.5 <= x <= 0.5 and -0.5 <= y <= 0.5)
        self.assertFalse(result.all_reached)
        self.assertTrue(result.at_bound)
        self.assertLess(result.bound_distance, 1e-6)

        free = find_base_placement(
            robot, targets, move_target=robot.rarm.end_coords,
            rotation_mask=False, base_bounds=((-5, 5), (-5, 5)))
        self.assertTrue(free.all_reached)
        self.assertFalse(free.at_bound)
        x, y = free.base_pose.worldpos()[:2]
        self.assertAlmostEqual(
            free.bound_distance, min(5 - abs(x), 5 - abs(y)))

    def test_base_frame_targets(self):
        for height in (0.0, -0.2):
            self._check_base_frame_targets(height)

    def _check_base_frame_targets(self, height):
        robot = self.robot
        world = _targets_from_base(
            robot, Coordinates(pos=[1.0, 0.5, height]).rotate(0.3, 'z'), 2,
            seed=5)
        # A pose of the hand relative to the root link, reachable from any
        # base.
        local = _targets_from_base(robot, Coordinates(), 1, seed=6)[0]
        result = find_base_placement(
            robot, world + [local], move_target=robot.rarm.end_coords,
            joint_list=robot.rarm.joint_list, base_height=height,
            initial_base_poses=Coordinates(pos=[0.8, 0.3, 0.0]),
            target_frame=['world', 'world', 'base'])
        self.assertTrue(result.all_reached, result)
        self.assertAlmostEqual(result.base_pose.worldpos()[2], height)
        set_base_pose(robot, result.base_pose)
        robot.angle_vector(result.angle_vectors[2])
        hand = robot.root_link.worldcoords().inverse_transformation()\
            .transform(robot.rarm.end_coords.worldcoords())
        np.testing.assert_allclose(hand.worldpos(), local.worldpos(),
                                   atol=2e-3)
        np.testing.assert_allclose(hand.worldrot(), local.worldrot(),
                                   atol=2e-2)
        with self.assertRaises(ValueError):
            find_base_placement(
                robot, world, move_target=robot.rarm.end_coords,
                target_frame=['world'])

    def test_targets_that_follow_the_base(self):
        robot = self.robot
        world = _targets_from_base(
            robot, Coordinates(pos=[1.0, 0.5, -0.2]).rotate(0.3, 'z'), 2,
            seed=8)
        fixture = _targets_from_base(
            robot, Coordinates(pos=[1.0, 0.5, -0.2]).rotate(0.3, 'z'), 1,
            seed=9)[0]
        offset = fixture.worldpos()[0] - 1.0

        def follow_x(base_pose):
            # Rides along x with the base; fixed in y and z, like a stand
            # on the front edge of a table.
            pos = fixture.worldpos().copy()
            pos[0] = base_pose.worldpos()[0] + offset
            return Coordinates(pos=pos, rot=fixture.worldrot())

        move_target = robot.rarm.end_coords
        link_list = _resolve_link_list(
            robot, move_target, None, robot.rarm.joint_list)
        fk_params = _chain_fk_params(robot, link_list, move_target)
        base = np.array([[0.9, 0.4, 0.2, -0.2]])  # x, y, yaw, height
        q = np.array([[[j.joint_angle() + 0.1 for j in
                        robot.rarm.joint_list]]])
        # A following target that sits on the hand at this base and q, so
        # the residual is zero and its derivative is the Jacobian.
        probe = _SharedBaseProblem(
            fk_params, np.zeros((1, 3)), np.eye(3)[None], np.ones(3),
            np.ones(3))
        pos, rot, _, _, _ = probe._ee_world(base, q, False)
        hand_pos, hand_rot = pos[0, 0], rot[0, 0]

        def on_hand(base_pose):
            p = hand_pos.copy()
            p[0] += base_pose.worldpos()[0] - 0.9
            return Coordinates(pos=p, rot=hand_rot)

        problem = _SharedBaseProblem(
            fk_params, np.zeros((1, 3)), np.eye(3)[None], np.ones(3),
            np.ones(3), target_fns={0: on_hand})
        err, base_jac, _ = problem.linearize(base, q)
        np.testing.assert_allclose(err, 0.0, atol=1e-9)
        for i in range(3):
            d = np.zeros((1, 4))
            d[0, i] = 1e-6
            num = -(problem.residual(base + d, q)
                    - problem.residual(base - d, q)) / 2e-6
            np.testing.assert_allclose(
                num[0, 0], base_jac[0, 0, :, i], atol=1e-4)

        # Two following targets and several starts, so the batch has more
        # than one of each.
        result = find_base_placement(
            robot, world + [follow_x, follow_x], move_target=move_target,
            joint_list=robot.rarm.joint_list, base_height=-0.2,
            initial_base_poses=[Coordinates(pos=[0.8, 0.3, 0.0]),
                                Coordinates(pos=[1.1, 0.6, 0.0])])
        self.assertTrue(result.all_reached, result)

        @vectorized_target
        def follow_x_all(base_matrices):
            out = np.tile(np.eye(4), (len(base_matrices), 1, 1))
            out[:, :3, :3] = fixture.worldrot()
            out[:, :3, 3] = fixture.worldpos()
            out[:, 0, 3] = base_matrices[:, 0, 3] + offset
            return out

        same = find_base_placement(
            robot, world + [follow_x_all, follow_x_all],
            move_target=move_target, joint_list=robot.rarm.joint_list,
            base_height=-0.2,
            initial_base_poses=[Coordinates(pos=[0.8, 0.3, 0.0]),
                                Coordinates(pos=[1.1, 0.6, 0.0])])
        np.testing.assert_allclose(
            same.base_pose.worldpos(), result.base_pose.worldpos(),
            atol=1e-9)
        np.testing.assert_allclose(
            same.target_coords[2].worldpos(),
            result.target_coords[2].worldpos(), atol=1e-9)
        solved = result.target_coords[2]
        self.assertAlmostEqual(
            solved.worldpos()[0], result.base_pose.worldpos()[0] + offset)
        set_base_pose(robot, result.base_pose)
        robot.angle_vector(result.angle_vectors[2])
        np.testing.assert_allclose(
            move_target.worldpos(), solved.worldpos(), atol=2e-3)

    def test_base_height_candidates(self):
        robot = self.robot
        targets = _targets_from_base(
            robot, Coordinates(pos=[1.0, 0.5, -0.2]), 2, seed=7)
        result = find_base_placement(
            robot, targets, move_target=robot.rarm.end_coords,
            joint_list=robot.rarm.joint_list, base_height=[0.0, -0.2],
            initial_base_poses=Coordinates(pos=[0.8, 0.3, 0.0]))
        heights = {round(c.base_pose.worldpos()[2], 6)
                   for c in result.candidates}
        self.assertEqual(heights, {0.0, -0.2})
        self.assertTrue(result.all_reached)
        self.assertIs(result.candidates[0], result)

    def test_maximize_margin(self):
        robot = self.robot
        targets = [Coordinates(pos=[1.5, 0.0, 0.8]),
                   Coordinates(pos=[1.6, 0.2, 0.9])]
        kwargs = dict(move_target=robot.rarm.end_coords,
                      joint_list=robot.rarm.joint_list, rotation_mask=False,
                      initial_base_poses=Coordinates(pos=[0.8, 0.0, 0.0]))
        hinge = find_base_placement(
            robot, targets, margin_target=0.1, **kwargs)
        climbed = find_base_placement(
            robot, targets, margin_target=0.1, maximize_margin=True,
            **kwargs)
        self.assertTrue(climbed.all_reached, climbed)
        self.assertGreater(climbed.min_joint_margin,
                           hinge.min_joint_margin + 0.02)
        capped = find_base_placement(
            robot, targets, margin_target=0.1, maximize_margin=True,
            max_margin_steps=1, **kwargs)
        self.assertLessEqual(capped.min_joint_margin,
                             climbed.min_joint_margin)
        self.assertGreaterEqual(capped.min_joint_margin,
                                hinge.min_joint_margin)

    def test_base_height(self):
        robot = self.robot
        true_base = Coordinates(pos=[1.0, 0.5, -0.2]).rotate(-0.4, 'z')
        targets = _targets_from_base(robot, true_base, 3, seed=2)
        result = find_base_placement(
            robot, targets, move_target=robot.rarm.end_coords,
            joint_list=robot.rarm.joint_list, base_height=-0.2,
            initial_base_poses=Coordinates(pos=[0.8, 0.3, 0.0]))
        self.assertTrue(result.all_reached, result)
        self.assertAlmostEqual(result.base_pose.worldpos()[2], -0.2)
        set_base_pose(robot, result.base_pose)
        for av, target in zip(result.angle_vectors, targets):
            robot.angle_vector(av)
            np.testing.assert_allclose(
                robot.rarm.end_coords.worldpos(), target.worldpos(),
                atol=2e-3)

    def test_per_target_masks(self):
        np.testing.assert_array_equal(
            _resolve_masks([1, 1, 0], 2), [[1, 1, 0], [1, 1, 0]])
        np.testing.assert_array_equal(
            _resolve_masks([True, 'z', False], 3),
            [[1, 1, 1], [0, 0, 1], [0, 0, 0]])
        with self.assertRaises(ValueError):
            _resolve_masks([True, False], 3)

        robot = self.robot
        true_base = Coordinates(pos=[1.0, 0.0, 0.0])
        targets = _targets_from_base(robot, true_base, 2, seed=3)
        # Turned about the hand's own y axis, which 'xz' leaves free, as
        # it does in inverse_kinematics.
        targets[0].rotate(0.8, 'y')
        # An orientation the arm cannot take at that position, but only
        # its position is asked for.
        targets[1].rotate(np.pi, 'x')
        result = find_base_placement(
            robot, targets, move_target=robot.rarm.end_coords,
            joint_list=robot.rarm.joint_list,
            initial_base_poses=Coordinates(pos=[0.8, 0.2, 0.0]),
            rotation_mask=['xz', False])
        self.assertTrue(result.all_reached, result)
        set_base_pose(robot, result.base_pose)
        robot.angle_vector(result.angle_vectors[0])
        hand = robot.rarm.end_coords
        np.testing.assert_allclose(
            hand.difference_rotation(targets[0], rotation_mask='xz'),
            0.0, atol=np.deg2rad(1.0))
        np.testing.assert_allclose(
            hand.difference_position(targets[0]), 0.0, atol=2e-3)
        self.assertGreater(
            np.linalg.norm(hand.difference_rotation(targets[0])), 0.1)

    def test_rotation_tolerance(self):
        robot = self.robot
        targets = _targets_from_base(
            robot, Coordinates(pos=[1.0, 0.0, 0.0]), 2, seed=4)
        targets[0].rotate(0.5, 'y')
        result = find_base_placement(
            robot, targets, move_target=robot.rarm.end_coords,
            joint_list=robot.rarm.joint_list,
            initial_base_poses=Coordinates(pos=[0.8, 0.2, 0.0]),
            rotation_tolerance=[(0, 0.6, 0), (0, 0, 0)])
        self.assertTrue(result.all_reached, result)
        set_base_pose(robot, result.base_pose)
        robot.angle_vector(result.angle_vectors[0])
        err = robot.rarm.end_coords.difference_rotation(targets[0])
        self.assertLessEqual(abs(err[1]), 0.6 + 1e-6)
        np.testing.assert_allclose(err[[0, 2]], 0.0, atol=np.deg2rad(1.0))

    def test_margin_target_keeps_joints_off_limits(self):
        robot = self.robot
        targets = [Coordinates(pos=[1.5, 0.0, 0.8]),
                   Coordinates(pos=[1.6, 0.2, 0.9])]
        kwargs = dict(move_target=robot.rarm.end_coords,
                      joint_list=robot.rarm.joint_list, rotation_mask=False,
                      initial_base_poses=Coordinates(pos=[0.8, 0.0, 0.0]))
        plain = find_base_placement(robot, targets, **kwargs)
        kept = find_base_placement(
            robot, targets, margin_target=0.15, **kwargs)
        self.assertTrue(kept.all_reached, kept)
        self.assertGreaterEqual(kept.min_joint_margin, 0.14)
        self.assertGreaterEqual(kept.min_joint_margin,
                                plain.min_joint_margin)

    def test_ranks_by_joint_margin_then_custom_score(self):
        robot = self.robot
        targets = [Coordinates(pos=[1.5, 0.0, 0.8]),
                   Coordinates(pos=[1.6, 0.2, 0.9])]
        starts = [Coordinates(pos=[x, y, 0.0]).rotate(yaw, 'z')
                  for x in (0.6, 1.0) for y in (-0.4, 0.4)
                  for yaw in (-0.5, 0.5)]
        result = find_base_placement(
            robot, targets, move_target=robot.rarm.end_coords,
            rotation_mask=False, initial_base_poses=starts)
        reached = [c for c in result.candidates if c.all_reached]
        self.assertGreater(len(reached), 1)
        self.assertTrue(result.all_reached)
        self.assertEqual(result.score, result.min_joint_margin)
        self.assertEqual(result.score, max(c.score for c in reached))
        self.assertTrue(0.0 <= result.min_joint_margin <= 0.5)

        result = find_base_placement(
            robot, targets, move_target=robot.rarm.end_coords,
            rotation_mask=False, initial_base_poses=starts,
            score=lambda r: -r.base_pose.worldpos()[0])
        reached = [c for c in result.candidates if c.all_reached]
        self.assertEqual(result.base_pose.worldpos()[0],
                         min(c.base_pose.worldpos()[0] for c in reached))

    def test_set_base_pose_does_not_drift(self):
        robot = self.robot
        robot.root_link.translate([0.1, 0.0, 0.05])
        try:
            for i in range(500):
                set_base_pose(robot, Coordinates(pos=[1.0, 2.0, 0.0])
                              .rotate(0.3 * i, 'z'))
            np.testing.assert_allclose(
                robot.root_link.worldpos(), [1.0, 2.0, 0.0], atol=1e-9)
            np.testing.assert_allclose(
                np.linalg.det(robot.worldrot()), 1.0, atol=1e-12)
        finally:
            robot.root_link.translate([-0.1, 0.0, -0.05])

    def test_apply_moves_robot(self):
        robot = self.robot
        targets = [Coordinates(pos=[1.5, 1.0, 0.8])]
        result = find_base_placement(
            robot, targets, move_target=robot.rarm.end_coords,
            rotation_mask=False, apply=True)
        self.assertTrue(result.all_reached)
        np.testing.assert_allclose(
            robot.root_link.worldpos(), result.base_pose.worldpos(),
            atol=1e-9)
        np.testing.assert_allclose(
            robot.rarm.end_coords.worldpos(), targets[0].worldpos(),
            atol=2e-3)


class TestJaxBackend(unittest.TestCase):

    def test_matches_numpy(self):
        try:
            import jax  # NOQA
        except ImportError:
            self.skipTest('jax is not installed')
        robot = Fetch()
        robot.reset_pose()
        targets = _targets_from_base(
            robot, Coordinates(pos=[1.0, 0.5, -0.2]).rotate(0.3, 'z'), 3,
            seed=10)
        offset = targets[2].worldpos()[0] - 1.0
        rot = targets[2].worldrot()
        pos = targets[2].worldpos()

        @vectorized_target
        def follow_x(base_matrices):
            out = np.tile(np.eye(4), (len(base_matrices), 1, 1))
            out[:, :3, :3] = rot
            out[:, :3, 3] = pos
            out[:, 0, 3] = base_matrices[:, 0, 3] + offset
            return out

        kwargs = dict(
            move_target=robot.rarm.end_coords,
            joint_list=robot.rarm.joint_list, base_height=-0.2,
            margin_target=0.1, maximize_margin=True,
            initial_base_poses=[Coordinates(pos=[0.8, 0.3, 0.0]),
                                Coordinates(pos=[1.2, 0.7, 0.0])])
        results = {}
        for backend in ('numpy', 'jax'):
            robot.reset_pose()
            results[backend] = find_base_placement(
                robot, targets[:2] + [follow_x], backend=backend, **kwargs)
        a, b = results['numpy'], results['jax']
        self.assertTrue(a.all_reached)
        np.testing.assert_array_equal(a.success, b.success)
        # Same steps, but summed in another order over many iterations.
        np.testing.assert_allclose(a.base_pose.worldpos(),
                                   b.base_pose.worldpos(), atol=1e-4)
        np.testing.assert_allclose(a.angle_vectors, b.angle_vectors,
                                   atol=1e-3)


if __name__ == '__main__':
    unittest.main()
