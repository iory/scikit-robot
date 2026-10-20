import types
import unittest

import numpy as np
import trimesh

from skrobot.collision import EnvironmentCollision
from skrobot.collision import RobotCollisionChecker
from skrobot.collision import sample_surface
from skrobot.coordinates import Coordinates
from skrobot.model import RobotModel
from skrobot.model.primitives import Box
from skrobot.models import Panda
from skrobot.sdf import BoxSDF


def _channel(gap, shift=0.0):
    """A U-shaped channel around a 0.02 x 0.02 x 0.05 m probe at the origin:
    walls ``gap`` beside its x faces and a floor ``gap`` under it, moved
    ``shift`` along +x. Its convex hull contains the probe."""
    parts = []
    for side in (-1, 1):
        wall = trimesh.creation.box([0.01, 0.06, 0.08])
        wall.apply_translation([side * (0.01 + gap + 0.005) + shift, 0, 0])
        parts.append(wall)
    floor = trimesh.creation.box([0.06, 0.06, 0.01])
    floor.apply_translation([shift, 0, -0.025 - gap - 0.005])
    parts.append(floor)
    return trimesh.util.concatenate(parts)


def _probe():
    box = Box(extents=[0.02, 0.02, 0.05], name='probe')
    return RobotModel(link_list=[box]), box


class TestSampleSurface(unittest.TestCase):

    def test_covers_a_sliver(self):
        mesh = trimesh.creation.box([0.5, 0.03, 0.002])
        spacing = 0.003
        samples = sample_surface(mesh, spacing)
        probe = trimesh.sample.sample_surface(mesh, 2000, seed=0)[0]
        gap = np.min(np.linalg.norm(
            probe[:, None] - samples[None], axis=2), axis=1)
        self.assertLess(gap.max(), spacing)
        # Rows follow the long edges, so a sliver costs about its area.
        self.assertLess(len(samples), 3 * mesh.area / spacing ** 2)


class TestEnvironmentCollision(unittest.TestCase):

    def test_concave_environment(self):
        robot, probe = _probe()
        free = EnvironmentCollision(robot, [('channel', _channel(0.002))],
                                    spacing=0.001)
        self.assertTrue(free.check().is_free)
        hit = EnvironmentCollision(
            robot, [('channel', _channel(0.002, shift=-0.005))],
            spacing=0.001)
        report = hit.check()
        self.assertFalse(report.is_free)
        self.assertAlmostEqual(report.max_depth, 0.003, delta=1e-6)
        contact = report.contacts()[0]
        self.assertEqual(contact['robot_link'], 'probe')
        self.assertEqual(contact['environment_link'], 'channel')

    def test_batch_labels_margin_and_ignore(self):
        robot, probe = _probe()
        checker = EnvironmentCollision(robot, [('channel', _channel(0.002))],
                                       spacing=0.001)
        poses = np.tile(np.eye(4), (4, 1, 1))
        poses[1, 0, 3] = 0.004     # 2 mm into the right wall
        poses[2, 2, 3] = 0.1       # lifted clear
        poses[3] = poses[1]        # the same pose again
        report = checker.check(link_transforms={'probe': poses},
                               labels=['in', 'pushed', 'up', 'again'],
                               margin=0.0025)
        np.testing.assert_allclose(report.min_distance,
                                   [0.002, -0.002, np.inf, -0.002],
                                   atol=1e-9)
        np.testing.assert_array_equal(report.in_collision,
                                      [False, True, False, True])
        pairs = report.pairs()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['labels'], ['pushed', 'again'])
        self.assertEqual(pairs[0]['frames'], [[1, 1], [3, 3]])
        self.assertIn('pushed', report.summary())

        ignoring = EnvironmentCollision(
            robot, [('channel', _channel(0.002))], spacing=0.001,
            ignore=[('probe', 'channel')])
        self.assertTrue(ignoring.check(
            link_transforms={'probe': poses}).is_free)
        with self.assertRaises(ValueError):
            EnvironmentCollision(robot, [('channel', _channel(0.002))],
                                 ignore=[('probe', 'wall')])

    def test_angle_vectors_match_link_transforms(self):
        robot = Panda()
        av = robot.reset_pose().copy()
        # The fingertips reach below its top at the reset pose; bending
        # the elbow lifts them clear.
        table = Box(extents=[0.2, 0.2, 0.02], pos=[0.25, 0.0, 0.64],
                    name='table')
        checker = EnvironmentCollision(robot, [table], spacing=0.003)
        up = av.copy()
        up[3] = -1.6
        avs = np.stack([av, up])
        base = Coordinates(pos=[0.0, 0.0, 0.02])
        report = checker.check(avs, base_pose=base, margin=0.01)
        np.testing.assert_array_equal(report.in_collision, [True, False])
        self.assertLessEqual(
            {'panda_leftfinger', 'panda_rightfinger'},
            {c['robot_link'] for c in report.contacts()})
        recorded = []
        before = robot.angle_vector().copy()
        for row in avs:
            robot.newcoords(base.copy_worldcoords())
            robot.angle_vector(row)
            recorded.append(
                np.stack([lk.worldcoords().T() for lk in robot.link_list]))
        robot.newcoords(Coordinates())
        robot.angle_vector(before)
        again = checker.check(link_transforms=np.stack(recorded), margin=0.01)
        np.testing.assert_allclose(report.min_distance, again.min_distance,
                                   atol=1e-9)
        self.assertEqual([(p['robot_link'], p['environment_link'])
                          for p in report.pairs()],
                         [(p['robot_link'], p['environment_link'])
                          for p in again.pairs()])
        # A placement result: the same postures on the same base.
        result = types.SimpleNamespace(angle_vectors=avs, base_pose=base)
        placed = checker.check_placement(result, margin=0.01)
        np.testing.assert_allclose(placed.min_distance, report.min_distance,
                                   atol=1e-9)
        self.assertEqual(placed.labels, ['target 0', 'target 1'])
        # The robot is left as it was.
        np.testing.assert_array_equal(robot.angle_vector(), before)
        np.testing.assert_array_equal(robot.worldpos(), np.zeros(3))


class TestRobotCollisionCheckerSDF(unittest.TestCase):

    def test_sdf_object_as_obstacle(self):
        robot = Panda()
        robot.reset_pose()
        checker = RobotCollisionChecker(robot)
        checker.add_link_sphere(robot.panda_hand, radius=0.05)
        # An SDF has a worldpos, which used to send it down the
        # primitive branch and fail with "Sphere-GridSDF pair".
        sdf = BoxSDF([0.1, 0.1, 0.1])
        sdf.translate(robot.panda_hand.worldpos() + [0, 0, 0.12])
        checker.add_world_obstacle(sdf)
        np.testing.assert_allclose(
            checker.compute_world_collision_distances(), [0.02], atol=1e-6)
