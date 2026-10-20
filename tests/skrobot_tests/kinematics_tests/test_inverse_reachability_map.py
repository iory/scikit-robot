import os
import tempfile
import unittest

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.kinematics import find_base_placement
from skrobot.kinematics import InverseReachabilityMap
from skrobot.kinematics import set_base_pose
from skrobot.models import Fetch


class TestInverseReachabilityMap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.robot = Fetch()
        cls.robot.reset_pose()
        cls.irm = InverseReachabilityMap(
            cls.robot, cls.robot.rarm.end_coords).compute(
                n_samples=200000)

    def setUp(self):
        self.robot.reset_pose()
        set_base_pose(self.robot, Coordinates())

    def _targets_from(self, base):
        """End-effector poses of a few stored samples, seen from base."""
        idx = np.random.RandomState(1).choice(
            self.irm.n_samples, 3, replace=False)
        targets = []
        for i in idx:
            hand = Coordinates(pos=self.irm.positions[i].astype(np.float64),
                               rot=self.irm.rotations[i].astype(np.float64))
            targets.append(base.copy_worldcoords().transform(hand))
        return targets

    def test_true_base_is_proposed(self):
        base = Coordinates(pos=[1.0, -2.0, 0.0]).rotate(2.0, 'z')
        candidates = self.irm.candidate_bases(
            self._targets_from(base), n_candidates=20)
        self.assertGreater(len(candidates), 0)
        self.assertEqual(candidates.coverage[0], 3)
        dist = np.hypot(candidates.bases[:, 0] - 1.0,
                        candidates.bases[:, 1] + 2.0)
        yaw = np.abs(np.arctan2(np.sin(candidates.bases[:, 2] - 2.0),
                                np.cos(candidates.bases[:, 2] - 2.0)))
        full = candidates.coverage == 3
        self.assertTrue(np.any(full & (dist < 0.2) & (yaw < 0.4)))

    def test_base_height_and_free_axis_mask(self):
        base = Coordinates(pos=[1.0, -2.0, -0.2]).rotate(2.0, 'z')
        targets = self._targets_from(base)
        for mask in (True, 'xz'):
            candidates = self.irm.candidate_bases(
                targets, n_candidates=20, base_height=-0.2,
                rotation_mask=mask)
            dist = np.hypot(candidates.bases[:, 0] - 1.0,
                            candidates.bases[:, 1] + 2.0)
            yaw = np.abs(np.arctan2(np.sin(candidates.bases[:, 2] - 2.0),
                                    np.cos(candidates.bases[:, 2] - 2.0)))
            full = candidates.coverage == 3
            self.assertTrue(np.any(full & (dist < 0.2) & (yaw < 0.4)),
                            mask)
            self.assertAlmostEqual(candidates.poses[0].worldpos()[2], -0.2)

    def test_hybrid_solve_respects_is_valid_base(self):
        base = Coordinates(pos=[1.0, -2.0, 0.0]).rotate(2.0, 'z')
        targets = self._targets_from(base)

        def is_valid_base(pose):
            return pose.worldpos()[0] < 1.2

        result = find_base_placement(
            self.robot, targets, move_target=self.robot.rarm.end_coords,
            irm=self.irm, is_valid_base=is_valid_base)
        self.assertTrue(result.all_reached, result)
        self.assertTrue(result.base_valid)
        self.assertLess(result.base_pose.worldpos()[0], 1.2)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'irm.npz')
            self.irm.save(path)
            loaded = InverseReachabilityMap(
                self.robot, self.robot.rarm.end_coords).load(path)
            np.testing.assert_array_equal(
                loaded.positions, self.irm.positions)
            other = InverseReachabilityMap(
                self.robot, self.robot.rarm.end_coords,
                joint_list=self.robot.rarm.joint_list[1:])
            with self.assertRaises(ValueError):
                other.load(path)


if __name__ == '__main__':
    unittest.main()
