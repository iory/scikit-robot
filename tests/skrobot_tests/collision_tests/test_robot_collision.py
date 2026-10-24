import unittest

import numpy as np

from skrobot.collision import RobotCollisionChecker
from skrobot.models import Panda


class TestSelfCollisionPairsTouchingAtRest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.robot = Panda()
        cls.links = [lk for lk in cls.robot.link_list
                     if getattr(lk, 'collision_mesh', None) is not None]

    def checker(self):
        checker = RobotCollisionChecker(self.robot)
        checker.add_links(self.links)
        return checker

    def test_rest_pose_overlaps_left_out(self):
        rest = self.robot.angle_vector().copy()
        checker = self.checker()
        checker.setup_self_collision_pairs()
        # the spheres and capsules are fatter than the links: some pairs
        # overlap in Panda's own rest pose (the hand on link7, the fingers)
        self.assertLess(checker.compute_self_collision_distances().min(), 0.0)
        n_all = len(checker._self_collision_pairs)

        margin = 0.002
        checker.setup_self_collision_pairs(ignore_touching_at=[rest],
                                           touching_margin=margin)
        d = checker.compute_self_collision_distances()
        self.assertGreater(len(d), 0)
        self.assertLess(len(d), n_all)
        self.assertGreaterEqual(d.min(), margin)
        np.testing.assert_allclose(self.robot.angle_vector(), rest)

    def test_other_poses_count_too(self):
        rest = self.robot.angle_vector().copy()
        bent = rest.copy()
        bent[3] = self.robot.joint_list[3].min_angle
        checker = self.checker()
        checker.setup_self_collision_pairs(ignore_touching_at=[rest])
        one = len(checker._self_collision_pairs)
        checker.setup_self_collision_pairs(ignore_touching_at=[rest, bent])
        self.assertLessEqual(len(checker._self_collision_pairs), one)
        np.testing.assert_allclose(self.robot.angle_vector(), rest)


if __name__ == '__main__':
    unittest.main()
