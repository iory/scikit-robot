import unittest

import numpy as np

import skrobot


class TestAero(unittest.TestCase):

    def test_init(self):
        skrobot.models.Aero()

    def test_init_nohand(self):
        skrobot.models.Aero(use_hand=False)

    def test_angle_vector_excludes_mimic_joints(self):
        for use_hand in (True, False):
            robot = skrobot.models.Aero(use_hand=use_hand)
            names = [j.name for j in robot.joint_list]
            self.assertFalse(any('mimic' in n for n in names))
            self.assertFalse(any('dummy' in n for n in names))
            self.assertEqual(
                any(('index' in n or 'thumb' in n) for n in names),
                use_hand)

    def test_reset_pose_drives_lifter_mimic(self):
        robot = skrobot.models.Aero()
        robot.reset_pose()
        np.testing.assert_allclose(
            robot.ankle_joint_mimic.joint_angle(),
            -robot.ankle_joint.joint_angle(), atol=1e-9)
        np.testing.assert_allclose(
            robot.knee_joint_mimic.joint_angle(),
            -robot.knee_joint.joint_angle(), atol=1e-9)

    def test_rarm_inverse_kinematics(self):
        robot = skrobot.models.Aero()
        robot.reset_pose()
        target = robot.rarm.end_coords.copy_worldcoords().translate(
            [0.03, 0.0, -0.05], 'world')
        result = robot.rarm.inverse_kinematics(
            target.copy_worldcoords(), rotation_axis=True, stop=300)
        self.assertIsNot(result, False)
        err = np.linalg.norm(
            robot.rarm.end_coords.worldpos() - target.worldpos())
        self.assertLess(err, 1e-3)
