import unittest

import numpy as np
from numpy import testing

import skrobot


class TestRobotModel(unittest.TestCase):

    def test_init(self):
        fetch = skrobot.robot_models.Fetch()
        fetch.angle_vector()

    def test_calc_union_link_list(self):
        fetch = skrobot.robot_models.Fetch()
        links = fetch.calc_union_link_list([fetch.rarm.link_list,
                                            fetch.rarm.link_list,
                                            fetch.link_list])
        self.assertEqual([l.name for l in links],
                         ['shoulder_pan_link',
                          'shoulder_lift_link',
                          'upperarm_roll_link',
                          'elbow_flex_link',
                          'forearm_roll_link',
                          'wrist_flex_link',
                          'wrist_roll_link',
                          'base_link',
                          'torso_lift_link',
                          'head_pan_link',
                          'head_tilt_link'])

    def test_find_link_route(self):
        fetch = skrobot.robot_models.Fetch()
        ret = fetch.find_link_route(fetch.torso_lift_link)
        self.assertEqual(ret,
                         [fetch.torso_lift_link])

        ret = fetch.find_link_route(fetch.wrist_roll_link)
        self.assertEqual(ret,
                         [fetch.torso_lift_link,
                          fetch.shoulder_pan_link,
                          fetch.shoulder_lift_link,
                          fetch.upperarm_roll_link,
                          fetch.elbow_flex_link,
                          fetch.forearm_roll_link,
                          fetch.wrist_flex_link,
                          fetch.wrist_roll_link])

    def test_inverse_kinematics(self):
        kuka = skrobot.robot_models.Kuka()
        move_target = kuka.rarm.end_coords
        link_list = kuka.rarm.link_list

        kuka.reset_manip_pose()
        target_coords = kuka.rarm.end_coords.copy_worldcoords().translate([
            100, -100, 100], 'local')
        kuka.inverse_kinematics(
            target_coords,
            move_target=move_target,
            link_list=link_list,
            translation_axis=True,
            rotation_axis=True)
        dif_pos = kuka.rarm.end_coords.difference_position(target_coords, True)
        dif_rot = kuka.rarm.end_coords.difference_rotation(target_coords, True)
        self.assertLess(np.linalg.norm(dif_pos), 0.001)
        self.assertLess(np.linalg.norm(dif_rot), np.deg2rad(1))

        kuka.reset_manip_pose()
        target_coords = kuka.rarm.end_coords.copy_worldcoords().\
            rotate(- np.pi / 6.0, 'y', 'local')
        kuka.inverse_kinematics(
            target_coords,
            move_target=move_target,
            link_list=link_list,
            translation_axis=True,
            rotation_axis=True)
        dif_pos = kuka.rarm.end_coords.difference_position(target_coords, True)
        dif_rot = kuka.rarm.end_coords.difference_rotation(target_coords, True)
        self.assertLess(np.linalg.norm(dif_pos), 0.001)
        self.assertLess(np.linalg.norm(dif_rot), np.deg2rad(1))

    def test_calc_target_joint_dimension(self):
        fetch = skrobot.robot_models.Fetch()
        joint_dimension = fetch.calc_target_joint_dimension(
            fetch.rarm.link_list)
        self.assertEqual(joint_dimension, 7)
        joint_dimension = fetch.calc_target_joint_dimension(
            [fetch.rarm.link_list, fetch.rarm.link_list])
        self.assertEqual(joint_dimension, 7)

    def test_calc_target_axis_dimension(self):
        fetch = skrobot.robot_models.Fetch()
        dimension = fetch.calc_target_axis_dimension(
            False, False)
        self.assertEqual(dimension, 0)
        dimension = fetch.calc_target_axis_dimension(
            [True, True], [True, True])
        self.assertEqual(dimension, 12)

        with self.assertRaises(ValueError):
            dimension = fetch.calc_target_axis_dimension(
                [True, False], True)

    def test_calc_jacobian_for_interlocking_joints(self):
        r = skrobot.robot_models.Fetch()
        jacobian = r.calc_jacobian_for_interlocking_joints(
            r.rarm.link_list,
            interlocking_joint_pairs=[
                (r.shoulder_pan_joint, r.elbow_flex_joint)])
        testing.assert_almost_equal(
            np.array([[1, 0, 0, -1, 0, 0, 0]],
                     dtype='f'),
            jacobian)
