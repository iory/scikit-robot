import unittest

import numpy as np
from numpy import testing

import skrobot
from skrobot.coordinates import make_coords
from skrobot.robot_model import joint_angle_limit_weight
from skrobot.robot_model import RotationalJoint


class TestRobotModel(unittest.TestCase):

    def test_init(self):
        fetch = skrobot.robot_models.Fetch()
        fetch.angle_vector()

    def test_calc_union_link_list(self):
        fetch = skrobot.robot_models.Fetch()
        links = fetch.calc_union_link_list([fetch.rarm.link_list,
                                            fetch.rarm.link_list,
                                            fetch.link_list])
        self.assertEqual(
            [l.name for l in links],
            [
                'shoulder_pan_link',
                'shoulder_lift_link',
                'upperarm_roll_link',
                'elbow_flex_link',
                'forearm_roll_link',
                'wrist_flex_link',
                'wrist_roll_link',
                'base_link',
                'r_wheel_link',
                'l_wheel_link',
                'torso_lift_link',
                'head_pan_link',
                'head_tilt_link',
                'gripper_link',
                'r_gripper_finger_link',
                'l_gripper_finger_link',
                'bellows_link2',
                'estop_link',
                'laser_link',
                'torso_fixed_link',
                'head_camera_link',
                'head_camera_rgb_frame',
                'head_camera_rgb_optical_frame',
                'head_camera_depth_frame',
                'head_camera_depth_optical_frame',
            ],
        )

    def test_find_joint_angle_limit_weight_from_union_link_list(self):
        fetch = skrobot.robot_models.Fetch()
        links = fetch.calc_union_link_list([fetch.rarm.link_list])

        # not set joint_angle_limit case
        fetch.reset_joint_angle_limit_weight(links)
        names, weights = fetch.\
            find_joint_angle_limit_weight_from_union_link_list(links)
        self.assertEqual(
            weights,
            False)

        # set joint_angle_limit case
        fetch.joint_angle_limit_weight_maps[names] = (
            names, np.ones(len(names), 'f'))
        names, weights = fetch.\
            find_joint_angle_limit_weight_from_union_link_list(links)
        testing.assert_almost_equal(
            weights,
            np.ones(len(names), 'f'))

    def test_reset_joint_angle_limit_weight(self):
        fetch = skrobot.robot_models.Fetch()
        links = fetch.calc_union_link_list([fetch.rarm.link_list])

        # not set joint_angle_limit case
        fetch.reset_joint_angle_limit_weight(links)
        names, weights = fetch.\
            find_joint_angle_limit_weight_from_union_link_list(links)
        self.assertEqual(
            weights,
            False)

        # set joint_angle_limit case
        fetch.joint_angle_limit_weight_maps[names] = (
            names, np.ones(len(names), 'f'))
        fetch.reset_joint_angle_limit_weight(links)
        names, weights = fetch.\
            find_joint_angle_limit_weight_from_union_link_list(links)
        self.assertEqual(
            weights,
            False)

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

        # ik failed case
        av = kuka.reset_manip_pose()
        target_coords = kuka.rarm.end_coords.copy_worldcoords().\
            translate([10000, 0, 0], 'local')
        ik_result = kuka.inverse_kinematics(
            target_coords,
            move_target=move_target,
            link_list=link_list,
            translation_axis=True,
            rotation_axis=True)
        self.assertEqual(ik_result, False)
        testing.assert_array_equal(
            av, kuka.angle_vector())

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

    def test_joint_angle_limit_weight(self):
        j1 = RotationalJoint(
            child_link=make_coords(),
            max_angle=32.3493,
            min_angle=-122.349)
        j1.joint_angle(-60.0)
        testing.assert_almost_equal(
            joint_angle_limit_weight([j1]),
            3.1019381e-01)

        j2 = RotationalJoint(
            child_link=make_coords(),
            max_angle=74.2725,
            min_angle=-20.2598)
        j2.joint_angle(74.0)
        testing.assert_almost_equal(
            joint_angle_limit_weight([j2]),
            1.3539208e+03)

        j3 = RotationalJoint(
            child_link=make_coords(),
            max_angle=float('inf'),
            min_angle=-float('inf'))
        j3.joint_angle(-20.0)
        testing.assert_almost_equal(
            joint_angle_limit_weight([j3]),
            0.0)
