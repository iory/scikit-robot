import copy
import unittest

import numpy as np
from numpy import testing
import trimesh

import skrobot
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates import make_coords
from skrobot.model import calc_dif_with_axis
from skrobot.model import joint_angle_limit_weight
from skrobot.model import RotationalJoint


class TestRobotModel(unittest.TestCase):

    fetch = None
    kuka = None

    @classmethod
    def setUpClass(cls):
        cls.pr2 = skrobot.models.PR2()
        cls.fetch = skrobot.models.Fetch()
        cls.kuka = skrobot.models.Kuka()

    def test_init(self):
        fetch = self.fetch
        fetch.angle_vector()

    def test_calc_dif_with_axis(self):
        dif = np.array([1, 2, 3])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'x'), [2, 3])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'xx'), [2, 3])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'y'), [1, 3])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'yy'), [1, 3])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'z'), [1, 2])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'zz'), [1, 2])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'xy'), [3])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'yx'), [3])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'yz'), [1])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'zy'), [1])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'xz'), [2])
        testing.assert_array_equal(calc_dif_with_axis(dif, 'zx'), [2])
        testing.assert_array_equal(calc_dif_with_axis(dif, True), [1, 2, 3])
        testing.assert_array_equal(calc_dif_with_axis(dif, False), [])
        testing.assert_array_equal(calc_dif_with_axis(dif, None), [])
        with self.assertRaises(ValueError):
            testing.assert_array_equal(calc_dif_with_axis(dif, [1, 2, 3]))

    def test_visual_mesh(self):
        fetch = self.fetch
        for link in fetch.link_list:
            assert isinstance(link.visual_mesh, list)
            assert all(isinstance(m, trimesh.Trimesh)
                       for m in link.visual_mesh)

    def test_calc_union_link_list(self):
        fetch = self.fetch
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

    def test_find_link_path(self):
        fetch = self.fetch
        links = fetch.find_link_path(
            fetch.torso_lift_link,
            fetch.rarm.end_coords.parent)
        ref_lists = ['torso_lift_link', 'shoulder_pan_link',
                     'shoulder_lift_link', 'upperarm_roll_link',
                     'elbow_flex_link', 'forearm_roll_link',
                     'wrist_flex_link', 'wrist_roll_link', 'gripper_link']
        self.assertEqual([l.name for l in links], ref_lists)
        links = fetch.find_link_path(fetch.rarm.end_coords.parent,
                                     fetch.torso_lift_link)
        self.assertEqual([l.name for l in links], ref_lists[::-1])

    def test_is_relevant(self):
        fetch = self.fetch
        self.assertTrue(fetch._is_relevant(
            fetch.shoulder_pan_joint, fetch.wrist_roll_link))
        self.assertFalse(fetch._is_relevant(
            fetch.shoulder_pan_joint, fetch.base_link))
        self.assertTrue(fetch._is_relevant(
            fetch.shoulder_pan_joint, fetch.rarm_end_coords))

        co = make_coords()
        with self.assertRaises(AssertionError):
            fetch._is_relevant(fetch.shoulder_pan_joint, co)

        # if it's not connected to the robot,
        casco = CascadedCoords()
        with self.assertRaises(AssertionError):
            fetch._is_relevant(fetch.shoulder_pan_joint, casco)

        # but, if casco connects to the robot,
        fetch.rarm_end_coords.assoc(casco)
        self.assertTrue(fetch._is_relevant(
            fetch.shoulder_pan_joint, casco))

    def test_calc_jacobian_from_link_list(self):
        # must be coincide with the one computed via numerical method
        fetch = self.fetch
        link_list = [fetch.torso_lift_link] + fetch.rarm.link_list
        joint_list = [l.joint for l in link_list]

        def set_angle_vector_util(av):
            for joint, angle in zip(joint_list, av):
                joint.joint_angle(angle)

        def compare(move_target):
            # first, compute jacobian numerically
            n_dof = len(joint_list)
            av0 = np.array([0.3] + [-0.4] * 7)
            set_angle_vector_util(av0)
            pos0 = move_target.worldpos()

            jac_numerical = np.zeros((3, n_dof))
            eps = 1e-7
            for idx in range(n_dof):
                av1 = copy.copy(av0)
                av1[idx] += eps
                set_angle_vector_util(av1)
                pos1 = move_target.worldpos()
                jac_numerical[:, idx] = (pos1 - pos0) / eps

            # second compute analytical jacobian
            base_link = fetch.link_list[0]
            jac_analytic = fetch.calc_jacobian_from_link_list(
                move_target, link_list,
                rotation_axis=None, transform_coords=base_link)
            testing.assert_almost_equal(jac_numerical, jac_analytic, decimal=5)
        for move_target in [fetch.rarm_end_coords] + link_list:
            compare(move_target)

        # check initialization of limb of RobotModel
        fetch.rarm.calc_jacobian_from_link_list(fetch.rarm.end_coords)

    def test_calc_inverse_kinematics_nspace_from_link_list(self):
        kuka = self.kuka
        kuka.calc_inverse_kinematics_nspace_from_link_list(
            kuka.rarm.link_list)

    def test_find_joint_angle_limit_weight_from_union_link_list(self):
        fetch = self.fetch
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
        fetch = self.fetch
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
        fetch = self.fetch
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

    def test_inverse_kinematics_args(self):
        kuka = self.kuka
        kuka.inverse_kinematics_args()
        d = kuka.inverse_kinematics_args(
            union_link_list=kuka.rarm.link_list,
            rotation_axis=[True],
            translation_axis=[True])
        self.assertEqual(d['dim'], 6)
        self.assertEqual(d['n_joint_dimension'], 7)

    def test_inverse_kinematics(self):
        kuka = self.kuka
        move_target = kuka.rarm.end_coords
        link_list = kuka.rarm.link_list

        kuka.reset_manip_pose()
        target_coords = kuka.rarm.end_coords.copy_worldcoords().translate([
            0.100, -0.100, 0.100], 'local')
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

        target_coords = kuka.rarm.end_coords.copy_worldcoords().\
            rotate(- np.pi / 6.0, 'y', 'local')

        for rotation_axis in [True,
                              'x', 'y', 'z',
                              'xx', 'yy', 'zz',
                              'xm', 'ym', 'zm']:
            kuka.reset_manip_pose()
            kuka.inverse_kinematics(
                target_coords,
                move_target=move_target,
                link_list=link_list,
                translation_axis=True,
                rotation_axis=rotation_axis)
            dif_pos = kuka.rarm.end_coords.difference_position(
                target_coords, True)
            dif_rot = kuka.rarm.end_coords.difference_rotation(
                target_coords, rotation_axis)
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

        # inverse kinematics with linear joint
        fetch = self.fetch
        fetch.reset_manip_pose()
        target_coords = make_coords(pos=[1.0, 0, 1.5])
        fetch.inverse_kinematics(
            target_coords,
            move_target=fetch.rarm.end_coords,
            link_list=[fetch.torso_lift_link] + fetch.rarm.link_list,
            stop=200)
        dif_pos = fetch.rarm.end_coords.difference_position(
            target_coords, True)
        dif_rot = fetch.rarm.end_coords.difference_rotation(
            target_coords, True)
        self.assertLess(np.linalg.norm(dif_pos), 0.001)
        self.assertLess(np.linalg.norm(dif_rot), np.deg2rad(1))

        # assoc coords
        robot = self.pr2
        assoc_coords = robot.larm.end_coords.copy_worldcoords().translate(
            [0.3, 0, 0])
        assoc_coords_axis = skrobot.model.Axis()
        assoc_coords_axis.newcoords(assoc_coords.copy_worldcoords())
        robot.larm.end_coords.parent.assoc(assoc_coords_axis, 'world')
        robot.reset_pose()
        ret = robot.larm.inverse_kinematics(
            assoc_coords_axis, move_target=assoc_coords_axis)
        self.assertIsNot(ret, False)
        robot.reset_pose()
        ret = robot.larm.inverse_kinematics(
            assoc_coords_axis.copy_worldcoords(),
            move_target=assoc_coords_axis)
        self.assertIsNot(ret, False)
        robot.reset_pose()
        ret = robot.larm.inverse_kinematics(
            assoc_coords_axis.copy_worldcoords(),
            move_target=robot.larm.end_coords)
        self.assertIsNot(ret, False)
        robot.reset_pose()
        ret = robot.larm.inverse_kinematics(
            assoc_coords_axis.copy_worldcoords(),
            move_target=robot.larm.end_coords.parent)
        self.assertIsNot(ret, False)

    def test_calc_target_joint_dimension(self):
        fetch = self.fetch
        joint_dimension = fetch.calc_target_joint_dimension(
            fetch.rarm.link_list)
        self.assertEqual(joint_dimension, 7)
        joint_dimension = fetch.calc_target_joint_dimension(
            [fetch.rarm.link_list, fetch.rarm.link_list])
        self.assertEqual(joint_dimension, 7)

    def test_calc_target_axis_dimension(self):
        fetch = self.fetch
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
        r = self.fetch
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
            max_angle=np.deg2rad(32.3493),
            min_angle=np.deg2rad(-122.349))
        j1.joint_angle(np.deg2rad(-60.0))
        testing.assert_almost_equal(
            joint_angle_limit_weight([j1]),
            np.float32(3.1019381e-01))

        j2 = RotationalJoint(
            child_link=make_coords(),
            max_angle=np.deg2rad(74.2725),
            min_angle=np.deg2rad(-20.2598))
        j2.joint_angle(np.deg2rad(74.0))
        testing.assert_almost_equal(
            joint_angle_limit_weight([j2]),
            np.float32(1.3539208e+03))

        j3 = RotationalJoint(
            child_link=make_coords(),
            max_angle=float('inf'),
            min_angle=-float('inf'))
        j3.joint_angle(np.deg2rad(-20.0))
        testing.assert_almost_equal(
            joint_angle_limit_weight([j3]),
            np.float32(0.0))
