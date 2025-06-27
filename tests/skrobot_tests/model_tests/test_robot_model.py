import copy
import os
import sys
import unittest

import numpy as np
from numpy import testing
import pytest
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

    @pytest.mark.skipif(sys.version_info[0] == 2, reason="Skip in Python 2")
    def test_from_urdf(self):
        import tempfile

        # Load from URDF string
        urdf_string = """
<robot name="test_robot">
  <link name="base_link"/>
  <joint name="joint1" type="fixed">
    <parent link="base_link"/>
    <child link="link1"/>
  </joint>
  <link name="link1"/>
</robot>
"""
        robot_from_string = skrobot.model.RobotModel.from_urdf(urdf_string)
        self.assertEqual(robot_from_string.name, "test_robot")
        self.assertIn("base_link", robot_from_string.__dict__)
        self.assertIn("link1", robot_from_string.__dict__)
        self.assertIn("joint1", robot_from_string.__dict__)

        # Create a temporary URDF file
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.urdf', delete=False) as tmp_file:
            tmp_file.write(urdf_string)
            urdf_path = tmp_file.name

        # Load from URDF file path
        robot_from_path = skrobot.model.RobotModel.from_urdf(urdf_path)
        self.assertEqual(robot_from_path.name, "test_robot")
        self.assertIn("base_link", robot_from_path.__dict__)
        self.assertIn("link1", robot_from_path.__dict__)
        self.assertIn("joint1", robot_from_path.__dict__)

        # Clean up the temporary file
        os.remove(urdf_path)

        # Test with a valid file path but different content
        urdf_string_2 = """
<robot name="another_robot">
  <link name="base"/>
</robot>
"""
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.urdf', delete=False) as tmp_file_2:
            tmp_file_2.write(urdf_string_2)
            urdf_path_2 = tmp_file_2.name

        robot_from_path_2 = skrobot.model.RobotModel.from_urdf(urdf_path_2)
        self.assertEqual(robot_from_path_2.name, "another_robot")
        self.assertIn("base", robot_from_path_2.__dict__)

        os.remove(urdf_path_2)

    def test_torque_vector_basic(self):
        """Basic test for torque_vector method."""
        pr2 = self.pr2
        pr2.init_pose()

        # Test that torque_vector method exists and returns correct shape
        torques = pr2.torque_vector()
        self.assertEqual(len(torques), len(pr2.joint_list))

        # Test with upward gravity
        torques_upward = pr2.torque_vector(gravity=np.array([0, 0, 9.80665]))
        self.assertEqual(len(torques_upward), len(pr2.joint_list))

        # Torques should have opposite signs due to gravity direction
        for i in range(len(torques)):
            if abs(torques[i]) > 1e-6:  # Skip near-zero values
                # Upward gravity should have opposite sign to default (downward gravity)
                self.assertLess(
                    torques[i] * torques_upward[i], 0,
                    msg="Joint {}: torques should have opposite signs".format(i)
                )

    def test_torque_vector_static(self):
        """Test static torque calculation (gravity only)."""
        robot = self.pr2
        robot.init_pose()

        # Test with default gravity (should be downward)
        torques_default = robot.torque_vector()

        # Test with explicit downward gravity
        torques_downward = robot.torque_vector(gravity=np.array([0, 0, -9.80665]))

        # Test with upward gravity
        torques_upward = robot.torque_vector(gravity=np.array([0, 0, 9.80665]))

        # Default should match explicit downward
        testing.assert_array_almost_equal(
            torques_default, torques_downward, decimal=10,
            err_msg="Default gravity should be downward"
        )

        # Torques should have opposite signs
        for i in range(len(torques_downward)):
            self.assertAlmostEqual(
                torques_downward[i] + torques_upward[i], 0.0,
                places=5,
                msg="Joint {}: torques with opposite gravity should sum to zero".format(i)
            )

    def test_torque_vector_with_external_forces(self):
        """Test torque calculation with external forces."""
        robot = self.pr2
        robot.init_pose()

        # Apply 10N force in -Z direction at end effector
        force = np.array([0, 0, -10.0])
        target_coords = robot.rarm.end_coords

        torques_no_force = robot.torque_vector()
        torques_with_force = robot.torque_vector(
            force_list=[force],
            target_coords=[target_coords]
        )

        # Torques should be different when external force is applied
        diff = np.linalg.norm(torques_with_force - torques_no_force)
        self.assertGreater(
            diff, 0.1,
            msg="External force should change joint torques"
        )

    def test_torque_vector_different_poses(self):
        """Test torque calculation at different robot poses."""
        robot = self.pr2

        # Test 1: Init pose
        robot.init_pose()
        torques_init = robot.torque_vector()

        # Test 2: Reset pose (arms down)
        robot.reset_pose()
        torques_reset = robot.torque_vector()

        # Shoulder lift joint should have different torques
        shoulder_idx = robot.joint_names.index('l_shoulder_lift_joint')
        self.assertNotAlmostEqual(
            torques_init[shoulder_idx],
            torques_reset[shoulder_idx],
            places=0,
            msg="Shoulder lift torque should change with pose"
        )

    def test_torque_vector_zero_gravity(self):
        """Test torque calculation with zero gravity."""
        robot = self.pr2
        robot.init_pose()

        # With zero gravity, static torques should be zero
        torques = robot.torque_vector(gravity=np.array([0, 0, 0]))

        for i, torque in enumerate(torques):
            self.assertAlmostEqual(
                torque, 0.0,
                places=10,
                msg="Joint {}: torque should be zero with no gravity".format(i)
            )

    def test_torque_vector_link_mass_effect(self):
        """Test that link masses affect torque calculation correctly."""
        robot = self.pr2
        robot.init_pose()

        # Get original torques
        torques_original = robot.torque_vector()

        # Double the mass of forearm link
        forearm_link = robot.l_forearm_link
        original_mass = forearm_link.mass
        forearm_link.mass = original_mass * 2.0

        # Recalculate torques
        torques_doubled = robot.torque_vector()

        # Restore original mass
        forearm_link.mass = original_mass

        # Shoulder and elbow joints should see increased torque
        shoulder_idx = robot.joint_names.index('l_shoulder_lift_joint')
        elbow_idx = robot.joint_names.index('l_elbow_flex_joint')

        self.assertGreater(
            abs(torques_doubled[shoulder_idx]),
            abs(torques_original[shoulder_idx]),
            msg="Shoulder torque should increase with heavier forearm"
        )
        self.assertGreater(
            abs(torques_doubled[elbow_idx]),
            abs(torques_original[elbow_idx]),
            msg="Elbow torque should increase with heavier forearm"
        )
