import copy
import os
import sys
import unittest

import numpy as np
from numpy import testing
import trimesh

import skrobot
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates import make_coords
from skrobot.model import calc_dif_with_axis
from skrobot.model import joint_angle_limit_weight
from skrobot.model import LinearJoint
from skrobot.model import Link
from skrobot.model import RotationalJoint
from skrobot.model.joint import calc_target_joint_dimension


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

    @unittest.skipIf(sys.version_info[0] == 2, "Skip in Python 2")
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

    def test_centroid(self):
        """Test centroid (center of gravity) calculation."""
        pr2 = self.pr2
        pr2.reset_pose()

        # Test that centroid method exists and returns correct shape
        cog = pr2.centroid()
        self.assertEqual(cog.shape, (3,))
        self.assertIsInstance(cog, np.ndarray)

    def test_batch_inverse_kinematics_basic(self):
        """Test basic batch inverse kinematics functionality."""
        fetch = self.fetch
        fetch.reset_pose()

        # Create test target poses using current position with small offsets
        current_pos = fetch.rarm.end_coords.worldpos()
        target_coords = [
            skrobot.coordinates.Coordinates(pos=current_pos + [0.01, 0.0, 0.0]),
            skrobot.coordinates.Coordinates(pos=current_pos + [0.0, 0.01, 0.0]),
            skrobot.coordinates.Coordinates(pos=current_pos + [0.0, 0.0, 0.01])
        ]

        # Test basic functionality
        solutions, success_flags, attempt_counts = fetch.batch_inverse_kinematics(
            target_coords,
            move_target=fetch.rarm.end_coords,
            stop=50,
            attempts_per_pose=1
        )

        # Check return types and shapes
        self.assertEqual(len(solutions), len(target_coords))
        self.assertEqual(len(success_flags), len(target_coords))
        self.assertEqual(len(attempt_counts), len(target_coords))

        for solution in solutions:
            self.assertEqual(len(solution), len(fetch.angle_vector()))
            self.assertIsInstance(solution, np.ndarray)

        for success in success_flags:
            self.assertIsInstance(success, bool)

        for attempts in attempt_counts:
            self.assertIsInstance(attempts, int)
            self.assertGreaterEqual(attempts, 1)

    def test_batch_inverse_kinematics_numpy_input(self):
        """Test batch IK with numpy array input."""
        fetch = self.fetch
        fetch.reset_pose()

        # Test with 6D numpy array input (x, y, z, roll, pitch, yaw)
        target_poses_6d = np.array([
            [0.7, -0.2, 0.9, 0.0, 0.0, 0.0],
            [0.6, -0.3, 1.0, 0.0, np.deg2rad(30), 0.0]
        ])

        solutions, success_flags, attempt_counts = fetch.batch_inverse_kinematics(
            target_poses_6d,
            move_target=fetch.rarm.end_coords,
            stop=50,
            attempts_per_pose=1
        )

        self.assertEqual(len(solutions), 2)
        self.assertEqual(len(success_flags), 2)
        self.assertEqual(len(attempt_counts), 2)

        # Test with 7D numpy array input (x, y, z, qw, qx, qy, qz)
        target_poses_7d = np.array([
            [0.7, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0],
            [0.6, -0.3, 1.0, 1.0, 0.0, 0.0, 0.0]
        ])

        solutions, success_flags, attempt_counts = fetch.batch_inverse_kinematics(
            target_poses_7d,
            move_target=fetch.rarm.end_coords,
            stop=50,
            attempts_per_pose=1
        )

        self.assertEqual(len(solutions), 2)
        self.assertEqual(len(success_flags), 2)
        self.assertEqual(len(attempt_counts), 2)

    def test_batch_inverse_kinematics_initial_angles(self):
        """Test batch IK with different initial_angles options."""
        fetch = self.fetch
        fetch.reset_pose()

        current_pos = fetch.rarm.end_coords.worldpos()
        target_coords = [skrobot.coordinates.Coordinates(pos=current_pos)]

        # Test with None (random)
        solutions1, success1, _ = fetch.batch_inverse_kinematics(
            target_coords,
            move_target=fetch.rarm.end_coords,
            initial_angles=None,
            stop=10,
            attempts_per_pose=1
        )

        # Test with "random"
        solutions2, success2, _ = fetch.batch_inverse_kinematics(
            target_coords,
            move_target=fetch.rarm.end_coords,
            initial_angles="random",
            stop=10,
            attempts_per_pose=1
        )

        # Test with "current"
        solutions3, success3, _ = fetch.batch_inverse_kinematics(
            target_coords,
            move_target=fetch.rarm.end_coords,
            initial_angles="current",
            stop=10,
            attempts_per_pose=1
        )

        # All should return valid results
        self.assertEqual(len(solutions1), 1)
        self.assertEqual(len(solutions2), 1)
        self.assertEqual(len(solutions3), 1)

        # Test with numpy array
        link_list = fetch.link_lists(fetch.rarm.end_coords.parent)
        joint_list_without_fixed = fetch.joint_list_from_link_list(link_list, ignore_fixed_joint=True)
        ndof = calc_target_joint_dimension(joint_list_without_fixed)

        initial_angles_array = np.random.uniform(-1, 1, (1, ndof))
        solutions4, success4, _ = fetch.batch_inverse_kinematics(
            target_coords,
            move_target=fetch.rarm.end_coords,
            initial_angles=initial_angles_array,
            stop=10,
            attempts_per_pose=1
        )
        self.assertEqual(len(solutions4), 1)

    def test_batch_inverse_kinematics_multiple_attempts(self):
        """Test batch IK with multiple attempts per pose."""
        fetch = self.fetch
        fetch.reset_pose()

        target_coords = [skrobot.coordinates.Coordinates(pos=[0.7, -0.2, 0.9])]

        # Test with multiple attempts
        solutions, success_flags, attempt_counts = fetch.batch_inverse_kinematics(
            target_coords,
            move_target=fetch.rarm.end_coords,
            stop=10,
            attempts_per_pose=5
        )

        self.assertEqual(len(solutions), 1)
        self.assertEqual(len(success_flags), 1)
        self.assertEqual(len(attempt_counts), 1)
        self.assertLessEqual(attempt_counts[0], 5)

    def test_batch_inverse_kinematics_axis_constraints(self):
        """Test batch IK with different axis constraints."""
        fetch = self.fetch
        fetch.reset_pose()

        target_coords = [skrobot.coordinates.Coordinates(pos=[0.7, -0.2, 0.9])]

        # Test different rotation axis constraints
        for rotation_axis in [True, False, 'x', 'y', 'z', 'xy', 'xyz']:
            solutions, success_flags, _ = fetch.batch_inverse_kinematics(
                target_coords,
                move_target=fetch.rarm.end_coords,
                rotation_axis=rotation_axis,
                stop=20,
                attempts_per_pose=1
            )
            self.assertEqual(len(solutions), 1)
            self.assertEqual(len(success_flags), 1)

        # Test different translation axis constraints
        for translation_axis in [True, False, 'x', 'y', 'z', 'xy', 'xyz']:
            solutions, success_flags, _ = fetch.batch_inverse_kinematics(
                target_coords,
                move_target=fetch.rarm.end_coords,
                translation_axis=translation_axis,
                stop=20,
                attempts_per_pose=1
            )
            self.assertEqual(len(solutions), 1)
            self.assertEqual(len(success_flags), 1)

    def test_batch_inverse_kinematics_parameters(self):
        """Test batch IK with different parameter values."""
        fetch = self.fetch
        fetch.reset_pose()

        target_coords = [skrobot.coordinates.Coordinates(pos=[0.7, -0.2, 0.9])]

        # Test different alpha values
        for alpha in [0.1, 0.5, 1.0]:
            solutions, success_flags, _ = fetch.batch_inverse_kinematics(
                target_coords,
                move_target=fetch.rarm.end_coords,
                alpha=alpha,
                stop=10,
                attempts_per_pose=1
            )
            self.assertEqual(len(solutions), 1)

        # Test different threshold values
        solutions, success_flags, _ = fetch.batch_inverse_kinematics(
            target_coords,
            move_target=fetch.rarm.end_coords,
            thre=0.01,
            rthre=np.deg2rad(5),
            stop=10,
            attempts_per_pose=1
        )
        self.assertEqual(len(solutions), 1)

    def test_batch_inverse_kinematics_error_cases(self):
        """Test batch IK error handling."""
        fetch = self.fetch
        fetch.reset_pose()

        # Test invalid target_coords shape
        with self.assertRaises(ValueError):
            fetch.batch_inverse_kinematics(
                np.array([[0.7, -0.2]]),  # Wrong shape
                move_target=fetch.rarm.end_coords
            )

        # Test invalid initial_angles type
        with self.assertRaises(ValueError):
            fetch.batch_inverse_kinematics(
                [skrobot.coordinates.Coordinates(pos=[0.7, -0.2, 0.9])],
                move_target=fetch.rarm.end_coords,
                initial_angles="invalid_string"
            )

        # Test invalid initial_angles shape
        with self.assertRaises(ValueError):
            fetch.batch_inverse_kinematics(
                [skrobot.coordinates.Coordinates(pos=[0.7, -0.2, 0.9])],
                move_target=fetch.rarm.end_coords,
                initial_angles=np.array([[1, 2, 3]])  # Wrong ndof
            )

    def test_batch_inverse_kinematics_solution_accuracy(self):
        """Test that batch IK solutions are accurate."""
        fetch = self.fetch
        fetch.reset_pose()

        # Create target coordinates
        target_coords = [skrobot.coordinates.Coordinates(pos=[0.7, -0.2, 0.9])]

        # Solve IK
        solutions, success_flags, _ = fetch.batch_inverse_kinematics(
            target_coords,
            move_target=fetch.rarm.end_coords,
            stop=100,
            attempts_per_pose=10,
            thre=0.001
        )

        # If IK succeeded, check accuracy
        if success_flags[0]:
            # Apply solution and check error
            original_angles = fetch.angle_vector()
            fetch.angle_vector(solutions[0])

            achieved_pos = fetch.rarm.end_coords.worldpos()
            target_pos = target_coords[0].worldpos()

            pos_error = np.linalg.norm(achieved_pos - target_pos)
            self.assertLess(pos_error, 0.01, "Position error should be small")

            # Restore original angles
            fetch.angle_vector(original_angles)

    def test_batch_inverse_kinematics_consistency(self):
        """Test that batch IK gives consistent results."""
        fetch = self.fetch
        fetch.reset_pose()

        target_coords = [skrobot.coordinates.Coordinates(pos=[0.7, -0.2, 0.9])]

        # Run same problem multiple times with current initial angles
        results = []
        for _ in range(3):
            solutions, success_flags, _ = fetch.batch_inverse_kinematics(
                target_coords,
                move_target=fetch.rarm.end_coords,
                initial_angles="current",
                stop=50,
                attempts_per_pose=1
            )
            results.append((solutions[0], success_flags[0]))

        # All should have same success status when starting from same initial pose
        success_statuses = [result[1] for result in results]
        if any(success_statuses):
            # If any succeeded, check that solutions are similar
            successful_solutions = [result[0] for result in results if result[1]]
            if len(successful_solutions) > 1:
                for i in range(1, len(successful_solutions)):
                    diff = np.linalg.norm(successful_solutions[i] - successful_solutions[0])
                    # Solutions should be identical when starting from same initial pose
                    self.assertLess(diff, 1e-6, "Solutions should be identical with same initial pose")

    def test_batch_inverse_kinematics_r8_6_mimic_joints(self):
        """Test batch IK with R8_6 robot which has mimic joints.

        R8_6 robot has mimic joints in its elbow mechanism, making it a good
        test case for verifying that mimic joints are handled correctly in
        batch inverse kinematics.
        """
        r8_6 = skrobot.models.R8_6()
        r8_6.reset_pose()

        # Create target poses near current position
        current_pos = r8_6.rarm.end_coords.worldpos()
        target_coords = [
            skrobot.coordinates.Coordinates(pos=current_pos + [0.05, 0.0, 0.0]),
            skrobot.coordinates.Coordinates(pos=current_pos + [0.0, 0.05, 0.0]),
            skrobot.coordinates.Coordinates(pos=current_pos + [0.0, 0.0, 0.05]),
        ]

        solutions, success_flags, _ = r8_6.batch_inverse_kinematics(
            target_coords,
            move_target=r8_6.rarm.end_coords,
            stop=100,
            attempts_per_pose=10
        )

        # Verify at least some poses were solved
        self.assertGreater(sum(success_flags), 0, "At least one pose should be solved")

        # Verify position accuracy for successful solutions
        for i, (solution, success) in enumerate(zip(solutions, success_flags)):
            if success:
                r8_6.angle_vector(solution)
                achieved_pos = r8_6.rarm.end_coords.worldpos()
                pos_error = np.linalg.norm(achieved_pos - target_coords[i].worldpos())
                self.assertLess(pos_error, 0.01, f"Position error too large for pose {i}: {pos_error}m")

    def test_batch_inverse_kinematics_rotated_base_with_linear_joint(self):
        """Test batch IK with rotated base for robot with LinearJoint.

        This test verifies that the batch Jacobian computation correctly
        transforms LinearJoint axes to world coordinates. When the robot base
        is rotated, the joint axes must be transformed to world coordinates
        for correct IK computation.

        Without proper coordinate transformation (bug in Line 3889), this test
        would fail because the Jacobian would use the local axis incorrectly.
        """
        r8_6 = skrobot.models.R8_6()
        r8_6.reset_pose()

        # Rotate the robot base by 90 degrees around Y axis FIRST
        r8_6.rotate(np.pi / 2, 'y')

        # Create target poses near current position (after rotation)
        current_pos = r8_6.rarm.end_coords.worldpos()
        target_coords = [
            skrobot.coordinates.Coordinates(pos=current_pos + [0.05, 0.0, 0.0]),
            skrobot.coordinates.Coordinates(pos=current_pos + [0.0, 0.05, 0.0]),
            skrobot.coordinates.Coordinates(pos=current_pos + [0.0, 0.0, 0.05]),
        ]

        # Run batch IK with rotated robot
        solutions, success_flags, _ = r8_6.batch_inverse_kinematics(
            target_coords,
            move_target=r8_6.rarm.end_coords,
            stop=100,
            attempts_per_pose=10
        )

        # Verify at least some poses were solved
        self.assertGreater(sum(success_flags), 0, "At least one pose should be solved with rotated base")

        # Verify position accuracy for successful solutions
        for i, (solution, success) in enumerate(zip(solutions, success_flags)):
            if success:
                r8_6.angle_vector(solution)
                achieved_pos = r8_6.rarm.end_coords.worldpos()
                pos_error = np.linalg.norm(achieved_pos - target_coords[i].worldpos())
                self.assertLess(pos_error, 0.01, f"Position error too large for pose {i}: {pos_error}m")

    def test_joint_type_alias(self):
        """Test that joint_type is an alias for type."""
        parent_link = Link(name='parent')
        child_link = Link(name='child')
        child_link.translate([0.1, 0, 0])

        joint = RotationalJoint(name='test_joint', parent_link=parent_link,
                                child_link=child_link, axis='z')
        self.assertIn(joint.type, ['revolute', 'continuous'])
        self.assertEqual(joint.type, joint.joint_type)

        linear_joint = LinearJoint(name='test_joint', parent_link=parent_link,
                                   child_link=child_link, axis='z')
        self.assertEqual(linear_joint.type, 'prismatic')
        self.assertEqual(linear_joint.joint_type, 'prismatic')

    def test_joint_axis_alias(self):
        """Test that joint_axis is an alias for axis."""
        parent_link = Link(name='parent')
        child_link = Link(name='child')
        child_link.translate([0.1, 0, 0])

        joint = RotationalJoint(name='test_joint', parent_link=parent_link,
                                child_link=child_link, axis='x')
        testing.assert_array_almost_equal(joint.axis, joint.joint_axis)
        testing.assert_array_almost_equal(joint.axis, [1, 0, 0])

        joint.joint_axis = 'y'
        testing.assert_array_almost_equal(joint.axis, [0, 1, 0])
        testing.assert_array_almost_equal(joint.joint_axis, [0, 1, 0])

    def test_joint_angle_limits_alias(self):
        """Test that min_joint_angle and max_joint_angle are aliases."""
        parent_link = Link(name='parent')
        child_link = Link(name='child')
        child_link.translate([0.1, 0, 0])

        joint = RotationalJoint(name='test_joint', parent_link=parent_link,
                                child_link=child_link, min_angle=-1.5, max_angle=1.5)
        self.assertEqual(joint.min_angle, joint.min_joint_angle)
        self.assertAlmostEqual(joint.min_angle, -1.5)

        joint.min_joint_angle = -2.0
        self.assertAlmostEqual(joint.min_angle, -2.0)
        self.assertAlmostEqual(joint.min_joint_angle, -2.0)

        self.assertEqual(joint.max_angle, joint.max_joint_angle)
        joint.max_joint_angle = 2.0
        self.assertAlmostEqual(joint.max_angle, 2.0)

    def test_batch_inverse_kinematics_with_offset_end_coords(self):
        """Test batch IK with offset end_coords (e.g., tool offset).

        This test verifies that batch IK correctly handles cases where
        end_coords has a local offset from the last link. This simulates
        scenarios like tool offsets or gripper tips.
        """
        pr2 = self.pr2
        pr2.init_pose()

        # Create a custom end_coords with 100mm offset in X direction
        offset_end_coords = CascadedCoords(
            parent=pr2.rarm.end_coords.parent,
            name='offset_tool'
        )
        offset_end_coords.translate([0.1, 0, 0])  # 100mm offset

        # Use current robot pose as the target (should be reachable)
        target_coords = [
            skrobot.coordinates.Coordinates().newcoords(offset_end_coords)
        ]

        # Run batch IK with offset end_coords
        solutions, success_flags, _ = pr2.batch_inverse_kinematics(
            target_coords,
            move_target=offset_end_coords,
            rotation_axis=False,  # Only position
            stop=200,
            attempts_per_pose=10,
            thre=0.001
        )

        # Verify at least some poses were solved
        self.assertGreater(sum(success_flags), 0, "At least one pose should be solved")

        # Verify position accuracy for successful solutions
        for i, (solution, success) in enumerate(zip(solutions, success_flags)):
            if success:
                pr2.angle_vector(solution)
                achieved_pos = offset_end_coords.worldpos()
                pos_error = np.linalg.norm(achieved_pos - target_coords[i].worldpos())
                self.assertLess(
                    pos_error, 0.01,
                    f"Position error too large for pose {i} with offset end_coords: {pos_error * 1000:.1f}mm"
                )

    def test_inverse_kinematics_translation_tolerance(self):
        """Test inverse kinematics with translation_tolerance parameter.

        translation_tolerance allows some error from target on specified axes.
        If error is within tolerance, that axis is treated as "reached".
        """
        kuka = self.kuka
        move_target = kuka.rarm.end_coords
        link_list = kuka.rarm.link_list

        # Create target that's reachable
        kuka.reset_manip_pose()
        target_coords = kuka.rarm.end_coords.copy_worldcoords().translate([
            0.05, -0.05, 0.05], 'local')

        # Without tolerance - should reach target precisely
        kuka.reset_manip_pose()
        result_no_tol = kuka.inverse_kinematics(
            target_coords,
            move_target=move_target,
            link_list=link_list,
            translation_axis=True,
            rotation_axis=True,
            stop=100)
        dif_pos_no_tol = kuka.rarm.end_coords.difference_position(
            target_coords, True)
        self.assertLess(np.linalg.norm(dif_pos_no_tol), 0.01)
        self.assertIsNot(result_no_tol, False)

        # With translation_tolerance parameter - verify it's accepted
        kuka.reset_manip_pose()
        result_with_tol = kuka.inverse_kinematics(
            target_coords,
            move_target=move_target,
            link_list=link_list,
            translation_axis=True,
            rotation_axis=True,
            translation_tolerance=[0.1, 0.1, 0.1],
            stop=100)
        # Parameter should be accepted without error
        self.assertIn(type(result_with_tol), [np.ndarray, bool])

    def test_inverse_kinematics_rotation_tolerance(self):
        """Test inverse kinematics with rotation_tolerance parameter.

        rotation_tolerance allows some error from target rotation.
        """
        kuka = self.kuka
        move_target = kuka.rarm.end_coords
        link_list = kuka.rarm.link_list

        # Create target with small rotation
        kuka.reset_manip_pose()
        target_coords = kuka.rarm.end_coords.copy_worldcoords().\
            rotate(np.deg2rad(15), 'x', 'local')

        # With rotation tolerance
        kuka.reset_manip_pose()
        kuka.inverse_kinematics(
            target_coords,
            move_target=move_target,
            link_list=link_list,
            translation_axis=True,
            rotation_axis=True,
            rotation_tolerance=[np.deg2rad(10), None, None],
            stop=100)

        dif_pos = kuka.rarm.end_coords.difference_position(target_coords, True)
        # Position should still be close
        self.assertLess(np.linalg.norm(dif_pos), 0.01)

    def test_inverse_kinematics_tolerance_params_exist(self):
        """Test that tolerance parameters are accepted without error."""
        kuka = self.kuka
        move_target = kuka.rarm.end_coords
        link_list = kuka.rarm.link_list

        kuka.reset_manip_pose()
        target_coords = kuka.rarm.end_coords.copy_worldcoords().translate([
            0.02, 0, 0], 'local')

        # Test that parameters are accepted
        kuka.reset_manip_pose()
        result = kuka.inverse_kinematics(
            target_coords,
            move_target=move_target,
            link_list=link_list,
            translation_axis=True,
            rotation_axis=True,
            translation_tolerance=[0.01, 0.01, 0.01],
            rotation_tolerance=[np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)],
            stop=50)

        # Should complete without error (result can be success or fail)
        self.assertIn(type(result), [np.ndarray, bool])
