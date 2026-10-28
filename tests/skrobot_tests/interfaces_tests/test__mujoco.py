import os
import shutil
import tempfile
import unittest

import numpy as np
import pytest

from skrobot.models.urdf import RobotModelFromURDF


mujoco = pytest.importorskip('mujoco')

from skrobot.interfaces import MujocoRobotInterface  # NOQA


# Two hinges in series, lying along x, so gravity loads both of them.
_ARM_URDF = """<?xml version="1.0"?>
<robot name="arm">
  <link name="base_link"/>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/><child link="upper"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="50.0" velocity="2.0"/>
  </joint>
  <link name="upper">
    <inertial><origin xyz="0.15 0 0" rpy="0 0 0"/><mass value="1.0"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-2" iyz="0" izz="1e-2"/>
    </inertial>
    <collision><origin xyz="0.15 0 0" rpy="0 0 0"/>
      <geometry><box size="0.3 0.04 0.04"/></geometry></collision>
  </link>
  <joint name="elbow" type="revolute">
    <parent link="upper"/><child link="fore"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="50.0" velocity="2.0"/>
  </joint>
  <link name="fore">
    <inertial><origin xyz="0.15 0 0" rpy="0 0 0"/><mass value="0.5"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="5e-3" iyz="0" izz="5e-3"/>
    </inertial>
    <collision><origin xyz="0.15 0 0" rpy="0 0 0"/>
      <geometry><box size="0.3 0.04 0.04"/></geometry></collision>
  </link>
</robot>
"""


class TestMujocoRobotInterface(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        urdf_path = os.path.join(self.tmpdir, 'arm.urdf')
        with open(urdf_path, 'w') as f:
            f.write(_ARM_URDF)
        self.robot = RobotModelFromURDF(urdf_file=urdf_path)
        self.ri = MujocoRobotInterface(self.robot)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reaches_goal(self):
        goal = np.array([0.5, -0.8])
        self.ri.angle_vector(goal, 2.0)
        self.assertTrue(self.ri.is_interpolating())
        self.assertEqual(self.ri.wait_interpolation(), [False])
        self.assertGreaterEqual(self.ri.time, 2.0)
        self.ri.step(250)
        np.testing.assert_allclose(self.ri.angle_vector(), goal, atol=1e-2)
        # ri works on its own copy of the model, as the ROS interface
        # does: the caller's stays where they left it.
        np.testing.assert_allclose(self.robot.angle_vector(), [0.0, 0.0])
        self.assertIsNot(self.ri.robot, self.robot)
        np.testing.assert_allclose(self.ri.robot.angle_vector(), goal,
                                   atol=1e-2)

    def test_holds_against_gravity(self):
        # Gravity compensation only works if it is in the XML; set on a
        # loaded model it is silently ignored and the arm sags.
        self.assertGreater(self.ri.model.ngravcomp, 0)
        self.ri.simulate(1.0)
        # Uncompensated, this arm sags about 4e-4 rad at kp=1000.
        np.testing.assert_allclose(self.ri.angle_vector(), [0.0, 0.0],
                                   atol=1e-5)

    def test_time_raised_to_velocity_limit(self):
        # 1.5 rad at 2.0 rad/s needs 0.75 s, so 0.1 s is not possible.
        self.ri.angle_vector(np.array([1.5, 0.0]), 0.1)
        self.ri.wait_interpolation()
        self.assertGreaterEqual(self.ri.time, 0.75)

    def test_sequence_passes_via_point(self):
        avs = [np.array([0.5, 0.0]), np.array([0.5, 0.5])]
        self.ri.angle_vector_sequence(avs, [1.0, 1.0])
        self.ri.wait_interpolation(timeout=1.0)
        self.assertTrue(self.ri.is_interpolating())
        np.testing.assert_allclose(self.ri.angle_vector(), avs[0],
                                   atol=5e-2)
        self.ri.wait_interpolation()
        self.ri.step(250)
        np.testing.assert_allclose(self.ri.angle_vector(), avs[1],
                                   atol=1e-2)


# A box on the floor with one hinge and a camera frame looking forward (+x),
# written the way ROS writes optical frames.
_MOBILE_URDF = """<?xml version="1.0"?>
<robot name="mobile">
  <link name="base_link">
    <inertial><origin xyz="0 0 0.1" rpy="0 0 0"/><mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <collision><origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry><box size="0.4 0.4 0.2"/></geometry></collision>
  </link>
  <joint name="pan" type="revolute">
    <parent link="base_link"/><child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-2" upper="2" effort="10.0" velocity="2.0"/>
  </joint>
  <link name="head">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.5"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
  </link>
  <joint name="camera_optical_joint" type="fixed">
    <parent link="head"/><child link="camera_optical_frame"/>
    <origin xyz="0.1 0 0" rpy="-1.5707963267948966 0 -1.5707963267948966"/>
  </joint>
  <link name="camera_optical_frame"/>
</robot>
"""


class TestMujocoMobileScene(unittest.TestCase):

    def setUp(self):
        from skrobot.model import Box

        self.tmpdir = tempfile.mkdtemp()
        urdf_path = os.path.join(self.tmpdir, 'mobile.urdf')
        with open(urdf_path, 'w') as f:
            f.write(_MOBILE_URDF)
        self.robot = RobotModelFromURDF(urdf_file=urdf_path)
        table = Box((0.5, 0.5, 0.4), name='table')
        table.translate((2.0, 0.0, 0.2))
        cube = Box((0.05, 0.05, 0.05), name='cube')
        cube.translate((2.0, 0.0, 0.45))
        self.ri = MujocoRobotInterface(
            self.robot, planar_base=True, objects=[table],
            movable_objects=[cube],
            cameras={'cam': {'link': 'camera_optical_frame', 'fovy': 60,
                             'width': 64, 'height': 48},
                     'cam_x': {'link': 'head', 'frame': 'x_forward'}})

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_go_pos_in_base_frame(self):
        self.ri.go_pos(0.5, 0.0, np.pi / 2)
        self.ri.go_pos(0.3, 0.0, 0.0)  # forward is now +y
        self.ri.simulate(0.5)
        np.testing.assert_allclose(self.ri.odom.worldpos(), [0.5, 0.3, 0.0],
                                   atol=5e-3)
        # The caller's model stays put until they move it.
        np.testing.assert_allclose(self.robot.worldpos(), [0.0, 0.0, 0.0])

    def test_go_velocity_follows_arc(self):
        # 0.2 m/s forward while turning 1 rad/s for a quarter turn ends
        # 0.2 m ahead and 0.2 m to the left.
        self.ri.go_velocity(0.2, 0.0, 1.0, sec=np.pi / 2, wait=True)
        self.ri.simulate(0.5)
        np.testing.assert_allclose(self.ri.odom.worldpos()[:2], [0.2, 0.2],
                                   atol=5e-3)

    def test_movable_object_rests_on_table(self):
        self.ri.simulate(1.0)
        pos = self.ri.object_coords('cube').worldpos()
        np.testing.assert_allclose(pos, [2.0, 0.0, 0.425], atol=2e-3)
        self.assertEqual(self.ri.contacts('cube'), ['table'])

    def test_camera_matches_optical_frame(self):
        link = self.robot.camera_optical_frame
        cam = self.ri.camera_coords('cam')
        np.testing.assert_allclose(cam.worldpos(), link.worldpos(),
                                   atol=1e-6)
        np.testing.assert_allclose(cam.worldrot(), link.worldrot(),
                                   atol=1e-6)
        # MuJoCo's camera looks along its -z; it must be the link's +z.
        mj_forward = -self.ri.data.cam_xmat[0].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(mj_forward, [1.0, 0.0, 0.0], atol=1e-6)
        K = self.ri.camera_info('cam')['K']
        self.assertAlmostEqual(K[1, 1], 24.0 / np.tan(np.deg2rad(30.0)))
        self.assertEqual((K[0, 2], K[1, 2]), (32.0, 24.0))

    def test_x_forward_camera_has_same_optical_frame(self):
        # Mounted on the head (x forward), it must look where the optical
        # frame link does.
        np.testing.assert_allclose(
            self.ri.camera_coords('cam_x').worldrot(),
            self.robot.camera_optical_frame.worldrot(), atol=1e-6)

    def test_depth_sees_table(self):
        try:
            depth = self.ri.get_depth('cam')
        except Exception as e:  # no OpenGL context on this machine
            pytest.skip('mujoco.Renderer unavailable ({}); run with '
                        'MUJOCO_GL=egl on a GPU machine'.format(e))
        rgb = self.ri.get_image('cam')
        self.assertEqual(rgb.shape, (48, 64, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(depth.shape, (48, 64))
        # The camera, 0.1 m ahead of the base at height 0.3, looks straight
        # at the table's near face at x = 1.75.
        self.assertAlmostEqual(float(depth[24, 32]), 1.65, delta=0.02)


# A box with one plain wheel on its right side, axle along y.
_WHEELED_URDF = """<?xml version="1.0"?>
<robot name="wheeled">
  <link name="base_link">
    <inertial><origin xyz="0 0 0.1" rpy="0 0 0"/><mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="wheel" type="continuous">
    <parent link="base_link"/><child link="wheel_link"/>
    <origin xyz="0 -0.2 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit effort="10.0" velocity="1.0"/>
  </joint>
  <link name="wheel_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.5"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
  </link>
</robot>
"""


class TestMujocoWheels(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        urdf_path = os.path.join(self.tmpdir, 'wheeled.urdf')
        with open(urdf_path, 'w') as f:
            f.write(_WHEELED_URDF)
        self.robot = RobotModelFromURDF(urdf_file=urdf_path)
        self.ri = MujocoRobotInterface(self.robot, planar_base=True,
                                       wheels={'wheel': (0.05, 0.0)})

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_wheel_rolls_with_base(self):
        # 0.5 m forward rolls a 5 cm wheel 10 rad.
        self.ri.go_pos(0.5, 0.0, 0.0)
        self.assertAlmostEqual(self.ri.angle_vector()[0], 10.0, delta=0.1)
        # Turning left in place carries the right wheel 0.2 m * pi/2
        # forward: 2 pi rad more.
        self.ri.go_pos(0.0, 0.0, np.pi / 2)
        self.assertAlmostEqual(self.ri.angle_vector()[0],
                               10.0 + 2 * np.pi, delta=0.1)

    def test_angle_vector_ignores_wheel(self):
        self.ri.go_pos(0.5, 0.0, 0.0)
        # An angle vector still holding the wheel at 0 neither turns it
        # back nor stretches the motion to the wheel's 1 rad/s limit.
        start = self.ri.time
        self.ri.angle_vector(np.zeros(1), 0.5)
        self.ri.wait_interpolation()
        self.assertAlmostEqual(self.ri.time - start, 0.5, delta=0.01)
        self.assertAlmostEqual(self.ri.angle_vector()[0], 10.0, delta=0.1)
