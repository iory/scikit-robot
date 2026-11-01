import os
import shutil
import tempfile
import unittest

import numpy as np
import pytest

from skrobot.models.urdf import RobotModelFromURDF


# Isaac Sim needs Python 3.12 and an RTX GPU; CI has neither.
pytest.importorskip('isaacsim')

from skrobot.interfaces import IsaacSimRobotInterface  # NOQA


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

# A box on the floor with one hinge and a camera frame looking forward (+x),
# written the way ROS writes optical frames.
_MOBILE_URDF = """<?xml version="1.0"?>
<robot name="mobile">
  <link name="base_link">
    <inertial><origin xyz="0 0 0.1" rpy="0 0 0"/><mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual><origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry><box size="0.4 0.4 0.2"/></geometry></visual>
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


def _robot(tmpdir, name, text):
    urdf_path = os.path.join(tmpdir, name + '.urdf')
    with open(urdf_path, 'w') as f:
        f.write(text)
    return RobotModelFromURDF(urdf_file=urdf_path)


class TestIsaacSimRobotInterface(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.robot = _robot(self.tmpdir, 'arm', _ARM_URDF)
        self.ri = IsaacSimRobotInterface(self.robot)

    def tearDown(self):
        self.ri.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reaches_goal(self):
        goal = np.array([0.5, -0.8])
        self.ri.angle_vector(goal, 2.0)
        self.assertTrue(self.ri.is_interpolating())
        self.assertEqual(self.ri.wait_interpolation(), [False])
        self.assertGreaterEqual(self.ri.time, 2.0)
        self.ri.simulate(0.5)
        np.testing.assert_allclose(self.ri.angle_vector(), goal, atol=1e-2)
        np.testing.assert_allclose(self.robot.angle_vector(), [0.0, 0.0])
        self.assertIsNot(self.ri.robot, self.robot)

    def test_holds_against_gravity(self):
        self.ri.simulate(1.0)
        np.testing.assert_allclose(self.ri.angle_vector(), [0.0, 0.0],
                                   atol=1e-3)

    def test_sequence_passes_via_point(self):
        avs = [np.array([0.5, 0.0]), np.array([0.5, 0.5])]
        self.ri.angle_vector_sequence(avs, [1.0, 1.0])
        self.ri.wait_interpolation(timeout=1.0)
        self.assertTrue(self.ri.is_interpolating())
        np.testing.assert_allclose(self.ri.angle_vector(), avs[0],
                                   atol=5e-2)
        self.ri.wait_interpolation()
        self.ri.simulate(0.5)
        np.testing.assert_allclose(self.ri.angle_vector(), avs[1],
                                   atol=1e-2)


class TestIsaacSimMobile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.robot = _robot(self.tmpdir, 'mobile', _MOBILE_URDF)
        self.ri = IsaacSimRobotInterface(
            self.robot, planar_base=True,
            cameras={'cam': {'link': 'camera_optical_frame', 'fovy': 60,
                             'width': 64, 'height': 48}})

    def tearDown(self):
        self.ri.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_go_pos_in_base_frame(self):
        self.ri.go_pos(0.5, 0.0, np.pi / 2)
        self.ri.go_pos(0.3, 0.0, 0.0)  # forward is now +y
        self.ri.simulate(0.5)
        np.testing.assert_allclose(self.ri.odom.worldpos(), [0.5, 0.3, 0.0],
                                   atol=5e-3)
        np.testing.assert_allclose(self.robot.worldpos(), [0.0, 0.0, 0.0])

    def test_camera_image(self):
        rgb = self.ri.get_image('cam')
        depth = self.ri.get_depth('cam')
        self.assertEqual(rgb.shape, (48, 64, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        # The scene is lit: an unlit stage renders black.
        self.assertGreater(rgb.mean(), 30.0)
        self.assertEqual(depth.shape, (48, 64))
        cam = self.ri.camera_coords('cam')
        link = self.robot.camera_optical_frame
        np.testing.assert_allclose(cam.worldpos(), link.worldpos(),
                                   atol=1e-6)
        # The camera at height 0.3 looks level along +x. The bottom row's
        # ray, 23.5 px below the centre at f = 24 / tan(30 deg), meets the
        # floor 0.3 / (23.5 / f) ahead.
        f = 24.0 / np.tan(np.deg2rad(30.0))
        self.assertAlmostEqual(float(depth[-1, 32]), 0.3 * f / 23.5,
                               delta=0.02)
        # Above the horizon there is no floor to hit.
        self.assertFalse(np.isfinite(depth[0, 32]) and depth[0, 32] < 100)


class TestIsaacSimViewer(unittest.TestCase):

    def setUp(self):
        pytest.importorskip('websockets')
        self.tmpdir = tempfile.mkdtemp()
        self.robot = _robot(self.tmpdir, 'arm', _ARM_URDF)
        self.port = 18765
        self.ri = IsaacSimRobotInterface(self.robot, viewer=True,
                                         viewer_port=self.port)

    def tearDown(self):
        self.ri.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_streams_frames_of_the_requested_view(self):
        import json
        import struct
        import threading
        import urllib.request

        from websockets.sync.client import connect

        page = urllib.request.urlopen(
            'http://localhost:{}/'.format(self.port)).read().decode()
        self.assertIn('WebSocket', page)
        frames = []

        def browser():
            with connect('ws://localhost:{}/ws'.format(self.port),
                         max_size=None) as ws:
                ws.send(json.dumps({'view': {
                    'az': 0.5, 'el': 0.3, 'dist': 2.0,
                    'target': [0, 0, 0.5], 'id': 7}}))
                while len(frames) < 10:
                    m = ws.recv(timeout=60)
                    if isinstance(m, bytes):
                        ws.send(json.dumps({'ack': 1}))
                        frames.append(m)

        thread = threading.Thread(target=browser, daemon=True)
        thread.start()
        # The simulation renders for the browser while it steps.
        while thread.is_alive() and self.ri.time < 120.0:
            self.ri.step(10)
        thread.join(timeout=1)
        self.assertEqual(len(frames), 10)
        view_id = struct.unpack('<iff', frames[-1][:12])[0]
        self.assertEqual(view_id, 7)
        # A JPEG follows the 12-byte header.
        self.assertEqual(frames[-1][12:14], b'\xff\xd8')
