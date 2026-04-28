"""Unit tests for the parameterised PandaROS2RobotInterface."""
import unittest

import pytest


try:
    import rclpy

    from skrobot.interfaces.ros2.panda import PandaROS2RobotInterface
    from skrobot.models import Panda
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not ROS2_AVAILABLE, reason="ROS 2 / rclpy not importable")


def _make(node_name, load_gripper=False, **kwargs):
    """Build an interface with controllers / gripper disabled for unit tests.

    Defaults: ``controller_timeout=0.1`` so the no-action-server controller
    spawn fails fast instead of blocking 3s; ``load_gripper=False`` so the
    gripper ``ActionClient.wait_for_server`` does not hang waiting for a
    franka_gripper node that is not running. Tests can override either.
    """
    return PandaROS2RobotInterface(
        robot=Panda(),
        node_name=node_name,
        controller_timeout=0.1,
        load_gripper=load_gripper,
        joint_states_topic='_test_panda_' + node_name,
        **kwargs,
    )


class TestPandaInterfaceDefaults(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    def test_default_controller_action_uses_panda_arm_controller(self):
        ri = _make('test_default_controller')
        try:
            spec = ri.arm_controller
            assert spec['controller_action'] == \
                '/panda_arm_controller/follow_joint_trajectory'
            assert spec['controller_state'] == '/panda_arm_controller/state'
            # Default limb_attr='rarm' on the Panda model gives 7 joints.
            assert len(spec['joint_names']) == 7
        finally:
            ri.destroy_node()

    def test_rarm_controller_alias_matches_arm_controller(self):
        ri = _make('test_rarm_alias')
        try:
            assert ri.rarm_controller == ri.arm_controller
        finally:
            ri.destroy_node()


class TestPandaInterfaceParameterisation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    def test_arm_id_drives_default_controller_name(self):
        ri = _make('test_arm_id_drives_name', arm_id='right_arm')
        try:
            spec = ri.arm_controller
            assert spec['controller_action'] == \
                '/right_arm_arm_controller/follow_joint_trajectory'
        finally:
            ri.destroy_node()

    def test_explicit_controller_name_wins_over_arm_id(self):
        ri = _make('test_explicit_controller_name',
                   arm_id='right_arm',
                   controller_name='dual_panda_joint_trajectory_controller')
        try:
            spec = ri.arm_controller
            assert spec['controller_action'] == \
                '/dual_panda_joint_trajectory_controller/follow_joint_trajectory'
        finally:
            ri.destroy_node()


class TestPandaInterfaceGripperDisable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    def test_load_gripper_false_skips_action_clients(self):
        ri = _make('test_load_gripper_false')
        try:
            assert ri.gripper_move is None
            assert ri.gripper_stop is None
        finally:
            ri.destroy_node()

    def test_grasp_warns_when_gripper_disabled(self):
        ri = _make('test_grasp_warn')
        try:
            # Should be a no-op (no exception, no real ActionClient call).
            ri.grasp()
            ri.ungrasp()
            ri.stop_gripper()
        finally:
            ri.destroy_node()
