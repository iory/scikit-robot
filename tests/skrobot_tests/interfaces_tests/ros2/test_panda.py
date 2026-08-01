"""Unit tests for the parameterised PandaROS2RobotInterface."""
from types import SimpleNamespace
import unittest
from unittest import mock

import pytest


try:
    import rclpy

    import skrobot.interfaces.ros2.panda as panda_module
    from skrobot.interfaces.ros2.panda import PandaROS2RobotInterface
    from skrobot.interfaces.ros2.panda import WIDTH_MAX
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


class _ImmediateFuture(object):

    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def add_done_callback(self, callback):
        callback(self)


class _FakeGoalHandle(object):

    def __init__(self, accepted=True):
        self.accepted = accepted
        self.cancel_called = False

    def get_result_async(self):
        return _ImmediateFuture('result')

    def cancel_goal_async(self):
        self.cancel_called = True
        return _ImmediateFuture('cancelled')


class _FakeActionClient(object):

    def __init__(self, node, action_type, action_name):
        self.node = node
        self.action_type = action_type
        self.action_name = action_name
        self.sent_goals = []
        self.goal_handle = _FakeGoalHandle()

    def wait_for_server(self, timeout_sec=None):
        return True

    def send_goal_async(self, goal):
        self.sent_goals.append(goal)
        return _ImmediateFuture(self.goal_handle)


class _FakeFrankaMoveGoal(object):

    def __init__(self):
        self.width = None
        self.speed = None


class _FakeFrankaStopGoal(object):
    pass


class _FakeFrankaMove(object):
    Goal = _FakeFrankaMoveGoal


class _FakeFrankaStop(object):
    Goal = _FakeFrankaStopGoal


class _FakeGripperCommandActionClient(object):

    def __init__(self,
                 node,
                 action_name,
                 width_scale=0.5,
                 max_effort=0.0):
        self.node = node
        self.action_name = action_name
        self.width_scale = width_scale
        self.max_effort = max_effort
        self.sent_joint_positions = []
        self.cancel_count = 0

    def move(self, width, max_effort=None, wait=True):
        effort = self.max_effort if max_effort is None else max_effort
        joint_value = width * self.width_scale
        self.sent_joint_positions.append((joint_value, effort, wait))
        return self.sent_joint_positions[-1]

    def cancel(self, wait=True):
        self.cancel_count += 1
        return wait


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


class TestPandaInterfaceGripperBackends(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError):
            PandaROS2RobotInterface(
                robot=None,
                node_name='test_unknown_backend',
                gripper_backend='not_a_backend')

    def test_gripper_command_backend_does_not_require_franka_gripper(self):
        with mock.patch.object(
            panda_module, 'FRANKA_GRIPPER_AVAILABLE', False
        ), mock.patch.object(
            panda_module,
            'GripperCommandActionClient',
            _FakeGripperCommandActionClient
        ):
            ri = _make(
                'test_gripper_command_without_franka',
                load_gripper=True,
                gripper_backend='gripper_command')
            try:
                assert ri.gripper_command is not None
                assert ri.gripper_move is None
                assert ri.gripper_stop is None
            finally:
                ri.destroy_node()

    def test_grasp_and_ungrasp_map_to_expected_joint_positions(self):
        with mock.patch.object(
            panda_module,
            'GripperCommandActionClient',
            _FakeGripperCommandActionClient
        ):
            ri = _make(
                'test_gripper_command_grasp_ungrasp',
                load_gripper=True,
                gripper_backend='gripper_command')
            try:
                ri.grasp(width=0.02, wait=True)
                ri.ungrasp(wait=True)
                positions = [
                    p for p, _, _ in ri.gripper_command.sent_joint_positions]
                assert positions[0] == pytest.approx(0.01)
                assert positions[1] == pytest.approx(WIDTH_MAX * 0.5)
            finally:
                ri.destroy_node()

    def test_default_backend_move_gripper_path_is_unchanged(self):
        fake_franka = SimpleNamespace(
            action=SimpleNamespace(Move=_FakeFrankaMove, Stop=_FakeFrankaStop))
        with mock.patch.object(
            panda_module, 'FRANKA_GRIPPER_AVAILABLE', True
        ), mock.patch.object(
            panda_module, 'franka_gripper', fake_franka
        ), mock.patch.object(
            panda_module, 'ActionClient', _FakeActionClient
        ), mock.patch.object(
            panda_module.rclpy, 'spin_until_future_complete', lambda n, f: None
        ):
            ri = _make(
                'test_default_backend_move',
                load_gripper=True)
            try:
                ri.move_gripper(width=0.03, speed=0.02, wait=True)
                move_goal = ri.gripper_move.sent_goals[-1]
                assert move_goal.width == pytest.approx(0.03)
                assert move_goal.speed == pytest.approx(0.02)

                ri.stop_gripper(wait=True)
                stop_goal = ri.gripper_stop.sent_goals[-1]
                assert isinstance(stop_goal, _FakeFrankaStopGoal)
            finally:
                ri.destroy_node()
