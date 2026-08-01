"""Unit tests for the generic ROS 2 GripperCommand action client."""
import unittest
from unittest import mock

import pytest


try:
    # No `import rclpy` here, unlike the sibling ROS 2 tests: those go on to
    # call rclpy.init(), while these run against a fake node, so the import
    # would be unused. Importing the module under test is enough of an
    # availability probe -- gripper.py imports rclpy itself.
    from skrobot.interfaces.ros2.gripper import GripperCommandActionClient
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not ROS2_AVAILABLE, reason="ROS 2 / rclpy not importable")


class _ImmediateFuture(object):

    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result

    def add_done_callback(self, callback):
        callback(self)


class _FakeGoalHandle(object):

    def __init__(self, accepted=True):
        self.accepted = accepted
        self.cancel_called = False

    def get_result_async(self):
        return _ImmediateFuture('goal_result')

    def cancel_goal_async(self):
        self.cancel_called = True
        return _ImmediateFuture('cancelled')


class _FakeActionClient(object):

    def __init__(self, node, action_type, action_name):
        self.node = node
        self.action_type = action_type
        self.action_name = action_name
        self.available = True
        self.sent_goals = []
        self.goal_handle = _FakeGoalHandle()

    def wait_for_server(self, timeout_sec=0.0):
        return self.available

    def send_goal_async(self, goal):
        self.sent_goals.append(goal)
        return _ImmediateFuture(self.goal_handle)


class _FakeNode(object):
    """Minimal stand-in for an rclpy node.

    The client only ever asks the node for its logger, and the action client is
    faked here anyway, so a real node buys nothing. It costs something, though:
    creating one initialises the global rclpy context and joins the DDS domain,
    which makes these tests order-dependent with the rest of the ROS 2 suite.
    """

    def __init__(self, executor=None):
        self.warnings = []
        # `None` means nobody else is spinning this node, so the client is
        # free to spin it itself -- which is the path these tests patch.
        self.executor = executor

    def get_logger(self):
        return self

    def warn(self, message):
        self.warnings.append(message)


class TestGripperCommandActionClient(unittest.TestCase):

    def setUp(self):
        self.node = _FakeNode()

    def test_move_default_width_scale_converts_gap_to_joint_value(self):
        with mock.patch(
            'skrobot.interfaces.ros2.gripper.ActionClient',
            _FakeActionClient
        ), mock.patch(
            'skrobot.interfaces.ros2.gripper.rclpy.spin_until_future_complete',
            lambda n, f, timeout_sec=None: None
        ):
            client = GripperCommandActionClient(
                self.node,
                'test_gripper_cmd')
            client.move(width=0.08, wait=True)

            goal = client._action_client.sent_goals[-1]
            assert goal.command.position == pytest.approx(0.04)
            assert goal.command.max_effort == pytest.approx(0.0)

    def test_move_custom_width_scale_is_applied(self):
        with mock.patch(
            'skrobot.interfaces.ros2.gripper.ActionClient',
            _FakeActionClient
        ), mock.patch(
            'skrobot.interfaces.ros2.gripper.rclpy.spin_until_future_complete',
            lambda n, f, timeout_sec=None: None
        ):
            client = GripperCommandActionClient(
                self.node,
                'test_gripper_cmd',
                width_scale=0.25,
                max_effort=12.0)
            client.move(width=0.08, wait=True)

            goal = client._action_client.sent_goals[-1]
            assert goal.command.position == pytest.approx(0.02)
            assert goal.command.max_effort == pytest.approx(12.0)

    def test_move_overrides_max_effort_when_passed(self):
        with mock.patch(
            'skrobot.interfaces.ros2.gripper.ActionClient',
            _FakeActionClient
        ), mock.patch(
            'skrobot.interfaces.ros2.gripper.rclpy.spin_until_future_complete',
            lambda n, f, timeout_sec=None: None
        ):
            client = GripperCommandActionClient(
                self.node,
                'test_gripper_cmd',
                max_effort=12.0)
            client.move(width=0.08, max_effort=5.0, wait=True)

            goal = client._action_client.sent_goals[-1]
            assert goal.command.max_effort == pytest.approx(5.0)


class TestGripperCommandActionClientExternalExecutor(unittest.TestCase):
    """The node is usually spun by an external executor, not by the client.

    Spinning it from both places shares one wait set between two spinners and
    rclpy raises "wait set index for status subscription is out of bounds".
    """

    def test_move_does_not_spin_a_node_owned_by_an_executor(self):
        node = _FakeNode(executor=object())
        with mock.patch(
                'skrobot.interfaces.ros2.gripper.ActionClient',
                _FakeActionClient), \
            mock.patch(
                'skrobot.interfaces.ros2.gripper.rclpy'
                '.spin_until_future_complete') as spin:
            client = GripperCommandActionClient(node, 'gripper_cmd')
            client.move(0.08)
        spin.assert_not_called()
