"""Unit tests for ROS2RobotInterfaceBase that do not require a live controller.

These cover the wait/update plumbing — `_received_joint_names`, the timeout
behaviour of `wait_until_update_all_joints`, the bool return of
`update_robot_state`, and the optional first-message wait in `__init__`.
"""
import threading
import time
import unittest

import numpy as np
import pytest


try:
    from action_msgs.msg import GoalStatus
    from action_msgs.msg import GoalStatusArray
    from control_msgs.action import FollowJointTrajectory
    from control_msgs.msg import JointTrajectoryControllerState
    import rclpy
    from rclpy.action import ActionServer
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState

    from skrobot.interfaces.ros2.base import ROS2RobotInterfaceBase
    from skrobot.models import Panda
    ROS2_AVAILABLE = True

    class _NoControllerInterface(ROS2RobotInterfaceBase):
        """Skip controller / action setup entirely so tests can focus on the
        joint_states bookkeeping path."""

        def default_controller(self):
            return []

    class _TwoControllerInterface(ROS2RobotInterfaceBase):
        """Two controllers, to check that callbacks keep them apart."""

        controller_names = ('_test_first_controller',
                            '_test_second_controller')

        def default_controller(self):
            return [dict(
                controller_type=name,
                controller_action=name + '/follow_joint_trajectory',
                controller_state=name + '/state',
                action_type=FollowJointTrajectory,
                joint_names=['panda_joint1'])
                for name in self.controller_names]
except ImportError:
    ROS2_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not ROS2_AVAILABLE, reason="ROS 2 / rclpy not importable")


def _make_interface(node_name, joint_states_topic=None, **kwargs):
    """Create an interface with an isolated joint_states topic by default.

    Each instance subscribes to its own per-test topic so concurrently running
    publishers (e.g. a panda controller from another shell) do not bleed into
    timeout tests via DDS discovery.
    """
    if joint_states_topic is None:
        joint_states_topic = '_test_joint_states_' + node_name
    return _NoControllerInterface(
        robot=Panda(),
        node_name=node_name,
        joint_states_topic=joint_states_topic,
        controller_timeout=0.1,  # don't actually wait for any controller
        **kwargs,
    )


def _make_joint_state(node, names, positions, stamp=None):
    msg = JointState()
    msg.name = list(names)
    msg.position = list(positions)
    msg.velocity = []
    msg.effort = []
    msg.header.stamp = (stamp or node.get_clock().now()).to_msg()
    return msg


class TestRos2BaseJointStateBookkeeping(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    def test_received_joint_names_tracks_callback_inputs(self):
        ri = _make_interface('test_received_names')
        try:
            assert ri._received_joint_names == set()
            ri.joint_state_callback(_make_joint_state(
                ri, ['panda_joint1', 'panda_joint2'], [0.1, 0.2]))
            assert ri._received_joint_names == {'panda_joint1', 'panda_joint2'}
            ri.joint_state_callback(_make_joint_state(
                ri, ['panda_finger_joint1'], [0.04]))
            assert ri._received_joint_names == {
                'panda_joint1', 'panda_joint2', 'panda_finger_joint1'}
        finally:
            ri.destroy_node()

    def test_wait_until_update_all_joints_times_out_with_reason(self):
        ri = _make_interface('test_timeout_reason')
        try:
            t0 = time.time()
            result = ri.wait_until_update_all_joints(True, timeout=0.2)
            elapsed = time.time() - t0
            assert result is False
            assert 0.15 < elapsed < 1.0  # bounded by the timeout
            # No joint_states arrived at all → reason should mention that.
            assert 'No joint_states' in ri._timeout_reason or \
                   'robot_state is empty' in ri._timeout_reason or \
                   'robot_state keys' in ri._timeout_reason
        finally:
            ri.destroy_node()

    def test_wait_returns_true_after_fresh_message(self):
        ri = _make_interface('test_wait_fresh')
        try:
            tgt = ri.get_clock().now()
            time.sleep(0.05)  # ensure the next stamp is strictly newer
            ri.joint_state_callback(_make_joint_state(
                ri, ['panda_joint1', 'panda_joint2'], [0.1, 0.2]))
            assert ri.wait_until_update_all_joints(tgt, timeout=1.0) is True
            assert ri._timeout_reason == ""
        finally:
            ri.destroy_node()

    def test_wait_ignores_unpublished_joints(self):
        """A joint declared in the URDF but never published must not block wait."""
        ri = _make_interface('test_filter_unpublished')
        try:
            tgt = ri.get_clock().now()
            time.sleep(0.05)
            # Publish only joint1+joint2; finger joints declared in the URDF
            # but never seen on the topic must be ignored by the wait logic.
            ri.joint_state_callback(_make_joint_state(
                ri, ['panda_joint1', 'panda_joint2'], [0.1, 0.2]))
            assert ri.wait_until_update_all_joints(tgt, timeout=1.0) is True
        finally:
            ri.destroy_node()


class TestRos2BaseUpdateRobotStateReturnValue(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    def test_returns_false_when_no_message_received(self):
        ri = _make_interface('test_update_no_msg')
        try:
            assert ri.update_robot_state() is False
            assert ri._timeout_reason  # populated for diagnosis
        finally:
            ri.destroy_node()

    def test_returns_true_after_message(self):
        ri = _make_interface('test_update_after_msg')
        try:
            ri.joint_state_callback(_make_joint_state(
                ri, ['panda_joint1', 'panda_joint2'], [0.5, -0.5]))
            assert ri.update_robot_state() is True
            assert np.isclose(ri.robot.panda_joint1.joint_angle(), 0.5)
            assert np.isclose(ri.robot.panda_joint2.joint_angle(), -0.5)
        finally:
            ri.destroy_node()

    def test_returns_false_when_wait_until_update_times_out(self):
        ri = _make_interface('test_update_timeout')
        try:
            assert ri.update_robot_state(wait_until_update=True) is False
        finally:
            ri.destroy_node()


class TestRos2BaseConstructorWaitsForJointStates(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    def test_wait_for_joint_states_default_off(self):
        # Default should not block in __init__ even when no publisher exists.
        t0 = time.time()
        ri = _make_interface('test_wait_default_off')
        elapsed = time.time() - t0
        try:
            assert elapsed < 1.0  # constructor returned promptly
        finally:
            ri.destroy_node()

    def test_wait_for_joint_states_raises_on_timeout(self):
        with pytest.raises(TimeoutError) as excinfo:
            _make_interface(
                'test_wait_timeout_ctor',
                wait_for_joint_states=True,
                joint_states_timeout=0.3)
        assert 'JointState' in str(excinfo.value)

    def test_wait_for_joint_states_returns_when_message_arrives(self):
        """A background publisher arriving before the timeout unblocks __init__."""
        topic = '_test_joint_states_arrives_ctor'
        publisher_node = rclpy.create_node('aux_publisher_for_wait_ctor')
        publisher = publisher_node.create_publisher(JointState, topic, 10)
        executor = SingleThreadedExecutor()
        executor.add_node(publisher_node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        stop_event = threading.Event()

        def publish_loop():
            while not stop_event.is_set():
                msg = JointState()
                msg.name = ['panda_joint1']
                msg.position = [0.0]
                msg.header.stamp = publisher_node.get_clock().now().to_msg()
                publisher.publish(msg)
                time.sleep(0.05)

        publish_thread = threading.Thread(target=publish_loop, daemon=True)
        publish_thread.start()

        try:
            ri = _make_interface(
                'test_wait_arrives_ctor',
                joint_states_topic=topic,
                wait_for_joint_states=True,
                joint_states_timeout=3.0)
            assert ri._joint_state_msg is not None
            ri.destroy_node()
        finally:
            stop_event.set()
            publish_thread.join(timeout=1.0)
            executor.shutdown()
            publisher_node.destroy_node()


class TestRos2BaseControllerCallbacks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    def setUp(self):
        # add_controller subscribes to the state and status topics only after
        # every action server has answered, so serve both actions.
        self.server_node = rclpy.create_node('aux_trajectory_servers')
        self.servers = [
            ActionServer(self.server_node, FollowJointTrajectory,
                         name + '/follow_joint_trajectory',
                         lambda goal_handle: FollowJointTrajectory.Result())
            for name in _TwoControllerInterface.controller_names]
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.server_node)
        self.spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.ri = _TwoControllerInterface(
            robot=Panda(), node_name='test_two_controllers',
            joint_states_topic='_test_joint_states_two_controllers',
            controller_timeout=5.0)
        assert self.ri.joint_action_enable

    def tearDown(self):
        self.ri.destroy_node()
        self.executor.shutdown()
        for server in self.servers:
            server.destroy()
        self.server_node.destroy_node()

    def _publish_until(self, msg_type, topic, msg, condition, timeout=5.0):
        publisher = self.server_node.create_publisher(msg_type, topic, 10)
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                publisher.publish(msg)
                rclpy.spin_once(self.ri, timeout_sec=0.1)
                if condition():
                    return True
            return False
        finally:
            self.server_node.destroy_publisher(publisher)

    def test_state_is_stored_under_its_own_controller(self):
        first, second = _TwoControllerInterface.controller_names
        received = self._publish_until(
            JointTrajectoryControllerState, first + '/state',
            JointTrajectoryControllerState(),
            lambda: first + '/state' in self.ri.robot_state
            or second + '/state' in self.ri.robot_state)
        assert received
        assert first + '/state' in self.ri.robot_state
        assert second + '/state' not in self.ri.robot_state

    def test_moving_status_is_stored_under_its_own_controller(self):
        first, second = _TwoControllerInterface.controller_names
        msg = GoalStatusArray(
            status_list=[GoalStatus(status=GoalStatus.STATUS_EXECUTING)])
        received = self._publish_until(
            GoalStatusArray, first + '/follow_joint_trajectory/status', msg,
            lambda: len(self.ri.moving_status) > 0)
        assert received
        assert self.ri.moving_status == {first: True}
        assert self.ri.is_moving(controller_type=first)
