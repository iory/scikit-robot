"""ROS2RobotInterfaceBase as a RobotInterfaceBase, without a live robot.

A FollowJointTrajectory action server in this process stands in for the
controller and records the goals it receives.
"""
import threading
import unittest

import pytest


try:
    from control_msgs.action import FollowJointTrajectory
    import rclpy
    from rclpy.action import ActionServer
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    from skrobot.interfaces.base import RobotInterfaceBase
    from skrobot.interfaces.ros2.base import ROS2RobotInterfaceBase
    from skrobot.models import Panda
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not ROS2_AVAILABLE, reason="ROS 2 / rclpy not importable")

_CONTROLLER = '_test_compat_controller'
_ARM = ['panda_joint{}'.format(i) for i in range(1, 8)]


if ROS2_AVAILABLE:
    class _ArmInterface(ROS2RobotInterfaceBase):

        def default_controller(self):
            return [dict(
                controller_type=_CONTROLLER,
                controller_action=_CONTROLLER + '/follow_joint_trajectory',
                controller_state=_CONTROLLER + '/state',
                action_type=FollowJointTrajectory,
                joint_names=list(_ARM))]


class TestRos2InterfaceIsRobotInterfaceBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    def setUp(self):
        self.goals = []
        self.goal_received = threading.Event()
        self.server_node = rclpy.create_node('aux_compat_server')
        self.server = ActionServer(
            self.server_node, FollowJointTrajectory,
            _CONTROLLER + '/follow_joint_trajectory', self._execute)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.server_node)
        self.spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.ri = _ArmInterface(
            robot=Panda(), node_name='test_compat',
            joint_states_topic='_test_joint_states_compat',
            default_controller=_CONTROLLER, controller_timeout=5.0)
        assert self.ri.joint_action_enable
        # Goal responses reach the interface through its own executor.
        self.executor.add_node(self.ri)

    def tearDown(self):
        self.ri.destroy_node()
        self.executor.shutdown()
        self.server.destroy()
        self.server_node.destroy_node()

    def _execute(self, goal_handle):
        self.goals.append(goal_handle.request.trajectory)
        self.goal_received.set()
        goal_handle.succeed()
        return FollowJointTrajectory.Result()

    def _publish_state(self, positions):
        msg = JointState()
        msg.name = list(positions)
        msg.position = [float(v) for v in positions.values()]
        msg.header.stamp = self.ri.get_clock().now().to_msg()
        self.ri.joint_state_callback(msg)

    def _last_goal(self):
        assert self.goal_received.wait(timeout=5.0), 'no goal arrived'
        return self.goals[-1]

    def test_is_a_robot_interface_and_a_node(self):
        assert isinstance(self.ri, RobotInterfaceBase)
        assert isinstance(self.ri, Node)
        assert self.ri.get_name() == 'test_compat'
        assert self.ri.controllers == {_CONTROLLER: _ARM}
        assert self.ri.joint_names() == _ARM

    def test_servo_hooks_default_to_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.ri.servo_on()
        with pytest.raises(NotImplementedError):
            self.ri.servo_off(['panda_joint1'])

    def test_out_of_limit_state_is_clamped_in_model_but_reported(self):
        joint4 = self.ri.robot.panda_joint4
        past = float(joint4.max_angle) + 0.3
        self._publish_state({'panda_joint4': past})
        assert self.ri.update_robot_state() is True
        names = [j.name for j in self.ri.robot.joint_list]
        i = names.index('panda_joint4')
        # Unchanged behaviour: the model and potentio_vector are clamped.
        assert joint4.joint_angle() == pytest.approx(float(joint4.max_angle))
        assert self.ri.potentio_vector()[i] == pytest.approx(
            float(joint4.max_angle))
        # The base class view is not.
        assert self.ri.measured_angle_vector()[i] == pytest.approx(past)
        assert self.ri.measured_joint_angles(['panda_joint4']) == {
            'panda_joint4': pytest.approx(past)}
        assert list(self.ri.outside_limits()) == ['panda_joint4']

    def test_dict_goal_fills_the_other_joints_from_the_model(self):
        self._publish_state({n: 0.0 for n in
                             [j.name for j in self.ri.robot.joint_list]})
        self._publish_state({'panda_joint4': -1.0})
        self.ri.update_robot_state()
        av = self.ri.angle_vector({'panda_joint1': 0.3}, 1.0)
        goal = self._last_goal()
        assert list(goal.joint_names) == _ARM
        positions = dict(zip(goal.joint_names, goal.points[-1].positions))
        assert positions['panda_joint1'] == pytest.approx(0.3)
        assert positions['panda_joint4'] == pytest.approx(-1.0)
        assert len(av) == len(self.ri.robot.joint_list)

    def test_unknown_controller_type_returns_false(self):
        assert self.ri.angle_vector(
            self.ri.robot.angle_vector(), 1.0,
            controller_type='no_such_controller') is False
        assert self.ri.angle_vector_sequence(
            [self.ri.robot.angle_vector()], 1.0,
            controller_type='no_such_controller') is False

    def test_wait_interpolation_waits_for_the_default_controller(self):
        self._publish_state({n: 0.0 for n in
                             [j.name for j in self.ri.robot.joint_list]})
        self.ri.angle_vector({'panda_joint1': 0.1}, 0.2)
        result = self.ri.wait_interpolation(
            timeout=5.0, wait_for_state_update=False)
        assert result == [False] * len(
            self.ri.controller_table[self.ri.controller_type])
        assert self.ri.is_interpolating() is False

    def test_base_send_hook_sends_only_named_joints(self):
        # The base class's own angle_vector, reached explicitly, sends a
        # partial goal through _send_trajectory.
        self._publish_state({n: 0.0 for n in _ARM})
        RobotInterfaceBase.angle_vector(
            self.ri, {'panda_joint2': 0.2}, 1.0)
        goal = self._last_goal()
        assert list(goal.joint_names) == ['panda_joint2']
        assert goal.points[-1].positions[0] == pytest.approx(0.2)
        assert goal.points[-1].time_from_start.sec == 1

    def test_cancel_reaches_the_action_clients(self):
        cancelled = []
        for action in self.ri.controller_table[_CONTROLLER]:
            action.cancel_all_goals = (
                lambda a=action: cancelled.append(a))
        self.ri.cancel_angle_vector()
        assert cancelled == self.ri.controller_table[_CONTROLLER]
