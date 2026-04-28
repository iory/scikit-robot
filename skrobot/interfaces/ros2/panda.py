import control_msgs.action
import rclpy
from rclpy.action import ActionClient

from skrobot.interfaces.ros2.base import ROS2RobotInterfaceBase


try:
    import franka_gripper.action
    FRANKA_GRIPPER_AVAILABLE = True
except ImportError:
    FRANKA_GRIPPER_AVAILABLE = False


WIDTH_MAX = 0.08


class PandaROS2RobotInterface(ROS2RobotInterfaceBase):
    """ROS 2 interface for a single Franka Panda arm.

    Parameters
    ----------
    robot : skrobot.model.RobotModel
        Robot model. Must expose a limb attribute (see ``limb_attr``)
        with a 7-joint arm.
    arm_id : str, default 'panda'
        Arm namespace from the URDF. Used to derive ``controller_name``
        when that argument is not given. Two arms on the same machine —
        e.g. a dual-Panda with arm_id 'right_arm' and 'left_arm' — can be
        controlled by instantiating this class twice with different
        arm_ids.
    controller_name : str or None
        Joint trajectory controller name. Defaults to
        ``f'{arm_id}_arm_controller'``. The action namespace is then
        ``/{controller_name}/follow_joint_trajectory``.
    gripper_action_prefix : str or None
        Prefix for the franka_gripper action namespaces. Defaults to
        ``'franka_gripper'``. For multi-arm setups pass e.g.
        ``'right_arm/franka_gripper'`` so move / stop go to per-arm
        action servers.
    limb_attr : str, default 'rarm'
        Attribute on ``robot`` whose ``joint_list`` defines the controlled
        joints. The default matches skrobot's ``Panda`` model, which
        exposes its 7-joint chain as ``robot.rarm``. For a multi-arm
        custom model whose limbs are named ``right_arm`` / ``left_arm``,
        instantiate the interface twice with the matching ``limb_attr``.
    load_gripper : bool, default True
        If False, skip creating gripper ActionClients entirely. Useful
        for sim setups (mock_components) where ``franka_gripper`` is not
        running, and for unit tests where blocking on
        ``wait_for_server`` is not desired.
    """

    def __init__(self, *args,
                 arm_id='panda',
                 controller_name=None,
                 gripper_action_prefix=None,
                 limb_attr='rarm',
                 load_gripper=True,
                 **kwargs):
        self._arm_id = arm_id
        self._controller_name = controller_name or '{}_arm_controller'.format(arm_id)
        self._gripper_action_prefix = gripper_action_prefix or 'franka_gripper'
        self._limb_attr = limb_attr

        super(PandaROS2RobotInterface, self).__init__(*args, **kwargs)

        self.gripper_move = None
        self.gripper_stop = None
        if load_gripper:
            if FRANKA_GRIPPER_AVAILABLE:
                self.gripper_move = ActionClient(
                    self,
                    franka_gripper.action.Move,
                    '{}/move'.format(self._gripper_action_prefix))
                self.gripper_move.wait_for_server()

                self.gripper_stop = ActionClient(
                    self,
                    franka_gripper.action.Stop,
                    '{}/stop'.format(self._gripper_action_prefix))
                self.gripper_stop.wait_for_server()
            else:
                self.get_logger().warn(
                    "franka_gripper package not available. "
                    "Gripper functions disabled.")

    @property
    def arm_controller(self):
        limb = getattr(self.robot, self._limb_attr)
        return dict(
            controller_type='{}_controller'.format(self._limb_attr),
            controller_action='/{}/follow_joint_trajectory'.format(
                self._controller_name),
            controller_state='/{}/state'.format(self._controller_name),
            action_type=control_msgs.action.FollowJointTrajectory,
            joint_names=[j.name for j in limb.joint_list],
        )

    @property
    def rarm_controller(self):
        # Backwards-compatible alias for the pre-parameterised API where
        # the only controller was always called rarm_controller. Prefer
        # `arm_controller` in new code.
        return self.arm_controller

    def default_controller(self):
        return [self.arm_controller]

    def grasp(self, width=0, **kwargs):
        self.move_gripper(width=width, **kwargs)

    def ungrasp(self, **kwargs):
        self.move_gripper(width=WIDTH_MAX, **kwargs)

    def move_gripper(self, width, speed=WIDTH_MAX, wait=True):
        if self.gripper_move is None:
            self.get_logger().warn(
                "Gripper ActionClient was not initialised "
                "(load_gripper=False or franka_gripper not available). "
                "Cannot move gripper.")
            return

        goal = franka_gripper.action.Move.Goal()
        goal.width = width
        goal.speed = speed

        if wait:
            future = self.gripper_move.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            if goal_handle.accepted:
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future)
                return result_future.result()
        else:
            self.gripper_move.send_goal_async(goal)

    def stop_gripper(self, wait=True):
        if self.gripper_stop is None:
            self.get_logger().warn(
                "Gripper ActionClient was not initialised "
                "(load_gripper=False or franka_gripper not available). "
                "Cannot stop gripper.")
            return

        goal = franka_gripper.action.Stop.Goal()

        if wait:
            future = self.gripper_stop.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            if goal_handle.accepted:
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future)
                return result_future.result()
        else:
            self.gripper_stop.send_goal_async(goal)
