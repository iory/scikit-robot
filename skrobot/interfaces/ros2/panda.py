import control_msgs.action
import rclpy
from rclpy.action import ActionClient

from .base import ROS2RobotInterfaceBase


try:
    import franka_gripper.action
    FRANKA_GRIPPER_AVAILABLE = True
except ImportError:
    FRANKA_GRIPPER_AVAILABLE = False


WIDTH_MAX = 0.08


class PandaROS2RobotInterface(ROS2RobotInterfaceBase):

    def __init__(self, *args, **kwargs):
        super(PandaROS2RobotInterface, self).__init__(*args, **kwargs)

        if FRANKA_GRIPPER_AVAILABLE:
            self.gripper_move = ActionClient(
                self,
                franka_gripper.action.Move,
                'franka_gripper/move')
            self.gripper_move.wait_for_server()

            self.gripper_stop = ActionClient(
                self,
                franka_gripper.action.Stop,
                'franka_gripper/stop')
            self.gripper_stop.wait_for_server()
        else:
            self.get_logger().warn("franka_gripper package not available. Gripper functions disabled.")

    @property
    def rarm_controller(self):
        return dict(
            controller_type='rarm_controller',
            controller_action='/panda_arm_controller/follow_joint_trajectory',
            controller_state='/panda_arm_controller/state',
            action_type=control_msgs.action.FollowJointTrajectory,
            joint_names=[j.name for j in self.robot.rarm.joint_list],
        )

    def default_controller(self):
        return [self.rarm_controller]

    def grasp(self, width=0, **kwargs):
        self.move_gripper(width=width, **kwargs)

    def ungrasp(self, **kwargs):
        self.move_gripper(width=WIDTH_MAX, **kwargs)

    def move_gripper(self, width, speed=WIDTH_MAX, wait=True):
        if not FRANKA_GRIPPER_AVAILABLE:
            self.get_logger().warn("franka_gripper package not available. Cannot move gripper.")
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
        if not FRANKA_GRIPPER_AVAILABLE:
            self.get_logger().warn("franka_gripper package not available. Cannot stop gripper.")
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
