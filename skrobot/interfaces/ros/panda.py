import actionlib
import control_msgs.msg
import franka_gripper.msg
import rospy

from skrobot.interfaces.ros.base import ROSRobotInterfaceBase


WIDTH_MAX = 0.08


class PandaROSRobotInterface(ROSRobotInterfaceBase):

    def __init__(self, *args, **kwargs):
        super(PandaROSRobotInterface, self).__init__(*args, **kwargs)

        if self.namespace:
            namespace_prefix = self.namespace.strip('/') + '/'
        else:
            namespace_prefix = ''

        gripper_move_action = namespace_prefix + 'franka_gripper/move'
        rospy.loginfo(
            'Waiting for action server: {}'.format(gripper_move_action))
        self.gripper_move = actionlib.SimpleActionClient(
            gripper_move_action,
            franka_gripper.msg.MoveAction)
        self.gripper_move.wait_for_server()
        rospy.loginfo(
            'Action server {} is ready'.format(gripper_move_action))

        gripper_stop_action = namespace_prefix + 'franka_gripper/stop'
        rospy.loginfo(
            'Waiting for action server: {}'.format(gripper_stop_action))
        self.gripper_stop = actionlib.SimpleActionClient(
            gripper_stop_action,
            franka_gripper.msg.StopAction)
        self.gripper_stop.wait_for_server()
        rospy.loginfo(
            'Action server {} is ready'.format(gripper_stop_action))

    @property
    def rarm_controller(self):
        return dict(
            controller_type='rarm_controller',
            controller_action='position_joint_trajectory_controller/follow_joint_trajectory',  # NOQA
            controller_state='position_joint_trajectory_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=[j.name for j in self.robot.rarm.joint_list],
        )

    def default_controller(self):
        return [self.rarm_controller]

    def grasp(self, width=0, **kwargs):
        self.move_gripper(width=width, **kwargs)

    def ungrasp(self, **kwargs):
        self.move_gripper(width=WIDTH_MAX, **kwargs)

    def move_gripper(self, width, speed=WIDTH_MAX, wait=True):
        goal = franka_gripper.msg.MoveGoal(width=width, speed=speed)
        if wait:
            self.gripper_move.send_goal_and_wait(goal)
        else:
            self.gripper_move.send_goal(goal)

    def stop_gripper(self, wait=True):
        goal = franka_gripper.msg.StopGoal()
        if wait:
            self.gripper_stop.send_goal_and_wait(goal)
        else:
            self.gripper_stop.send_goal(goal)
