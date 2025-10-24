import actionlib
import control_msgs.msg

from .base import ROSRobotInterfaceBase


class NextageROSRobotInterface(ROSRobotInterfaceBase):

    def __init__(self, *args, **kwargs):
        super(NextageROSRobotInterface, self).__init__(*args, **kwargs)

        self.rarm_move = actionlib.SimpleActionClient(
            '/rarm_controller/follow_joint_trajectory_action',
            control_msgs.msg.FollowJointTrajectoryAction
        )
        self.rarm_move.wait_for_server()

        self.larm_move = actionlib.SimpleActionClient(
            '/larm_controller/follow_joint_trajectory_action',
            control_msgs.msg.FollowJointTrajectoryAction
        )
        self.larm_move.wait_for_server()

    @property
    def rarm_controller(self):
        return dict(
            controller_type='rarm_controller',
            controller_action='/rarm_controller/follow_joint_trajectory_action',
            controller_state='/rarm_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=[j.name for j in self.robot.rarm.joint_list],
        )

    @property
    def larm_controller(self):
        return dict(
            controller_type='larm_controller',
            controller_action='/larm_controller/follow_joint_trajectory_action',
            controller_state='/larm_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=[j.name for j in self.robot.larm.joint_list],
        )

    def default_controller(self):
        return [self.rarm_controller, self.larm_controller]

    def move_arm(self, trajectory, arm='rarm', wait=True):
        if arm == 'rarm':
            self.send_trajectory(self.rarm_move, trajectory, wait)
        elif arm == 'larm':
            self.send_trajectory(self.larm_move, trajectory, wait)

    def send_trajectory(self, client, trajectory, wait=True):
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory = trajectory
        if wait:
            client.send_goal_and_wait(goal)
        else:
            client.send_goal(goal)
