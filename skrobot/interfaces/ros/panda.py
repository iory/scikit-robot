import control_msgs.msg

from .base import ROSRobotInterfaceBase


class PandaROSRobotInterface(ROSRobotInterfaceBase):

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
