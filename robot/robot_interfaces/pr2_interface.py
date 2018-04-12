import rospy
import control_msgs.msg
import pr2_controllers_msgs.msg

from robot.robot_interface import RobotInterface
from robot.robot_model import RotationalJoint


class PR2Interface(RobotInterface):
    """
    pr2 robot interface
    """

    def __init__(self, *args, **kwargs):
        RobotInterface.__init__(
            self, *args, **kwargs)

        # add controllers
        for ctype, name in [(self.larm_controller,
                             "l_arm_controller/follow_joint_trajectory"),
                            (self.rarm_controller,
                             "r_arm_controller/follow_joint_trajectory"),
                            (self.head_controller,
                             "head_traj_controller/follow_joint_trajectory"),
                            (self.torso_controller,
                             "torso_controller/follow_joint_trajectory")]:
            pass

        # rospy.Subscriber('/r_gripper_controller/state',
        #                  pr2_controllers_msgs.msg.JointControllerState,
        #                  lambda msg: self.pr2_fingertip_callback('rarm'))
        # rospy.Subscriber('/l_gripper_controller/state',
        #                  pr2_controllers_msgs.msg.JointControllerState,
        #                  lambda msg: self.pr2_fingertip_callback('larm'))

    def wait_interpolation(self, controller_type=None, timeout=0):
        """
        Overwrite for pr2
        because some joint is still moving after joint-trajectory-action stops.

        Parameters
        ----------
        controller_type : None or string
            controller to be wait
        timeout : float
            max time of for waiting

        Returns
        -------
            return values are a list of is_interpolating for all controllers.
            if all interpolation has stopped, return True.
        """
        if self.is_simulation_mode():
            return super(PR2Interface, self).wait_interpolation(
                controller_type, timeout)
        super(PR2Interface, self).wait_interpolation(controller_type, timeout)
        while not rospy.is_shutdown():
            self.update_robot_state(wait_until_update=True)
            if all(map(lambda j: abs(j.joint_velocity) < 0.05
                       if isinstance(j, RotationalJoint) else
                       abs(j.joint_velocity) < 0.001,
                       self.robot.joint_list)):
                break
        # TODO Fix return value
        return True

    @property
    def larm_controller(self):
        return dict(
            controller_action="l_arm_controller/follow_joint_trajectory",
            controller_state="l_arm_controller/state",
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=["l_shoulder_pan_joint",
                         "l_shoulder_lift_joint",
                         "l_upper_arm_roll_joint",
                         "l_elbow_flex_joint",
                         "l_forearm_roll_joint",
                         "l_wrist_flex_joint",
                         "l_wrist_roll_joint"])

    @property
    def rarm_controller(self):
        return dict(
            controller_action="r_arm_controller/follow_joint_trajectory",
            controller_state="r_arm_controller/state",
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=["r_shoulder_pan_joint",
                         "r_shoulder_lift_joint",
                         "r_upper_arm_roll_joint",
                         "r_elbow_flex_joint",
                         "r_forearm_roll_joint",
                         "r_wrist_flex_joint",
                         "r_wrist_roll_joint"])

    @property
    def head_controller(self):
        return dict(
            controller_action="head_traj_controller/follow_joint_trajectory",
            controller_state="head_traj_controller/state",
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=["head_pan_joint", "head_tilt_joint"])

    @property
    def torso_controller(self):
        return dict(
            controller_action="torso_controller/follow_joint_trajectory",
            controller_state="torso_controller/state",
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=["torso_lift_joint"])

    def default_controller(self):
        """
        Overriding default_controller

        Returns
        -------
        List of limb controller : list
        """
        return [self.larm_controller,
                self.rarm_controller,
                self.head_controller,
                self.torso_controller]
