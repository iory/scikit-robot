import actionlib
import control_msgs.msg
import pr2_controllers_msgs.msg
import rospy

from ...model import RotationalJoint
from .move_base import ROSRobotMoveBaseInterface


class PR2ROSRobotInterface(ROSRobotMoveBaseInterface):

    """pr2 robot interface."""

    def __init__(self, *args, **kwargs):
        super(PR2ROSRobotInterface, self).__init__(*args, **kwargs)

        self.gripper_states = dict(larm=None, rarm=None)
        rospy.Subscriber('/r_gripper_controller/state',
                         pr2_controllers_msgs.msg.JointControllerState,
                         lambda msg: self.pr2_gripper_state_callback(
                             'rarm', msg))
        rospy.Subscriber('/l_gripper_controller/state',
                         pr2_controllers_msgs.msg.JointControllerState,
                         lambda msg: self.pr2_gripper_state_callback(
                             'larm', msg))

        self.l_gripper_action = actionlib.SimpleActionClient(
            "/l_gripper_controller/gripper_action",
            pr2_controllers_msgs.msg.Pr2GripperCommandAction)
        self.r_gripper_action = actionlib.SimpleActionClient(
            "/r_gripper_controller/gripper_action",
            pr2_controllers_msgs.msg.Pr2GripperCommandAction)
        for action in [self.l_gripper_action, self.r_gripper_action]:
            if not (self.joint_action_enable and action.wait_for_server(
                    rospy.Duration(3))):
                self.joint_action_enable = False
                rospy.logwarn(
                    '{} is not respond, PR2ROSRobotInterface is disabled')
                break

        self.ignore_joint_list = ['laser_tilt_mount_joint', ]

    def wait_interpolation(self, controller_type=None, timeout=0):
        """Overwrite wait_interpolation

        Overwrite for pr2 because some joint is still moving after joint-
        trajectory-action stops.

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
        super(PR2ROSRobotInterface, self).wait_interpolation(
            controller_type, timeout)
        while not rospy.is_shutdown():
            self.update_robot_state(wait_until_update=True)
            if all(map(lambda j: j.name in self.ignore_joint_list
                       or abs(j.joint_velocity) < 0.05
                       if isinstance(j, RotationalJoint) else
                       abs(j.joint_velocity) < 0.001,
                       self.robot.joint_list)):
                break
        # TODO(Fix return value)
        return True

    @property
    def larm_controller(self):
        return dict(
            controller_type='larm_controller',
            controller_action='l_arm_controller/follow_joint_trajectory',
            controller_state='l_arm_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['l_shoulder_pan_joint',
                         'l_shoulder_lift_joint',
                         'l_upper_arm_roll_joint',
                         'l_elbow_flex_joint',
                         'l_forearm_roll_joint',
                         'l_wrist_flex_joint',
                         'l_wrist_roll_joint'])

    @property
    def rarm_controller(self):
        return dict(
            controller_type='rarm_controller',
            controller_action='r_arm_controller/follow_joint_trajectory',
            controller_state='r_arm_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['r_shoulder_pan_joint',
                         'r_shoulder_lift_joint',
                         'r_upper_arm_roll_joint',
                         'r_elbow_flex_joint',
                         'r_forearm_roll_joint',
                         'r_wrist_flex_joint',
                         'r_wrist_roll_joint'])

    @property
    def head_controller(self):
        return dict(
            controller_type='head_controller',
            controller_action='head_traj_controller/follow_joint_trajectory',
            controller_state='head_traj_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['head_pan_joint', 'head_tilt_joint'])

    @property
    def torso_controller(self):
        return dict(
            controller_type='torso_controller',
            controller_action='torso_controller/follow_joint_trajectory',
            controller_state='torso_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['torso_lift_joint'])

    def move_gripper(self, arm, pos, effort=25,
                     wait=True,
                     ignore_stall=False):
        """Move gripper function

        Parameters
        ----------
        arm : str
            can take 'larm', 'rarm', 'arms'
        pos : float
            position of gripper.
            if pos is 0.0, gripper closed.
        effort : float
            effort of grasp.
        wait : bool
            if wait is True, wait until gripper action ends.
        """
        if arm == 'larm':
            action_clients = [self.l_gripper_action]
        elif arm == 'rarm':
            action_clients = [self.r_gripper_action]
        elif arm == 'arms':
            action_clients = [self.l_gripper_action,
                              self.r_gripper_action]
        else:
            return
        for action_client in action_clients:
            goal = pr2_controllers_msgs.msg.Pr2GripperCommandActionGoal()
            goal.goal.command.position = pos
            goal.goal.command.max_effort = effort
            action_client.send_goal(goal.goal)
        results = []
        if wait:
            for action_client in action_clients:
                results.append(action_client.wait_for_result())
        else:
            for action_client in action_clients:
                results.append(action_client.wait_for_result())
        return results

    def pr2_gripper_state_callback(self, arm, msg):
        self.gripper_states[arm] = msg

    def default_controller(self):
        """Overriding default_controller.

        Returns
        -------
        List of limb controller : list
        """
        return [self.larm_controller,
                self.rarm_controller,
                self.head_controller,
                self.torso_controller]
