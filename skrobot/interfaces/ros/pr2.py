import math

import actionlib
import control_msgs.msg
import numpy as np
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

    def _continuous_joint_largest_movement(self, av_diff, controller_type):
        assert isinstance(controller_type, str)

        controller_params = self.controller_param_table[controller_type]
        controlling_joint_names = []
        for param in controller_params:
            controlling_joint_names.extend(param['joint_names'])

        assert len(av_diff) == len(self.robot.joint_list)

        abs_diff_largeest = - np.Inf
        ret_joint_name = None
        for joint, angle_diff in zip(self.robot.joint_list, av_diff):

            if joint.name not in controlling_joint_names:
                continue

            is_continuous = (joint.max_angle - joint.min_angle) > 2 * math.pi
            if is_continuous and abs_diff_largeest < abs(angle_diff):
                abs_diff_largeest = abs(angle_diff)
                ret_joint_name = joint.name

        assert ret_joint_name is not None
        return abs_diff_largeest, ret_joint_name

    def angle_vector(self,
                     av=None,
                     time=None,
                     controller_type=None,
                     start_time=0.0,
                     time_scale=5.0,
                     velocities=None):

        if controller_type is None:
            controller_type = self.controller_type  # use default controller

        if av is not None:
            av_diff = av - self.angle_vector()
            diff_max, joint_name = self._continuous_joint_largest_movement(
                av_diff, controller_type)
            if diff_max > math.pi:
                rospy.logwarn(
                    "continuous joint {} movement over 180 degree detected"
                    .format(joint_name))
                rospy.logwarn(
                    "angle_vector_sequence will be used as a workaround")

                if time is None:
                    default_duration = 3.0  # same as pr2eus
                    time = default_duration
                return self.angle_vector_sequence(
                    [av], [time],
                    controller_type=controller_type,
                    start_time=start_time,
                    time_scale=time_scale)

        return super(PR2ROSRobotInterface, self).angle_vector(
            av=av,
            time=time,
            controller_type=controller_type,
            start_time=start_time,
            time_scale=time_scale,
            velocities=velocities)

    def angle_vector_sequence(self,
                              avs,
                              times=None,
                              controller_type=None,
                              start_time=0.0,
                              time_scale=5.0):

        if controller_type is None:
            controller_type = self.controller_type  # use default controller

        if times is None:
            default_duration = 3.0  # same as pr2eus
            times = [default_duration for _ in range(len(avs))]

        assert isinstance(times, list), 'times must be None or list'
        assert len(avs) == len(times), 'length of times and avs must be equal'

        av_initial = self.angle_vector()
        avs_with_initial = [av_initial] + avs

        avs_reformed = []
        times_reformed = []
        for i in range(len(avs)):
            av_here = avs_with_initial[i]
            av_next = avs_with_initial[i + 1]
            av_diff = av_next - av_here

            diff_max, joint_name = self._continuous_joint_largest_movement(
                av_diff, controller_type)
            if diff_max > math.pi:
                rospy.logwarn(
                    "continuous joint {} movement over 180 degree detected"
                    .format(joint_name))
                rospy.logwarn('interval will be split')
                n_split = int(math.ceil(diff_max / (math.pi * 2 / 3)))
                av_diff_partial = av_diff / n_split
                for j in range(n_split):
                    avs_reformed.append(av_here + av_diff_partial * (j + 1))
                times_reformed.extend(
                    [times[i] / n_split for _ in range(n_split)])
            else:
                avs_reformed.append(av_next)
                times_reformed.append(times[i])

        assert len(avs_reformed) == len(times_reformed)
        assert abs(sum(times_reformed) - sum(times)) < 1e-8

        return super(PR2ROSRobotInterface, self).angle_vector_sequence(
            avs_reformed,
            times=times_reformed,
            controller_type=controller_type,
            start_time=start_time,
            time_scale=time_scale)

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
