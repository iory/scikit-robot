import datetime
from functools import reduce
from logging import getLogger
from numbers import Number
import sys

import numpy as np

import actionlib
import control_msgs.msg
import rospy
from sensor_msgs.msg import JointState
from skrobot.model import LinearJoint
from skrobot.model import RotationalJoint
import trajectory_msgs.msg

logger = getLogger(__name__)


def _flatten(xlst):
    """Flatten list.

    Parameters
    ----------
    xlst : list of object
        [[c1], [c2], ..., [cn]]

    Returns
    -------
    flatten list : list of object
        [c1, c2, ..., cn]
    """
    return reduce(lambda x, y: x + y, xlst)


class RobotInterface(object):
    """RobotInterface is class for interacting real robot.

    RobotInterface is class for interacting real robot thorugh
    JointTrajectoryAction servers and JointState topics.

    """

    def __init__(self, robot=None,
                 default_controller=None,
                 joint_states_topic='joint_states',
                 joint_states_queue_size=1,
                 controller_timeout=3,
                 namespace=None):
        """Initialization of RobotInterface

        Parameters
        ----------
        robot : robot_model.RobotModel
            instance of robotmodel
        default_controller : TODO
            TODO
        joint_states_topic : string
            topic name of joint_states
        joint_states_queue_size : int
            queue size of joint_states
        controller_timeout : int
            time out
        namespace : string or None
            namespace of controller

        """
        if rospy.rostime._rostime_initialized is False:
            rospy.init_node('default_robot_interface')

        wait_seconds = 180
        start_time = datetime.datetime.now()
        ros_current_time = rospy.Time.now()
        if rospy.get_param('use_sim_time', False) and \
           ros_current_time.to_sec() == 0 and \
           ros_current_time.to_nsec() == 0:
            rospy.logdebug(
                '[{}] /use_sim_time is TRUE, check if /clock is published'.
                format(rospy.get_name()))
            while (ros_current_time.to_sec() == 0 and
                   ros_current_time.to_nsec() == 0):
                diff_time = datetime.datetime.now() - start_time
                if diff_time.seconds > wait_seconds:
                    rospy.logfatal(
                        '[{}] /use_sim_time is TRUE '
                        'but /clock is NOT PUBLISHED'.format(rospy.get_name()))
                    rospy.logfatal('[{}] {} seconds elapsed. aborting...'.
                                   format(rospy.get_name(), wait_seconds))
                    sys.exit(1)
                rospy.logwarn(
                    '[{}] waiting /clock... {} seconds elapsed.'.
                    format(rospy.get_name(),
                           diff_time.seconds + 1e-6 * diff_time.microseconds))
                ros_current_time = rospy.Time.now()
        rospy.loginfo('[{}] /clock is now published.'.format(rospy.get_name()))

        self.robot = robot
        self.robot_state = dict()
        self.controller_timeout = controller_timeout
        self.joint_action_enable = True
        self.namespace = namespace
        if self.namespace:
            rospy.Subscriber('{}/{}'.format(
                self.namespace, joint_states_topic),
                JointState)
        else:
            rospy.Subscriber(joint_states_topic, JointState,
                             self.joint_state_callback, queue_size=1)

        if default_controller is None:
            default_controller = 'default_controller'
        self.controller_table = {}
        self.controller_type = default_controller
        self.controller_actions = self.add_controller(
            self.controller_type, create_actions=True, joint_enable_check=True)

    def _check_time(self, time, fastest_time):
        """Check and Return send angle vector time

        Parameters
        ----------
        time : float or str or None
            time of send angle vector.
            If time is 'fast' and 'fastest',
            set fastest_time.
        fastest_time : float
            fastest time

        Returns
        -------
        time : float
            time of send angle vector.
        """
        if time in ['fast', 'fastest']:
            # Fastest time Mode
            time = fastest_time
        elif isinstance(time, Number):
            # Normal Number disgnated Mode
            if time < fastest_time:
                time = fastest_time
        elif time is None:
            # Safe Mode (Speed will be 5 * fastest_time)
            time = 5.0 * fastest_time
        else:
            raise ValueError(
                'time is invalid type. {}'.format(time))
        return time

    def wait_until_update_all_joints(self, tgt_tm):
        """TODO"""
        if isinstance(tgt_tm, rospy.Time):
            initial_time = tgt_tm.to_nsec()
        else:
            initial_time = rospy.Time.now().to_nsec()
        while True:
            if all(map(lambda ts: ts.to_nsec() > initial_time,
                       self.robot_state['stamp_list'])):
                return
            if self.is_simulation_mode():
                # to update robot_state
                self.robot_interface_simulation_callback()

    def set_robot_state(self, key, msg):
        self.robot_state[key] = msg

    def update_robot_state(self, wait_until_update=False):
        """Update robot state.

        Parameters
        ----------
        wait_until_update : bool
            if True TODO

        Returns
        -------
        TODO
        """
        if wait_until_update:
            self.wait_until_update_all_joints(wait_until_update)
        if not self.robot_state:
            return False
        joint_names = self.robot_state['name']
        positions = self.robot_state['position']
        velocities = self.robot_state['velocity']
        efforts = self.robot_state['effort']
        joint_num = len(joint_names)
        if not (joint_num == len(velocities)):
            velocities = np.zeros(joint_num)
        if not (joint_num == len(efforts)):
            efforts = np.zeros(joint_num)
        for jn, position, velocity, effort in zip(
                joint_names,
                positions,
                velocities,
                efforts):
            if not hasattr(self.robot, jn):
                continue
            joint = getattr(self.robot, jn)
            joint.joint_angle(position)
            joint.joint_velocity = velocity
            joint.joint_torque = effort

    def joint_state_callback(self, msg):
        if 'name' in self.robot_state:
            robot_state_names = self.robot_state['name']
        else:
            robot_state_names = msg.name
            self.robot_state['name'] = robot_state_names
            for key in ['position', 'velocity', 'effort']:
                self.robot_state[key] = np.zeros(len(robot_state_names))
            self.robot_state['stamp_list'] = [None for _ in robot_state_names]

        # set joint data
        joint_names = msg.name
        stamp_list = self.robot_state['stamp_list']
        for key in ['position', 'velocity', 'effort']:
            joint_data = getattr(msg, key)
            index = 0
            if len(joint_names) == len(joint_data):
                data = self.robot_state[key]
                for jn in joint_names:
                    joint_index = robot_state_names.index(jn)
                    data[joint_index] = joint_data[index]
                    index += 1

                    # update stamp
                    if key == 'position':
                        stamp_list[joint_index] = msg.header.stamp
        self.robot_state['stamp_list'] = stamp_list
        self.robot_state['name'] = robot_state_names
        self.set_robot_state('stamp', msg.header.stamp)

    def add_controller(self, controller_type, joint_enable_check=True,
                       create_actions=None):
        """Add controller

        Parameters
        ----------
        controller_type : string
            type of contrfoller
        joint_enable_check : bool
            TODO
        create_actions : bool
            TODO

        Returns
        -------
        actions : TODO
            TODO
        """
        tmp_actions = []
        tmp_actions_name = []
        if create_actions:
            for controller in self.default_controller():
                controller_action = controller['controller_action']
                if self.namespace is not None:
                    controller_action = '{}/{}'.format(
                        self.namespace,
                        controller_action)
                action = ControllerActionClient(self,
                                                controller_action,
                                                controller['action_type'])
                tmp_actions.append(action)
                tmp_actions_name.append(controller_action)
            for action, action_name in zip(tmp_actions, tmp_actions_name):
                if self.controller_timeout is None:
                    rospy.logwarn(
                        'Waiting for actionlib interface forever '
                        'because controler-timeout is None')
                if not (
                    self.joint_action_enable and action.wait_for_server(
                        rospy.Duration(
                            self.controller_timeout))):
                    rospy.logwarn('{} is not respond, {}_interface is disable'.
                                  format(action, self.robot.name))
                    rospy.logwarn('make sure that you can run '
                                  "'rostopic echo /{0}/status' "
                                  "and 'rostopic info /{0}/status'".
                                  format(action_name))
                    if joint_enable_check:
                        self.joint_action_enable = False
                        return []
            for param in self.default_controller():
                controller_state = param['controller_state']
                if self.namespace is not None:
                    topic_name = '{}/{}'.format(
                        self.namespace,
                        controller_state)
                else:
                    topic_name = controller_state
                rospy.Subscriber(
                    topic_name,
                    control_msgs.msg.JointTrajectoryControllerState,
                    lambda msg: self.set_robot_state(controller_state, msg))
        else:  # not creating actions, just search
            self.controller_type = controller_type
        self.controller_table[controller_type] = tmp_actions
        return self.controller_table[controller_type]

    def default_controller(self):
        return [dict(
            controller_action='fullbody_controller/'
            'follow_joint_trajectory_action',
            controller_state='fullbody_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=[
                    joint.name for joint in self.robot.joint_list])]

    def sub_angle_vector(self, v0, v1):
        """Return subtraction of angle vector

        Parameters
        ----------
        v0 : numpy.ndarray
            angle vector
        v1 : numpy.ndarray
            angle vector

        Returns
        -------
        ret : numpy.ndarray
            Diff of given angle_vector v0 and v1.
        """
        ret = v0 - v1
        joint_list = self.robot.joint_list
        for i in range(len(v0)):
            joint = joint_list[i]
            if np.isinf(joint.min_angle) and \
               np.isinf(joint.max_angle):
                if ret[i] > 180.0:
                    ret[i] = ret[i] - 360.0
                elif ret[i] < -180.0:
                    ret[i] = ret[i] + 360.0
        return ret

    def angle_vector(self,
                     av=None,
                     time=None,
                     controller_type=None,
                     start_time=0.0,
                     scale=1.0,
                     min_time=1.0,
                     velocities=None):
        """Send joint angle to robot

        Send joint angle to robot. this method retuns immediately, so use
        self.wait_interpolation to block until the motion stops.

        Parameters
        ----------
        av : list or numpy.ndarray
            joint angle vector [rad]
        time : None or float or string
            time to goal in [msec]
            if designated time is faster than fastest speed, use fastest speed
            if not specified(None), it will use 1 / scale of the fastest speed.
            if 'fastest' is specefied use fastest speed calcurated from
            max speed
        controller_type : string
            controller method name
        start_time : float
            time to start moving
        scale : float
            if time is not specified, it will use 1/scale of the fastest speed.
        min_time : float
            minimum time for time to goal

        Returns
        -------
        av : np.ndarray
            angle-vector of real robots
        """
        if av is None:
            self.update_robot_state(wait_until_update=True)
            return self.robot.angle_vector()
        if controller_type is None:
            controller_type = self.controller_type
        if not (controller_type in self.controller_table):
            rospy.logwarn(
                'controller_type {} not found'.format(controller_type))
            return False

        # check and decide time
        fastest_time = 1000 * self.angle_vector_duration(
            # self.state.potentio_vector,
            self.potentio_vector(),
            av,
            scale,
            min_time,
            controller_type)
        time = self._check_time(time, fastest_time)

        # for simulation mode
        if self.is_simulation_mode():
            if av:
                self.angle_vector_simulation(av, time, controller_type)
        self.robot.angle_vector(av)
        cacts = self.controller_table[controller_type]

        if velocities is not None:
            angle_velocities = velocities
        else:
            angle_velocities = np.zeros_like(av)
        duration = time / 1000.0
        traj_points = [(av, angle_velocities, duration), ]
        self.traj_points = traj_points
        for action, controller_param in zip(cacts, self.default_controller()):
            self.send_ros_controller(
                action,
                controller_param['joint_names'],
                start_time,
                traj_points)
        return av

    def potentio_vector(self):
        """Retuns current robot angle vector, This method uses caced data."""
        return self.robot.angle_vector()

    def send_ros_controller(
            self,
            action,
            joint_names,
            start_time,
            traj_points):
        """Send angle vector to ROS controller

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        if self.is_simulation_mode():
            return False

        goal = action.action_client.ActionGoal()
        goal_points = []
        if isinstance(start_time, Number):
            st = rospy.Time.now() + rospy.Duration(start_time)
        joints = [getattr(self.robot, jn) for jn in joint_names]

        goal.header.seq = 1
        goal.header.stamp = st

        if isinstance(goal, control_msgs.msg.SingleJointPositionActionGoal):
            raise NotImplementedError
        else:
            goal.goal.trajectory.joint_names = joint_names
            goal.goal.trajectory.header.stamp = st
            for traj_point in traj_points:
                all_positions = traj_point[0]
                all_velocities = traj_point[1]
                duration = traj_point[2]
                positions = np.zeros(len(joint_names))
                velocities = np.zeros(len(joint_names))

                for i in range(len(joints)):
                    joint = joints[i]
                    idx = self.robot.joint_list.index(joint)
                    p = all_positions[idx]
                    v = all_velocities[idx]

                    positions[i] = p
                    velocities[i] = v
                goal_points.append(
                    trajectory_msgs.msg.JointTrajectoryPoint(
                        positions=positions,
                        velocities=velocities,
                        time_from_start=rospy.Duration(duration)))
            goal.goal.trajectory.points = goal_points
        return action.send_goal(goal.goal)

    def angle_vector_sequence(self,
                              avs,
                              times=[3000],
                              controller_type=None,
                              start_time=0.0,
                              scale=1,
                              min_time=0.0):
        """Send sequence of joint angles to robot

        Send sequence of joint angle to robot, this method retuns
        immediately, so use self.wait_interpolation to block until the motion
        stops.

        Parameters
        ----------
        avs : list or numpy.ndarray
            [av0, av1, ..., avn]
            sequence of joint angles in [rad]
        times : list of float or float
            [list tm0 tm1 ... tmn]
            sequence of duration(float) from previous angle-vector
            to next goal [msec].
            if times is atom, then use
                (list (make-list (length avs) :initial-element times)))
                for times
            if designated each tmn is faster than fastest speed,
                use fastest speed
            if tmn is nil, then it will use 1/scale of the fastest speed .
            if :fastest is specefied, use fastest speed calcurated
                from max speed
        ctype : string
            controller method name
        start_time : float
            time to start moving
        scale : float
            if times is not specified, it will use 1 / scale of the
            fastest speed
        min_time : float
            minimum time for time to goal

        Returns
        -------
        avs : list of numpy.ndarray
            list of angle vector.
        """
        if controller_type is None:
            # use default self.controller_type if controller_type is None
            controller_type = self.controller_type

        if not (controller_type in self.controller_table):
            rospy.logwarn('controller_type: {} not found'.
                          format(controller_type))
            return False

        if not isinstance(times, list):
            times = len(avs) * [times]

        prev_av = self.angle_vector()
        traj_points = []
        total_steps = len(avs)
        next_start_time = start_time
        for i_step in range(total_steps):
            av = avs[i_step]
            fastest_time = 1000 * self.angle_vector_duration(
                prev_av, av, scale, min_time, controller_type)
            time = times[i_step]
            time = self._check_time(time, fastest_time)

            vel = np.zeros_like(prev_av)
            if i_step != total_steps - 1:
                next_time = times[i_step + 1]
                next_av = avs[i_step + 1]
                fastest_next_tiem = 1000 * self.angle_vector_duration(
                    av, next_av, scale, min_time, controller_type)
                next_time = self._check_time(next_time, fastest_next_tiem)
                if time > 0.0 and next_time > 0.0:
                    v0 = self.sub_angle_vector(av, prev_av)
                    v1 = self.sub_angle_vector(next_av, av)
                    vel = np.zeros_like(prev_av)
                    indices = v0 * v1 >= 0.0
                    vel[indices] = 0.5 * ((1000.0 / time) * v0[indices] +
                                          (1000.0 / next_time) * v1[indices])
            traj_points.append((av, vel, (time + next_start_time) / 1000.0))
            next_start_time = time
            prev_av = av
        self.traj_points = traj_points

        cacts = self.controller_table[controller_type]
        for action, controller_param in zip(cacts, self.default_controller()):
            self.send_ros_controller(
                action,
                controller_param['joint_names'],
                start_time,
                traj_points)
        return avs

    def wait_interpolation(self, controller_type=None, timeout=0):
        """Wait until last sent motion is finished.

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
            while self.interpolating():
                self.robot_interface_simulation_callback()
        else:
            if controller_type:
                controller_actions = self.controller_table[controller_type]
            else:
                controller_actions = self.controller_actions
            for action in controller_actions:
                action.wait_for_result(timeout=rospy.Duration(timeout))
        # TODO(Fix return value)
        return True

    def angle_vector_duration(
            self, start, end,
            scale=1.0, min_time=1.0,
            controller_type=None):
        """Calculate maximum time to reach goal for all joint.

        Parameters
        ----------
        start : list or np.ndarray
            start angle-vector
        end : list or np.ndarray
            end angle-vector (target position)
        scale : float
            TODO
        min_time : float
            TODO
        controller_type : None or string
            type of controller
        """
        unordered_joint_names = set(
            _flatten([c['joint_names'] for c in self.default_controller()]))
        joint_list = self.robot.joint_list
        diff_avs = end - start
        time_list = []
        for diff_angle, joint in zip(diff_avs, joint_list):
            if joint.name in unordered_joint_names:
                if isinstance(joint, LinearJoint):
                    time = scale * abs(diff_angle) / \
                        joint.max_joint_velocity
                else:
                    time = scale * abs(diff_angle) / \
                        joint.max_joint_velocity
            else:
                time = 0
            time_list.append(time)
        return max(max(time_list), min_time)

    def is_simulation_mode(self):
        """Check if simulation mode.

        Returns
        -------
        not joint_action_enable : bool
            if joint_action is enabled, not simulation mode.
        """
        return not self.joint_action_enable


class ControllerActionClient(actionlib.SimpleActionClient):

    def __init__(self, robot_interface, ns, ActionSpec):
        self.ri = robot_interface
        self.time_to_finish = 0
        self.last_feedback_msg_stamp = rospy.Time.now()
        actionlib.SimpleActionClient.__init__(self, ns, ActionSpec)

    def action_feedback_cb(self, msg):
        rospy.debug('action_feedback_cb {}'.format(msg))
        self.last_feedback_msg_stamp = msg.header.stamp

    def is_interpolating(self):
        return self.time_sequence is None
