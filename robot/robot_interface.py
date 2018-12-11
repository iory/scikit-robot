from logging import getLogger
import datetime
import sys
from numbers import Number

import numpy as np
from numpy import deg2rad

import rospy
import actionlib
from sensor_msgs.msg import JointState
import control_msgs.msg
import trajectory_msgs.msg

from robot.robot_model import LinearJoint
from robot.robot_model import RotationalJoint


logger = getLogger(__name__)


class RobotInterface(object):
    """
    RobotInterface is class for interacting real robot thorugh
    JointTrajectoryAction servers and JointState topics.
    """

    def __init__(self, robot=None,
                 default_controller=None,
                 joint_states_topic='joint_states',
                 joint_states_queue_size=1,
                 controller_timeout=3,
                 namespace=None):
        """
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

        Returns
        -------
        """
        if rospy.rostime._rostime_initialized is False:
            rospy.init_node("default_robot_interface")

        wait_seconds = 180
        start_time = datetime.datetime.now()
        ros_current_time = rospy.Time.now()
        if rospy.get_param('use_sim_time', False) and \
           ros_current_time.to_sec() == 0 and \
           ros_current_time.to_nsec() == 0:
            rospy.logdebug(
                '[{}] /use_sim_time is TRUE, check if /clock is published'.format(rospy.get_name()))
            while ros_current_time.to_sec() == 0 and ros_current_time.to_nsec():
                diff_time = datetime.datetime.now() - start_time
                if diff_time.seconds > wait_seconds:
                    rospy.logfatal(
                        "[{}] /use_sim_time is TRUE but /clock is NOT PUBLISHED".format(rospy.get_name()))
                    rospy.logfatal("[{}] {} seconds elapsed. aborting...".
                                   format(rospy.get_name(), wait_seconds))
                    sys.exit(1)
                rospy.logwarn("[{}] waiting /clock... {} seconds elapsed.".
                              format(rospy.get_name(), diff_time.seconds + 1e-6 * diff_time.microseconds))
                ros_current_time = rospy.Time.now()
        rospy.loginfo('[{}] /clock is now published.'.format(rospy.get_name()))

        self.robot = robot
        self.robot_state = dict()
        self.controller_timeout = controller_timeout
        self.joint_action_enable = True
        self.namespace = namespace
        if self.namespace:
            rospy.Subscriber("{}/{}".format(
                self.namespace, joint_states_topic),
                JointState)
        else:
            rospy.Subscriber(joint_states_topic, JointState,
                             self.joint_state_callback, queue_size=1)

        if default_controller is None:
            default_controller = "default_controller"
        self.controller_table = {}
        self.controller_type = default_controller
        self.controller_actions = self.add_controller(
            self.controller_type, create_actions=True, joint_enable_check=True)

    def wait_until_update_all_joints(self, tgt_tm):
        """
        TODO
        """
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
        """
        Update robot state

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
            joint.joint_angle(deg2rad(position))
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
        """

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
                controller_action = controller["controller_action"]
                if self.namespace is not None:
                    controller_action = "{}/{}".format(
                        self.namespace,
                        controller_action)
                action = ControllerActionClient(self,
                                                controller_action,
                                                controller["action_type"])
                tmp_actions.append(action)
                tmp_actions_name.append(controller_action)
            for action, action_name in zip(tmp_actions, tmp_actions_name):
                if self.controller_timeout is None:
                    rospy.logwarn(
                        "Waiting for actionlib interface forever because controler-timeout is None")
                if not (self.joint_action_enable and
                        action.wait_for_server(rospy.Duration(self.controller_timeout))):
                    rospy.logwarn("{} is not respond, {}_interface is disable".
                                  format(action, self.robot.name))
                    rospy.logwarn("make sure that you can run "
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
        return [dict(controller_action="fullbody_controller/follow_joint_trajectory_action",
                     controller_state="fullbody_controller/state",
                     action_type=control_msgs.msg.FollowJointTrajectoryAction,
                     joint_names=[joint.name for joint in self.robot.joint_list])]

    def sub_angle_vector(self, v0, v1):
        ret = v0 - v1
        joint_list = self.robot.joint_list
        i = 0
        while joint_list:
            joint = joint_list[0]
            joint_list = joint_list[1:]
            if np.isinf(joint.min_angle()) and \
               np.isinf(joint.max_angle()):
                if ret[i] > 180.0:
                    ret[i] = ret[i] - 360.0
                elif ret[i] < -180.0:
                    ret[i] = ret[i] + 360.0
            i += 1
        return ret

    def angle_vector(self,
                     av,
                     time=None,
                     controller_type=None,
                     start_time=0.0,
                     scale=1.0,
                     min_time=1.0,
                     end_coords_interpolation=None,
                     end_coords_interpolation_steps=10):
        """
        Send joind angle to robot, this method retuns immediately,
        so use self.wait_interpolation to block until the motion stops.

        Parameters
        ----------
        av : list or numpy.ndarray
            joint angle vector [deg]
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
        end_coords_interpolation :
            set True if you want to move robot in cartesian space interpolation
        end_coords_interpolation_steps : int
            number of divisions when interpolating end-coords

        Returns
        -------
        av : np.ndarray
            angle-vector of real robots
        """
        if end_coords_interpolation is not None:
            return self.angle_vector_sequence(
                [av], [time], controller_type,
                start_time, scale, min_time, end_coords_interpolation=True)
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
                "angle_vector time is invalid args: {}".format(time))

        # for simulation mode
        if self.is_simulation_mode():
            if av:
                self.angle_vector_simulation(av, time, controller_type)
        self.robot.angle_vector(av)

        cacts = self.controller_table[controller_type]

        for action, controller_param in zip(cacts, self.default_controller()):
            angle_velocities = np.zeros_like(av)
            duration = time / 1000.0
            self.send_ros_controller(
                action,
                controller_param['joint_names'],
                start_time,
                [[av, angle_velocities, duration]])
        return av

    def potentio_vector(self):
        """
        Retuns current robot angle vector, This method uses caced data
        """
        return self.robot.angle_vector()

    def send_ros_controller(self, action, joint_names, start_time, traj_points):
        """
        TODO

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

                    if isinstance(joint, RotationalJoint):
                        p = deg2rad(p)
                        v = deg2rad(v)
                    else:
                        p = 0.001 * p
                        v = 0.001 * v
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
                              start_time=0.1,
                              scale=1,
                              min_time=0.0,
                              end_coords_interpolation=None,
                              end_coords_interpolation_steps=10):
        """
        Send sequence of joind angle to robot, this method retuns immediately,
        so use self.wait_interpolation to block until the motion stops.

        Parameters
        ----------
        avs : list [av0, av1, ..., avn]
            sequence of joint angles(float-vector) [deg]
        times : list [list tm0 tm1 ... tmn]
            sequence of duration(float) from previous angle-vector
            to next goal [msec].
            if times is atom, then use (list (make-list (length avs) :initial-element times))) for times
            if designated each tmn is faster than fastest speed, use fastest speed
            if tmn is nil, then it will use 1/scale of the fastest speed .
            if :fastest is specefied, use fastest speed calcurated from max speed
        ctype : string
            controller method name
        start_time : float
            time to start moving
        scale : float
            if times is not specified, it will use 1 / scale of the fastest speed
        min_time : float
            minimum time for time to goal
        end_coords_interpolation : TODO
            set t if you want to move robot in cartesian space interpolation
        end_coords_interpolation_steps : int
            number of divisions when interpolating end-coords
        """
        if controller_type is None:
            # use default controller-type if ctype is nil
            controller_type = self.controller_type

        if not (controller_type in self.controller_table):
            # (warn ";; controller-type: ~A not found" ctype)
            return False

        traj_points = []
        st = 0
        av_prev = self.state.potentio_vector()

    def wait_interpolation(self, controller_type=None, timeout=0):
        """
        Wait until last sent motion is finished.

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
        # TODO Fix return value
        return True

    def angle_vector_duration(
            self, start, end,
            scale=1.0, min_time=1.0,
            controller_type=None):
        """
        Calculate maximum time to reach goal for all joint

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

        def flatten(xlst):
            """
            Flatten list

            Parameters
            ----------
            xlst : list ([[c1], [c2], ..., [cn]])

            Returns
            -------
            flatten list : list [c1, c2, ..., cn]
            """
            return reduce(lambda x, y: x + y, xlst)

        unordered_joint_names = set(
            flatten([c['joint_names'] for c in self.default_controller()]))
        joint_list = self.robot.joint_list
        diff_avs = end - start
        time_list = []
        for diff_angle, joint in zip(diff_avs, joint_list):
            if joint.name in unordered_joint_names:
                if isinstance(joint, LinearJoint):
                    time = scale * (0.001 * abs(diff_angle)) * \
                        joint.max_joint_velocity
                else:
                    time = scale * deg2rad(abs(diff_angle)) * \
                        joint.max_joint_velocity
            else:
                time = 0
            time_list.append(time)
        return max(max(time_list), min_time)

    def is_simulation_mode(self):
        """
        Check if simulation mode

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
        rospy.debug("action_feedback_cb {}".format(msg))
        self.last_feedback_msg_stamp = msg.header.stamp

    def is_interpolating(self):
        return self.time_sequence is None
