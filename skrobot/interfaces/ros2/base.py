import copy
from functools import reduce
from logging import getLogger
from numbers import Number
import time

import action_msgs.msg
import control_msgs.action
import control_msgs.msg
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
import trajectory_msgs.msg

from skrobot.model import LinearJoint
from skrobot.model import RotationalJoint


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


class ROS2RobotInterfaceBase(Node):

    """ROS2RobotInterface is a class for interacting with a real robot.

    ROS2RobotInterface is a class for interacting with a real robot through
    JointTrajectoryAction servers and JointState topics.

    """

    def __init__(self, robot,
                 default_controller='default_controller',
                 joint_states_topic='joint_states',
                 joint_states_queue_size=1,
                 controller_timeout=3,
                 namespace=None,
                 node_name='robot_interface'):
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
        node_name : string
            name of the ROS2 node

        """
        super().__init__(node_name)

        self.robot = copy.deepcopy(robot)
        self.robot_state = dict()
        self.moving_status = dict()
        self.prev_moving_status = dict()
        self.controller_timeout = controller_timeout
        self.joint_action_enable = True
        self.namespace = namespace
        self._joint_state_msg = None

        if self.namespace:
            topic_name = f'{self.namespace}/{joint_states_topic}'
        else:
            topic_name = joint_states_topic

        self.joint_state_sub = self.create_subscription(
            JointState,
            topic_name,
            self.joint_state_callback,
            joint_states_queue_size
        )

        self.controller_table = {}
        self.controller_param_table = {}
        self.controller_type = default_controller
        self.controller_actions = self.add_controller(
            self.controller_type, create_actions=True, joint_enable_check=True)

    def _check_time(self, time, fastest_time, time_scale):
        """Check and Return send angle vector time

        Parameters
        ----------
        time : float or None or list of tuple
            time of send angle vector.
        fastest_time : float
            fastest time
        time_scale : float
            Time will use 1/time_scale of the fastest speed.
            time_scale must be >=1.

        Returns
        -------
        time : float
            time of send angle vector.
        """
        if time_scale < 1:
            raise ValueError(
                'time_scale must be >=1, but given: {}'.format(time_scale))
        if isinstance(time, Number):
            if isinstance(fastest_time, list) and len(fastest_time) > 0:
                joints_violating_minimum_time = []
                org_time = time
                for fastest_time_each_joint, joint_name in fastest_time:
                    if time < fastest_time_each_joint:
                        joints_violating_minimum_time.append(joint_name)
                        time = max(time, fastest_time_each_joint)
                if len(joints_violating_minimum_time) > 0:
                    self.get_logger().warn(
                        'Time has been changed from {} to {} '.format(
                            org_time, time)
                        + 'due to joint velocity limit. '
                        + 'Make sure that joint limit is correctly set in urdf'
                        + ' and the following joints had requested times'
                        + ' shorter than the minimum allowed: {}'.format(
                            joints_violating_minimum_time))
                return time
            if time < fastest_time:
                self.get_logger().warn(
                    'Time has been changed from {} to {} '
                    'due to joint velocity limit. '
                    'Make sure that joint limit is correctly set in urdf'.
                    format(time, fastest_time))
                time = fastest_time
        elif time is None:
            if isinstance(fastest_time, list) and len(fastest_time) > 0:
                fastest_time = max([t for t, _ in fastest_time])
            time = time_scale * fastest_time
            self.get_logger().warn(
                'Time of send angle vector is set to {}. '
                'If the speed seems slow, check if '
                'joint velocity limit is correctly set in urdf'.
                format(time))
        else:
            raise ValueError(
                'time is invalid type. {}'.format(time))
        return time

    def wait_until_update_all_joints(self, tgt_tm):
        """Wait until all joints have been updated with timestamps newer than target time.

        This method handles both rclpy.time.Time objects (with nanoseconds attribute)
        and builtin_interfaces.msg.Time objects (with sec and nanosec attributes).

        Parameters
        ----------
        tgt_tm : rclpy.time.Time or bool
            Target time to wait for. If True, uses current time.
        """
        if hasattr(tgt_tm, 'nanoseconds'):
            initial_time = tgt_tm.nanoseconds
        else:
            initial_time = self.get_clock().now().nanoseconds

        while True:
            if 'stamp_list' in self.robot_state:
                all_valid = True
                for ts in self.robot_state['stamp_list']:
                    if ts is None:
                        all_valid = False
                        break

                    # Handle rclpy.time.Time objects
                    if hasattr(ts, 'nanoseconds'):
                        if ts.nanoseconds <= initial_time:
                            all_valid = False
                            break
                    # Handle builtin_interfaces.msg.Time objects
                    elif hasattr(ts, 'sec') and hasattr(ts, 'nanosec'):
                        ts_nano = ts.sec * 1e9 + ts.nanosec
                        if ts_nano <= initial_time:
                            all_valid = False
                            break
                    else:
                        # Unknown timestamp type
                        all_valid = False
                        break

                if all_valid:
                    return

            # Small sleep to avoid busy waiting and allow other threads to run
            time.sleep(0.001)

    def set_robot_state(self, key, msg):
        self.robot_state[key] = msg

    def set_moving_status(self, key, msg):
        """Update the moving status based on follow_joint_trajectory.

        While the robot is moving, the latest goal status is ACTIVE (1).
        When the goal is achieved, its status changes to SUCCEEDED (4).
        """
        moving = any(
            goal_status.status in (action_msgs.msg.GoalStatus.STATUS_EXECUTING,
                                   action_msgs.msg.GoalStatus.STATUS_CANCELING)
            for goal_status in msg.status_list
        )
        if moving:
            self.moving_status[key] = True
            self.prev_moving_status[key] = True
        elif self.prev_moving_status.get(key, False):
            self.moving_status[key] = True
            self.prev_moving_status[key] = False
        else:
            self.moving_status[key] = False

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
        self._joint_state_msg = msg
        if 'name' in self.robot_state:
            robot_state_names = self.robot_state['name']
        else:
            robot_state_names = msg.name
            self.robot_state['name'] = robot_state_names
            for key in ['position', 'velocity', 'effort']:
                self.robot_state[key] = np.zeros(len(robot_state_names))
            self.robot_state['stamp_list'] = [None for _ in robot_state_names]

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
        controller_type_actions = {}
        controller_type_params = {}
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
                if 'controller_type' in controller:
                    controller_type_actions[
                        controller['controller_type']] = [action]
                    controller_type_params[
                        controller['controller_type']] = [controller]
            for action, action_name in zip(tmp_actions, tmp_actions_name):
                if self.controller_timeout is None:
                    self.get_logger().warn(
                        'Waiting for actionlib interface forever '
                        'because controller-timeout is None')
                self.get_logger().info('Waiting for actionlib interface {}'
                                       .format(action_name))
                if not (
                    self.joint_action_enable and action.wait_for_server(
                        timeout_sec=self.controller_timeout)):
                    self.get_logger().warn('{} is not respond, {}_interface is disable'.
                                           format(action, self.robot.name))
                    self.get_logger().warn('make sure that you can run '
                                           "'ros2 topic echo {0}/status' "
                                           "and 'ros2 topic info {0}/status'".
                                           format(action_name))
                    if joint_enable_check:
                        self.joint_action_enable = False
                        return []
                else:
                    self.get_logger().info('Actionlib interface {} is enabled'
                                           .format(action_name))
            for param in self.default_controller():
                controller_state = param['controller_state']
                trajectory_status = '{}/status'.format(
                    param['controller_action'])
                if self.namespace is not None:
                    controller_state_topic_name = '{}/{}'.format(
                        self.namespace,
                        controller_state)
                    trajectory_status_topic_name = '{}/{}'.format(
                        self.namespace,
                        trajectory_status)
                else:
                    controller_state_topic_name = controller_state
                    trajectory_status_topic_name = trajectory_status
                self.create_subscription(
                    control_msgs.msg.JointTrajectoryControllerState,
                    controller_state_topic_name,
                    lambda msg: self.set_robot_state(controller_state, msg),
                    10)
                self.create_subscription(
                    action_msgs.msg.GoalStatusArray,
                    trajectory_status_topic_name,
                    lambda msg: self.set_moving_status(
                        param['controller_type'], msg),
                    10)
        else:
            self.controller_type = controller_type
        self.controller_table[controller_type] = tmp_actions
        self.controller_param_table[controller_type] \
            = self.default_controller()
        self.controller_table.update(controller_type_actions)
        self.controller_param_table.update(controller_type_params)
        return self.controller_table[controller_type]

    def default_controller(self):
        return [dict(
            controller_type='fullbody_controller',
            controller_action='fullbody_controller/'
            'follow_joint_trajectory',
            controller_state='fullbody_controller/state',
            action_type=control_msgs.action.FollowJointTrajectory,
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
                ret[i] = ret[i] % (2 * np.pi)
                if ret[i] > np.pi:
                    ret[i] = ret[i] - 2 * np.pi
                elif ret[i] < - np.pi:
                    ret[i] = ret[i] + 2 * np.pi
        return ret

    def angle_vector(self,
                     av=None,
                     time=None,
                     controller_type=None,
                     start_time=0.0,
                     time_scale=5.0,
                     velocities=None):
        """Send joint angle to robot

        Send joint angle to robot. this method returns immediately, so use
        self.wait_interpolation to block until the motion stops.

        Parameters
        ----------
        av : list or numpy.ndarray
            joint angle vector
        time : None or float
            time to goal in [sec]
            if designated time is faster than fastest speed, use fastest speed
            if not specified(None),
            it will use 1 / time_scale of the fastest speed.
        controller_type : string
            controller method name
        start_time : float
            time to start moving
        time_scale : float
            if time is not specified,
            it will use 1/time_scale of the fastest speed.
            time_scale must be >=1. (default: 5.0)

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
        if controller_type not in self.controller_table:
            self.get_logger().warn(
                'controller_type {} not found'.format(controller_type))
            return False

        fastest_time = self.angle_vector_duration(
            self.angle_vector(),
            av,
            controller_type,
            return_joint_names=True)
        time = self._check_time(time, fastest_time, time_scale=time_scale)

        self.robot.angle_vector(av)
        cacts = self.controller_table[controller_type]

        if velocities is not None:
            angle_velocities = velocities
        else:
            angle_velocities = np.zeros_like(av)
        duration = time
        traj_points = [(av, angle_velocities, duration), ]
        controller_params = self.controller_param_table[controller_type]
        for action, controller_param in zip(cacts, controller_params):
            self.send_ros_controller(
                action,
                controller_param['joint_names'],
                start_time,
                traj_points)
        return av

    def potentio_vector(self):
        """Returns current robot angle vector, This method uses caced data."""
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
        goal_msg = control_msgs.action.FollowJointTrajectory.Goal()
        goal_points = []
        if isinstance(start_time, Number):
            current_time = self.get_clock().now()
            start_duration = Duration(seconds=start_time)
            st = Time(nanoseconds=current_time.nanoseconds + start_duration.nanoseconds)
        joints = [getattr(self.robot, jn) for jn in joint_names]

        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.header.stamp = st.to_msg()
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
                    positions=positions.tolist(),
                    velocities=velocities.tolist(),
                    time_from_start=Duration(seconds=duration).to_msg()))
        goal_msg.trajectory.points = goal_points
        future = action.send_goal_async(goal_msg)

        # Store the future for wait_interpolation
        action._current_future = future

        # Set up callback to store goal handle when future completes
        def goal_response_callback(future):
            try:
                goal_handle = future.result()
                if goal_handle.accepted:
                    action._current_goal_handle = goal_handle
                else:
                    self.get_logger().warn("Goal was rejected")
            except Exception as e:
                self.get_logger().error(f"Error in goal response: {e}")

        future.add_done_callback(goal_response_callback)
        return future

    def angle_vector_sequence(self,
                              avs,
                              times=None,
                              controller_type=None,
                              start_time=0.0,
                              time_scale=5.0):
        """Send sequence of joint angles to robot

        Send sequence of joint angle to robot, this method returns
        immediately, so use self.wait_interpolation to block until the motion
        stops.

        Parameters
        ----------
        avs : list or numpy.ndarray
            [av0, av1, ..., avn]
            sequence of joint angles
        times : list of float or float
            [list tm0 tm1 ... tmn]
            sequence of duration(float) from previous angle-vector
            to next goal [sec].
            if times is atom, then use
                (list (make-list (length avs) :initial-element times))
                for times
            if designated each tmn is faster than fastest speed,
                use fastest speed
            if tmn is nil, then it will use 1/time_scale of the fastest speed.
        ctype : string
            controller method name
        start_time : float
            time to start moving
        time_scale : float
            if time is not specified,
            it will use 1/time_scale of the fastest speed.
            time_scale must be >=1. (default: 5.0)

        Returns
        -------
        avs : list of numpy.ndarray
            list of angle vector.
        """
        if controller_type is None:
            controller_type = self.controller_type

        if controller_type not in self.controller_table:
            self.get_logger().warn('controller_type: {} not found'.
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
            fastest_time = self.angle_vector_duration(
                prev_av, av, controller_type,
                return_joint_names=True)
            time = times[i_step]
            time = self._check_time(time, fastest_time, time_scale=time_scale)

            vel = np.zeros_like(prev_av)
            if i_step != total_steps - 1:
                next_time = times[i_step + 1]
                next_av = avs[i_step + 1]
                fastest_next_time = self.angle_vector_duration(
                    av, next_av, controller_type,
                    return_joint_names=True)
                next_time = self._check_time(
                    next_time, fastest_next_time, time_scale=time_scale)
                if time > 0.0 and next_time > 0.0:
                    v0 = self.sub_angle_vector(av, prev_av)
                    v1 = self.sub_angle_vector(next_av, av)
                    indices = v0 * v1 >= 0.0
                    vel[indices] = 0.5 * ((1.0 / time) * v0[indices]
                                          + (1.0 / next_time) * v1[indices])
            traj_points.append((av, vel, time + next_start_time))
            next_start_time += time
            prev_av = av

        cacts = self.controller_table[controller_type]
        controller_params = self.controller_param_table[controller_type]
        for action, controller_param in zip(cacts, controller_params):
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
        list[bool]
            return values are a list of is_interpolating for all controllers.
        """
        if controller_type:
            controller_actions = self.controller_table[controller_type]
        else:
            controller_actions = self.controller_table[self.controller_type]

        results = []
        start_time = time.time()

        for action in controller_actions:
            action_completed = False

            # First, wait for goal to be accepted
            if hasattr(action, '_current_future') and action._current_future:
                self.get_logger().info("Waiting for goal to be accepted...")

                # Wait for goal acceptance
                while not action._current_future.done():
                    time.sleep(0.01)
                    if timeout > 0 and (time.time() - start_time) > timeout:
                        self.get_logger().warn(f"Goal acceptance timed out after {timeout}s")
                        results.append(True)
                        action_completed = True
                        break

                if action_completed:
                    continue

                # Check if goal was accepted
                try:
                    goal_handle = action._current_future.result()
                    if not goal_handle.accepted:
                        self.get_logger().warn("Goal was rejected")
                        results.append(False)
                        continue
                except Exception as e:
                    self.get_logger().error(f"Error getting goal handle: {e}")
                    results.append(False)
                    continue

            # Now wait for result
            result_future = action.get_result_async()
            if result_future is None:
                self.get_logger().info("No active goal to wait for")
                results.append(False)
                continue

            self.get_logger().info("Waiting for motion to complete...")

            # Wait for the motion to complete
            while not result_future.done():
                time.sleep(0.01)  # 10ms polling

                # Check timeout
                if timeout > 0 and (time.time() - start_time) > timeout:
                    self.get_logger().warn(f"wait_interpolation timed out after {timeout}s")
                    results.append(True)  # Still interpolating
                    action_completed = True
                    break

            if not action_completed:
                # Future completed normally
                try:
                    result_future.result()
                    self.get_logger().info("Motion completed successfully")
                    results.append(False)  # Not interpolating anymore
                except Exception as e:
                    self.get_logger().error(f"Error getting result: {e}")
                    results.append(False)

        return results

    def is_interpolating(self, controller_type=None):
        if controller_type:
            controller_actions = self.controller_table[controller_type]
        else:
            controller_actions = self.controller_table[self.controller_type]
        is_interpolatings = map(
            lambda action: action.is_interpolating(), controller_actions)
        return any(list(is_interpolatings))

    def is_moving(self, controller_type=None):
        """"Check whether the robot is moving due to follow_joint_trajectory.

        This is not limited to goals sent from the same instance.
        """
        if controller_type is None or controller_type == self.controller_type:
            is_movings = list(self.moving_status.values())
        else:
            is_movings = [self.moving_status[controller_type]]
        return any(is_movings)

    def angle_vector_duration(self, start_av, end_av, controller_type=None,
                              return_joint_names=False):
        """Calculate maximum time to reach goal for all joint.

        Parameters
        ----------
        start_av : list or np.ndarray
            start angle-vector
        end_av : list or np.ndarray
            end angle-vector (target position)
        controller_type : None or string
            type of controller
        return_joint_names : bool
            if True, return list of tuple (time, joint_name)

        Returns
        -------
        av_duration : float or list of tuple
            if return_joint_names is False,
                return max time of angle vector.
            if return_joint_names is True,
                return list of tuple (time, joint_name)
                where time is the time to reach goal for each joint
                and joint_name is the name of the joint at the same
        """
        if controller_type is None:
            controller_type = self.controller_type
        unordered_joint_names = set(
            _flatten([c['joint_names']
                      for c in self.controller_param_table[controller_type]]))
        joint_list = self.robot.joint_list
        diff_avs = self.sub_angle_vector(end_av, start_av)
        time_list = []
        for diff_angle, joint in zip(diff_avs, joint_list):
            if joint.name in unordered_joint_names:
                if (isinstance(joint, RotationalJoint)
                    and abs(diff_angle) < 0.0017453292519943296) \
                    or (isinstance(joint, LinearJoint)
                        and abs(diff_angle) < 0.01):
                    time = 0
                else:
                    time = 1. * abs(diff_angle) / joint.max_joint_velocity
            else:
                time = 0
            if return_joint_names:
                time_list.append((time, joint.name))
            else:
                time_list.append(time)
        if return_joint_names:
            return time_list
        return max(time_list)

    def cancel_angle_vector(self, controller_type=None):
        """Stop motion via follow joint trajectory cancel

        Parameters
        ----------
        controller_type : str, optional
            Controller type to cancel. If None, cancels all controllers.
        """
        if controller_type is None:
            # Cancel all controllers
            for controller_type_key in self.controller_table:
                for action in self.controller_table[controller_type_key]:
                    action.cancel_all_goals()
        else:
            # Cancel specific controller
            if controller_type in self.controller_table:
                for action in self.controller_table[controller_type]:
                    action.cancel_all_goals()
            else:
                self.get_logger().warn(
                    f'Controller type {controller_type} not found for cancellation')


class ControllerActionClient:

    def __init__(self, node, ns, ActionSpec):
        self.node = node
        self.time_to_finish = 0
        self.last_feedback_msg_stamp = node.get_clock().now()
        self._action_client = ActionClient(node, ActionSpec, ns)
        self._current_goal_handle = None

    def wait_for_server(self, timeout_sec=None):
        return self._action_client.wait_for_server(timeout_sec=timeout_sec)

    def send_goal_async(self, goal):
        future = self._action_client.send_goal_async(goal)
        return future

    def send_goal(self, goal):
        future = self.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future)
        self._current_goal_handle = future.result()
        return self._current_goal_handle

    def get_result_async(self):
        if self._current_goal_handle:
            return self._current_goal_handle.get_result_async()
        return None

    def cancel_all_goals(self):
        # Cancel current goal if it exists
        if self._current_goal_handle:
            return self._current_goal_handle.cancel_goal_async()
        else:
            # No active goal to cancel
            self.node.get_logger().info("No active goal to cancel")
            return None

    def is_interpolating(self):
        if self._current_goal_handle:
            return not self._current_goal_handle.accepted
        return False
