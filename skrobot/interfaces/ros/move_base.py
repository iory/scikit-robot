import actionlib
import actionlib_msgs.msg
import control_msgs.msg
import dynamic_reconfigure.msg
import dynamic_reconfigure.srv
import geometry_msgs.msg
import move_base_msgs.msg
import nav_msgs.msg
import numpy as np
import rospy
import std_srvs.srv
import trajectory_msgs.msg

from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rotate_vector
from skrobot.coordinates.math import rotation_distance

from .base import ROSRobotInterfaceBase
from .tf_utils import coords_to_geometry_pose
from .tf_utils import geometry_pose_to_coords
from .tf_utils import tf_pose_to_coords
from .transform_listener import TransformListener


class ROSRobotMoveBaseInterface(ROSRobotInterfaceBase):

    def __init__(self, *args, **kwargs):
        self.move_base_action = None
        self.move_base_trajectory_action = None
        self.move_base_goal_msg = None
        self.move_base_goal_coords = None
        self.move_base_goal_map_to_frame = None
        self.odom_topic = None
        self.go_pos_unsafe_goal_msg = None
        self.current_goal_coords = None
        self.map_frame_id = kwargs.pop(
            'map_frame_id', 'map')
        self.move_base_action_name = kwargs.pop(
            'move_base_action_name', 'move_base')
        self.base_frame_id = kwargs.pop(
            'base_frame_id',
            'base_footprint')
        self.base_controller_action_name = kwargs.pop(
            'base_controller_action_name',
            "/base_controller/follow_joint_trajectory")
        self.move_base_trajectory_joint_names = kwargs.pop(
            'base_controller_joint_names',
            ["base_link_x", "base_link_y", "base_link_pan"])
        self.move_base_simple_name = kwargs.pop(
            'move_base_simple_name',
            'move_base_simple')
        self.odom_topic = kwargs.pop('odom_topic', '/base_odometry/odom')
        self.use_tf2 = kwargs.pop('use_tf2', False)

        super(ROSRobotMoveBaseInterface, self).__init__(
            *args, **kwargs)

        self.tf_listener = TransformListener(
            use_tf2=self.use_tf2)

        self.move_base_action = actionlib.SimpleActionClient(
            self.move_base_action_name,
            move_base_msgs.msg.MoveBaseAction)

        if self.base_controller_action_name:
            self.move_base_trajectory_action = actionlib.SimpleActionClient(
                self.base_controller_action_name,
                control_msgs.msg.FollowJointTrajectoryAction)
            if self.move_base_trajectory_action.wait_for_server(
                    rospy.Duration(3)) is False:
                rospy.logwarn('{} is not found'.format(
                    self.base_controller_action_name))
                self.move_base_trajectory_action = None

        self.go_pos_unsafe_goal_msg = None

        self.move_base_simple_publisher = rospy.Publisher(
            "{}/goal".format(self.move_base_simple_name),
            geometry_msgs.msg.PoseStamped,
            queue_size=1)

        self.odom_msg = None
        self.odom_subscriber = rospy.Subscriber(
            self.odom_topic, nav_msgs.msg.Odometry,
            callback=self.odom_callback,
            queue_size=1)

    @property
    def odom(self):
        """Return Coordinates of this odom

        Returns
        -------
        odom_coords : skrobot.coordinates.Coordinates
            coordinates of odom.
        """
        if self.odom_msg is None:
            raise RuntimeError(
                'odom is not set. Please check odom topic {}'
                ' is published.'.format(self.odom_topic))
        pos = (self.odom_msg.pose.pose.position.x,
               self.odom_msg.pose.pose.position.y,
               self.odom_msg.pose.pose.position.z)
        q_wxyz = (self.odom_msg.pose.pose.orientation.w,
                  self.odom_msg.pose.pose.orientation.x,
                  self.odom_msg.pose.pose.orientation.y,
                  self.odom_msg.pose.pose.orientation.z)
        return Coordinates(pos=pos, rot=q_wxyz)

    def odom_callback(self, msg):
        """ROS's subscriber callback for odom.

        Parameters
        ----------
        msg : nav_msgs.msg.Odometry
            odometry message.
        """
        self.odom_msg = msg

    def go_stop(self, force_stop=True):
        """Cancel move_base.

        Parameters
        ----------
        force_stop : bool
            if force_stop is True, send go_velocity(0, 0, 0)
        """
        if self.joint_action_enable is True:
            ret = self.move_base_action.cancel_all_goals()
        if force_stop is True:
            self.go_velocity(0.0, 0.0, 0.0)
        return ret

    def move_to(self, coords, wait=True,
                frame_id=None):
        """Move Robot to target coords.

        Parameters
        ----------
        coords : str or skrobot.coordinates.Coordinates
            target tf name or target coords.
        wait : bool
            if True, wait until move end.
        frame_id : None or str

        Returns
        -------
        result : bool
            if move_base is succeeded, return True.
        """
        if frame_id is None:
            frame_id = self.base_frame_id
        if isinstance(coords, str):
            base_to_target = self.tf_listener.lookup_transform(
                self.base_frame_id, coords,
                rospy.Time(0),
                rospy.Duration(3))
            if base_to_target is False:
                rospy.logwarn('Could not lookup transform {} to {}'.
                              format(self.base_frame_id, coords))
                return False
            coords = tf_pose_to_coords(base_to_target)
        if self.move_to_send(coords, frame_id=frame_id) is False:
            return False
        if wait:
            return self.move_to_wait(frame_id=frame_id)
        return True

    def move_to_send(self,
                     coords,
                     frame_id=None,
                     wait_for_server_timeout=5.0):
        """Send MoveBaseAction

        Parameters
        ----------
        coords : skrobot.coordinates.Coordinates

        Return
        ------
        result : bool
            False or True. If False, could not send MoveBaseAction

        """
        if frame_id is None:
            frame_id = self.base_frame_id
        self.move_base_goal_msg = move_base_msgs.msg.MoveBaseActionGoal()
        self.move_base_goal_coords = coords
        count = 0
        ros_time = rospy.Time.now()
        map_to_frame = self.tf_listener.lookup_transform(
            self.map_frame_id, frame_id, rospy.Time(0))
        if map_to_frame is False:
            rospy.logwarn("Could not lookup transform {} to {}".format(
                self.map_frame_id, frame_id))
            return False
        map_to_frame_transform = tf_pose_to_coords(map_to_frame)
        # store in member variable for self.move_to_wait()
        self.move_base_goal_map_to_frame = map_to_frame_transform

        if self.move_base_action.wait_for_server(
                rospy.Duration(wait_for_server_timeout)) is False:
            return False

        self.move_base_goal_msg.header.stamp = ros_time
        self.move_base_goal_msg.goal.target_pose.header.stamp = ros_time
        if map_to_frame:
            self.move_base_goal_msg.goal.target_pose.header.frame_id = \
                self.map_frame_id
            self.move_base_goal_msg.goal.target_pose.pose = \
                coords_to_geometry_pose(
                    coords.copy_worldcoords().transform(
                        map_to_frame_transform, 'world'))
        else:
            self.move_base_goal_msg.goal.target_pose.header.frame_id = frame_id
            self.move_base_goal_msg.goal.target_pose.pose = \
                coords_to_geometry_pose(coords)
        self.move_base_goal_msg.header.seq = count
        self.move_base_action.send_goal(self.move_base_goal_msg.goal)
        return self.move_base_goal_msg

    def move_to_wait(self,
                     retry=10,
                     frame_id='world'):
        count = 0
        if self.move_base_goal_msg is None:
            return False

        ret = False
        while ret is False and count < retry:
            if count > 0:
                self.clear_costmap()
                self.move_base_goal_msg.header.seq = count
                self.move_base_goal_msg.goal.target_pose.header.seq = count
                self.move_base_action.send_goal(self.move_base_goal_msg.goal)
            self.move_base_action.wait_for_result()
            if self.move_base_action.get_state() == \
               actionlib_msgs.msg.GoalStatus.PREEMPTED:
                ret = False
                continue
            if self.move_base_action.get_state() == \
               actionlib_msgs.msg.GoalStatus.SUCCEEDED:
                ret = True
            count += 1
        return ret

    def _calc_move_diff_coords(self, frame_id):
        if frame_id == self.base_frame_id:
            map_goal_coords = self.move_base_goal_map_to_frame.\
                copy_worldcoords().transform(
                    self.move_base_goal_coords.copy_worldcoords())
        else:
            map_to_frame = self.tf_listener.lookup_transform(
                self.map_frame_id, frame_id,
                rospy.Time(0))
            if map_to_frame is False:
                rospy.logwarn("Could not lookup transform {} to {}".format(
                    self.map_frame_id, frame_id))
                return False
            map_to_frame_transform = tf_pose_to_coords(
                map_to_frame)
            map_goal_coords = map_to_frame_transform.transform(
                self.move_base_goal_coords.copy_worldcoords())

        current_coords = self.tf_listener.lookup_transform(
            self.map_frame_id, self.base_frame_id,
            rospy.Time(0))
        if current_coords is False:
            rospy.logwarn("Could not lookup transform {} to {}".format(
                self.map_frame_id, self.base_frame_id))
            return False
        current_coords = tf_pose_to_coords(current_coords)
        diff_coords = current_coords.transformation(
            map_goal_coords).copy_worldcoords()
        rospy.logwarn("move_to error translation: {}, rotation: {}".
                      format(diff_coords.translation,
                             diff_coords.rpy_angle()[0]))
        return diff_coords

    def go_pos(self, x=0.0, y=0.0, yaw=0.0, wait=True):
        """Move Robot using MoveBase

        Parameters
        ----------
        x : float
            move distance with respect to x axis. unit is [m].
        y : float
            move distance with respect to y axis. unit is [m].
        yaw : float
            rotate angle. unit is [rad].
        wait : bool
            if wait is True, wait until move base done.
        """
        c = Coordinates(pos=(x, y, 0)).rotate(yaw, 'z')
        return self.move_to(c, wait=wait, frame_id=self.base_frame_id)

    def go_pos_unsafe(self, x=0.0, y=0.0, yaw=0.0, wait=False):
        """Move Robot using MoveBase

        Parameters
        ----------
        x : float
            move distance with respect to x axis. unit is [m].
        y : float
            move distance with respect to y axis. unit is [m].
        yaw : float
            rotate angle. unit is [rad].
        wait : bool
            if wait is True, wait until stop go_pos_unsafe
        """
        self.go_pos_unsafe_no_wait(x=x, y=y, yaw=yaw)
        if wait is True:
            return self.go_pos_unsafe_wait()
        else:
            return True

    def go_pos_unsafe_no_wait(self, x=0.0, y=0.0, yaw=0.0):
        """Move Robot using MoveBase

        Parameters
        ----------
        x : float
            move distance with respect to x axis. unit is [m].
        y : float
            move distance with respect to y axis. unit is [m].
        yaw : float
            rotate angle. unit is [rad].
        """
        if self.move_base_trajectory_action is None:
            rospy.logwarn("go_pos_unsafe is disabled. "
                          'move_base_trajectory_action is not found')
            return True
        maxvel = 0.295
        maxrad = 0.495
        ratio = 0.8
        sec = max(np.linalg.norm([x, y]) / (maxvel * ratio),
                  abs(yaw) / (maxrad * ratio),
                  1.0)
        self.go_pos_unsafe_goal_msg = self.move_trajectory(
            x, y, yaw,
            sec, stop=True)
        return self.move_base_trajectory_action.send_goal(
            self.go_pos_unsafe_goal_msg.goal)

    def go_pos_unsafe_wait(self, wait_counts=3):
        maxvel = 0.295
        maxrad = 0.495
        ratio = 0.8
        counter = 0
        if self.go_pos_unsafe_goal_msg is None:
            return False
        if self.move_base_trajectory_action is None:
            rospy.logwarn("go_pos_unsafe_wait is disabled. "
                          'move_base_trajectory_action is not found')
            return True
        while counter < wait_counts:
            action_ret = self.move_base_trajectory_action.wait_for_result()
            if action_ret is False:
                return False

            goal_position = np.array(
                self.go_pos_unsafe_goal_msg.goal.
                trajectory.points[1].positions,
                dtype=np.float32)
            odom = self.odom
            odom_pos = odom.translation
            odom_angle = odom.rpy_angle()[0][0]
            diff_position = goal_position - (
                odom_pos + np.array((0, 0, odom_angle)))
            v = rotate_vector(
                np.array((diff_position[0], diff_position[1], 0.0)),
                -odom_angle, 'z') - np.array((0, 0, odom_angle))
            x = v[0]
            y = v[1]
            yaw = diff_position[2]
            if yaw > 2 * np.pi:
                yaw = yaw - 2 * np.pi
            if yaw < - 2 * np.pi:
                yaw = yaw + 2 * np.pi

            sec = max(np.linalg.norm([x, y]) / (maxvel * ratio),
                      abs(yaw) / (maxrad * ratio))
            sec = max(sec, 1.0)
            step = 1.0 / sec

            rospy.loginfo("                diff-pos {} {}, diff-angle {}".
                          format(x, y, yaw))
            if np.sqrt(x * x + y * y) <= 0.025 and \
               abs(yaw) <= np.deg2rad(2.5) and \
               counter > 0:  # try at least 1 time
                self.go_pos_unsafe_goal_msg = None
                return True

            self.go_pos_unsafe_goal_msg = self.move_trajectory(
                x * step, y * step, yaw * step, sec, stop=True)
            self.move_base_trajectory_action.send_goal(
                self.go_pos_unsafe_goal_msg.goal)
            counter += 1
        self.go_pos_unsafe_goal_msg = None
        return True

    def go_velocity(self, x=0.0, y=0.0, yaw=0.0,
                    sec=1.0, stop=True, wait=False):
        """Move Robot using MoveBase

        Parameters
        ----------
        x : float
            move velocity with respect to x axis. unit is [m/s].
        y : float
            move velocity with respect to y axis. unit is [m/s].
        yaw : float
            rotate angle. unit is [rad/s].
        sec : float
            time.
        wait : bool
            if wait is True, wait until move base done.
        """
        if self.move_base_trajectory_action is None:
            rospy.logwarn("go_velocity is disabled. "
                          'move_base_trajectory_action is not found')
            return True
        x = x * sec
        y = y * sec
        yaw = yaw * sec
        goal = self.move_trajectory(x, y, yaw, sec, stop=stop)
        ret = self.move_base_trajectory_action.send_goal(goal.goal)
        if wait:
            self.move_base_trajectory_action.wait_for_result()
        return ret

    def move_trajectory_sequence(self, trajectory_points, time_list, stop=True,
                                 start_time=None, send_action=None, wait=True):
        """Move base following the trajectory points at each time points

        trajectory-points [ list of #f(x y yaw) ([m] for x, y; [rad] for yaw) ]
        time-list [list of time span [sec] ]
        stop [ stop after msec moveing ]
        start-time [ robot will move at start-time [sec or ros::Time] ]
        send-action [ send message to action server, it means robot will move ]

        """
        if len(trajectory_points) != len(time_list):
            raise ValueError
        while self.odom_msg is None:
            rospy.sleep(0.01)
        odom_coords = geometry_pose_to_coords(self.odom_msg.pose.pose)
        goal = control_msgs.msg.FollowJointTrajectoryActionGoal()
        msg = trajectory_msgs.msg.JointTrajectory()
        msg.joint_names = self.move_base_trajectory_joint_names
        if start_time is not None:
            msg.header.stamp = start_time
        else:
            msg.header.stamp = rospy.Time.now()
        coords_list = [odom_coords]
        for traj_point in trajectory_points:
            cds = Coordinates(pos=odom_coords.translation,
                              rot=odom_coords.rotation)
            cds.translate((traj_point[0], traj_point[1], 0.0))
            cds.rotate(traj_point[2], 'z')
            coords_list.append(cds)
        cur_cds = odom_coords
        cur_yaw = cur_cds.rpy_angle()[0][0]
        cur_time = 0
        pts_msg_list = []
        for i, (cur_cds, next_cds) in enumerate(zip(
                coords_list[:-1],
                coords_list[1:])):
            next_time = time_list[i]
            tra = cur_cds.transformation(next_cds)
            rot = cur_cds.rotate_vector(tra.translation)
            diff_yaw = rotation_distance(cur_cds.rotation, next_cds.rotation)
            pts_msg_list.append(
                trajectory_msgs.msg.JointTrajectoryPoint(
                    positions=[cur_cds.translation[0],
                               cur_cds.translation[1],
                               cur_yaw],
                    velocities=[rot[0] / next_time,
                                rot[1] / next_time,
                                tra.rpy_angle()[0][0] / next_time],
                    time_from_start=rospy.Time(cur_time)))
            cur_time += next_time
            cur_cds = next_cds
            if tra.rpy_angle()[0][0] > 0:
                cur_yaw += abs(diff_yaw)
            else:
                cur_yaw -= abs(diff_yaw)

        # append last point
        if stop:
            velocities = [0, 0, 0]
        else:
            velocities = [rot[0] / next_time,
                          rot[1] / next_time,
                          tra.rpy_angle()[0][0] / next_time]
        pts_msg_list.append(
            trajectory_msgs.msg.JointTrajectoryPoint(
                positions=[cur_cds.translation[0],
                           cur_cds.translation[1],
                           cur_yaw],
                velocities=velocities,
                time_from_start=rospy.Time(cur_time)))

        msg.points = pts_msg_list
        goal.goal.trajectory = msg

        if not send_action:
            return goal

        if self.move_base_trajectory_action is None:
            rospy.logerror(
                'send_action is True, '
                'but move_base_trajectory_action is not found')
            return False

        self.move_base_trajectory_action.send_goal(goal.goal)
        if not wait:
            return goal

        if self.move_base_trajectory_action.wait_for_result():
            return self.move_base_trajectory_action.get_result()
        else:
            return False

    def move_trajectory(self, x, y, yaw, sec=1.0, stop=True,
                        start_time=None,
                        send_action=None):
        """Move trajectory.

        This function call move_trajectory_sequence internally.

        x : float
            move distance with respect to x axis. unit is [m].
        y : float
            move distance with respect to y axis. unit is [m].
        yaw : float
            rotate angle. unit is [rad].
        sec : float
            time. unit is [sec].
        """
        return self.move_trajectory_sequence([[x, y, yaw]], [sec],
                                             stop=stop,
                                             start_time=start_time,
                                             send_action=send_action)

    def clear_costmap(self):
        """Send signal to clear costmap for obstacle avoidance to move_base.

        """
        service_name = "{}/clear_costmaps".format(self.move_base_action_name)
        rospy.wait_for_service(service_name)
        empty_service = rospy.ServiceProxy(service_name, std_srvs.srv.Empty)
        return empty_service()

    def change_inflation_range(self,
                               inflation_range=0.2,
                               node_name='move_base_node',
                               costmap_name='local_costmap',
                               inflation_name='inflation'):
        """Changes inflation range of local costmap for obstacle avoidance.

        Parameters
        ----------
        inflation_range : float
            range of inflation
        node_name : str
            name of move_base_node
        costmap_name : str
            name of costmap
        inflation_name : str
            name of inflation
        """
        service_name = "{}/{}/{}/set_parameters".format(
            node_name, costmap_name, inflation_name)
        request = dynamic_reconfigure.srv.ReconfigureRequest()
        request.config.doubles.append(
            dynamic_reconfigure.msg.DoubleParameter(
                name='inflation_radius',
                value=inflation_range))
        rospy.wait_for_service(service_name)
        return rospy.ServiceProxy(
            service_name,
            dynamic_reconfigure.srv.Reconfigure)()
