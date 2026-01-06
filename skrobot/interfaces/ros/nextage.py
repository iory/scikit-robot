import actionlib
import control_msgs.msg
import rospy
import trajectory_msgs.msg

from skrobot.interfaces.ros.base import ROSRobotInterfaceBase


class NextageROSRobotInterface(ROSRobotInterfaceBase):
    def __init__(self, *args, **kwargs):
        self.on_gazebo = rospy.get_param('/gazebo/time_step', None) is not None \
                         and rospy.get_param('/torso_controller/type', None) is not None

        if self.on_gazebo:
            rospy.loginfo("Gazebo environment detected")

        self.lhand = None
        self.rhand = None

        super(NextageROSRobotInterface, self).__init__(*args, **kwargs)

    def _init_lhand(self):
        if self.lhand is None:
            self.lhand = LHandInterface()
        return self.lhand

    def _init_rhand(self):
        if self.rhand is None:
            self.rhand = RHandInterface()
        return self.rhand

    @property
    def fullbody_controller(self):
        return dict(
            controller_type='fullbody_controller',
            controller_action='/fullbody_controller/follow_joint_trajectory_action',
            controller_state='/fullbody_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=[j.name for j in self.robot.joint_list],
        )

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

    @property
    def torso_controller(self):
        return dict(
            controller_type='torso_controller',
            controller_action='/torso_controller/follow_joint_trajectory_action',
            controller_state='/torso_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=[j.name for j in self.robot.torso.joint_list],
        )

    @property
    def head_controller(self):
        return dict(
            controller_type='head_controller',
            controller_action='/head_controller/follow_joint_trajectory_action',
            controller_state='/head_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=[j.name for j in self.robot.head.joint_list],
        )

    def default_controller(self):
        if self.on_gazebo:
            return [self.larm_controller, self.rarm_controller,
                    self.torso_controller, self.head_controller]
        else:
            return [self.fullbody_controller]

    def start_grasp(self, arm='arms', **kwargs):
        if arm == 'larm':
            self._init_lhand().start_grasp(**kwargs)
        elif arm == 'rarm':
            self._init_rhand().start_grasp(**kwargs)
        elif arm == 'arms':
            self._init_lhand().start_grasp(**kwargs)
            self._init_rhand().start_grasp(**kwargs)

    def stop_grasp(self, arm='arms', **kwargs):
        if arm == 'larm':
            self._init_lhand().stop_grasp(**kwargs)
        elif arm == 'rarm':
            self._init_rhand().stop_grasp(**kwargs)
        elif arm == 'arms':
            self._init_lhand().stop_grasp(**kwargs)
            self._init_rhand().stop_grasp(**kwargs)

    def open_forceps(self, arm='arms', **kwargs):
        if arm == 'larm':
            self._init_lhand().open_forceps(**kwargs)
        elif arm == 'arms':
            self._init_lhand().open_forceps(**kwargs)

    def close_forceps(self, arm='arms', **kwargs):
        if arm == 'larm':
            self._init_lhand().close_forceps(**kwargs)
        elif arm == 'arms':
            self._init_lhand().close_forceps(**kwargs)

    def open_holder(self, arm='arms', **kwargs):
        if arm == 'rarm':
            self._init_rhand().open_holder(**kwargs)
        elif arm == 'arms':
            self._init_rhand().open_holder(**kwargs)

    def close_holder(self, arm='arms', **kwargs):
        if arm == 'rarm':
            self._init_rhand().close_holder(**kwargs)
        elif arm == 'arms':
            self._init_rhand().close_holder(**kwargs)


class LHandInterface:
    def __init__(self):
        self.action_client = actionlib.SimpleActionClient(
            "/lhand/position_joint_trajectory_controller/follow_joint_trajectory",
            control_msgs.msg.FollowJointTrajectoryAction
        )
        if not self.action_client.wait_for_server(rospy.Duration(5)):
            rospy.logwarn("LHand action server not available")

    def move_hand(self, grasp_angle, wait=True, tm=1.0):
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["lhand_joint"]
        point = trajectory_msgs.msg.JointTrajectoryPoint()
        point.positions = [grasp_angle]
        point.time_from_start = rospy.Duration(tm)
        goal.trajectory.points = [point]
        self.action_client.send_goal(goal)
        if wait:
            self.action_client.wait_for_result(rospy.Duration(tm + 5.0))

    def start_grasp(self, **kwargs):
        return self.move_hand(-2.7, **kwargs)

    def stop_grasp(self, **kwargs):
        return self.move_hand(0.0, **kwargs)

    def open_forceps(self, wait=False, tm=0.2):
        return self.move_hand(-2.2, wait, tm)

    def close_forceps(self, wait=False, tm=0.2):
        return self.move_hand(-3.2, wait, tm)


class RHandInterface:
    def __init__(self):
        self.action_client = actionlib.SimpleActionClient(
            "/rhand/position_joint_trajectory_controller/follow_joint_trajectory",
            control_msgs.msg.FollowJointTrajectoryAction
        )
        if not self.action_client.wait_for_server(rospy.Duration(5)):
            rospy.logwarn("RHand action server not available")

    def move_hand(self, grasp_angle, wait=True, tm=1.0):
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["rhand_joint"]
        point = trajectory_msgs.msg.JointTrajectoryPoint()
        point.positions = [grasp_angle]
        point.time_from_start = rospy.Duration(tm)
        goal.trajectory.points = [point]
        self.action_client.send_goal(goal)
        if wait:
            self.action_client.wait_for_result(rospy.Duration(tm + 5.0))

    def start_grasp(self, **kwargs):
        return self.move_hand(-2.7, **kwargs)

    def stop_grasp(self, **kwargs):
        return self.move_hand(0.0, **kwargs)

    def open_holder(self, wait=True, tm=0.2):
        return self.move_hand(-0.20, wait, tm)

    def close_holder(self, wait=True, tm=0.2):
        return self.move_hand(0.08, wait, tm)
