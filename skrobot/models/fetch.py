from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.data import fetch_urdfpath
from skrobot.model import RobotModel

from .urdf import RobotModelFromURDF


class Fetch(RobotModelFromURDF):
    """Fetch Robot Model.

    http://docs.fetchrobotics.com/robot_hardware.html
    """

    def __init__(self, *args, **kwargs):
        super(Fetch, self).__init__(*args, **kwargs)
        self.rarm_end_coords = CascadedCoords(parent=self.gripper_link,
                                              name='rarm_end_coords')
        self.rarm_end_coords.translate([0, 0, 0])
        self.rarm_end_coords.rotate(0, axis='z')
        self.end_coords = [self.rarm_end_coords]

    @cached_property
    def default_urdf_path(self):
        return fetch_urdfpath()

    def reset_pose(self):
        self.torso_lift_joint.joint_angle(0)
        self.shoulder_pan_joint.joint_angle(np.deg2rad(75.6304))
        self.shoulder_lift_joint.joint_angle(np.deg2rad(80.2141))
        self.upperarm_roll_joint.joint_angle(np.deg2rad(-11.4592))
        self.elbow_flex_joint.joint_angle(np.deg2rad(98.5487))
        self.forearm_roll_joint.joint_angle(0.0)
        self.wrist_flex_joint.joint_angle(np.deg2rad(95.111))
        self.wrist_roll_joint.joint_angle(0.0)
        self.head_pan_joint.joint_angle(0.0)
        self.head_tilt_joint.joint_angle(0.0)
        return self.angle_vector()

    def reset_manip_pose(self):
        self.torso_lift_joint.joint_angle(0)
        self.shoulder_pan_joint.joint_angle(0)
        self.shoulder_lift_joint.joint_angle(0)
        self.upperarm_roll_joint.joint_angle(0)
        self.elbow_flex_joint.joint_angle(np.pi / 2.0)
        self.forearm_roll_joint.joint_angle(0)
        self.wrist_flex_joint.joint_angle(- np.pi / 2.0)
        self.wrist_roll_joint.joint_angle(0)
        self.head_pan_joint.joint_angle(0)
        self.head_tilt_joint.joint_angle(0)
        return self.angle_vector()

    @cached_property
    def rarm(self):
        rarm_links = [self.shoulder_pan_link,
                      self.shoulder_lift_link,
                      self.upperarm_roll_link,
                      self.elbow_flex_link,
                      self.forearm_roll_link,
                      self.wrist_flex_link,
                      self.wrist_roll_link]
        rarm_joints = []
        for link in rarm_links:
            rarm_joints.append(link.joint)
        r = RobotModel(link_list=rarm_links,
                       joint_list=rarm_joints)
        r.end_coords = self.rarm_end_coords
        return r
