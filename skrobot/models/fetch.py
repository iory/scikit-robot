from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.data import fetch_urdfpath
from skrobot.model import RobotModel


class Fetch(RobotModel):
    """Fetch Robot Model.

    http://docs.fetchrobotics.com/robot_hardware.html
    """

    def __init__(self, urdf=None, urdf_file=None):
        # For backward compatibility, support both urdf and urdf_file
        if urdf is not None and urdf_file is not None:
            raise ValueError(
                "'urdf' and 'urdf_file' cannot be given at the same time"
            )
        urdf_input = urdf or urdf_file or fetch_urdfpath()
        super(Fetch, self).__init__(urdf=urdf_input)
        self.rarm_end_coords = CascadedCoords(parent=self.gripper_link,
                                              name='rarm_end_coords')
        self.rarm_end_coords.translate([0, 0, 0])
        self.rarm_end_coords.rotate(0, axis='z')
        self.end_coords = [self.rarm_end_coords]

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

    @cached_property
    def rarm_with_torso(self):
        rarm_with_torso_links = [self.torso_lift_link,
                                 self.shoulder_pan_link,
                                 self.shoulder_lift_link,
                                 self.upperarm_roll_link,
                                 self.elbow_flex_link,
                                 self.forearm_roll_link,
                                 self.wrist_flex_link,
                                 self.wrist_roll_link]
        rarm_with_torso_joints = []
        for link in rarm_with_torso_links:
            rarm_with_torso_joints.append(link.joint)
        r = RobotModel(link_list=rarm_with_torso_links,
                       joint_list=rarm_with_torso_joints)
        r.end_coords = self.rarm_end_coords
        return r
