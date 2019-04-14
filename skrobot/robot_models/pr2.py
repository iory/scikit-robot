import os
import tarfile

from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.robot_model import RobotModel
from skrobot.utils.download import _default_cache_dir
from skrobot.utils.download import cached_gdown_download

pr2_description_url = 'https://drive.google.com/uc?id='\
    '1zy4C665o6efPko7eMk4XBdHbvgFfdC-6'
pr2_description_md5sum = '892958fddcd58147869b17a78e06a172'


class PR2(RobotModel):

    """PR2 Robot Model.

    """

    def __init__(self, *args, **kwargs):
        self.urdf_path = kwargs.pop('urdf_path', None)
        RobotModel.__init__(self, *args, **kwargs)

        if self.urdf_path is None:
            self.urdf_path = os.path.join(_default_cache_dir,
                                          'pr2_description',
                                          'pr2.urdf')
            if not os.path.exists(self.urdf_path):
                download_filepath = cached_gdown_download(
                    pr2_description_url,
                    pr2_description_md5sum)
                extract_file = tarfile.open(download_filepath, 'r:gz')
                extract_file.extractall(_default_cache_dir)
        self.load_urdf(self.urdf_path)

        self.rarm_end_coords = CascadedCoords(
            parent=self.r_gripper_tool_frame,
            name='rarm_end_coords')
        self.larm_end_coords = CascadedCoords(
            parent=self.l_gripper_tool_frame,
            name='larm_end_coords')
        self.head_end_coords = CascadedCoords(
            pos=[0.08, 0.0, 0.13],
            parent=self.head_tilt_link,
            name='head_end_coords').rotate(np.pi / 2.0, 'y')
        self.torso_end_coords = CascadedCoords(
            parent=self.torso_lift_link,
            name='head_end_coords')

        # limbs
        self.torso = [self.torso_lift_link]
        self.torso_root_link = self.torso_lift_link
        self.larm_root_link = self.l_shoulder_pan_link
        self.rarm_root_link = self.r_shoulder_pan_link
        self.head_root_link = self.head_pan_link

        # custom min_angle and max_angle for joints
        joint_list = [
            self.torso_lift_joint, self.l_shoulder_pan_joint,
            self.l_shoulder_lift_joint, self.l_upper_arm_roll_joint,
            self.l_elbow_flex_joint, self.l_forearm_roll_joint,
            self.l_wrist_flex_joint, self.l_wrist_roll_joint,
            self.r_shoulder_pan_joint, self.r_shoulder_lift_joint,
            self.r_upper_arm_roll_joint, self.r_elbow_flex_joint,
            self.r_forearm_roll_joint, self.r_wrist_flex_joint,
            self.r_wrist_roll_joint, self.head_pan_joint, self.head_tilt_joint
        ]
        for j, min_angle, max_angle in zip(
                joint_list,
                (11.5, -32.3493, -20.2598, -37.2423, -121.542, -float('inf'),
                 -114.592, -float('inf'), -122.349,
                 -20.2598, -214.859, -121.542,
                 -float('inf'), -114.592, -float('inf'),
                 -163.694, -21.2682),
                (325.0, 122.349, 74.2725, 214.859,
                 -8.59437, float('inf'), -5.72958,
                 float('inf'), 32.3493, 74.2725,
                 37.2423, -8.59437, float('inf'),
                 -5.72958, float('inf'), 163.694, 74.2702)):
            j.min_angle = min_angle
            j.max_angle = max_angle

    @cached_property
    def rarm(self):
        rarm_links = [
            self.r_shoulder_pan_link, self.r_shoulder_lift_link,
            self.r_upper_arm_roll_link, self.r_elbow_flex_link,
            self.r_forearm_roll_link, self.r_wrist_flex_link,
            self.r_wrist_roll_link
        ]

        rarm_joints = []
        for link in rarm_links:
            rarm_joints.append(link.joint)
        r = RobotModel(link_list=rarm_links, joint_list=rarm_joints)
        r.end_coords = self.rarm_end_coords
        return r

    @cached_property
    def larm(self):
        larm_links = [
            self.l_shoulder_pan_link,
            self.l_shoulder_lift_link,
            self.l_upper_arm_roll_link,
            self.l_elbow_flex_link,
            self.l_forearm_roll_link,
            self.l_wrist_flex_link,
            self.l_wrist_roll_link,
        ]
        larm_joints = []
        for link in larm_links:
            larm_joints.append(link.joint)
        r = RobotModel(link_list=larm_links, joint_list=larm_joints)
        r.end_coords = self.larm_end_coords
        return r

    @cached_property
    def head(self):
        links = [
            self.head_pan_link,
            self.head_tilt_link]
        joints = []
        for link in links:
            joints.append(link.joint)
        r = RobotModel(link_list=links, joint_list=joints)
        r.end_coords = self.head_end_coords
        return r

    def reset_manip_pose(self):
        self.torso_lift_joint.joint_angle(300)
        self.l_shoulder_pan_joint.joint_angle(75)
        self.l_shoulder_lift_joint.joint_angle(50)
        self.l_upper_arm_roll_joint.joint_angle(110)
        self.l_elbow_flex_joint.joint_angle(-110)
        self.l_forearm_roll_joint.joint_angle(-20)
        self.l_wrist_flex_joint.joint_angle(-10)
        self.l_wrist_roll_joint.joint_angle(-10)
        self.r_shoulder_pan_joint.joint_angle(-75)
        self.r_shoulder_lift_joint.joint_angle(50)
        self.r_upper_arm_roll_joint.joint_angle(-110)
        self.r_elbow_flex_joint.joint_angle(-110)
        self.r_forearm_roll_joint.joint_angle(20)
        self.r_wrist_flex_joint.joint_angle(-10)
        self.r_wrist_roll_joint.joint_angle(-10)
        self.head_pan_joint.joint_angle(0)
        self.head_tilt_joint.joint_angle(50)
        return self.angle_vector()

    def reset_pose(self):
        self.torso_lift_joint.joint_angle(50)
        self.l_shoulder_pan_joint.joint_angle(60)
        self.l_shoulder_lift_joint.joint_angle(74)
        self.l_upper_arm_roll_joint.joint_angle(70)
        self.l_elbow_flex_joint.joint_angle(-120)
        self.l_forearm_roll_joint.joint_angle(20)
        self.l_wrist_flex_joint.joint_angle(-30)
        self.l_wrist_roll_joint.joint_angle(180)
        self.r_shoulder_pan_joint.joint_angle(-60)
        self.r_shoulder_lift_joint.joint_angle(74)
        self.r_upper_arm_roll_joint.joint_angle(-70)
        self.r_elbow_flex_joint.joint_angle(-120)
        self.r_forearm_roll_joint.joint_angle(-20)
        self.r_wrist_flex_joint.joint_angle(-30)
        self.r_wrist_roll_joint.joint_angle(180)
        self.head_pan_joint.joint_angle(0)
        self.head_tilt_joint.joint_angle(0)
        return self.angle_vector()
