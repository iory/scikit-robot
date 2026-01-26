import os.path as osp

import skrobot
from skrobot.coordinates import CascadedCoords
from skrobot.data import r8_6_urdfpath


class R8_6(skrobot.model.RobotModel):

    def __init__(self, urdf_path=None, *args, **kwargs):
        super(R8_6, self).__init__(*args, **kwargs)
        urdf_path = r8_6_urdfpath() if urdf_path is None else urdf_path
        if osp.exists(urdf_path):
            self.load_urdf_file(open(urdf_path, 'r'))
        else:
            raise ValueError()

        # Define end effector coordinates for left arm
        self.larm_end_coords = CascadedCoords(
            parent=self.l_hand_y_link,
            name='larm_end_coords'
        )

        # Define end effector coordinates for right arm
        self.rarm_end_coords = CascadedCoords(
            parent=self.r_hand_y_link,
            name='rarm_end_coords'
        )

        # Define camera coordinates for hands
        self.l_hand_camera_end_coords = CascadedCoords(
            parent=self.l_wrist_camera_link,
            name='l_hand_camera_end_coords')

        self.r_hand_camera_end_coords = CascadedCoords(
            parent=self.r_wrist_camera_link,
            name='r_hand_camera_end_coords')

        # Define elevator camera coordinates
        self.elv_camera_end_coords = CascadedCoords(
            parent=self.elv_camera_link,
            name='elv_camera_end_coords')

        # Define joint lists (arm only, without torso z-axis)
        self.larm_links = [
            self.l_shoulder_y_link,
            self.l_elbow_p1_link,
            self.l_elbow_connector_link,
            self.l_elbow_p2_link,
            self.l_elbow_tip_link,
            self.l_upper_arm_y_link,
            self.l_wrist_link,
            self.l_hand_y_link
        ]

        self.rarm_links = [
            self.r_shoulder_y_link,
            self.r_elbow_p1_link,
            self.r_elbow_connector_link,
            self.r_elbow_p2_link,
            self.r_elbow_tip_link,
            self.r_upper_arm_y_link,
            self.r_wrist_link,
            self.r_hand_y_link
        ]

        self.larm_joints = [
            self.l_shoulder_y_joint,
            self.l_elbow_p1_joint,
            self.l_elbow_p2_joint,
            self.l_upper_arm_y_joint,
            self.l_wrist_p_joint,
            self.l_wrist_r_joint
        ]

        self.rarm_joints = [
            self.r_shoulder_y_joint,
            self.r_elbow_p1_joint,
            self.r_elbow_p2_joint,
            self.r_upper_arm_y_joint,
            self.r_wrist_p_joint,
            self.r_wrist_r_joint
        ]

        # Define joint lists with torso (including z-axis)
        self.larm_with_torso_links = [
            self.l_zaxis_link,
        ] + self.larm_links

        self.rarm_with_torso_links = [
            self.r_zaxis_link,
        ] + self.rarm_links

        self.larm_with_torso_joints = [
            self.l_zaxis_joint,
        ] + self.larm_joints

        self.rarm_with_torso_joints = [
            self.r_zaxis_joint,
        ] + self.rarm_joints

        self.elv_joints = [
            self.elv_zaxis_joint,
            self.elv_storage_y_joint
        ]

        # Joint groups
        self.joint_list = (
            self.larm_with_torso_joints + self.rarm_with_torso_joints +
            self.elv_joints)

        # Cache for arm models
        self._larm = None
        self._rarm = None
        self._larm_with_torso = None
        self._rarm_with_torso = None

    @property
    def larm(self):
        """Left arm model (without torso z-axis)"""
        if self._larm is None:
            link_list = [self.l_zaxis_link] + self.larm_links
            joint_list = self.larm_joints
            self._larm = skrobot.model.RobotModel(link_list=link_list,
                                                   joint_list=joint_list)
            self._larm.name = self.name + '_larm'
            self._larm.end_coords = self.larm_end_coords

            # Setup inverse kinematics defaults
            self._larm.inverse_kinematics_defaults = {
                'link_list': self.larm_links,
                'move_target': self._larm.end_coords,
                'rotation_axis': True,
                'translation_axis': True,
            }
        return self._larm

    @property
    def rarm(self):
        """Right arm model (without torso z-axis)"""
        if self._rarm is None:
            link_list = [self.r_zaxis_link] + self.rarm_links
            joint_list = self.rarm_joints
            self._rarm = skrobot.model.RobotModel(link_list=link_list,
                                                   joint_list=joint_list)
            self._rarm.name = self.name + '_rarm'
            self._rarm.end_coords = self.rarm_end_coords

            # Setup inverse kinematics defaults
            self._rarm.inverse_kinematics_defaults = {
                'link_list': self.rarm_links,
                'move_target': self._rarm.end_coords,
                'rotation_axis': True,
                'translation_axis': True,
            }
        return self._rarm

    @property
    def larm_with_torso(self):
        """Left arm model with torso z-axis"""
        if self._larm_with_torso is None:
            link_list = [self.l_arm_link] + self.larm_with_torso_links
            joint_list = self.larm_with_torso_joints
            self._larm_with_torso = skrobot.model.RobotModel(
                link_list=link_list,
                joint_list=joint_list)
            self._larm_with_torso.name = self.name + '_larm_with_torso'
            self._larm_with_torso.end_coords = self.larm_end_coords

            # Setup inverse kinematics defaults
            self._larm_with_torso.inverse_kinematics_defaults = {
                'link_list': self.larm_with_torso_links,
                'move_target': self._larm_with_torso.end_coords,
                'rotation_axis': True,
                'translation_axis': True,
            }
        return self._larm_with_torso

    @property
    def rarm_with_torso(self):
        """Right arm model with torso z-axis"""
        if self._rarm_with_torso is None:
            link_list = [self.r_arm_link] + self.rarm_with_torso_links
            joint_list = self.rarm_with_torso_joints
            self._rarm_with_torso = skrobot.model.RobotModel(
                link_list=link_list,
                joint_list=joint_list)
            self._rarm_with_torso.name = self.name + '_rarm_with_torso'
            self._rarm_with_torso.end_coords = self.rarm_end_coords

            # Setup inverse kinematics defaults
            self._rarm_with_torso.inverse_kinematics_defaults = {
                'link_list': self.rarm_with_torso_links,
                'move_target': self._rarm_with_torso.end_coords,
                'rotation_axis': True,
                'translation_axis': True,
            }
        return self._rarm_with_torso

    def reset_pose(self):
        """Reset robot to initial pose"""
        # Left arm initial pose
        self.l_zaxis_joint.joint_angle(0.7)  # Middle of range
        self.l_shoulder_y_joint.joint_angle(0.0)
        self.l_elbow_p1_joint.joint_angle(0.785)  # 45 degrees
        self.l_elbow_p2_joint.joint_angle(0.785)  # 45 degrees
        self.l_upper_arm_y_joint.joint_angle(0.0)
        self.l_wrist_p_joint.joint_angle(0.0)
        self.l_wrist_r_joint.joint_angle(0.0)

        # Right arm initial pose
        self.r_zaxis_joint.joint_angle(0.7)  # Middle of range
        self.r_shoulder_y_joint.joint_angle(0.0)
        self.r_elbow_p1_joint.joint_angle(0.785)  # 45 degrees
        self.r_elbow_p2_joint.joint_angle(0.785)  # 45 degrees
        self.r_upper_arm_y_joint.joint_angle(0.0)
        self.r_wrist_p_joint.joint_angle(0.0)
        self.r_wrist_r_joint.joint_angle(0.0)

        # Elevator initial pose
        self.elv_zaxis_joint.joint_angle(0.2)  # Middle of range
        self.elv_storage_y_joint.joint_angle(0.0)

        return self.angle_vector()

    @property
    def elv(self):
        """Return elevator as a sub-robot model"""
        link_list = [self.elv_link, self.elv_zaxis_link, self.elv_storage_link]
        joint_list = self.elv_joints
        elv_model = skrobot.model.RobotModel(link_list=link_list,
                                              joint_list=joint_list)
        elv_model.name = self.name + '_elv'
        elv_model.end_coords = self.elv_attach_obj_link
        return elv_model

    # New naming convention aliases (backward compatible)
    @property
    def right_arm(self):
        return self.rarm

    @property
    def left_arm(self):
        return self.larm

    @property
    def right_arm_with_torso(self):
        return self.rarm_with_torso

    @property
    def left_arm_with_torso(self):
        return self.larm_with_torso

    @property
    def right_arm_end_coords(self):
        return self.rarm_end_coords

    @property
    def left_arm_end_coords(self):
        return self.larm_end_coords
