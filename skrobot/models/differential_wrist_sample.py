from cached_property import cached_property
import numpy as np
import yaml

from skrobot.coordinates import CascadedCoords
from skrobot.data import differential_wrist_sample_joint_limit_table_path
from skrobot.data import differential_wrist_sample_urdfpath
from skrobot.model import create_joint_limit_table
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF


class DifferentialWristSample(RobotModelFromURDF):
    """Differential Wrist Sample Robot Model.

    A sample robot arm with a differential wrist mechanism where
    WRIST_JOINT_Y and WRIST_JOINT_R have coupled joint limits.

    Parameters
    ----------
    use_joint_limit_table : bool
        If True (default), apply joint limit tables for the wrist joints.
        This enables dynamic joint limits where each wrist joint's limits
        depend on the other wrist joint's current angle.
    """

    def __init__(self, use_joint_limit_table=True):
        super(DifferentialWristSample, self).__init__()

        self.end_coords = CascadedCoords(
            parent=self.end_effector_link,
            name='end_coords')

        # limbs
        self.arm_root_link = self.ARM_LINK0

        # Apply joint limit tables for coupled wrist constraints
        if use_joint_limit_table:
            self._apply_joint_limit_tables()

    def _apply_joint_limit_tables(self):
        """Apply joint limit tables from YAML configuration."""
        yaml_path = differential_wrist_sample_joint_limit_table_path()
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        for table_config in config['joint_limit_tables']:
            create_joint_limit_table(
                self,
                table_config['target_joint'],
                table_config['dependent_joint'],
                table_config['target_min_angle'],
                table_config['target_max_angle'],
                table_config['min_angles'],
                table_config['max_angles'],
            )

    @cached_property
    def default_urdf_path(self):
        return differential_wrist_sample_urdfpath()

    @cached_property
    def arm(self):
        """Get arm link list for IK."""
        arm_links = [
            self.ARM_LINK0, self.ARM_LINK1, self.ARM_LINK2,
            self.ARM_LINK3, self.ARM_LINK4,
            self.WRIST_GEAR, self.WRIST_END
        ]
        arm_joints = [link.joint for link in arm_links if link.joint]
        r = RobotModel(link_list=arm_links, joint_list=arm_joints)
        r.end_coords = self.end_coords
        return r

    @cached_property
    def wrist(self):
        """Get wrist link list."""
        wrist_links = [self.WRIST_GEAR, self.WRIST_END]
        wrist_joints = [link.joint for link in wrist_links if link.joint]
        r = RobotModel(link_list=wrist_links, joint_list=wrist_joints)
        r.end_coords = self.end_coords
        return r

    def reset_pose(self):
        """Reset robot to default pose."""
        self.ARM_JOINT0.joint_angle(0)
        self.ARM_JOINT1.joint_angle(np.deg2rad(30))
        self.ARM_JOINT2.joint_angle(0)
        self.ARM_JOINT3.joint_angle(np.deg2rad(60))
        self.ARM_JOINT4.joint_angle(0)
        self.WRIST_JOINT_Y.joint_angle(0)
        self.WRIST_JOINT_R.joint_angle(0)
        return self.angle_vector()

    def reset_manip_pose(self):
        """Reset robot to manipulation pose."""
        self.ARM_JOINT0.joint_angle(0)
        self.ARM_JOINT1.joint_angle(np.deg2rad(45))
        self.ARM_JOINT2.joint_angle(0)
        self.ARM_JOINT3.joint_angle(np.deg2rad(90))
        self.ARM_JOINT4.joint_angle(0)
        self.WRIST_JOINT_Y.joint_angle(0)
        self.WRIST_JOINT_R.joint_angle(0)
        return self.angle_vector()
