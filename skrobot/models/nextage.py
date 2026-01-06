from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.data import nextage_urdfpath
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF


class Nextage(RobotModelFromURDF):
    """
    - Nextage Open Official Information.

      https://nextage.kawadarobot.co.jp/open

    - Nextage Open Robot Description

      https://github.com/tork-a/rtmros_nextage/tree/indigo-devel/nextage_description/urdf
    """

    def __init__(self, *args, **kwargs):
        super(Nextage, self).__init__(*args, **kwargs)

        # End effector coordinates
        self.rarm_end_coords = CascadedCoords(
            pos=[-0.185, 0.0, -0.01],
            parent=self.RARM_JOINT5_Link,
            name='rarm_end_coords')
        self.rarm_end_coords.rotate(-np.pi / 2.0, 'y')

        self.larm_end_coords = CascadedCoords(
            pos=[-0.185, 0.0, -0.01],
            parent=self.LARM_JOINT5_Link,
            name='larm_end_coords')
        self.larm_end_coords.rotate(-np.pi / 2.0, 'y')

        self.head_end_coords = CascadedCoords(
            pos=[0.06, 0, 0.025],
            parent=self.HEAD_JOINT1_Link,
            name='head_end_coords')
        self.head_end_coords.rotate(np.deg2rad(90), 'y')

        self.reset_pose()

    @cached_property
    def default_urdf_path(self):
        return nextage_urdfpath()

    def reset_pose(self):
        angle_vector = [
            0.0,
            0.0,
            0.0,
            np.deg2rad(0.6),
            0.0,
            np.deg2rad(-100),
            np.deg2rad(-15.2),
            np.deg2rad(9.4),
            np.deg2rad(-3.2),
            np.deg2rad(-0.6),
            0.0,
            np.deg2rad(-100),
            np.deg2rad(15.2),
            np.deg2rad(9.4),
            np.deg2rad(3.2),
        ]
        self.angle_vector(angle_vector)
        return self.angle_vector()

    def reset_manip_pose(self):
        """Reset robot to manipulation pose (same as reset_pose for Nextage)"""
        return self.reset_pose()

    @cached_property
    def rarm(self):
        link_names = ["RARM_JOINT{}_Link".format(i) for i in range(6)]
        links = [getattr(self, n) for n in link_names]
        joints = [l.joint for l in links]
        model = RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.rarm_end_coords
        return model

    @cached_property
    def larm(self):
        link_names = ["LARM_JOINT{}_Link".format(i) for i in range(6)]
        links = [getattr(self, n) for n in link_names]
        joints = [l.joint for l in links]
        model = RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.larm_end_coords
        return model

    @cached_property
    def head(self):
        link_names = ["HEAD_JOINT{}_Link".format(i) for i in range(2)]
        links = [getattr(self, n) for n in link_names]
        joints = [l.joint for l in links]
        model = RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.head_end_coords
        return model

    @cached_property
    def torso(self):
        link_names = ["CHEST_JOINT0_Link"]
        links = [getattr(self, n) for n in link_names]
        joints = [l.joint for l in links]
        model = RobotModel(link_list=links, joint_list=joints)
        return model

    # New naming convention aliases (backward compatible)
    @property
    def right_arm(self):
        return self.rarm

    @property
    def left_arm(self):
        return self.larm

    @property
    def right_arm_end_coords(self):
        return self.rarm_end_coords

    @property
    def left_arm_end_coords(self):
        return self.larm_end_coords
