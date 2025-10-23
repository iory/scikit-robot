from cached_property import cached_property

from ..data import nextage_urdfpath
from ..model import RobotModel
from .urdf import RobotModelFromURDF


class Nextage(RobotModelFromURDF):

    """
    - Nextage Open Official Information.

      https://nextage.kawadarobot.co.jp/open

    - Nextage Open Robot Description

      https://github.com/tork-a/rtmros_nextage/tree/indigo-devel/nextage_description/urdf
    """

    def __init__(self, *args, **kwargs):
        super(Nextage, self).__init__(*args, **kwargs)
        self.reset_pose()

    @cached_property
    def default_urdf_path(self):
        return nextage_urdfpath()

    def reset_pose(self):
        import numpy as np
        angle_vector = [
            0.0, 
            -0.5, 0.0, np.deg2rad(-100), np.deg2rad(15.2), np.deg2rad(9.4), np.deg2rad(3.2),
            0.5, 0.0, np.deg2rad(-100), np.deg2rad(-15.2), np.deg2rad(9.4), np.deg2rad(-3.2),
            0.0, 0.0
        ]
        self.angle_vector(angle_vector)
        return self.angle_vector()

    def reset_manip_pose(self):
        """Reset robot to manipulation pose (same as reset_pose for Nextage)"""
        return self.reset_pose()

    @cached_property
    def rarm(self):
        link_names = ['RARM_JOINT{}_Link'.format(i) for i in range(6)]
        links = [getattr(self, n) for n in link_names]
        joints = [l.joint for l in links]
        model = RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.rarm_end_coords
        return model
    
    @cached_property
    def larm(self):
        link_names = ['LARM_JOINT{}_Link'.format(i) for i in range(6)]
        links = [getattr(self, n) for n in link_names]
        joints = [l.joint for l in links]
        model = RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.larm_end_coords
        return model
    
    @cached_property
    def head(self):
        link_names = ['HEAD_JOINT{}_Link'.format(i) for i in range(2)]
        links = [getattr(self, n) for n in link_names]
        joints = [l.joint for l in links]
        model = RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.head_end_coords
        return model
    
    @cached_property
    def rarm_end_coords(self):
        return self.RARM_JOINT5_Link
    
    @cached_property
    def larm_end_coords(self):
        return self.LARM_JOINT5_Link
    
    @cached_property
    def head_end_coords(self):
        return self.HEAD_JOINT1_Link
