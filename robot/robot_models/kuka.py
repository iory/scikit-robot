import os

from cached_property import cached_property
import numpy as np

import robot
from robot.robot_model import RobotModel
from robot.coordinates import CascadedCoords


class Kuka(RobotModel):
    """
    Kuka Robot Model
    """

    def __init__(self, urdf_path=None, *args, **kwargs):
        super(Kuka, self).__init__(*args, **kwargs)
        if urdf_path is None:
            urdf_path = os.path.join(os.path.dirname(robot.__file__),
                                     'models_robot/kuka_description/kuka.urdf')
        self.urdf_path = urdf_path
        self.load_urdf(urdf_path)

        self.rarm_end_coords = CascadedCoords(
            parent=self.lbr_iiwa_with_wsg50__lbr_iiwa_link_7,
            name='rarm_end_coords')
        self.rarm_end_coords.translate(
            np.array([0, 30, 250], dtype=np.float32))
        self.rarm_end_coords.rotate(- np.pi / 2.0, axis='y')
        self.rarm_end_coords.rotate(- np.pi / 2.0, axis='x')
        self.end_coords = [self.rarm_end_coords]

    def reset_manip_pose(self):
        return self.angle_vector([0, 10, 0, -90, 0, 90,
                                  0, 0, 0, 0, 0, 0])

    @cached_property
    def rarm(self):
        rarm_links = [self.lbr_iiwa_with_wsg50__lbr_iiwa_link_1,
                      self.lbr_iiwa_with_wsg50__lbr_iiwa_link_2,
                      self.lbr_iiwa_with_wsg50__lbr_iiwa_link_3,
                      self.lbr_iiwa_with_wsg50__lbr_iiwa_link_4,
                      self.lbr_iiwa_with_wsg50__lbr_iiwa_link_5,
                      self.lbr_iiwa_with_wsg50__lbr_iiwa_link_6,
                      self.lbr_iiwa_with_wsg50__lbr_iiwa_link_7]
        rarm_joints = []
        for link in rarm_links:
            if hasattr(link, 'joint'):
                rarm_joints.append(link.joint)
        r = RobotModel(link_list=rarm_links,
                       joint_list=rarm_joints)
        r.end_coords = self.rarm_end_coords
        return r

    def close_hand(self, av=None):
        if av is None:
            av = self.angle_vector()
        av[-2] = 0
        av[-4] = 0
        return self.angle_vector(av)

    def open_hand(self, default_angle=10, av=None):
        if av is None:
            av = self.angle_vector()
        av[-2] = default_angle
        av[-4] = -default_angle
        return self.angle_vector(av)
