from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.data import panda_urdfpath
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF


class Panda(RobotModelFromURDF):

    """Panda Robot Model.

    https://frankaemika.github.io/docs/control_parameters.html
    """

    def __init__(self, *args, **kwargs):
        super(Panda, self).__init__(*args, **kwargs)

        # End effector coordinate frame
        # Based on franka_ros configuration:
        # https://github.com/frankaemika/franka_ros/blob/0.10.1/franka_description/robots/common/franka_robot.xacro#L8
        self.rarm_end_coords = CascadedCoords(
            pos=[0, 0, 0.1034],
            parent=self.panda_hand,
            name='rarm_end_coords')
        self.rarm_end_coords.rotate(np.deg2rad(-90), 'y')
        self.reset_pose()

    @cached_property
    def default_urdf_path(self):
        return panda_urdfpath()

    def reset_pose(self):
        angle_vector = [
            0.03942226991057396,
            -0.9558116793632507,
            -0.014800949953496456,
            -2.130282163619995,
            -0.013104429468512535,
            1.1745525598526,
            0.8112226724624634,
        ]
        for link, angle in zip(self.rarm.link_list, angle_vector):
            link.joint.joint_angle(angle)
        return self.angle_vector()

    def reset_manip_pose(self):
        """Reset robot to manipulation pose (same as reset_pose for Panda)"""
        return self.reset_pose()

    @cached_property
    def rarm(self):
        link_names = ['panda_link{}'.format(i) for i in range(1, 8)]
        links = [getattr(self, n) for n in link_names]
        joints = [l.joint for l in links]
        model = RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.rarm_end_coords
        return model

    # New naming convention aliases (backward compatible)
    @property
    def arm(self):
        return self.rarm

    @property
    def arm_end_coords(self):
        return self.rarm_end_coords
