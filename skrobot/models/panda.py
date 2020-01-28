from cached_property import cached_property

from ..data import panda_urdfpath
from ..model import RobotModel
from .urdf import RobotModelFromURDF


class Panda(RobotModelFromURDF):

    """Panda Robot Model.

    https://frankaemika.github.io/docs/control_parameters.html
    """

    def __init__(self, *args, **kwargs):
        super(Panda, self).__init__(*args, **kwargs)
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

    @cached_property
    def rarm(self):
        link_names = ['panda_link{}'.format(i) for i in range(1, 8)]
        links = [getattr(self, n) for n in link_names]
        joints = [l.joint for l in links]
        model = RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.panda_hand
        return model
