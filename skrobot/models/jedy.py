from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.data import jedy_urdfpath
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF


class Jedy(RobotModelFromURDF):
    """Jedy dual-arm mobile robot model.

    The robot has two 7-DOF arms (``{r,l}arm_joint0`` .. ``joint6``) each
    terminated by a parallel gripper, a 2-DOF head (``head_joint0`` pan /
    ``head_joint1`` tilt) and a 4-wheel mobile base.

    An Intel RealSense D405 is mounted on ``head_link1``. ``head_end_coords``
    is attached to ``camera_depth_optical_frame``, whose origin RealSense
    colocates with the D405's **left** (infra1) imager, so the head end
    coordinates coincide with the left camera. It follows the optical-frame
    convention, so its z-axis points along the camera's viewing direction.
    """

    def __init__(self, *args, **kwargs):
        super(Jedy, self).__init__(*args, **kwargs)

        self.rarm_end_coords = CascadedCoords(
            parent=self.rarm_link6,
            pos=[0.0, 0.0, -0.102466],
            name='rarm_end_coords')
        self.rarm_end_coords.rotate(np.pi / 2.0, 'y')
        self.larm_end_coords = CascadedCoords(
            parent=self.larm_link6,
            pos=[0.0, 0.0, -0.102466],
            name='larm_end_coords')
        self.larm_end_coords.rotate(np.pi / 2.0, 'y')

        self.head_end_coords = CascadedCoords(
            parent=self.camera_depth_optical_frame,
            name='head_end_coords')

        self.reset_pose()

    @cached_property
    def default_urdf_path(self):
        return jedy_urdfpath()

    def reset_pose(self):
        self.rarm_joint0.joint_angle(np.deg2rad(30))
        self.rarm_joint1.joint_angle(np.deg2rad(0))
        self.rarm_joint2.joint_angle(np.deg2rad(0))
        self.rarm_joint3.joint_angle(np.deg2rad(-120))
        self.rarm_joint4.joint_angle(np.deg2rad(0))
        self.rarm_joint5.joint_angle(np.deg2rad(0))
        self.rarm_joint6.joint_angle(np.deg2rad(0))

        self.larm_joint0.joint_angle(np.deg2rad(-30))
        self.larm_joint1.joint_angle(np.deg2rad(0))
        self.larm_joint2.joint_angle(np.deg2rad(0))
        self.larm_joint3.joint_angle(np.deg2rad(-120))
        self.larm_joint4.joint_angle(np.deg2rad(0))
        self.larm_joint5.joint_angle(np.deg2rad(0))
        self.larm_joint6.joint_angle(np.deg2rad(0))

        self.head_joint0.joint_angle(np.deg2rad(0))
        self.head_joint1.joint_angle(np.deg2rad(0))
        return self.angle_vector()

    @cached_property
    def rarm(self):
        links = [getattr(self, 'rarm_link{}'.format(i)) for i in range(7)]
        joints = [link.joint for link in links]
        r = RobotModel(link_list=links, joint_list=joints)
        r.end_coords = self.rarm_end_coords
        return r

    @cached_property
    def larm(self):
        links = [getattr(self, 'larm_link{}'.format(i)) for i in range(7)]
        joints = [link.joint for link in links]
        r = RobotModel(link_list=links, joint_list=joints)
        r.end_coords = self.larm_end_coords
        return r

    @cached_property
    def head(self):
        links = [self.head_link0, self.head_link1]
        joints = [link.joint for link in links]
        r = RobotModel(link_list=links, joint_list=joints)
        r.end_coords = self.head_end_coords
        return r
