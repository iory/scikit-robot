from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.data import rover_armed_tycoon_urdfpath
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF


class RoverArmedTycoon(RobotModelFromURDF):
    """Rover Armed Tycoon Robot Model."""

    def __init__(self, *args, **kwargs):
        super(RoverArmedTycoon, self).__init__(*args, **kwargs)

        self._arm_end_coords = CascadedCoords(
            parent=self.tycoon_7_roll_link,
            pos=[-0.05, 0, 0],
            name='arm_end_coords')
        self._arm_end_coords.rotate(np.pi, 'z')

    @cached_property
    def default_urdf_path(self):
        return rover_armed_tycoon_urdfpath()

    @cached_property
    def arm(self):
        """Arm kinematic chain."""
        links = [
            self.tycoon_0_pitch_link,
            self.tycoon_0_roll_link,
            self.tycoon_1_pitch_link,
            self.tycoon_1_roll_link,
            self.tycoon_2_pitch_link,
            self.tycoon_2_roll_link,
            self.tycoon_3_pitch_link,
            self.tycoon_3_roll_link,
            self.tycoon_4_pitch_link,
            self.tycoon_4_roll_link,
            self.tycoon_5_pitch_link,
            self.tycoon_5_roll_link,
            self.tycoon_6_pitch_link,
            self.tycoon_6_roll_link,
            self.tycoon_7_pitch_link,
            self.tycoon_7_roll_link,
        ]
        joints = [link.joint for link in links]
        model = RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self._arm_end_coords
        return model

    @property
    def arm_end_coords(self):
        """End coordinates for arm."""
        return self._arm_end_coords

    def reset_pose(self):
        self.init_pose()
        self.tycoon_3_pitch_joint.joint_angle(0.5)
        self.tycoon_4_pitch_joint.joint_angle(0.5)
        self.tycoon_5_pitch_joint.joint_angle(-0.5)
        self.tycoon_6_pitch_joint.joint_angle(-0.5)


if __name__ == '__main__':
    from skrobot.models import Axis
    from skrobot.models import RoverArmedTycoon
    from skrobot.viewers import PyrenderViewer

    robot = RoverArmedTycoon()
    robot.reset_pose()

    viewer = PyrenderViewer()
    viewer.add(robot)

    axis_arm = Axis.from_coords(robot.arm_end_coords, axis_radius=0.01, axis_length=0.1)
    viewer.add(axis_arm)

    viewer.show()

    while viewer.is_active:
        viewer.redraw()
