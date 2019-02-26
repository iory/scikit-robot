import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.robot_models import Kuka
from skrobot.pybullet_robot_interface import PybulletRobotInterface
from skrobot.pybullet_robot_interface import draw


if __name__ == '__main__':
    r = Kuka()
    pri = PybulletRobotInterface(r)
    r.reset_manip_pose()
    pri.angle_vector(r.angle_vector())

    target_coords = Coordinates(pos=[0.5, 0, 0]).\
        rotate(np.pi / 2.0, 'y', 'local')
    draw(target_coords)
    r.inverse_kinematics(target_coords,
                         link_list=r.rarm.link_list,
                         move_target=r.rarm_end_coords,
                         rotation_axis=True,
                         stop=1000,
                         inverse_kinematics_hook=[
                             lambda: pri.angle_vector(r.angle_vector())])
    pri.angle_vector(r.angle_vector())
