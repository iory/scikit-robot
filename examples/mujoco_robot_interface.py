#!/usr/bin/env python
"""Pick a cube with Jedy, simulated in MuJoCo.

``robot`` is the model you plan with; ``ri`` sends its posture to the
simulated robot and reads back what the robot did, as it would on the real
Jedy. Needs ``pip install scikit-robot[mujoco] coacd``.
"""

import argparse
import importlib.util
import sys

import numpy as np

import skrobot
from skrobot.coordinates import Coordinates
from skrobot.interfaces import MujocoRobotInterface
from skrobot.model import Box


TABLE_HEIGHT = 0.065
CUBE = 0.035  # fits between the closed fingers' bars (about 3.8 cm apart)
# Wheel joint -> (radius [m], roller angle [rad]), so the mecanum wheels
# turn while the base drives.
WHEELS = {
    'front_left_wheel_joint': (0.03, -np.pi / 4),
    'front_right_wheel_joint': (0.03, np.pi / 4),
    'rear_left_wheel_joint': (0.03, np.pi / 4),
    'rear_right_wheel_joint': (0.03, -np.pi / 4),
}
# MuJoCo collides meshes by their convex hull, which would fill the space
# between the fingers, so these links are split into convex parts (CoACD).
FINGERS = ['rarm_link7', 'rarm_link8', 'larm_link7', 'larm_link8']


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--no-interactive', action='store_true',
                        help='run without opening the MuJoCo viewer')
    args = parser.parse_args()
    if importlib.util.find_spec('mujoco') is None:
        print('mujoco is not installed: pip install scikit-robot[mujoco]')
        sys.exit(0)

    robot = skrobot.models.Jedy(light=True)
    robot.reset_pose()
    table = Box((0.2, 0.4, TABLE_HEIGHT), face_colors=(160, 110, 60),
                name='table')
    table.translate((0.55, 0.0, TABLE_HEIGHT / 2))
    cube = Box((CUBE, CUBE, CUBE), face_colors=(220, 30, 30), name='cube')
    cube.translate((0.47, -0.08, TABLE_HEIGHT + CUBE / 2))

    ri = MujocoRobotInterface(
        robot, planar_base=True, wheels=WHEELS,
        base_max_velocity=(0.15, 1.0),
        objects=[table], movable_objects=[cube],
        converter_kwargs={'convex_decompose_links': FINGERS,
                          'coacd_quality': 'fine'},
        viewer=not args.no_interactive)

    def move_arm(coords, time=1.5):
        if robot.inverse_kinematics(coords, link_list=robot.rarm.link_list,
                                    move_target=robot.rarm_end_coords,
                                    stop=300) is False:
            raise RuntimeError('IK found no solution')
        ri.angle_vector(robot.angle_vector(), time)
        ri.wait_interpolation()

    def gripper(angle, time=1.0):
        # 0 closes the hand, negative opens it; both fingers move.
        robot.rarm_gripper_joint.joint_angle(angle)
        robot.rarm_mimic_gripper_joint.joint_angle(-angle)
        ri.angle_vector(robot.angle_vector(), time)
        ri.wait_interpolation()

    side = Coordinates()  # hand pointing forward, fingers closing sideways
    gripper(-1.0)
    move_arm(side.copy_worldcoords().translate((0.12, -0.08, 0.10), 'world'))

    ri.go_pos(0.30, 0.0, 0.0)
    # ri drove the simulated robot only; move the model to where it stands.
    robot.newcoords(ri.odom)

    # Where the cube is, from ri. The hand's tip goes a little past its
    # centre and above it, so the fingers' bars hold it clear of the table.
    p = ri.object_coords('cube').worldpos()
    grasp = p + [0.0125, 0.0, 0.008]
    move_arm(side.copy_worldcoords().translate(grasp + [-0.05, 0, 0], 'world'))
    move_arm(side.copy_worldcoords().translate(grasp, 'world'))
    gripper(0.3)
    move_arm(side.copy_worldcoords().translate(grasp + [0, 0, 0.04], 'world'))

    rise = ri.object_coords('cube').worldpos()[2] - p[2]
    print('lifted by {:.3f} m: {}'.format(
        rise, 'success' if rise > 0.02 else 'failed'))


if __name__ == '__main__':
    main()
