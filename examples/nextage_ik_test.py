#!/usr/bin/env python
import argparse
import time

import numpy as np
import rospy

import skrobot
from skrobot.interfaces.ros import NextageROSRobotInterface
from skrobot.utils.visualization import ik_visualization


def main():
    parser = argparse.ArgumentParser(description='Nextage IK test with Gazebo')
    parser.add_argument('--no-interactive', action='store_true')
    parser.add_argument('--viewer', type=str, choices=['trimesh', 'pyrender'], default='trimesh')
    parser.add_argument('--no-ik-visualization', action='store_true')
    parser.add_argument('--no-gazebo', action='store_true', help='Visualization only, no Gazebo')
    args = parser.parse_args()

    robot_model = skrobot.models.Nextage()
    robot_model.reset_manip_pose()

    if not args.no_gazebo:
        rospy.init_node('nextage_ik_test')
        ri = NextageROSRobotInterface(robot_model)
        ri.angle_vector(robot_model.angle_vector(), time=3.0)
        ri.wait_interpolation()

    if args.viewer == 'trimesh':
        viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    else:
        viewer = skrobot.viewers.PyrenderViewer(resolution=(640, 480))

    viewer.add(robot_model)
    viewer.show()
    viewer.set_camera([np.deg2rad(45), -np.deg2rad(0), np.deg2rad(135)], distance=1.5)

    target_pos = [0.35, -0.15, -0.1]
    print("Solving IK for rarm target: {}".format(target_pos))

    target_coords = skrobot.coordinates.Coordinates(target_pos, [0, 0, 0])
    target_axis = skrobot.model.Axis(axis_radius=0.01, axis_length=0.15,
                                     pos=target_coords.translation,
                                     rot=target_coords.rotation)
    viewer.add(target_axis)
    viewer.redraw()

    if not args.no_ik_visualization:
        with ik_visualization(viewer, sleep_time=0.5):
            result = robot_model.rarm.inverse_kinematics(target_coords, rotation_axis=False)
    else:
        result = robot_model.rarm.inverse_kinematics(target_coords, rotation_axis=False)

    success = result is not False and result is not None
    if success:
        print("IK succeeded!")
        if not args.no_gazebo:
            ri.angle_vector(robot_model.angle_vector(), time=5.0)
            ri.wait_interpolation()
    else:
        print("IK failed")

    viewer.redraw()

    target_pos = [0.35, 0.15, -0.1]
    print("Solving IK for larm target: {}".format(target_pos))

    target_coords = skrobot.coordinates.Coordinates(target_pos, [0, 0, 0])
    target_axis = skrobot.model.Axis(axis_radius=0.01, axis_length=0.15,
                                     pos=target_coords.translation,
                                     rot=target_coords.rotation)
    target_axis.set_color([0, 255, 0])
    viewer.add(target_axis)
    viewer.redraw()

    if not args.no_ik_visualization:
        with ik_visualization(viewer, sleep_time=0.5):
            result = robot_model.larm.inverse_kinematics(target_coords, rotation_axis=False)
    else:
        result = robot_model.larm.inverse_kinematics(target_coords, rotation_axis=False)

    success = result is not False and result is not None
    if success:
        print("IK succeeded!")
        if not args.no_gazebo:
            ri.angle_vector(robot_model.angle_vector(), time=5.0)
            ri.wait_interpolation()
    else:
        print("IK failed")

    viewer.redraw()

    robot_model.reset_manip_pose()
    if not args.no_gazebo:
        ri.angle_vector(robot_model.angle_vector(), time=3.0)
        ri.wait_interpolation()

    viewer.redraw()

    if not args.no_interactive:
        print('==> Press [q] to close window')
        while viewer.is_active:
            time.sleep(0.1)
            viewer.redraw()

    viewer.close()
    time.sleep(1.0)


if __name__ == '__main__':
    main()
