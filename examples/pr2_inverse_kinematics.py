#!/usr/bin/env python

import argparse
import time

import numpy as np

import skrobot
from skrobot.utils.visualization import ik_visualization


def main():
    parser = argparse.ArgumentParser(
        description='Simple PR2 inverse kinematics example.')
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help="Run in non-interactive mode (do not wait for user input)"
    )
    parser.add_argument(
        '--viewer', type=str,
        choices=['trimesh', 'pyrender'], default='trimesh',
        help='Choose the viewer type: trimesh or pyrender')
    parser.add_argument(
        '--no-ik-visualization',
        action='store_true',
        help="Disable inverse kinematics visualization during solving"
    )
    args = parser.parse_args()

    # Create robot model
    robot_model = skrobot.models.PR2()
    robot_model.reset_pose()

    # Define joint list for right arm
    link_list = [
        robot_model.r_shoulder_pan_link,
        robot_model.r_shoulder_lift_link,
        robot_model.r_upper_arm_roll_link,
        robot_model.r_elbow_flex_link,
        robot_model.r_forearm_roll_link,
        robot_model.r_wrist_flex_link,
        robot_model.r_wrist_roll_link
    ]

    # Create viewer
    if args.viewer == 'trimesh':
        viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    elif args.viewer == 'pyrender':
        viewer = skrobot.viewers.PyrenderViewer(resolution=(640, 480))

    viewer.add(robot_model)
    viewer.show()
    viewer.set_camera([np.deg2rad(45), -np.deg2rad(0),
                       np.deg2rad(135)], distance=2.5)

    # Define target position
    target_pos = [0.7, -0.2, 0.8]
    print("Solving inverse kinematics for target: {}".format(target_pos))

    # Create target coordinates
    target_coords = skrobot.coordinates.Coordinates(target_pos, [0, 0, 0])

    # Solve inverse kinematics with optional visualization
    if not args.no_ik_visualization:
        with ik_visualization(viewer, sleep_time=0.5):
            result = robot_model.inverse_kinematics(
                target_coords,
                link_list=link_list,
                move_target=robot_model.rarm_end_coords,
                rotation_axis=True
            )
    else:
        result = robot_model.inverse_kinematics(
            target_coords,
            link_list=link_list,
            move_target=robot_model.rarm_end_coords,
            rotation_axis=True
        )

    if result is False:
        print("Failed to reach target")
    else:
        print("Successfully reached target!")

    viewer.redraw()
    print("IK solving completed!")

    if not args.no_interactive:
        print('==> Press [q] to close window')
        while not viewer.has_exit:
            time.sleep(0.1)
            viewer.redraw()


if __name__ == '__main__':
    main()
