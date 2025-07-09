#!/usr/bin/env python

import argparse
import time

import numpy as np

import skrobot
from skrobot.coordinates.base import Coordinates
from skrobot.coordinates.geo import midcoords
from skrobot.model.primitives import Box


def init_pose():
    robot_model.reset_pose()

    target_coords = skrobot.coordinates.Coordinates(
        [0.5, -0.3, 0.7], [0, 0, 0])
    robot_model.inverse_kinematics(
        target_coords,
        link_list=link_list,
        move_target=move_target)

    open_right_gripper()


def open_right_gripper():
    robot_model.gripper_distance(0.093, arm='rarm')
    viewer.redraw()


def close_right_gripper():
    robot_model.gripper_distance(0.0, arm='rarm')
    viewer.redraw()


def open_left_gripper():
    robot_model.gripper_distance(0.093, arm='larm')
    viewer.redraw()


def close_left_gripper():
    robot_model.gripper_distance(0.0, arm='larm')
    viewer.redraw()


def add_box(box_center):
    box = Box(extents=[0.1, 0.1, 0.2], with_sdf=True)
    box.translate(box_center)
    viewer.add(box)
    return box


def move_to_box(box):
    robot_model.look_at(box)
    start_coords = move_target.copy_worldcoords()
    target_coords = box.copy_worldcoords()

    for i in range(20):
        robot_model.inverse_kinematics(
            midcoords(i / 20.0, start_coords, target_coords),
            link_list=link_list,
            move_target=move_target)
        viewer.redraw()
        time.sleep(0.1)


def grasp_box(box):
    robot_model.gripper_distance(0.036, arm='rarm')
    move_target.assoc(box)
    viewer.redraw()


def pull_box(box):
    target_coords = Coordinates([0.5, -0.3, 0.7])
    start_coords = box

    for i in range(20):
        robot_model.inverse_kinematics(
            midcoords(i / 20.0, start_coords, target_coords),
            link_list=link_list,
            move_target=move_target)
        robot_model.look_at(box)
        viewer.redraw()
        time.sleep(0.1)


parser = argparse.ArgumentParser(
    description='Scikit-robot grasp and pull box example.')
parser.add_argument(
    '--no-interactive',
    action='store_true',
    help="Run in non-interactive mode (do not wait for user input)"
)
parser.add_argument(
    '--viewer', type=str,
    choices=['trimesh', 'pyrender'], default='trimesh',
    help='Choose the viewer type: trimesh or pyrender')
args = parser.parse_args()


# Create robot model
robot_model = skrobot.models.PR2()
link_list = [
    robot_model.r_shoulder_pan_link,
    robot_model.r_shoulder_lift_link,
    robot_model.r_upper_arm_roll_link,
    robot_model.r_elbow_flex_link,
    robot_model.r_forearm_roll_link,
    robot_model.r_wrist_flex_link,
    robot_model.r_wrist_roll_link]
rarm_end_coords = skrobot.coordinates.CascadedCoords(
    parent=robot_model.r_gripper_tool_frame,
    name='rarm_end_coords')
move_target = rarm_end_coords

# Create viewer
if args.viewer == 'trimesh':
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480),
                                                update_interval=0.1)
elif args.viewer == 'pyrender':
    viewer = skrobot.viewers.PyrenderViewer(resolution=(640, 480),
                                            update_interval=0.1)
viewer.add(robot_model)
viewer.show()
viewer.set_camera([np.deg2rad(45), -np.deg2rad(0),
                  np.deg2rad(135)], distance=2.5)

# Move robot
init_pose()
box_center = np.array([0.9, -0.2, 0.9])
box = add_box(box_center)
move_to_box(box)
grasp_box(box)
pull_box(box)

if not args.no_interactive:
    print('==> Press [q] to close window')
    while not viewer.has_exit:
        time.sleep(0.1)
        viewer.redraw()
viewer.close()
time.sleep(1.0)
