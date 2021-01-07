#!/usr/bin/env python

import argparse
import time

import numpy as np

import skrobot
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.planner import sqp_plan_trajectory
from skrobot.planner import SweptSphereSdfCollisionChecker
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--without-base',
    action='store_true',
    help='Solve motion planning without base.'
)
args = parser.parse_args()

# initialization stuff
np.random.seed(0)
robot_model = skrobot.models.PR2()
robot_model.init_pose()

box_center = np.array([0.9, -0.2, 0.9])
box = Box(extents=[0.7, 0.5, 0.6], with_sdf=True)
box.translate(box_center)

link_list = [
    robot_model.r_shoulder_pan_link, robot_model.r_shoulder_lift_link,
    robot_model.r_upper_arm_roll_link, robot_model.r_elbow_flex_link,
    robot_model.r_forearm_roll_link, robot_model.r_wrist_flex_link,
    robot_model.r_wrist_roll_link]
joint_list = [link.joint for link in link_list]

coll_link_list = [
    robot_model.r_upper_arm_link, robot_model.r_forearm_link,
    robot_model.r_gripper_palm_link, robot_model.r_gripper_r_finger_link,
    robot_model.r_gripper_l_finger_link]

# obtain av_start (please try both with_base=True, False)
with_base = not args.without_base
av_start = np.array([0.564, 0.35, -0.74, -0.7, -0.7, -0.17, -0.63])
if with_base:
    av_start = np.hstack([av_start, [0, 0, 0]])

# solve inverse kinematics to obtain av_goal
joint_angles = np.deg2rad([-60, 74, -70, -120, -20, -30, 180])
set_robot_config(robot_model, joint_list, joint_angles)

rarm_end_coords = skrobot.coordinates.CascadedCoords(
    parent=robot_model.r_gripper_tool_frame,
    name='rarm_end_coords')
robot_model.inverse_kinematics(
    target_coords=skrobot.coordinates.Coordinates([0.8, -0.6, 0.8], [0, 0, 0]),
    link_list=link_list,
    move_target=robot_model.rarm_end_coords, rotation_axis=True)
av_goal = get_robot_config(robot_model, joint_list, with_base=with_base)

# collision checker setup
sscc = SweptSphereSdfCollisionChecker(lambda X: box.sdf(X), robot_model)
for link in coll_link_list:
    sscc.add_collision_link(link)

# motion planning
ts = time.time()
av_seq = sqp_plan_trajectory(
    sscc, av_start, av_goal, joint_list, 10,
    safety_margin=1e-2, with_base=with_base)
print("solving time : {0} sec".format(time.time() - ts))

# visualizatoin
print("show trajectory")
viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(robot_model)
viewer.add(box)
viewer.add(Axis(pos=[0.8, -0.6, 0.8]))
sscc.add_coll_spheres_to_viewer(viewer)
viewer.show()
for av in av_seq:
    set_robot_config(robot_model, joint_list, av, with_base=with_base)
    sscc.update_color()
    viewer.redraw()
    time.sleep(1.0)

print('==> Press [q] to close window')
while not viewer.has_exit:
    time.sleep(0.1)
    viewer.redraw()
