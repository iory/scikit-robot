#!/usr/bin/env python
import argparse
import time

import skrobot
from skrobot.model import Box
from skrobot.planner import SweptSphereSdfCollisionChecker
from skrobot.planner.utils import set_robot_config
from skrobot.viewers import VIEWER_HELP
from skrobot.viewers import VIEWER_TYPES


try:
    robot_model  # noqa
except:  # noqa
    robot_model = skrobot.models.PR2()
    table = Box(extents=[0.7, 1.0, 0.05], with_sdf=True)
    table.translate([0.8, 0.0, 0.655])

    parser = argparse.ArgumentParser(
        description='Swept spheres visualization.')
    parser.add_argument(
        '--viewer', type=str,
        choices=VIEWER_TYPES, default='pyrender',
        help=VIEWER_HELP)
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help="Run in non-interactive mode (do not wait for user input)"
    )
    args = parser.parse_args()

    viewer = skrobot.viewers.create_viewer(
        args.viewer, resolution=(640, 480))

    link_idx_table = {}
    for link_idx in range(len(robot_model.link_list)):
        name = robot_model.link_list[link_idx].name
        link_idx_table[name] = link_idx

coll_link_names = [
    "r_upper_arm_link",
    "r_forearm_link",
    "r_gripper_palm_link",
    "r_gripper_r_finger_link",
    "r_gripper_l_finger_link"]

coll_link_list = [robot_model.link_list[link_idx_table[lname]]
                  for lname in coll_link_names]

move_link_names = [
    "r_shoulder_pan_link",
    "r_shoulder_lift_link",
    "r_upper_arm_roll_link",
    "r_elbow_flex_link",
    "r_forearm_roll_link",
    "r_wrist_flex_link",
    "r_wrist_roll_link"]
link_list = [robot_model.link_list[link_idx_table[lname]]
             for lname in move_link_names]
joint_list = [link.joint for link in link_list]

sscc = SweptSphereSdfCollisionChecker(table.sdf, robot_model)
for lname in coll_link_names:
    link = robot_model.link_list[link_idx_table[lname]]
    sscc.add_collision_link(link)

with_base = True
av = [0.4, 0.6] + [-0.7] * 5 + [0.1, 0.0, 0.3]
set_robot_config(robot_model, joint_list, av, with_base=with_base)
dists = sscc.update_color()
sscc.add_coll_spheres_to_viewer(viewer)
viewer.add(robot_model)
viewer.add(table)
viewer.show()

if not args.no_interactive:
    viewer.wait_until_close()
else:
    viewer.close()
time.sleep(1.0)
