#!/usr/bin/env python

import argparse
import time

import numpy as np

import skrobot
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.model.primitives import LineString
from skrobot.planner import sqp_plan_trajectory
from skrobot.planner import SweptSphereSdfCollisionChecker
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config
from skrobot.utils.visualization import trajectory_visualization


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '-n', type=int, default=10,
    help='number of waypoints.')
parser.add_argument(
    '--without-base',
    action='store_true',
    help='Solve motion planning without base.'
)
parser.add_argument(
    '--viewer', type=str,
    choices=['trimesh', 'pyrender'], default='trimesh',
    help='Choose the viewer type: trimesh or pyrender')
parser.add_argument(
    '--no-interactive',
    action='store_true',
    help="Run in non-interactive mode (do not wait for user input)"
)
parser.add_argument(
    '--trajectory-visualization',
    action='store_true',
    help="Enable trajectory optimization visualization"
)
args = parser.parse_args()

# initialization stuff
np.random.seed(0)
robot_model = skrobot.models.PR2()
robot_model.init_pose()

# create obstacle's visual element with the corresponding sdf.
# The sdf is stored as a member variable of box.
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
    # base pose is specified by [x, y, theta]
    base_pose_start = [-0.5, 0.8, 0]
    av_start = np.hstack([av_start, base_pose_start])

# solve inverse kinematics to obtain av_goal
joint_angles = np.deg2rad([-60, 74, -70, -120, -20, -30, 180])
set_robot_config(robot_model, joint_list, joint_angles)
target_coords = skrobot.coordinates.Coordinates([0.8, -0.6, 0.8], [0, 0, 0])

rarm_end_coords = skrobot.coordinates.CascadedCoords(
    parent=robot_model.r_gripper_tool_frame,
    name='rarm_end_coords')
robot_model.inverse_kinematics(
    target_coords=target_coords,
    link_list=link_list,
    move_target=robot_model.rarm_end_coords, rotation_axis=True)
av_goal = get_robot_config(robot_model, joint_list, with_base=with_base)

# collision checker setup
sscc = SweptSphereSdfCollisionChecker(box.sdf, robot_model)
for link in coll_link_list:
    sscc.add_collision_link(link)

# visualization
print("show trajectory")
if args.viewer == 'trimesh':
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
elif args.viewer == 'pyrender':
    viewer = skrobot.viewers.PyrenderViewer(resolution=(640, 480))

viewer.add(robot_model)
viewer.add(box)
viewer.add(Axis(pos=target_coords.worldpos(), rot=target_coords.worldrot()))

sscc.add_coll_spheres_to_viewer(viewer)
viewer.show()
viewer.set_camera([0, 0, np.pi / 2.0])

# motion planning
ts = time.time()
n_waypoint = args.n

# Trajectory planning with optional visualization
if args.trajectory_visualization:
    with trajectory_visualization(
        viewer=viewer,
        robot_model=robot_model,
        joint_list=joint_list,
        sleep_time=0.3,
        with_base=with_base,
        update_every_n_iterations=1,
        debug=False,  # Disable debug for normal use
        waypoint_mode='cycle',  # Cycle through waypoints to see trajectory evolution
    ):
        av_seq = sqp_plan_trajectory(
            sscc, av_start, av_goal, joint_list, n_waypoint,
            safety_margin=5.0e-2, with_base=with_base)
else:
    av_seq = sqp_plan_trajectory(
        sscc, av_start, av_goal, joint_list, n_waypoint,
        safety_margin=5.0e-2, with_base=with_base)

print("solving time : {0} sec".format(time.time() - ts))

rarm_point_history = []
line_string = None
for av in av_seq:
    set_robot_config(robot_model, joint_list, av, with_base=with_base)
    rarm_point_history.append(rarm_end_coords.worldpos())

    # update rarm trajectory visualization
    if line_string is not None:
        viewer.delete(line_string)
    if len(rarm_point_history) > 1:
        line_string = LineString(np.array(rarm_point_history))
        viewer.add(line_string)

    sscc.update_color()
    viewer.redraw()
    time.sleep(1.0)

if not args.no_interactive:
    print('==> Press [q] to close window')
    while not viewer.has_exit:
        time.sleep(0.1)
        viewer.redraw()
