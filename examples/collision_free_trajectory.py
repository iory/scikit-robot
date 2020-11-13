#!/usr/bin/env python

import skrobot
import numpy as np
import time 
import copy
from skrobot.utils import sdf_box

if __name__ == '__main__':
    robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=skrobot.data.pr2_urdfpath())
    robot_model.init_pose()
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    viewer.add(robot_model)
    viewer.show()

    link_idx_table = {}
    for link_idx in range(len(robot_model.link_list)):
        name = robot_model.link_list[link_idx].name
        link_idx_table[name] = link_idx

    link_names = ["r_shoulder_pan_link", "r_shoulder_lift_link", \
            "r_upper_arm_roll_link", "r_elbow_flex_link", \
            "r_forearm_roll_link", "r_wrist_flex_link", "r_wrist_roll_link"]

    link_list = [robot_model.link_list[link_idx_table[name]] for name in link_names]
    joint_list = [link.joint for link in link_list]
    def set_joint_angles(av):
        return [j.joint_angle(a) for j, a in zip(joint_list, av)]

    rarm_end_coords = skrobot.coordinates.CascadedCoords(
            parent=robot_model.r_gripper_tool_frame) 
    forarm_coords = skrobot.coordinates.CascadedCoords(
            parent=robot_model.r_forearm_link) 

    # set initial angle vector of rarm
    av_init = [0.58, 0.35, -0.74, -0.70, -0.17, -0.63, 0.0]
    set_joint_angles(av_init)

    target_coords = skrobot.coordinates.Coordinates([0.7, -0.7, 1.0], [0, 0, 0])
    box_center = np.array([0.9, -0.2, 0.9])
    box_width = np.array([0.5, 0.5, 0.6])
    box = skrobot.models.Box(
        extents=box_width, face_colors=(1., 0, 0)
    )
    box.translate(box_center)
    viewer.add(box)
    margin = 0.1
    def sdf(X):
        return sdf_box(X, box_width * 0.5, box_center) - margin

    traj = robot_model.plan_trajectory(target_coords, 10, link_list, rarm_end_coords,
            [rarm_end_coords, forarm_coords], sdf,
            )

    time.sleep(1.0)
    print("show trajectory")
    for av in traj:
        set_joint_angles(av)
        viewer.redraw()
        time.sleep(1.0)
