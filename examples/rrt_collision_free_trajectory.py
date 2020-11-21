#!/usr/bin/env python

import skrobot
import numpy as np
import time
from skrobot.utils import sdf_box

if __name__ == '__main__':
    robot_model = skrobot.models.urdf.RobotModelFromURDF(
        urdf_file=skrobot.data.pr2_urdfpath())
    robot_model.init_pose()
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    viewer.add(robot_model)
    viewer.show()

    link_idx_table = {}
    for link_idx in range(len(robot_model.link_list)):
        name = robot_model.link_list[link_idx].name
        link_idx_table[name] = link_idx

    link_names = ["r_shoulder_pan_link", "r_shoulder_lift_link",
                  "r_upper_arm_roll_link", "r_elbow_flex_link",
                  "r_forearm_roll_link", "r_wrist_flex_link",
                  "r_wrist_roll_link"]

    link_list = [robot_model.link_list[link_idx_table[lname]]
                 for lname in link_names]
    joint_list = [link.joint for link in link_list]

    def set_joint_angles(av):
        return [j.joint_angle(a) for j, a in zip(joint_list, av)]

    rarm_end_coords = skrobot.coordinates.CascadedCoords(
        parent=robot_model.r_gripper_tool_frame)
    forarm_coords = skrobot.coordinates.CascadedCoords(
        parent=robot_model.r_forearm_link)

    # set initial angle vector of rarm
    av_init = np.array([-0.7] * 7)
    av_goal = av_init + 0.5
    set_joint_angles(av_init)
    traj = robot_model.plan_trajectory_rrt(
        av_goal, link_list, rarm_end_coords, debug_plot=True)

    time.sleep(1.0)
    print("show trajectory")
    for av in traj:
        set_joint_angles(av)
        viewer.redraw()
        time.sleep(0.1)
