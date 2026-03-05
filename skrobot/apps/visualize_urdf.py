#!/usr/bin/env python

import argparse
import os.path as osp
import time

import skrobot
from skrobot.model import RobotModel


def main():
    parser = argparse.ArgumentParser(description='Visualize URDF')
    parser.add_argument('input_urdfpath', type=str, nargs='?', help='Input URDF path')
    parser.add_argument(
        '--viewer', type=str,
        choices=['trimesh', 'pyrender', 'viser'], default='pyrender',
        help='Choose the viewer type: trimesh, pyrender, or viser')
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='enter interactive shell'
    )
    parser.add_argument(
        '--ros', type=str, nargs='?', const='/robot_description', default=None,
        help='Load URDF from ROS parameter server (specify parameter name, default: /robot_description)'
    )
    parser.add_argument(
        '--show-joints', '-j',
        action='store_true',
        help='Show joint axes visualization (pyrender viewer only, press j key to toggle)'
    )
    parser.add_argument(
        '--no-joints-on-top',
        dest='joints_always_on_top',
        action='store_false',
        help='Disable always-on-top rendering for joint markers'
    )
    args = parser.parse_args()

    # Validate arguments
    if args.ros is not None and args.input_urdfpath is not None:
        parser.error("Cannot specify both URDF file path and --ros option")
    if args.ros is None and args.input_urdfpath is None:
        parser.error("Must specify either URDF file path or --ros option")

    if args.viewer == 'trimesh':
        viewer = skrobot.viewers.TrimeshSceneViewer(update_interval=0.1)
    elif args.viewer == 'pyrender':
        viewer = skrobot.viewers.PyrenderViewer(update_interval=0.1)
    elif args.viewer == 'viser':
        viewer = skrobot.viewers.ViserViewer(enable_ik=True)

    # Load robot model from ROS parameter or file
    if args.ros is not None:
        robot_model = RobotModel.from_robot_description(args.ros,
                                                        include_mimic_joints=False)
    else:
        if not osp.exists(args.input_urdfpath):
            parser.error(f"URDF file not found: {args.input_urdfpath}")
        robot_model = RobotModel.from_urdf(osp.abspath(args.input_urdfpath),
                                           include_mimic_joints=False)

    viewer.add(robot_model)

    # Configure joint axes display for pyrender viewer
    if args.viewer == 'pyrender':
        viewer.joint_axes_always_on_top = args.joints_always_on_top
        if args.show_joints:
            viewer.show_joint_axes = True
            viewer._toggle_joint_axes()

    viewer.show()
    if args.interactive:
        try:
            import IPython
        except Exception as e:
            print("IPython is not installed. {}".format(e))
            return
        IPython.embed()
    else:
        while viewer.is_active:
            viewer.redraw()
            time.sleep(0.1)
        viewer.close()


if __name__ == '__main__':
    main()
