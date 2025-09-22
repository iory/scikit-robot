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
        choices=['trimesh', 'pyrender'], default='pyrender',
        help='Choose the viewer type: trimesh or pyrender')
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='enter interactive shell'
    )
    parser.add_argument(
        '--ros', type=str, nargs='?', const='/robot_description', default=None,
        help='Load URDF from ROS parameter server (specify parameter name, default: /robot_description)'
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

    # Load robot model from ROS parameter or file
    if args.ros is not None:
        robot_model = RobotModel.from_robot_description(args.ros)
    else:
        robot_model = RobotModel.from_urdf(osp.abspath(args.input_urdfpath))

    viewer.add(robot_model)
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
