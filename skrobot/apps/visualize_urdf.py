#!/usr/bin/env python

import argparse
import os.path as osp

import skrobot
from skrobot.models.urdf import RobotModelFromURDF


def main():
    parser = argparse.ArgumentParser(description='Visualize URDF')
    parser.add_argument('input_urdfpath', type=str, help='Input URDF path')
    parser.add_argument(
        '--viewer', type=str,
        choices=['trimesh', 'pyrender'], default='trimesh',
        help='Choose the viewer type: trimesh or pyrender')
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='enter interactive shell'
    )
    args = parser.parse_args()

    if args.viewer == 'trimesh':
        viewer = skrobot.viewers.TrimeshSceneViewer()
    elif args.viewer == 'pyrender':
        viewer = skrobot.viewers.PyrenderViewer()
    robot_model = RobotModelFromURDF(
        urdf_file=osp.abspath(args.input_urdfpath))
    viewer.add(robot_model)
    viewer._init_and_start_app()
    if args.interactive:
        try:
            import IPython
        except Exception as e:
            print("IPython is not installed. {}".format(e))
            return
        IPython.embed()


if __name__ == '__main__':
    main()
