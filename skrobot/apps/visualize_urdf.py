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
    args = parser.parse_args()

    if args.viewer == 'trimesh':
        viewer = skrobot.viewers.TrimeshSceneViewer()
    elif args.viewer == 'pyrender':
        viewer = skrobot.viewers.PyrenderViewer()
    model = RobotModelFromURDF(urdf_file=osp.abspath(args.input_urdfpath))
    viewer.add(model)
    viewer._init_and_start_app()


if __name__ == '__main__':
    main()
