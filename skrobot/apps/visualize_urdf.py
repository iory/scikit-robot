#!/usr/bin/env python

import argparse
import os.path as osp

import skrobot
from skrobot.models.urdf import RobotModelFromURDF


def main():
    parser = argparse.ArgumentParser(description='Visualize URDF')
    parser.add_argument('input_urdfpath', type=str, help='Input URDF path')
    args = parser.parse_args()

    viewer = skrobot.viewers.TrimeshSceneViewer()
    model = RobotModelFromURDF(urdf_file=osp.abspath(args.input_urdfpath))
    viewer.add(model)
    viewer._init_and_start_app()


if __name__ == '__main__':
    main()
