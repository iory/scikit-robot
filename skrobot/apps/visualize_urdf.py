#!/usr/bin/env python

import argparse
import os.path as osp
import time

import skrobot
from skrobot.models.urdf import RobotModelFromURDF


def main():
    parser = argparse.ArgumentParser(description='Visualize URDF')
    parser.add_argument('input_urdfpath', type=str, help='Input URDF path')
    parser.add_argument(
        '--viewer', type=str,
        choices=['trimesh', 'pyrender'], default='pyrender',
        help='Choose the viewer type: trimesh or pyrender')
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='enter interactive shell'
    )
    args = parser.parse_args()

    if args.viewer == 'trimesh':
        viewer = skrobot.viewers.TrimeshSceneViewer(update_interval=0.1)
    elif args.viewer == 'pyrender':
        viewer = skrobot.viewers.PyrenderViewer(update_interval=0.1)
    robot_model = RobotModelFromURDF(
        urdf_file=osp.abspath(args.input_urdfpath))
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
