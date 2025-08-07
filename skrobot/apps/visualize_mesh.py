#!/usr/bin/env python

import argparse
import os.path as osp
import time

import numpy as np

import skrobot
from skrobot.model import Axis
from skrobot.utils.urdf import load_meshes


def main():
    parser = argparse.ArgumentParser(description='Visualize Meshes and Origin Coordinate System')
    parser.add_argument('input_mesh_paths', type=str, nargs='+',
                        help='Input mesh paths (e.g., .stl, .3dxml, .dae, ...)')
    parser.add_argument(
        '--axis_length', type=float, default=1,
        help='Length of the coordinate system axes. Default is 1.')
    parser.add_argument(
        '--axis_radius', type=float, default=0.01,
        help='Radius (thickness) of the coordinate system axes. Default is 0.01.')
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
        viewer = skrobot.viewers.TrimeshSceneViewer()
    elif args.viewer == 'pyrender':
        viewer = skrobot.viewers.PyrenderViewer()
    for mesh_path in args.input_mesh_paths:
        if not osp.exists(mesh_path):
            print(f"Warning: Mesh file not found at {mesh_path}. Skipping.")
            continue
        try:
            print(mesh_path)
            mesh = load_meshes(osp.abspath(mesh_path))
            print(mesh)
            link = skrobot.model.Link(collision_mesh=mesh, visual_mesh=mesh)
            viewer.add(link)
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}. Skipping.")

    axis = Axis(pos=np.array([0, 0, 0]),
                rot=np.eye(3),
                axis_length=args.axis_length,
                axis_radius=args.axis_radius)
    viewer.add(axis)
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
