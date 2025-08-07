#!/usr/bin/env python

import argparse
import time

import numpy as np

import skrobot


def _get_tile_shape(num, hw_ratio=1):
    r_num = int(round(np.sqrt(num / hw_ratio)))  # weighted by wh_ratio
    c_num = 0
    while r_num * c_num < num:
        c_num += 1
    while (r_num - 1) * c_num >= num:
        r_num -= 1
    return r_num, c_num


def main():
    parser = argparse.ArgumentParser(
        description='Set viewer for skrobot.')
    parser.add_argument(
        '--viewer', type=str,
        choices=['trimesh', 'pyrender'], default='pyrender',
        help='Choose the viewer type: trimesh or pyrender')
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help="Run in non-interactive mode (do not wait for user input)"
    )
    args = parser.parse_args()

    if args.viewer == 'trimesh':
        viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    elif args.viewer == 'pyrender':
        viewer = skrobot.viewers.PyrenderViewer(resolution=(640, 480))

    robots = [
        skrobot.models.Kuka(),
        skrobot.models.Fetch(),
        skrobot.models.PR2(),
        skrobot.models.Panda(),
    ]
    nrow, ncol = _get_tile_shape(len(robots))
    row, col = 2, 2

    for i in range(nrow):
        for j in range(ncol):
            try:
                robot = robots[i * nrow + j]
            except IndexError:
                break
            plane = skrobot.model.Box(extents=(row - 0.01, col - 0.01, 0.01))
            plane.translate((row * i, col * j, -0.01))
            viewer.add(plane)
            robot.translate((row * i, col * j, 0))
            viewer.add(robot)

    viewer.set_camera(angles=[np.deg2rad(30), 0, 0])
    viewer.show()

    if not args.no_interactive:
        print('==> Press [q] to close window')
        while viewer.is_active:
            time.sleep(0.1)
            viewer.redraw()
    viewer.close()
    time.sleep(1.0)


if __name__ == '__main__':
    main()
