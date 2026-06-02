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
        choices=['trimesh', 'pyrender', 'viser'], default='pyrender',
        help='Choose the viewer type: trimesh, pyrender or viser')
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help="Run in non-interactive mode (do not wait for user input)"
    )
    args = parser.parse_args()

    viewer = skrobot.viewers.create_viewer(
        args.viewer, resolution=(640, 480))

    robots = [
        skrobot.models.Kuka(),
        skrobot.models.Fetch(),
        skrobot.models.Nextage(),
        skrobot.models.PR2(),
        skrobot.models.Panda(),
    ]
    nrow, ncol = _get_tile_shape(len(robots))
    row, col = 2, 2

    for i in range(nrow):
        for j in range(ncol):
            try:
                robot = robots[i * ncol + j]
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
        viewer.wait_until_close()
    else:
        viewer.close()
    time.sleep(1.0)


if __name__ == '__main__':
    main()
