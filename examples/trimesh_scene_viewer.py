#!/usr/bin/env python

import argparse
import time

import numpy as np

import skrobot


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='enter interactive shell'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help="Run in non-interactive mode (do not wait for user input)"
    )
    args = parser.parse_args()

    robot = skrobot.models.Kuka()

    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))

    # base plane
    plane = skrobot.model.Box(
        extents=(2, 2, 0.01), face_colors=(0.75, 0.75, 0.75)
    )
    viewer.add(plane)

    viewer.add(robot)

    viewer.set_camera(angles=[np.deg2rad(45), 0, 0], distance=4)

    viewer.show()

    box = skrobot.model.Box(
        extents=(0.05, 0.05, 0.05), face_colors=(1., 0, 0)
    )
    box.translate((0.5, 0, 0.3))
    viewer.add(box)

    if args.interactive:
        print('''\
>>> # Usage

>>> robot.reset_manip_pose()
>>> viewer.redraw()
>>> robot.init_pose()
>>> robot.inverse_kinematics(box, rotation_axis='y')
''')

        import IPython

        IPython.embed()
    else:
        print('==> Waiting 3 seconds')
        time.sleep(3)

        print('==> Moving to reset_manip_pose')
        robot.reset_manip_pose()
        print(robot.angle_vector())
        time.sleep(1)
        viewer.redraw()

        print('==> Waiting 3 seconds')
        time.sleep(3)

        print('==> Moving to init_pose')
        robot.init_pose()
        print(robot.angle_vector())
        time.sleep(1)
        viewer.redraw()

        print('==> Waiting 3 seconds')
        time.sleep(3)

        print('==> IK to box')
        robot.reset_manip_pose()
        robot.inverse_kinematics(box, rotation_axis='y')
        print(robot.angle_vector())
        time.sleep(1)
        viewer.redraw()

        if not args.no_interactive:
            print('==> Press [q] to close window')
            while viewer.is_active:
                time.sleep(0.1)
                viewer.redraw()
        viewer.close()
        time.sleep(1.0)


if __name__ == '__main__':
    main()
