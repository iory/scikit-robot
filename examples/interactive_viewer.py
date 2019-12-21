import argparse
import time

import numpy as np
import trimesh
import trimesh.viewer

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
    args = parser.parse_args()

    robot = skrobot.models.Kuka()

    scene = trimesh.Scene()

    # base plane
    geom = trimesh.creation.box((2, 2, 0.01))
    geom.visual.face_colors = (0.75, 0.75, 0.75)
    scene.add_geometry(geom)

    viewer = skrobot.visualization.trimesh.SceneViewer(
        scene, resolution=(1280, 960)
    )

    viewer.add(robot)

    viewer.set_camera(angles=[np.deg2rad(45), 0, 0], distance=4)

    box = skrobot.models.Box(
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

        print('==> Press [q] to close window')
        while not viewer.has_exit:
            time.sleep(0.1)
            viewer.redraw()


if __name__ == '__main__':
    main()
